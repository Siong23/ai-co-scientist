"""Offline tests for SemanticScholarSearchTool.

Schema / formatting logic and error-handling run fully offline.
Live API calls are marked ``@pytest.mark.network`` and excluded from
``make test`` (run with ``make test-all``).
"""

import pytest
from unittest.mock import Mock, patch

from app.tools.semantic_scholar_search import SemanticScholarSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _s2_paper(
    paper_id: str = "abc123",
    arxiv_id: str = "",
    abstract: str = "A detailed research abstract.",
    title: str = "Test Paper",
) -> dict:
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "year": 2024,
        "publicationDate": "2024-03-01",
        "authors": [{"name": "Alice Researcher"}, {"name": "Bob Scientist"}],
        "externalIds": {
            "ArXiv": arxiv_id,
            "DOI": "10.1234/test.2024",
        },
        "fieldsOfStudy": ["Medicine", "Biology"],
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
    }


def _mock_response(papers: list, status_code: int = 200) -> Mock:
    mock = Mock()
    mock.status_code = status_code
    mock.json.return_value = {"data": papers}
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Schema / formatting – offline
# ---------------------------------------------------------------------------


def test_format_paper_uses_arxiv_id_for_dedup_when_available():
    """When a paper has an arXiv ID, it should be the dedup key."""
    tool = SemanticScholarSearchTool()
    result = tool._format_paper(_s2_paper(paper_id="xyz", arxiv_id="2301.00001"))
    assert result["arxiv_id"] == "2301.00001"
    assert result["arxiv_url"] == "https://arxiv.org/abs/2301.00001"
    assert result["source"] == "semantic_scholar"


def test_format_paper_uses_s2_prefix_without_arxiv_id():
    """Papers not on arXiv get a stable ``s2:`` prefixed dedup ID."""
    tool = SemanticScholarSearchTool()
    result = tool._format_paper(_s2_paper(paper_id="biomedxyz99", arxiv_id=""))
    assert result["arxiv_id"] == "s2:biomedxyz99"
    assert "semanticscholar.org" in result["arxiv_url"]
    assert result["source"] == "semantic_scholar"


def test_format_paper_fields_populated():
    """All schema fields required by the RAG pipeline are present."""
    tool = SemanticScholarSearchTool()
    result = tool._format_paper(_s2_paper())
    required_keys = {
        "arxiv_id", "entry_id", "title", "abstract", "authors",
        "primary_category", "categories", "published", "doi",
        "pdf_url", "arxiv_url", "source",
    }
    assert required_keys.issubset(result.keys())
    assert isinstance(result["authors"], list)
    assert isinstance(result["categories"], list)


def test_format_paper_empty_arxiv_id_string_uses_s2_prefix():
    """An empty string for ArXiv ID (not None) still triggers s2: prefix."""
    tool = SemanticScholarSearchTool()
    result = tool._format_paper({**_s2_paper(), "externalIds": {"ArXiv": ""}})
    assert result["arxiv_id"].startswith("s2:")


# ---------------------------------------------------------------------------
# Network-error / rate-limit handling – offline via mocks
# ---------------------------------------------------------------------------


def test_search_papers_returns_empty_list_on_network_error():
    tool = SemanticScholarSearchTool()
    with patch(
        "app.tools.semantic_scholar_search.requests.get",
        side_effect=Exception("Network error"),
    ):
        results = tool.search_papers("quantum biology")
    assert results == []


def test_search_papers_returns_empty_list_on_429():
    tool = SemanticScholarSearchTool()
    with patch(
        "app.tools.semantic_scholar_search.requests.get",
        return_value=_mock_response([], status_code=429),
    ):
        results = tool.search_papers("CRISPR off-target effects")
    assert results == []


def test_search_papers_filters_out_papers_without_abstract():
    """Papers with None or missing abstract must be excluded."""
    tool = SemanticScholarSearchTool()
    papers_from_api = [
        _s2_paper("p1", abstract="Has a real abstract."),
        {**_s2_paper("p2"), "abstract": None},
        {**_s2_paper("p3"), "abstract": ""},
    ]
    with patch(
        "app.tools.semantic_scholar_search.requests.get",
        return_value=_mock_response(papers_from_api),
    ):
        results = tool.search_papers("sickle cell therapy")
    # Only p1 has an abstract; p2 (None) and p3 ("") are filtered at API level
    # Note: empty string is falsy so also filtered.
    assert len(results) == 1
    assert results[0]["arxiv_id"] == "s2:p1"


def test_search_papers_sends_correct_query_params():
    """The correct query parameters and limit must be sent to the API."""
    tool = SemanticScholarSearchTool(max_results=5)
    with patch(
        "app.tools.semantic_scholar_search.requests.get",
        return_value=_mock_response([]),
    ) as mock_get:
        tool.search_papers("CAR-T cell immunotherapy", max_results=3)

    call_kwargs = mock_get.call_args
    params = call_kwargs.kwargs.get("params") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["params"]
    assert params["query"] == "CAR-T cell immunotherapy"
    assert params["limit"] == 3


def test_search_papers_uses_instance_max_results_when_not_overridden():
    tool = SemanticScholarSearchTool(max_results=7)
    with patch(
        "app.tools.semantic_scholar_search.requests.get",
        return_value=_mock_response([]),
    ) as mock_get:
        tool.search_papers("stem cell differentiation")

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or {}
    assert params.get("limit") == 7


# ---------------------------------------------------------------------------
# Live API – network marker, excluded from make test
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_search_returns_academic_papers_with_abstracts():
    tool = SemanticScholarSearchTool(max_results=3)
    results = tool.search_papers("CRISPR-Cas9 off-target effects hematopoietic stem cells")
    assert len(results) > 0
    for r in results:
        assert r["title"], "title must not be empty"
        assert r["abstract"], "abstract must not be empty"
        assert r["source"] == "semantic_scholar"
        assert r["arxiv_id"], "arxiv_id (dedup key) must not be empty"


@pytest.mark.network
def test_live_search_dedup_key_format():
    """arXiv papers from S2 should get plain arXiv IDs; others get s2: prefix."""
    tool = SemanticScholarSearchTool(max_results=5)
    results = tool.search_papers("attention is all you need transformer")
    ids = [r["arxiv_id"] for r in results]
    # At least one result should be a real arXiv paper (Vaswani et al.)
    assert any(not rid.startswith("s2:") for rid in ids), (
        "Expected at least one result with a real arXiv ID"
    )
