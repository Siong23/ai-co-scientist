"""Offline tests for the Semantic Scholar search integration."""

from unittest.mock import Mock, patch

from app.tools.semantic_scholar_search import SemanticScholarSearchTool


def _paper(
    paper_id: str = "abc123",
    arxiv_id: str = "",
    abstract: str | None = "A detailed research abstract.",
) -> dict:
    return {
        "paperId": paper_id,
        "title": "Test Paper",
        "abstract": abstract,
        "year": 2024,
        "publicationDate": "2024-03-01",
        "authors": [{"name": "Alice Researcher"}],
        "externalIds": {"ArXiv": arxiv_id, "DOI": "10.1234/test.2024"},
        "fieldsOfStudy": ["Medicine"],
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
    }


def _response(papers: list[dict], status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = {"data": papers}
    response.raise_for_status.return_value = None
    return response


def test_format_paper_prefers_arxiv_id_for_cross_source_deduplication():
    result = SemanticScholarSearchTool._format_paper(_paper(arxiv_id="2301.00001"))

    assert result["arxiv_id"] == "2301.00001"
    assert result["arxiv_url"] == "https://arxiv.org/abs/2301.00001"
    assert result["source"] == "semantic_scholar"


def test_format_paper_uses_semantic_scholar_id_when_arxiv_id_is_absent():
    result = SemanticScholarSearchTool._format_paper(_paper(paper_id="s2-paper"))

    assert result["arxiv_id"] == "s2:s2-paper"
    assert result["arxiv_url"].endswith("/s2-paper")


def test_search_uses_api_key_and_filters_papers_without_abstract(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
    tool = SemanticScholarSearchTool(max_results=5)

    with patch(
        "app.tools.semantic_scholar_search.requests.get",
        return_value=_response([_paper("usable"), _paper("missing", abstract=None)]),
    ) as mock_get:
        results = tool.search_papers("cell therapy", max_results=3)

    assert [paper["arxiv_id"] for paper in results] == ["s2:usable"]
    assert mock_get.call_args.kwargs["params"]["limit"] == 3
    assert mock_get.call_args.kwargs["headers"] == {"x-api-key": "test-key"}


def test_search_returns_empty_list_on_network_error():
    tool = SemanticScholarSearchTool()

    with patch(
        "app.tools.semantic_scholar_search.requests.get",
        side_effect=RuntimeError("network unavailable"),
    ):
        assert tool.search_papers("quantum biology") == []


def test_search_retries_rate_limit_then_returns_empty_list():
    tool = SemanticScholarSearchTool()

    with (
        patch(
            "app.tools.semantic_scholar_search.requests.get",
            return_value=_response([], status_code=429),
        ) as mock_get,
        patch("app.tools.semantic_scholar_search.time.sleep") as mock_sleep,
    ):
        results = tool.search_papers("CRISPR effects")

    assert results == []
    assert mock_get.call_count == 3
    assert [call.args[0] for call in mock_sleep.call_args_list] == [2.0, 4.0]
    assert tool.last_error_status == 429
