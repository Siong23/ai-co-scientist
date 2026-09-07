"""Tests for the arXiv integration.

Category-mapping logic is pure and runs offline; everything hitting the live
arXiv API is marked `network` (run with `make test-all`).
"""

import pytest

from app.tools.arxiv_search import ArxivSearchTool, build_arxiv_query, get_categories_for_field

# --- Offline: pure category-mapping logic ---


def test_known_fields_map_to_categories():
    assert len(get_categories_for_field("computer_science")) > 0
    assert len(get_categories_for_field("physics")) > 0


def test_client_page_size_matches_requested_result_limit():
    tool = ArxivSearchTool(max_results=6)

    assert tool.client.page_size == 6


def test_natural_language_query_uses_fielded_and_connected_concepts():
    query = build_arxiv_query(
        "Develop a closed-loop multi-agent AI framework to dynamically allocate "
        "5G slice bandwidth during traffic spikes"
    )

    assert 'all:"closed loop"' in query
    assert 'all:"multi agent"' in query
    assert "all:5G" in query
    assert "all:slice" in query
    assert " AND " in query
    assert "framework" not in query


def test_existing_arxiv_field_syntax_is_preserved():
    query = '(ti:"network slicing" OR abs:"network slicing") AND cat:cs.NI'

    assert build_arxiv_query(query) == query


def test_category_filter_wraps_field_aware_query(monkeypatch):
    tool = ArxivSearchTool(max_results=2)
    captured = {}

    class FakeSearch:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.tools.arxiv_search.arxiv.Search", FakeSearch)
    monkeypatch.setattr(tool.client, "results", lambda _search: [])

    tool.search_papers("network slicing latency", categories=["cs.NI", "cs.AI"])

    assert captured["query"] == "(all:network AND all:slicing AND all:latency) AND (cat:cs.NI OR cat:cs.AI)"


# --- Live arXiv API ---


@pytest.mark.network
def test_basic_search_returns_papers_with_metadata():
    tool = ArxivSearchTool(max_results=5)
    papers = tool.search_papers("machine learning", max_results=3)

    assert len(papers) > 0
    for paper in papers:
        assert paper["title"]
        assert paper["arxiv_id"]
        assert isinstance(paper["authors"], list)
        assert isinstance(paper["categories"], list)


@pytest.mark.network
def test_category_filtered_search():
    tool = ArxivSearchTool(max_results=3)
    papers = tool.search_papers("neural networks", categories=["cs.AI", "cs.LG"])
    assert len(papers) > 0


@pytest.mark.network
def test_recent_papers_search_does_not_error():
    tool = ArxivSearchTool(max_results=3)
    papers = tool.search_recent_papers("transformer", days_back=30)
    assert isinstance(papers, list)  # may legitimately be empty


@pytest.mark.network
def test_specific_paper_retrieval():
    tool = ArxivSearchTool()
    paper = tool.get_paper_details("1706.03762")  # Attention Is All You Need
    assert paper is not None
    assert "attention" in paper["title"].lower()


@pytest.mark.network
def test_trends_analysis_shape():
    tool = ArxivSearchTool()
    trends = tool.analyze_research_trends("quantum computing", days_back=60)
    assert "total_papers" in trends
    assert "top_categories" in trends
    assert "top_authors" in trends
