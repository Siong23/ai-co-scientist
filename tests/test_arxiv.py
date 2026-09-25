"""Tests for the arXiv integration.

Category-mapping logic is pure and runs offline; everything hitting the live
arXiv API is marked `network` (run with `make test-all`).
"""

import socket
import urllib.error

import feedparser
import pytest

from app.tools.arxiv_search import (
    _ARXIV_API_ENDPOINT,
    ArxivSearchTool,
    build_arxiv_query,
    get_categories_for_field,
)

_ATOM_ENTRY = """
  <entry>
    <id>http://arxiv.org/abs/2203.01590v1</id>
    <title>5G Network Slice Isolation</title>
    <summary>Isolation  keeps
  slices independent.</summary>
    <published>2022-03-03T09:30:00Z</published>
    <updated>2022-03-04T09:30:00Z</updated>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">10.1000/example</arxiv:doi>
    <arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">8 pages</arxiv:comment>
    <arxiv:journal_ref xmlns:arxiv="http://arxiv.org/schemas/atom">J. Netw. 2022</arxiv:journal_ref>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CR"/>
    <link href="http://arxiv.org/abs/2203.01590v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2203.01590v1" rel="related" type="application/pdf"/>
    <category term="cs.CR"/>
    <category term="cs.NI"/>
  </entry>
"""

_ATOM_ERROR = """
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
    <summary>incorrect id format for abc</summary>
  </entry>
"""


def _feed(entries):
    """Build a parsed arXiv Atom feed from raw entry fragments."""

    body = "".join(entries)
    return feedparser.parse(
        f'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'
    )


# --- Offline: pure category-mapping logic ---


def test_known_fields_map_to_categories():
    assert len(get_categories_for_field("computer_science")) > 0
    assert len(get_categories_for_field("physics")) > 0


def test_requested_result_limit_is_retained():
    tool = ArxivSearchTool(max_results=6)

    assert tool.max_results == 6


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


def test_compound_phrases_are_combined_with_or_not_and():
    """Requiring several exact phrases at once returns nothing from arXiv."""

    query = build_arxiv_query(
        "Develop an AI-driven self-optimizing 5G network architecture with energy-saving mechanisms"
    )

    assert '(all:"ai driven" OR all:"self optimizing" OR all:"energy saving")' in query
    assert 'all:"ai driven" AND all:"self optimizing"' not in query


def test_the_phrase_group_counts_as_one_concept_against_the_limit():
    query = build_arxiv_query(
        "Develop an AI-driven self-optimizing 5G network architecture with energy-saving mechanisms",
        max_concepts=2,
    )

    assert query.count(" AND ") == 1
    assert query.startswith('(all:"ai driven" OR ')


def test_existing_arxiv_field_syntax_is_preserved():
    query = '(ti:"network slicing" OR abs:"network slicing") AND cat:cs.NI'

    assert build_arxiv_query(query) == query


def test_category_filter_wraps_field_aware_query(monkeypatch):
    tool = ArxivSearchTool(max_results=2)
    captured = {}

    def fake_fetch(params, timeout=None):
        captured.update(params)
        return _feed([])

    monkeypatch.setattr("app.tools.arxiv_search._fetch_feed", fake_fetch)

    tool.search_papers("network slicing latency", categories=["cs.NI", "cs.AI"])

    assert captured["search_query"] == "(all:network AND all:slicing AND all:latency) AND (cat:cs.NI OR cat:cs.AI)"


def test_atom_entry_is_mapped_onto_the_paper_contract(monkeypatch):
    """The Atom feed must fill every field the retrieval pipeline reads."""

    tool = ArxivSearchTool(max_results=1)
    monkeypatch.setattr(
        "app.tools.arxiv_search._fetch_feed",
        lambda params, timeout=None: _feed([_ATOM_ENTRY]),
    )

    (paper,) = tool.search_papers("slice isolation")

    assert paper["arxiv_id"] == "2203.01590v1"
    assert paper["title"] == "5G Network Slice Isolation"
    assert paper["abstract"] == "Isolation keeps slices independent."
    assert paper["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert paper["primary_category"] == "cs.CR"
    assert paper["categories"] == ["cs.CR", "cs.NI"]
    assert paper["pdf_url"] == "http://arxiv.org/pdf/2203.01590v1"
    assert paper["arxiv_url"] == "https://arxiv.org/abs/2203.01590v1"
    assert paper["doi"] == "10.1000/example"
    assert paper["comment"] == "8 pages"
    assert paper["journal_ref"] == "J. Netw. 2022"
    assert paper["published"] == "2022-03-03T09:30:00Z"
    assert paper["source"] == "arxiv"


def test_rejected_request_records_its_status_for_backoff(monkeypatch):
    """A 406 must reach the caller as a status, not an unclassified error."""

    tool = ArxivSearchTool(max_results=1)

    def reject(params, timeout=None):
        raise urllib.error.HTTPError(_ARXIV_API_ENDPOINT, 406, "Not Acceptable", {}, None)

    monkeypatch.setattr("app.tools.arxiv_search._fetch_feed", reject)

    assert tool.search_papers("network slicing") == []
    assert tool.last_error_status == 406
    assert tool.last_error_kind == "provider_error"


def test_read_timeout_is_classified_as_a_timeout(monkeypatch):
    tool = ArxivSearchTool(max_results=1)

    def time_out(params, timeout=None):
        raise urllib.error.URLError(socket.timeout("timed out"))

    monkeypatch.setattr("app.tools.arxiv_search._fetch_feed", time_out)

    assert tool.search_papers("network slicing") == []
    assert tool.last_error_kind == "timeout"


def test_error_feed_is_not_reported_as_a_result(monkeypatch):
    """arXiv returns errors as a one-entry feed, which must not become a paper."""

    tool = ArxivSearchTool(max_results=1)
    monkeypatch.setattr(
        "app.tools.arxiv_search._fetch_feed",
        lambda params, timeout=None: _feed([_ATOM_ERROR]),
    )

    assert tool.search_papers("network slicing") == []
    assert tool.last_error_kind == "provider_error"


def test_requests_share_one_three_second_slot(monkeypatch):
    """arXiv's terms allow one request every three seconds for the whole client."""

    from app.tools import arxiv_search

    clock = [100.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(arxiv_search, "_next_request_at", 0.0)
    monkeypatch.setattr(arxiv_search.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(arxiv_search.time, "sleep", sleep)

    arxiv_search._wait_for_request_slot()
    clock[0] += 1.0
    arxiv_search._wait_for_request_slot()
    clock[0] += 5.0
    arxiv_search._wait_for_request_slot()

    assert sleeps == [pytest.approx(2.0)]


def test_paper_details_requests_the_identifier_without_sorting(monkeypatch):
    tool = ArxivSearchTool()
    captured = {}

    def fake_fetch(params, timeout=None):
        captured.update(params)
        return _feed([_ATOM_ENTRY])

    monkeypatch.setattr("app.tools.arxiv_search._fetch_feed", fake_fetch)

    paper = tool.get_paper_details("2203.01590")

    assert captured["id_list"] == "2203.01590"
    assert "sortBy" not in captured
    assert paper["arxiv_id"] == "2203.01590v1"


def test_fetch_feed_uses_explicit_ssl_context_without_alpn(monkeypatch):
    """Avoid sending ALPN http/1.1 extension which triggers arXiv Fastly/Varnish HTTP 406."""
    import ssl
    from unittest.mock import MagicMock

    from app.tools.arxiv_search import _fetch_feed

    captured = {}

    def fake_urlopen(request, timeout=None, context=None):
        captured["request"] = request
        captured["timeout"] = timeout
        captured["context"] = context
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<feed></feed>"
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("app.tools.arxiv_search._wait_for_request_slot", lambda: None)

    _fetch_feed({"search_query": "all:test"})

    assert isinstance(captured["context"], ssl.SSLContext)


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
