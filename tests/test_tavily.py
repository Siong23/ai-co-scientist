"""Offline tests for the Tavily web-search integration."""

from unittest.mock import Mock, patch

from app.tools.tavily_search import TavilySearchTool


def _response(results: list[dict], status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = {"results": results}
    response.raise_for_status.return_value = None
    return response


def test_search_requires_an_api_key():
    assert TavilySearchTool().search_papers("edge security") == []


def test_search_normalizes_web_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    tool = TavilySearchTool(max_results=4)
    result = {
        "title": "Edge security study",
        "url": "https://example.com/paper",
        "content": "A relevant abstract from the web.",
        "published_date": "2025-01-01",
    }
    with patch("app.tools.tavily_search.requests.post", return_value=_response([result])) as mock_post:
        evidence = tool.search("edge security")

    assert evidence[0]["source_id"].startswith("web:")
    assert evidence[0]["source_type"] == "web"
    assert evidence[0]["provider"] == "tavily"
    assert evidence[0]["url"] == "https://example.com/paper"
    assert evidence[0]["domain"] == "example.com"
    assert evidence[0]["snippet"] == "A relevant abstract from the web."
    assert evidence[0]["content"] == ""
    assert evidence[0]["content_extracted"] is False
    assert "arxiv_id" not in evidence[0]
    assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer tvly-test"}
    assert mock_post.call_args.kwargs["json"] == {
        "query": "edge security",
        "search_depth": "advanced",
        "chunks_per_source": 3,
        "max_results": 4,
        "include_answer": False,
        "include_raw_content": False,
    }


def test_extract_returns_bounded_content_keyed_by_canonical_url(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    tool = TavilySearchTool(extract_chunks_per_source=2)
    extracted_result = {
        "url": "https://example.com/guidance?utm_source=search",
        "raw_content": "<chunk 1> relevant evidence",
    }

    with patch(
        "app.tools.tavily_search.requests.post",
        return_value=_response([extracted_result]),
    ) as mock_post:
        extracted = tool.extract(
            ["https://example.com/guidance"],
            query="MEC security constraints",
        )

    assert extracted == {
        "https://example.com/guidance": "<chunk 1> relevant evidence"
    }
    assert mock_post.call_args.args[0] == "https://api.tavily.com/extract"
    assert mock_post.call_args.kwargs["json"] == {
        "urls": ["https://example.com/guidance"],
        "query": "MEC security constraints",
        "chunks_per_source": 2,
        "extract_depth": "basic",
        "format": "text",
    }


def test_search_applies_planner_domain_freshness_and_news_filters(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    tool = TavilySearchTool(max_results=6)
    result = {
        "title": "Recent model release",
        "url": "https://qwenlm.github.io/blog/release",
        "content": "A current release announcement.",
    }

    with patch(
        "app.tools.tavily_search.requests.post",
        return_value=_response([result]),
    ) as mock_post:
        tool.search(
            "Qwen model release",
            include_domains=("QwenLM.github.io", "qwenlm.github.io"),
            time_range="year",
            topic="news",
        )

    assert mock_post.call_args.kwargs["json"] == {
        "query": "Qwen model release",
        "search_depth": "advanced",
        "chunks_per_source": 3,
        "max_results": 6,
        "include_answer": False,
        "include_raw_content": False,
        "include_domains": ["qwenlm.github.io"],
        "time_range": "year",
        "topic": "news",
    }


def test_search_records_rate_limit(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    tool = TavilySearchTool()
    with patch("app.tools.tavily_search.requests.post", return_value=_response([], status_code=429)):
        assert tool.search_papers("edge security") == []

    assert tool.last_error_status == 429


def test_search_preserves_direct_pdf_url_for_full_text_indexing(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    result = {
        "title": "Open paper",
        "url": "https://arxiv.org/pdf/2501.01234",
        "content": "A relevant abstract from an open paper.",
    }

    with patch("app.tools.tavily_search.requests.post", return_value=_response([result])):
        evidence = TavilySearchTool().search("open paper")

    assert evidence[0]["pdf_url"] == "https://arxiv.org/pdf/2501.01234"
    assert evidence[0]["page_type"] == "pdf"


def test_search_derives_pdf_url_from_arxiv_abstract_result(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    result = {
        "title": "Open paper",
        "url": "https://arxiv.org/abs/2501.01234",
        "content": "A relevant abstract from an open paper.",
    }

    with patch("app.tools.tavily_search.requests.post", return_value=_response([result])):
        evidence = TavilySearchTool().search("open paper")

    assert evidence[0]["pdf_url"] == "https://arxiv.org/pdf/2501.01234"


def test_historical_search_papers_alias_returns_web_schema(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    result = {
        "title": "Web guidance",
        "url": "https://example.com/guidance?utm_source=test",
        "content": "Current technical guidance.",
    }

    with patch("app.tools.tavily_search.requests.post", return_value=_response([result])):
        evidence = TavilySearchTool().search_papers("guidance")

    assert evidence[0]["canonical_url"] == "https://example.com/guidance"
    assert "abstract" not in evidence[0]
