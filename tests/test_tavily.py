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
        papers = tool.search_papers("edge security")

    assert papers[0]["arxiv_id"].startswith("tavily:")
    assert papers[0]["abstract"] == "A relevant abstract from the web."
    assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer tvly-test"}
    assert mock_post.call_args.kwargs["json"]["max_results"] == 4


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
        papers = TavilySearchTool().search_papers("open paper")

    assert papers[0]["pdf_url"] == "https://arxiv.org/pdf/2501.01234"


def test_search_derives_pdf_url_from_arxiv_abstract_result(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    result = {
        "title": "Open paper",
        "url": "https://arxiv.org/abs/2501.01234",
        "content": "A relevant abstract from an open paper.",
    }

    with patch("app.tools.tavily_search.requests.post", return_value=_response([result])):
        papers = TavilySearchTool().search_papers("open paper")

    assert papers[0]["pdf_url"] == "https://arxiv.org/pdf/2501.01234"
