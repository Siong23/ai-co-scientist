"""Offline tests for the Tavily web-search integration."""

from unittest.mock import Mock, patch

from app.tools.tavily_search import TavilySearchTool


def _response(results: list[dict]) -> Mock:
    response = Mock()
    response.json.return_value = {"results": results}
    response.raise_for_status.return_value = None
    return response


def test_search_uses_environment_api_key_and_normalizes_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    tool = TavilySearchTool(max_results=5)
    result = {
        "url": "https://example.com/evidence",
        "title": "Evidence page",
        "content": "A concise evidence summary.",
        "published_date": "2025-01-02",
    }

    with patch(
        "app.tools.tavily_search.requests.post",
        return_value=_response([result]),
    ) as mock_post:
        results = tool.search("cell therapy", max_results=3)

    assert results[0]["arxiv_id"].startswith("tavily:")
    assert results[0]["arxiv_url"] == "https://example.com/evidence"
    assert results[0]["abstract"] == "A concise evidence summary."
    assert mock_post.call_args.kwargs["headers"] == {"Authorization": "Bearer tvly-test-key"}
    assert mock_post.call_args.kwargs["json"]["max_results"] == 3


def test_search_skips_request_without_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = TavilySearchTool()

    with patch("app.tools.tavily_search.requests.post") as mock_post:
        assert tool.search("cell therapy") == []

    mock_post.assert_not_called()


def test_search_returns_empty_list_on_network_error(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    tool = TavilySearchTool()

    with patch(
        "app.tools.tavily_search.requests.post",
        side_effect=RuntimeError("network unavailable"),
    ):
        assert tool.search("cell therapy") == []
