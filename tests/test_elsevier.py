"""Offline tests for the Elsevier Scopus API search integration."""

from unittest.mock import Mock, patch

from app.tools.elsevier_search import ElsevierSearchTool


def _entry() -> dict:
    return {
        "eid": "2-s2.0-85123456789",
        "dc:title": "Elsevier Test Paper",
        "dc:description": "A detailed Scopus research abstract.",
        "dc:creator": "Alice Researcher, Bob Scientist",
        "prism:doi": "10.1016/j.test.2025.001",
        "prism:coverDate": "2025-01-15",
        "prism:publicationName": "Journal of Test Research",
        "link": [{"@href": "https://doi.org/10.1016/j.test.2025.001"}],
    }


def _response(entries: list[dict], status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = {"search-results": {"entry": entries}}
    response.raise_for_status.return_value = None
    return response


def test_search_skips_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("ELSEVIER_API_KEY", raising=False)
    with patch("app.tools.elsevier_search.requests.get") as mock_get:
        assert ElsevierSearchTool().search_papers("cell therapy") == []
    mock_get.assert_not_called()


def test_search_uses_api_key_and_normalizes_scopus_entries(monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "elsevier-test-key")
    tool = ElsevierSearchTool(max_results=5)
    with patch("app.tools.elsevier_search.requests.get", return_value=_response([_entry()])) as mock_get:
        papers = tool.search_papers("immunotherapy", max_results=3)

    assert papers[0]["arxiv_id"] == "elsevier:2-s2.0-85123456789"
    assert papers[0]["authors"] == ["Alice Researcher", "Bob Scientist"]
    assert papers[0]["arxiv_url"] == "https://doi.org/10.1016/j.test.2025.001"
    assert mock_get.call_args.kwargs["params"] == {"query": "immunotherapy", "count": 3}
    assert mock_get.call_args.kwargs["headers"] == {
        "Accept": "application/json",
        "X-ELS-APIKey": "elsevier-test-key",
    }


def test_search_records_rate_limit(monkeypatch):
    monkeypatch.setenv("ELSEVIER_API_KEY", "elsevier-test-key")
    response = _response([], status_code=429)
    response.raise_for_status.side_effect = RuntimeError("rate limit")
    with patch("app.tools.elsevier_search.requests.get", return_value=response):
        assert ElsevierSearchTool().search_papers("quantum biology") == []
