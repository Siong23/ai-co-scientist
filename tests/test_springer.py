"""Offline tests for the Springer Nature API search integration."""

from unittest.mock import Mock, patch

from app.tools.springer_search import SpringerSearchTool


def _springer_record(
    doi: str = "10.1007/test-doi",
    title: str = "Springer Test Paper",
    abstract: str | dict | None = "A detailed Springer research abstract.",
) -> dict:
    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "publicationDate": "2024-05-15",
        "publicationName": "Nature Communications",
        "creators": [{"creator": "Dr. Alice Springer"}],
        "url": [
            {"format": "html", "platform": "web", "value": f"https://doi.org/{doi}"},
            {"format": "pdf", "platform": "web", "value": f"https://example.com/{doi}.pdf"},
        ],
    }


def _response(records: list[dict], status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = {"records": records}
    response.raise_for_status.return_value = None
    return response


def test_format_paper_normalizes_springer_record():
    result = SpringerSearchTool._format_paper(_springer_record())

    assert result["arxiv_id"] == "springer:10.1007/test-doi"
    assert result["title"] == "Springer Test Paper"
    assert result["abstract"] == "A detailed Springer research abstract."
    assert result["authors"] == ["Dr. Alice Springer"]
    assert result["journal_ref"] == "Nature Communications"
    assert result["pdf_url"] == "https://example.com/10.1007/test-doi.pdf"
    assert result["source"] == "springer"


def test_extract_abstract_handles_dict_structure():
    record = _springer_record(abstract={"p": "Structured paragraph abstract."})
    abstract = SpringerSearchTool._extract_abstract(record)
    assert abstract == "Structured paragraph abstract."


def test_search_skips_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("SPRINGER_API_KEY", raising=False)
    monkeypatch.delenv("SPRINGER_OPEN_ACCESS_API_KEY", raising=False)
    monkeypatch.delenv("SPRINGER_META_API_KEY", raising=False)
    tool = SpringerSearchTool()

    with patch("app.tools.springer_search.requests.get") as mock_get:
        assert tool.search_papers("cell therapy") == []

    mock_get.assert_not_called()


def test_search_uses_api_key_and_returns_formatted_papers(monkeypatch):
    monkeypatch.delenv("SPRINGER_OPEN_ACCESS_API_KEY", raising=False)
    monkeypatch.delenv("SPRINGER_META_API_KEY", raising=False)
    monkeypatch.setenv("SPRINGER_API_KEY", "springer-test-key")
    tool = SpringerSearchTool(max_results=5)

    with patch(
        "app.tools.springer_search.requests.get",
        return_value=_response([_springer_record()]),
    ) as mock_get:
        results = tool.search_papers("immunotherapy", max_results=3)

    assert len(results) == 1
    assert results[0]["arxiv_id"] == "springer:10.1007/test-doi"
    assert mock_get.call_args.kwargs["params"] == {
        "q": "immunotherapy",
        "api_key": "springer-test-key",
        "p": 3,
    }


def test_search_returns_empty_list_on_error(monkeypatch):
    monkeypatch.setenv("SPRINGER_API_KEY", "springer-test-key")
    tool = SpringerSearchTool()

    with patch(
        "app.tools.springer_search.requests.get",
        side_effect=RuntimeError("network error"),
    ):
        assert tool.search_papers("quantum biology") == []
