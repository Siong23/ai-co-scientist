"""Offline tests for the OpenAlex scholarly search provider."""

from unittest.mock import Mock

import pytest
import requests

from app.config import config
from app.evidence import evidence_from_result
from app.tools import openalex_search
from app.tools.openalex_search import OpenAlexSearchTool, reconstruct_abstract


@pytest.fixture(autouse=True)
def no_request_pacing(monkeypatch):
    monkeypatch.setattr(openalex_search, "_wait_for_request_slot", lambda: None)


def _work(**overrides):
    work = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1109/example.2024.1",
        "title": "Energy saving for 5G radio access networks",
        "publication_date": "2024-05-01",
        "publication_year": 2024,
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "primary_location": {
            "landing_page_url": "https://ieeexplore.ieee.org/document/1",
            "pdf_url": None,
            "source": {"display_name": "IEEE Access"},
        },
        "best_oa_location": {"pdf_url": "https://ieeexplore.ieee.org/stamp/1.pdf"},
        "locations": [],
        "abstract_inverted_index": {"Cell": [0], "sleep": [1], "saves": [2], "energy.": [3]},
    }
    work.update(overrides)
    return work


def _response(status=200, payload=None, headers=None):
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    response.text = "" if payload is None else str(payload)
    response.json.return_value = payload if payload is not None else {}
    return response


def test_inverted_abstract_is_rebuilt_in_word_order():
    assert reconstruct_abstract({"energy": [2], "Cell": [0], "sleep": [1, 3]}) == "Cell sleep energy sleep"
    assert reconstruct_abstract(None) == ""


def test_journal_work_keeps_its_doi_and_open_access_pdf():
    paper = OpenAlexSearchTool._format_work(_work())

    assert paper["arxiv_id"] == "doi:10.1109/example.2024.1"
    assert paper["doi"] == "10.1109/example.2024.1"
    assert paper["abstract"] == "Cell sleep saves energy."
    assert paper["pdf_url"] == "https://ieeexplore.ieee.org/stamp/1.pdf"
    assert paper["venue"] == "IEEE Access"
    assert paper["authors"] == ["Ada Lovelace"]
    assert paper["published"] == "2024-05-01"
    assert paper["source"] == "openalex"


def test_arxiv_preprint_merges_with_the_same_paper_found_through_arxiv():
    """The DataCite DOI would otherwise keep two copies of one preprint apart."""

    paper = OpenAlexSearchTool._format_work(
        _work(
            doi="https://doi.org/10.48550/arxiv.2411.03326",
            primary_location={"landing_page_url": "http://arxiv.org/abs/2411.03326", "source": {}},
            best_oa_location={"pdf_url": "https://arxiv.org/pdf/2411.03326"},
        )
    )
    openalex_evidence = evidence_from_result(paper, "openalex", "academic")
    arxiv_evidence = evidence_from_result(
        {"arxiv_id": "2411.03326v2", "title": "Same paper", "abstract": "Text.", "source": "arxiv"},
        "arxiv",
        "academic",
    )

    assert paper["doi"] is None
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2411.03326"
    assert openalex_evidence.source_id == "arXiv:2411.03326"
    assert openalex_evidence.canonical_key == arxiv_evidence.canonical_key


def test_journal_work_with_an_arxiv_copy_downloads_from_arxiv():
    paper = OpenAlexSearchTool._format_work(_work(best_oa_location={"pdf_url": "https://arxiv.org/pdf/2305.07117"}))

    assert paper["doi"] == "10.1109/example.2024.1"
    assert paper["arxiv_id"] == "2305.07117"
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2305.07117"


def test_work_without_an_abstract_is_skipped():
    assert OpenAlexSearchTool._format_work(_work(abstract_inverted_index=None)) is None


def test_search_uses_semantic_search_and_sends_the_key_only_as_a_header(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "offline-test-key")
    get = Mock(return_value=_response(payload={"results": [_work(), _work(abstract_inverted_index=None)]}))
    monkeypatch.setattr(openalex_search.requests, "get", get)

    papers = OpenAlexSearchTool(max_results=80).search_papers("  5G  energy saving ")

    assert [paper["title"] for paper in papers] == ["Energy saving for 5G radio access networks"]
    params = get.call_args.kwargs["params"]
    assert params["search.semantic"] == "5G energy saving"
    assert params["per-page"] == 50
    assert "offline-test-key" not in str(params)
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer offline-test-key"


def test_search_without_a_key_sends_no_authorization(monkeypatch):
    get = Mock(return_value=_response(payload={"results": []}))
    monkeypatch.setattr(openalex_search.requests, "get", get)

    OpenAlexSearchTool().search_papers("5G handover")

    assert "Authorization" not in get.call_args.kwargs["headers"]


@pytest.mark.parametrize(
    "remaining,kind",
    [("0", "quota_or_plan_rejection"), ("0.05", "rate_limited"), (None, "rate_limited")],
)
def test_spent_daily_budget_is_told_apart_from_a_burst(monkeypatch, remaining, kind):
    headers = {} if remaining is None else {"X-RateLimit-Remaining-USD": remaining}
    monkeypatch.setattr(openalex_search.requests, "get", Mock(return_value=_response(429, headers=headers)))
    tool = OpenAlexSearchTool()

    assert tool.search_papers("5G slicing") == []
    assert tool.last_error_status == 429
    assert tool.last_error_kind == kind


def test_timeout_is_classified(monkeypatch):
    monkeypatch.setattr(openalex_search.requests, "get", Mock(side_effect=requests.Timeout("slow")))
    tool = OpenAlexSearchTool()

    assert tool.search_papers("5G slicing") == []
    assert tool.last_error_kind == "timeout"


def test_retriever_searches_openalex_when_enabled(monkeypatch):
    from app.rag_retriever import ResearchRetriever

    monkeypatch.setitem(config.setdefault("openalex", {}), "enabled", True)
    retriever = ResearchRetriever()

    assert ("OpenAlex", "openalex", retriever.openalex) in retriever._academic_sources()
    assert isinstance(retriever.openalex, OpenAlexSearchTool)


def test_offline_retrievers_leave_openalex_out():
    from app.rag_retriever import ResearchRetriever

    assert ResearchRetriever().openalex is None
