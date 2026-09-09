"""Focused Phase A tests for abstract-first full-text acquisition."""

import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.agents import AbstractScreeningResult, call_llm_for_abstract_screening
from app.agents_modules.generation import GenerationAgent
from app.models import ContextMemory, ResearchGoal
from app.paper_library import ChromaPaperLibrary
from app.rag_retriever import EvidenceAspect, ProvisionalHypothesis


class GateEmbeddings(Embeddings):
    """Small deterministic embeddings keep the focused tests offline."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [1.0, 0.0] if "latency" in text.casefold() else [0.0, 1.0]


def _library(tmp_path) -> ChromaPaperLibrary:
    library = ChromaPaperLibrary(
        embeddings=GateEmbeddings(),
        enabled=True,
        persist_directory=tmp_path / "chroma",
        pdf_directory=tmp_path / "papers",
    )
    library.require_indexed_sources_for_generation = True
    library.candidate_download_limit = 1
    library.per_requirement_acquisition_limit = 1
    library.chunk_size = 500
    library.chunk_overlap = 50
    library.top_k_chunks = 2
    return library


def _paper(source_id: str = "arXiv:2609.00001") -> Document:
    return Document(
        page_content="Title: Adaptive scheduling\nAbstract: Latency measurements under traffic spikes.",
        metadata={
            "source_id": source_id,
            "source_type": "academic",
            "provider": "arxiv",
            "title": "Adaptive scheduling",
            "abstract": "Latency measurements under traffic spikes.",
            "pdf_url": f"https://arxiv.org/pdf/{source_id.removeprefix('arXiv:')}",
            "evidence_requirement_id": "req_latency",
            "reserved_requirement_ids": ["req_latency"],
        },
    )


def _screening(
    source_id: str,
    decision: str,
    *,
    full_text_needed: bool,
) -> AbstractScreeningResult:
    return AbstractScreeningResult(
        source_id=source_id,
        decision=decision,
        relevance_score={"ACCEPT": 9.0, "MAYBE": 5.0, "REJECT": 1.0}[decision],
        reason=f"{decision.title()} based on the supplied abstract.",
        evidence_requirement_ids=("req_latency",),
        provisional_hypothesis_ids=("primary_latency",),
        full_text_needed=full_text_needed,
        full_text_questions=("What latency measurements and limitations were reported?",),
    )


def _requirements():
    return (EvidenceAspect("req_latency", "latency measurements under traffic spikes"),)


def _hypotheses():
    return (
        ProvisionalHypothesis(
            "primary_latency",
            "primary",
            "Adaptive scheduling may reduce traffic-spike latency.",
            "traffic spikes",
        ),
    )


def _install_pdf_stubs(library, monkeypatch, downloads):
    def download(url, destination):
        downloads.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-phase-a")

    monkeypatch.setattr(library, "_download_pdf", download)
    monkeypatch.setattr(
        library,
        "_extract_pages",
        lambda _path: [(1, "Measured latency under traffic spikes. " * 40)],
    )


def test_structured_abstract_screening_captures_acquisition_fields():
    source_id = "arXiv:2609.00001"
    payload = json.dumps(
        {
            "screening_results": [
                {
                    "source_id": source_id,
                    "decision": "MAYBE",
                    "relevance_score": 6.5,
                    "reason": "The abstract is relevant but omits the measured result.",
                    "evidence_requirement_ids": ["req_latency"],
                    "provisional_hypothesis_ids": ["primary_latency"],
                    "full_text_needed": True,
                    "full_text_questions": ["What latency result was measured?"],
                }
            ]
        }
    )

    with patch("app.agents.call_llm", return_value=payload):
        results, error = call_llm_for_abstract_screening(
            "Reduce latency during traffic spikes.",
            [{"source_id": source_id, "title": "A", "abstract": "Relevant abstract"}],
            {source_id},
            explicit_requirements=_requirements(),
            provisional_hypotheses=_hypotheses(),
        )

    assert error is None
    assert results == (
        AbstractScreeningResult(
            source_id=source_id,
            decision="MAYBE",
            relevance_score=6.5,
            reason="The abstract is relevant but omits the measured result.",
            evidence_requirement_ids=("req_latency",),
            provisional_hypothesis_ids=("primary_latency",),
            full_text_needed=True,
            full_text_questions=("What latency result was measured?",),
        ),
    )


def test_reject_never_downloads_or_indexes(tmp_path, monkeypatch):
    library = _library(tmp_path)
    document = _paper()
    monkeypatch.setattr(
        library,
        "_download_pdf",
        lambda *_args: pytest.fail("REJECT triggered a PDF download"),
    )
    agent = GenerationAgent(paper_library=library)

    with patch(
        "app.agents_modules.generation.call_llm_for_abstract_screening",
        return_value=((_screening(document.metadata["source_id"], "REJECT", full_text_needed=False),), None),
    ):
        retained = agent._prepare_candidate_documents(
            [document],
            ResearchGoal("Reduce latency during traffic spikes."),
            _requirements(),
            _hypotheses(),
        )

    assert retained == []
    assert document.metadata["evidence_mode"] == "abstract_only"
    assert document.metadata["evidence_refs"][0]["evidence_type"] == "abstract_only"
    assert document.metadata["strict_gate_rejection_reason"] == "abstract_screen_rejected"
    assert library.get_index_status(document.metadata["source_id"]) == "MISSING"
    assert library.acquisition_funnel == {
        "full_text_requested": 0,
        "full_text_cache_hits": 0,
        "full_text_downloads": 0,
    }


def test_accept_can_download_index_and_report_funnel(tmp_path, monkeypatch):
    library = _library(tmp_path)
    document = _paper()
    downloads = []
    _install_pdf_stubs(library, monkeypatch, downloads)
    agent = GenerationAgent(paper_library=library)

    with patch(
        "app.agents_modules.generation.call_llm_for_abstract_screening",
        return_value=((_screening(document.metadata["source_id"], "ACCEPT", full_text_needed=True),), None),
    ):
        retained = agent._prepare_candidate_documents(
            [document],
            ResearchGoal("Reduce latency during traffic spikes."),
            _requirements(),
            _hypotheses(),
        )

    assert downloads == ["https://arxiv.org/pdf/2609.00001"]
    assert len(retained) == 1
    assert retained[0].metadata["full_text_indexed"] is True
    assert retained[0].metadata["full_text_chunks_used"] > 0
    assert retained[0].metadata["abstract_screen_decision"] == "ACCEPT"

    context = ContextMemory()
    context.last_generation_diagnostics = {}
    agent._persist_evidence_diagnostics(context, [document], retained)
    funnel = context.last_generation_diagnostics["evidence_funnel"]
    assert funnel["abstract_candidates"] == 1
    assert funnel["abstract_screened"] == 1
    assert funnel["abstract_accepted"] == 1
    assert funnel["abstract_maybe"] == 0
    assert funnel["abstract_rejected"] == 0
    assert funnel["full_text_requested"] == 1
    assert funnel["full_text_cache_hits"] == 0
    assert funnel["full_text_downloads"] == 1


def test_maybe_is_promoted_only_for_missing_evidence_coverage(tmp_path, monkeypatch):
    library = _library(tmp_path)
    document = _paper()
    downloads = []
    _install_pdf_stubs(library, monkeypatch, downloads)
    agent = GenerationAgent(paper_library=library)
    result = _screening(document.metadata["source_id"], "MAYBE", full_text_needed=True)

    with patch(
        "app.agents_modules.generation.call_llm_for_abstract_screening",
        return_value=((result,), None),
    ) as screen:
        initially_retained = agent._prepare_candidate_documents(
            [document],
            ResearchGoal("Reduce latency during traffic spikes."),
            _requirements(),
            _hypotheses(),
        )
        promoted = agent._prepare_candidate_documents(
            [document],
            ResearchGoal("Reduce latency during traffic spikes."),
            _requirements(),
            _hypotheses(),
            uncovered_requirement_ids=("req_latency",),
        )

    assert initially_retained == []
    assert downloads == ["https://arxiv.org/pdf/2609.00001"]
    assert len(promoted) == 1
    assert promoted[0].metadata["abstract_screen_promoted"] is True
    assert promoted[0].metadata["abstract_acquisition_reason"] == "maybe_promoted_for_uncovered_requirement"
    assert library.acquisition_funnel["full_text_requested"] == 1
    assert screen.call_count == 1


def test_verified_cached_full_text_is_reused_without_rescreen_or_download(tmp_path, monkeypatch):
    library = _library(tmp_path)
    document = _paper()
    seeded_downloads = []
    _install_pdf_stubs(library, monkeypatch, seeded_downloads)
    assert library.ensure_indexed(document) is True
    assert seeded_downloads == ["https://arxiv.org/pdf/2609.00001"]
    library.begin_run()
    monkeypatch.setattr(
        library,
        "_download_pdf",
        lambda *_args: pytest.fail("verified cached full text was downloaded again"),
    )
    agent = GenerationAgent(paper_library=library)

    with patch(
        "app.agents_modules.generation.call_llm_for_abstract_screening",
        side_effect=AssertionError("verified cache should bypass abstract screening"),
    ):
        retained = agent._prepare_candidate_documents(
            [document],
            ResearchGoal("Reduce latency during traffic spikes."),
            _requirements(),
            _hypotheses(),
        )

    assert len(retained) == 1
    assert retained[0].metadata["full_text_cache_hit"] is True
    assert library.acquisition_funnel == {
        "full_text_requested": 1,
        "full_text_cache_hits": 1,
        "full_text_downloads": 0,
    }


def test_strict_evidence_gate_still_rejects_index_without_retrieved_passage():
    document = _paper()

    class PassageLessLibrary:
        enabled = True
        require_indexed_sources_for_generation = True
        acquisition_funnel = {}
        last_evidence_diagnostics = []

        @staticmethod
        def has_indexed_source(_source_id):
            return False

        @staticmethod
        def enrich_documents(documents, _queries):
            for item in documents:
                item.metadata.update(
                    {
                        "full_text_indexed": True,
                        "full_text_chunks_used": 0,
                        "index_status": "COMMITTED",
                        "evidence_refs": [
                            {
                                "source_id": item.metadata["source_id"],
                                "evidence_type": "abstract_only",
                            }
                        ],
                    }
                )
            return list(documents)

    agent = GenerationAgent(paper_library=PassageLessLibrary())
    with patch(
        "app.agents_modules.generation.call_llm_for_abstract_screening",
        return_value=((_screening(document.metadata["source_id"], "ACCEPT", full_text_needed=True),), None),
    ):
        retained = agent._prepare_candidate_documents(
            [document],
            ResearchGoal("Reduce latency during traffic spikes."),
            _requirements(),
            _hypotheses(),
        )

    assert retained == []
    assert document.metadata["strict_gate_rejection_reason"] == "no_retrieved_full_text_passage"
