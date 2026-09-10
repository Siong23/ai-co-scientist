from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.evidence import EvidenceDocument
from app.lexical_retrieval import BM25PassageIndex, LexicalPassage, passage_rank_fusion
from app.paper_library import ChromaPaperLibrary, IndexIntegrityReport, PaperChunk
from app.rag_retriever import reciprocal_rank_fusion


class StaticEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _library(tmp_path: Path) -> ChromaPaperLibrary:
    library = ChromaPaperLibrary(
        embeddings=StaticEmbeddings(),
        enabled=True,
        persist_directory=tmp_path / "chroma",
        pdf_directory=tmp_path / "papers",
    )
    library.context_neighbor_window = 1
    library.context_parent_chunk_limit = 0
    return library


def _chunk(
    chunk_id: str,
    text: str,
    *,
    index: int = 0,
    dense_score: float | None = None,
    lexical_score: float | None = None,
    parent_id: str = "section-results",
    previous_chunk_id: str = "",
    next_chunk_id: str = "",
) -> PaperChunk:
    return PaperChunk(
        source_id="paper-1",
        title="Scientific Retrieval",
        page=index + 1,
        text=text,
        chunk_id=chunk_id,
        section="Results",
        section_path=("Results",),
        raw_text=text,
        retrieval_text=text,
        parent_id=parent_id,
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=next_chunk_id,
        chunk_index=index,
        chunk_count=3,
        dense_score=dense_score,
        lexical_score=lexical_score,
    )


def _records(*chunks: PaperChunk) -> dict[str, tuple[str, dict]]:
    return {
        chunk.chunk_id: (
            chunk.retrieval_text,
            {
                "source_id": chunk.source_id,
                "title": chunk.title,
                "page": chunk.page,
                "page_start": chunk.page,
                "page_end": chunk.page,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "chunk_count": chunk.chunk_count,
                "section": chunk.section,
                "section_path": " > ".join(chunk.section_path),
                "raw_text": chunk.raw_text,
                "parent_id": chunk.parent_id,
                "previous_chunk_id": chunk.previous_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
                "evidence_type": "full_text",
            },
        )
        for chunk in chunks
    }


def test_exact_token_lexical_match_survives_poor_dense_rank(tmp_path, monkeypatch):
    library = _library(tmp_path)
    dense = [
        _chunk("generic-a", "Generic evaluation prose.", dense_score=0.93),
        _chunk("generic-b", "Broad benchmark discussion.", dense_score=0.89),
        _chunk("exact", "The MMLU-Pro score increased by 4.2 points.", dense_score=0.11),
    ]
    monkeypatch.setattr(library, "_dense_search", lambda *_args: dense)
    monkeypatch.setattr(library, "_stored_source_records", lambda _source_id: _records(*dense))

    results = library.search("MMLU-Pro", ["paper-1"], top_k=2)

    assert results[0].chunk_id == "exact"
    assert results[0].lexical_score is not None
    assert {chunk.chunk_id for chunk in results} == {"exact", "generic-a"}


def test_dense_only_passage_survives_hybrid_fusion(tmp_path, monkeypatch):
    library = _library(tmp_path)
    dense_only = _chunk("dense-only", "Semantically relevant mechanism.", dense_score=0.91)
    lexical_only = _chunk("lexical-only", "Exact ZNF804A identifier.", lexical_score=3.7)
    monkeypatch.setattr(library, "_dense_search", lambda *_args: [dense_only])
    monkeypatch.setattr(library, "_lexical_search", lambda *_args: [lexical_only])

    results = library.search("ZNF804A mechanism", ["paper-1"], top_k=2)

    assert {chunk.chunk_id for chunk in results} == {"dense-only", "lexical-only"}
    assert next(chunk for chunk in results if chunk.chunk_id == "dense-only").lexical_score is None


def test_dense_and_lexical_candidate_pools_are_broadened_independently(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.dense_candidate_factor = 3
    library.lexical_candidate_factor = 4
    candidate_limits = {}
    monkeypatch.setattr(
        library,
        "_dense_search",
        lambda _query, _source_ids, candidate_k: candidate_limits.setdefault("dense", candidate_k) and [],
    )
    monkeypatch.setattr(
        library,
        "_lexical_search",
        lambda _query, _source_ids, candidate_k: candidate_limits.setdefault("lexical", candidate_k) and [],
    )

    library.search("scientific query", ["paper-1"], top_k=2)

    assert candidate_limits == {"dense": 6, "lexical": 8}


def test_passage_hybrid_fusion_is_deterministic_for_ties():
    expected = ["dense-z", "lexical-a"]

    for _attempt in range(5):
        fused = passage_rank_fusion(["dense-z"], ["lexical-a"], k=60)
        assert [result.chunk_id for result in fused] == expected


def test_duplicate_dense_and_bm25_hits_resolve_to_one_chunk(tmp_path, monkeypatch):
    library = _library(tmp_path)
    dense_hit = _chunk("shared", "TP53 response evidence.", dense_score=0.8)
    lexical_hit = _chunk("shared", "TP53 response evidence.", lexical_score=4.0)
    monkeypatch.setattr(library, "_dense_search", lambda *_args: [dense_hit])
    monkeypatch.setattr(library, "_lexical_search", lambda *_args: [lexical_hit])

    results = library.search("TP53", ["paper-1"], top_k=5)

    assert [chunk.chunk_id for chunk in results] == ["shared"]


def test_dense_lexical_and_hybrid_scores_remain_distinguishable(tmp_path, monkeypatch):
    library = _library(tmp_path)
    dense_hit = _chunk("shared", "MMLU-Pro evidence.", dense_score=0.42)
    lexical_hit = _chunk("shared", "MMLU-Pro evidence.", lexical_score=6.25)
    monkeypatch.setattr(library, "_dense_search", lambda *_args: [dense_hit])
    monkeypatch.setattr(library, "_lexical_search", lambda *_args: [lexical_hit])

    result = library.search("MMLU-Pro", ["paper-1"], top_k=1)[0]

    assert result.dense_score == 0.42
    assert result.lexical_score == 6.25
    assert result.hybrid_score == pytest.approx(2 / 61)
    assert library.last_passage_retrieval_diagnostics[0]["selected"][0] == {
        "chunk_id": "shared",
        "dense_score": 0.42,
        "lexical_score": 6.25,
        "hybrid_score": pytest.approx(2 / 61),
    }


@pytest.mark.parametrize(
    ("query", "expected_chunk_id"),
    [
        ("TP53", "gene"),
        ("Qwen2.5-72B", "model"),
        ("MMLU-Pro", "benchmark"),
        ("SWE-bench Verified", "dataset"),
        ("AUROC", "metric"),
        ("CUDA_ERROR_OUT_OF_MEMORY", "error"),
    ],
)
def test_scientific_identifiers_and_exact_names_are_lexically_retrievable(query, expected_chunk_id):
    passages = [
        LexicalPassage("gene", "TP53 and BRCA1 expression was measured."),
        LexicalPassage("model", "We evaluated Qwen2.5-72B under constrained decoding."),
        LexicalPassage("benchmark", "The reported benchmark was MMLU-Pro."),
        LexicalPassage("dataset", "SWE-bench Verified supplied the test instances."),
        LexicalPassage("metric", "AUROC was the prespecified primary metric."),
        LexicalPassage("error", "The run failed with CUDA_ERROR_OUT_OF_MEMORY."),
    ]

    results = BM25PassageIndex(passages).search(query, top_k=1)

    assert results[0].chunk_id == expected_chunk_id
    assert results[0].lexical_score > 0


def test_context_expansion_respects_added_character_budget(tmp_path, monkeypatch):
    library = _library(tmp_path)
    previous = _chunk("c0", "p" * 20, index=0, next_chunk_id="c1")
    anchor = _chunk("c1", "anchor", index=1, previous_chunk_id="c0", next_chunk_id="c2")
    following = _chunk("c2", "n" * 20, index=2, previous_chunk_id="c1")
    source_chunks = {chunk.chunk_id: chunk for chunk in (previous, anchor, following)}
    monkeypatch.setattr(library, "_source_chunks_for_expansion", lambda _source_id: source_chunks)

    expanded = library.expand_context([anchor], max_expansion_chars=20)

    assert [chunk.chunk_id for chunk in expanded] == ["c0", "c1"]
    assert sum(len(chunk.text) for chunk in expanded if chunk.is_context_expansion) <= 20


def test_previous_anchor_next_order_is_preserved(tmp_path, monkeypatch):
    library = _library(tmp_path)
    previous = _chunk("c0", "previous", index=0, next_chunk_id="c1")
    anchor = _chunk("c1", "anchor", index=1, previous_chunk_id="c0", next_chunk_id="c2")
    following = _chunk("c2", "next", index=2, previous_chunk_id="c1")
    source_chunks = {chunk.chunk_id: chunk for chunk in (previous, anchor, following)}
    monkeypatch.setattr(library, "_source_chunks_for_expansion", lambda _source_id: source_chunks)

    expanded = library.expand_context([anchor], max_expansion_chars=100)

    assert [chunk.chunk_id for chunk in expanded] == ["c0", "c1", "c2"]
    assert [chunk.context_relation for chunk in expanded] == ["previous", "selected", "next"]


def test_expansion_preserves_anchor_and_each_passage_provenance(tmp_path, monkeypatch):
    library = _library(tmp_path)
    previous = _chunk("c0", "previous", index=0, next_chunk_id="c1")
    anchor = _chunk("c1", "anchor", index=1, previous_chunk_id="c0", next_chunk_id="c2")
    following = _chunk("c2", "next", index=2, previous_chunk_id="c1")
    source_chunks = {chunk.chunk_id: chunk for chunk in (previous, anchor, following)}
    monkeypatch.setattr(library, "_source_chunks_for_expansion", lambda _source_id: source_chunks)

    expanded = library.expand_context([anchor], max_expansion_chars=100)

    selected = next(chunk for chunk in expanded if not chunk.is_context_expansion)
    neighbors = [chunk for chunk in expanded if chunk.is_context_expansion]
    assert selected.chunk_id == selected.selected_anchor_chunk_id == "c1"
    assert [chunk.chunk_id for chunk in neighbors] == ["c0", "c2"]
    assert {chunk.selected_anchor_chunk_id for chunk in neighbors} == {"c1"}


def test_parent_expansion_uses_own_chunk_id(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.context_neighbor_window = 0
    library.context_parent_chunk_limit = 1
    anchor = _chunk("c1", "anchor", index=1)
    parent_sibling = _chunk("c3", "same section context", index=3)
    source_chunks = {chunk.chunk_id: chunk for chunk in (anchor, parent_sibling)}
    monkeypatch.setattr(library, "_source_chunks_for_expansion", lambda _source_id: source_chunks)

    expanded = library.expand_context([anchor], max_expansion_chars=100)

    assert [chunk.chunk_id for chunk in expanded] == ["c1", "c3"]
    assert expanded[1].context_relation == "parent"
    assert expanded[1].selected_anchor_chunk_id == "c1"


def test_duplicate_neighbor_is_not_inserted_twice(tmp_path, monkeypatch):
    library = _library(tmp_path)
    first = _chunk("c0", "first anchor", index=0, next_chunk_id="c1")
    shared = _chunk("c1", "shared neighbor", index=1, previous_chunk_id="c0", next_chunk_id="c2")
    second = _chunk("c2", "second anchor", index=2, previous_chunk_id="c1")
    source_chunks = {chunk.chunk_id: chunk for chunk in (first, shared, second)}
    monkeypatch.setattr(library, "_source_chunks_for_expansion", lambda _source_id: source_chunks)

    expanded = library.expand_context([first, second], max_expansion_chars=100)

    assert [chunk.chunk_id for chunk in expanded] == ["c0", "c1", "c2"]
    assert sum(chunk.chunk_id == "c1" for chunk in expanded) == 1


def test_lexical_failure_degrades_to_dense_with_explicit_diagnostic(tmp_path, monkeypatch):
    library = _library(tmp_path)
    dense = _chunk("dense", "Semantic evidence.", dense_score=0.88)
    monkeypatch.setattr(library, "_dense_search", lambda *_args: [dense])
    monkeypatch.setattr(
        library,
        "_lexical_search",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("lexical store unavailable")),
    )

    results = library.search("semantic evidence", ["paper-1"], top_k=1)

    assert [chunk.chunk_id for chunk in results] == ["dense"]
    assert results[0].dense_score == 0.88
    assert results[0].lexical_score is None
    assert library.last_passage_retrieval_diagnostics[0]["lexical_available"] is False
    assert "unavailable" in library.last_passage_retrieval_diagnostics[0]["lexical_error"]


def test_enrichment_keeps_selected_and_expanded_provenance_separate(tmp_path, monkeypatch):
    library = _library(tmp_path)
    library.max_prompt_chars = 1000
    previous = _chunk("c0", "previous context", index=0, next_chunk_id="c1")
    anchor = _chunk(
        "c1",
        "selected evidence",
        index=1,
        dense_score=0.7,
        lexical_score=2.5,
        previous_chunk_id="c0",
        next_chunk_id="c2",
    )
    anchor = replace(anchor, hybrid_score=2 / 61)
    following = _chunk("c2", "following context", index=2, previous_chunk_id="c1")
    source_chunks = {chunk.chunk_id: chunk for chunk in (previous, anchor, following)}
    document = Document(
        page_content="Abstract evidence",
        metadata={"source_id": "paper-1", "title": "Scientific Retrieval", "pdf_url": "https://arxiv.org/pdf/1"},
    )
    monkeypatch.setattr(library, "has_indexed_source", lambda _source_id: True)
    monkeypatch.setattr(library, "get_index_status", lambda _source_id: "COMMITTED")
    monkeypatch.setattr(
        library,
        "verify_indexed_source",
        lambda source_id: IndexIntegrityReport(source_id, "COMMITTED", 3, 3),
    )
    monkeypatch.setattr(library, "search_many", lambda *_args, **_kwargs: [anchor])
    monkeypatch.setattr(library, "_source_chunks_for_expansion", lambda _source_id: source_chunks)

    enriched = library.enrich_documents([document], "MMLU-Pro")

    metadata = enriched[0].metadata
    assert metadata["selected_chunk_ids"] == ["c1"]
    assert metadata["expanded_chunk_ids"] == ["c0", "c2"]
    full_text_refs = [ref for ref in metadata["evidence_refs"] if ref["evidence_type"] == "full_text"]
    assert [ref["chunk_id"] for ref in full_text_refs] == ["c0", "c1", "c2"]
    assert next(ref for ref in full_text_refs if ref["chunk_id"] == "c1")["is_context_expansion"] is False
    assert {ref["selected_anchor_chunk_id"] for ref in full_text_refs} == {"c1"}


def test_provider_rrf_score_is_separate_from_passage_hybrid_score():
    evidence = EvidenceDocument(
        source_id="paper-1",
        source_family="academic",
        document_type="academic_paper",
        provider="arxiv",
        title="Paper",
        url="https://arxiv.org/abs/1",
        summary="Abstract",
    )

    fused = reciprocal_rank_fusion([[evidence]], k=60)[0]

    assert fused.provider_rrf_score == pytest.approx(1 / 61)
    assert fused.rrf_score == fused.provider_rrf_score
