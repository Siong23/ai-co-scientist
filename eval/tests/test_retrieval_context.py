"""Retrieval-context extraction must never turn citation metadata into evidence."""

import pytest
from rubrics.errors import LLMEvaluationError
from rubrics.retrieval_context import (
    METADATA_ONLY_FIELDS,
    MIN_PASSAGE_CHARS,
    PASSAGE_TEXT_FIELD,
    SOURCE_TEXT_FIELDS,
    extract_retrieval_context,
)

PASSAGE_A = "Encapsulation barrier results. " * 20
PASSAGE_B = "Damp-heat ageing measurements. " * 20
ABSTRACT = "We report a hydrophobic encapsulation study. " * 20


def metadata_only_source(source_id="doi:example"):
    """A citation stub: it identifies a paper but says nothing about it."""
    return {
        "source_id": source_id,
        "title": "Humidity barrier study",
        "doi": "10.1000/example",
        "url": "https://example.org/paper",
        "authors": ["Ng, A.", "Tan, B."],
        "published": "2026-01-01",
        "abstract": None,
        "summary": None,
        "content": "",
        "evidence_refs": [
            {
                "source_id": source_id,
                "chunk_id": f"abstract:{source_id}",
                "section": "Abstract",
                "evidence_type": "abstract_only",
            }
        ],
    }


def sourced_evidence(source_id="doi:example", passages=(PASSAGE_A, PASSAGE_B)):
    """A source persisted with real retrieved chunk text plus its abstract."""
    source = metadata_only_source(source_id)
    source["abstract"] = ABSTRACT
    source["summary"] = ABSTRACT
    source["evidence_refs"] += [
        {
            "source_id": source_id,
            "chunk_id": f"chunk-{index}",
            "evidence_type": "full_text",
            "page": index,
            PASSAGE_TEXT_FIELD: passage,
        }
        for index, passage in enumerate(passages)
    ]
    return source


def test_metadata_fields_are_never_read_as_evidence_text():
    assert METADATA_ONLY_FIELDS.isdisjoint(SOURCE_TEXT_FIELDS)
    assert PASSAGE_TEXT_FIELD not in METADATA_ONLY_FIELDS


def test_no_sources_is_not_substantive():
    context = extract_retrieval_context([])

    assert context.is_substantive is False
    assert context.as_list() is None
    assert "no persisted evidence sources" in context.reason
    assert context.summary()["passage_count"] == 0


def test_metadata_only_sources_are_not_substantive():
    context = extract_retrieval_context([metadata_only_source(), metadata_only_source("doi:other")])

    assert context.is_substantive is False
    assert context.as_list() is None
    assert "only citation metadata" in context.reason
    assert context.summary() == {
        "substantive": False,
        "evidence_source_count": 2,
        "sources_with_text": 0,
        "passage_count": 0,
        "reason": context.reason,
    }


def test_each_persisted_passage_stays_a_separate_string():
    context = extract_retrieval_context([sourced_evidence()])

    assert context.is_substantive is True
    assert context.as_list() == [PASSAGE_A.strip(), PASSAGE_B.strip(), ABSTRACT.strip()]
    assert context.summary()["sources_with_text"] == 1


def test_abstract_persisted_twice_is_counted_once():
    source = sourced_evidence(passages=())
    assert source["abstract"] == source["summary"]

    context = extract_retrieval_context([source])

    assert context.as_list() == [ABSTRACT.strip()]


def test_short_passages_are_not_treated_as_evidence():
    source = sourced_evidence(passages=("x" * (MIN_PASSAGE_CHARS - 1),))
    source["abstract"] = None
    source["summary"] = None

    context = extract_retrieval_context([source])

    assert context.is_substantive is False


def test_mixed_sources_keep_only_the_substantive_one():
    context = extract_retrieval_context(
        [metadata_only_source("doi:stub"), sourced_evidence("doi:full", passages=(PASSAGE_A,))]
    )

    assert context.as_list() == [PASSAGE_A.strip(), ABSTRACT.strip()]
    assert context.summary() == {
        "substantive": True,
        "evidence_source_count": 2,
        "sources_with_text": 1,
        "passage_count": 2,
        "reason": None,
    }


def test_malformed_evidence_sources_are_rejected():
    with pytest.raises(LLMEvaluationError, match="list of objects"):
        extract_retrieval_context(["not-an-object"])
