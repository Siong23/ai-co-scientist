import json
from unittest.mock import patch

import numpy as np

from app.agents import (
    EvidenceAspect,
    call_llm_for_full_text_evidence_coverage,
    call_llm_for_literature_synthesis,
)
from app.agents import (
    call_llm_for_grounded_hypothesis_audit as call_llm_for_hypothesis_audit,
)
from app.agents_modules.generation_helpers import classify_numeric_specificity
from app.rag_retriever import SharedSentenceTransformerEmbeddings

SOURCE_ID = "arXiv:1111.1111"


def _scores(**overrides):
    values = {
        "evidence_validity": 8,
        "claim_evidence_entailment": 8,
        "novelty_against_prior_art": 8,
        "cross_paper_synthesis": 8,
        "mechanistic_plausibility": 8,
        "operational_falsifiability": 8,
        "unsupported_specificity": 8,
    }
    values.update(overrides)
    return values


def _audit_response(
    final_hypothesis,
    *,
    remaining_claims=None,
    remaining_numbers=None,
    scores=None,
    claim_assessments=None,
):
    return json.dumps(
        {
            "audited_hypotheses": [
                {
                    "candidate_index": 0,
                    "scores": scores or _scores(),
                    "claim_assessments": claim_assessments or [],
                    "closest_prior_art": [],
                    "draft_unsupported_claims": [],
                    "draft_unsupported_numbers": [],
                    "remaining_unsupported_claims": remaining_claims or [],
                    "remaining_unsupported_numbers": remaining_numbers or [],
                    "verdict": "accept",
                    "revision_instruction": "",
                    "final_hypothesis": final_hypothesis,
                }
            ]
        }
    )


def _hypothesis(hypothesis, rationale="The cited evidence grounds the premise.", feasibility="Compare baselines."):
    return {
        "title": "Candidate",
        "hypothesis": hypothesis,
        "rationale": rationale,
        "feasibility": feasibility,
        "source_ids": [SOURCE_ID],
    }


def test_number_component_transplantation_is_a_hard_failure():
    final = _hypothesis("PRB assignment executes within 25 μs.")
    evidence = "The MLP inference stage executes in 10–25 μs."
    with patch(
        "app.agents.call_llm",
        return_value=_audit_response(
            final,
            remaining_claims=[
                "The 25 μs measurement applies to MLP inference, not PRB assignment."
            ],
        ),
    ):
        audits, error = call_llm_for_hypothesis_audit(
            "Reduce scheduling latency.",
            [final],
            evidence,
            {SOURCE_ID},
        )

    assert error is None
    assert audits[0]["passed"] is False
    assert "unsupported claims" in " ".join(audits[0]["audit_report"]["hard_failures"])


def test_atomic_claim_entailment_requires_an_exact_known_chunk():
    chunk_id = "results-chunk"
    evidence_text = "The evaluated method reduced median latency relative to the baseline."
    evidence_ref = {
        "source_id": SOURCE_ID,
        "chunk_id": chunk_id,
        "section": "Results",
        "page": 7,
        "evidence_type": "full_text",
    }
    final = {
        **_hypothesis("A new controller will be evaluated against the baseline."),
        "evidence_refs": [chunk_id],
    }
    claims = [
        {
            "claim_id": "claim_1",
            "claim": "The evaluated method reduced median latency relative to the baseline.",
            "source_id": SOURCE_ID,
            "chunk_ids": [chunk_id],
            "evidence_spans": [evidence_text],
            "support_status": "entailed",
            "reason": "The subject, metric, and comparator match.",
        }
    ]
    with patch(
        "app.agents.call_llm",
        return_value=_audit_response(final, claim_assessments=claims),
    ):
        audits, error = call_llm_for_hypothesis_audit(
            "Reduce latency.",
            [final],
            evidence_text,
            {SOURCE_ID},
            available_evidence_refs={chunk_id: evidence_ref},
        )

    assert error is None
    assert audits[0]["passed"] is True
    assessment = audits[0]["audit_report"]["claim_assessments"][0]
    assert assessment["support_status"] == "entailed"
    assert assessment["section"] == "Results"
    assert assessment["page"] == 7


def test_atomic_claim_with_invented_chunk_is_rejected():
    final = {
        **_hypothesis("A new controller will be evaluated against the baseline."),
        "evidence_refs": ["known-chunk"],
    }
    claims = [
        {
            "claim_id": "claim_1",
            "claim": "An established result.",
            "source_id": SOURCE_ID,
            "chunk_ids": ["invented-chunk"],
            "evidence_spans": [],
            "support_status": "entailed",
            "reason": "Claimed support.",
        }
    ]
    known_ref = {
        "source_id": SOURCE_ID,
        "chunk_id": "known-chunk",
        "section": "Results",
        "page": 3,
        "evidence_type": "full_text",
    }
    with patch(
        "app.agents.call_llm",
        return_value=_audit_response(final, claim_assessments=claims),
    ):
        audits, error = call_llm_for_hypothesis_audit(
            "Test a controller.",
            [final],
            "Known evidence.",
            {SOURCE_ID},
            available_evidence_refs={"known-chunk": known_ref},
        )

    assert error is None
    assert audits[0]["passed"] is False
    assert audits[0]["audit_report"]["claim_assessments"][0]["support_status"] == "unsupported"


def test_invented_effect_sizes_are_rejected():
    for claim in (
        "The method will reduce signaling overhead by >40%.",
        "The method will improve spectral efficiency by >15%.",
    ):
        final = _hypothesis(claim)
        with patch("app.agents.call_llm", return_value=_audit_response(final)):
            audits, error = call_llm_for_hypothesis_audit(
                "Improve network performance.",
                [final],
                "The method may improve resource allocation qualitatively.",
                {SOURCE_ID},
            )

        assert error is None
        assert audits[0]["passed"] is False
        assert audits[0]["audit_report"]["unsupported_numbers"]


def test_design_parameter_is_allowed_but_empirical_stability_claim_is_not():
    design = _hypothesis(
        "The method will be compared with a baseline.",
        feasibility="Run 10,000 TTIs and compare latency with the baseline.",
    )
    design_classes = classify_numeric_specificity(design, "")
    assert design_classes == [
        {
            "value": "10,000 TTIs",
            "field": "feasibility",
            "classification": "experimental_design_parameter",
        }
    ]

    empirical = _hypothesis("The method remains stable for 10,000 TTIs.")
    empirical_classes = classify_numeric_specificity(empirical, "")
    assert empirical_classes[0]["classification"] == "unsupported"


def test_malformed_audit_gets_one_format_only_repair():
    final = _hypothesis("The method will outperform the baseline.")
    with patch(
        "app.agents.call_llm",
        side_effect=["{unterminated", _audit_response(final)],
    ) as mock_call:
        audits, error = call_llm_for_hypothesis_audit(
            "Improve network performance.",
            [final],
            "The source describes a baseline limitation.",
            {SOURCE_ID},
        )

    assert error is None
    assert audits[0]["passed"] is True
    assert mock_call.call_count == 2
    assert "Reformat the audit response" in mock_call.call_args_list[1].args[0]


def test_low_prior_art_novelty_score_rejects_duplicate_architecture():
    final = _hypothesis("Use PPO between slices and deterministic scheduling within each slice.")
    with patch(
        "app.agents.call_llm",
        return_value=_audit_response(final, scores=_scores(novelty_against_prior_art=4)),
    ):
        audits, error = call_llm_for_hypothesis_audit(
            "Develop a resource allocation framework.",
            [final],
            "Prior work uses PPO between slices and deterministic scheduling within slices.",
            {SOURCE_ID},
        )

    assert error is None
    assert audits[0]["passed"] is False
    assert "Novelty score" in " ".join(audits[0]["audit_report"]["hard_failures"])


def test_standards_composition_requires_direct_entailment():
    final = _hypothesis("3GPP Rel-18 E2 service models provide the control interface.")
    with patch(
        "app.agents.call_llm",
        return_value=_audit_response(
            final,
            remaining_claims=[
                "No supplied authoritative evidence establishes a 3GPP Rel-18 E2 service model."
            ],
        ),
    ):
        audits, error = call_llm_for_hypothesis_audit(
            "Study standards-based control.",
            [final],
            "One paper discusses Release 18. Another discusses O-RAN E2.",
            {SOURCE_ID},
        )

    assert error is None
    assert audits[0]["passed"] is False


def test_synthesis_prompt_prioritizes_full_text_qualification():
    evidence_ref = {
        "source_id": SOURCE_ID,
        "chunk_id": "chunk-results",
        "section": "Limitations",
        "page": 11,
        "evidence_type": "full_text",
    }
    payload = json.dumps(
        {
            "established_findings": [
                {
                    "claim": "The positive result holds only under restricted conditions.",
                    "source_ids": [SOURCE_ID],
                    "evidence_refs": [
                        {"source_id": SOURCE_ID, "chunk_id": "chunk-results"}
                    ],
                }
            ],
            "contradictions": [],
            "knowledge_gaps": ["Generalization remains unresolved."],
            "analytical_rationale": "The limitation motivates broader validation.",
        }
    )
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        synthesis, error = call_llm_for_literature_synthesis(
            "Test generalization.",
            (EvidenceAspect("generalization", "generalization evidence"),),
            (),
            "Abstract: strong result. Full text: restricted conditions.",
            {SOURCE_ID},
            available_evidence_refs={"chunk-results": evidence_ref},
        )

    assert error is None
    assert synthesis.established_findings[0].evidence_refs[0]["page"] == 11
    normalized_prompt = " ".join(mock_call.call_args.args[0].split())
    assert "full-text evidence takes precedence" in normalized_prompt


def test_full_text_coverage_accepts_only_exact_chunk_provenance():
    known_ref = {
        "source_id": SOURCE_ID,
        "chunk_id": "known-chunk",
        "section": "Results",
        "page": 7,
        "evidence_type": "full_text",
    }
    payload = json.dumps(
        {
            "aspect_coverage": [
                {
                    "aspect_id": "latency",
                    "evidence_refs": [
                        {"source_id": SOURCE_ID, "chunk_id": "known-chunk"},
                        {"source_id": SOURCE_ID, "chunk_id": "invented-chunk"},
                    ],
                }
            ],
            "gap_queries": [],
            "reason": "The Results passage supports the requirement.",
        }
    )
    with patch("app.agents.call_llm", return_value=payload):
        coverage, error = call_llm_for_full_text_evidence_coverage(
            "Reduce latency.",
            (EvidenceAspect("latency", "latency evidence"),),
            "retrieved evidence",
            {"known-chunk": known_ref},
        )

    assert error is None
    assert coverage.sufficient is True
    assert coverage.stage == "full_text"
    assert coverage.aspect_source_ids == {"latency": (SOURCE_ID,)}
    assert [ref["chunk_id"] for ref in coverage.aspect_evidence_refs["latency"]] == [
        "known-chunk"
    ]


def test_qwen_style_query_instruction_is_query_side_only():
    class FakeModel:
        def __init__(self):
            self.inputs = []

        def encode(self, value, **_kwargs):
            self.inputs.append(value)
            if isinstance(value, list):
                return np.array([[1.0, 0.0] for _ in value])
            return np.array([1.0, 0.0])

    model = FakeModel()
    embeddings = SharedSentenceTransformerEmbeddings(
        query_instruction_enabled=True,
        query_instruction="retrieve direct scientific evidence",
    )
    with patch("app.rag_retriever.get_sentence_transformer_model", return_value=model):
        embeddings.embed_documents(["indexed passage"])
        embeddings.embed_query("latency evidence")

    assert model.inputs[0] == ["indexed passage"]
    assert model.inputs[1] == (
        "Instruct: retrieve direct scientific evidence\nQuery: latency evidence"
    )
