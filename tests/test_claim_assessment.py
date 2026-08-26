from langchain_core.documents import Document

from app.agents_modules.reflection_helpers import (
    calculate_claim_confidence,
    compute_overall_confidence,
    evaluate_claims,
    function_to_extract_claim,
    function_to_get_supporting_evidence,
)
from app.models import ClaimAssessment, Hypothesis, ReflectionReport


class FakeRetriever:
    def retrieve(self, original_query, query_plan, *, force_web=False):
        assert original_query in {
            "The method improves recall in noisy data.",
            "The method reduces false positives in noisy data.",
        }
        assert query_plan.queries[0].search_intent == "counterevidence"
        assert force_web is True
        return [
            Document(
                page_content="A controlled study reports lower recall in noisy data.",
                metadata={
                    "source_id": "arXiv:9999.00001",
                    "source_type": "academic",
                    "title": "A counterexample",
                    "abstract": "A controlled study reports lower recall in noisy data.",
                },
            )
        ]


def test_claim_extraction_uses_llm_to_split_hypothesis_section(monkeypatch):
    hypothesis = Hypothesis(
        "G1",
        "Recall claim",
        "Hypothesis: The method improves recall in noisy data.\n\nRationale: It combines two signals.",
    )
    monkeypatch.setattr(
        "app.agents_modules.reflection_helpers._call_llm",
        lambda *args, **kwargs: '{"sub_claims": ["The method improves recall in noisy data.", "The method reduces false positives in noisy data."]}',
    )

    assert function_to_extract_claim(hypothesis) == [
        "The method improves recall in noisy data.",
        "The method reduces false positives in noisy data.",
    ]


def test_supporting_evidence_requires_valid_source_id_and_matches_claim():
    hypothesis = Hypothesis("G1", "Recall claim", "Hypothesis: The method improves recall in noisy data.")
    hypothesis.evidence_source_ids = ["paper-1"]
    hypothesis.evidence_sources = [
        {"source_id": "paper-1", "title": "Recall in noisy data", "abstract": "The method improves recall."},
        {"source_id": "not-cited", "title": "Recall in noisy data", "abstract": "The method improves recall."},
        {"source_id": "paper-1", "title": "Unrelated topic", "abstract": "A different experiment."},
    ]

    evidence = function_to_get_supporting_evidence(hypothesis, "The method improves recall in noisy data.")

    assert [item["source_id"] for item in evidence] == ["paper-1"]
    assert evidence[0]["claim_relevance_score"] > 0


def test_evaluate_claims_returns_sub_claim_assessments_and_report_confidence(monkeypatch):
    hypothesis = Hypothesis(
        "G1",
        "Recall claim",
        "Hypothesis: The method improves recall in noisy data.\n\nRationale: It combines two signals.",
    )
    hypothesis.evidence_source_ids = ["paper-1"]
    hypothesis.evidence_sources = [
        {"source_id": "paper-1", "title": "Recall in noisy data", "abstract": "The method improves recall."},
    ]

    monkeypatch.setattr(
        "app.agents_modules.reflection_helpers._call_llm",
        lambda *args, **kwargs: '{"sub_claims": ["The method improves recall in noisy data."]}',
    )

    assessment = evaluate_claims(
        hypothesis,
        evidence_quality_score=8,
        plausibility_score=7,
        retriever=FakeRetriever(),
    )

    assert 1.0 <= assessment["overall_confidence"] <= 10.0
    assert len(assessment["claims"]) == 1
    sub_claim = assessment["claims"][0]
    assert sub_claim["claim"] == "The method improves recall in noisy data."
    assert sub_claim["status"] == "MIXED"
    assert 1.0 <= sub_claim["confidence"] <= 10.0
    assert sub_claim["contradictory_evidence"][0]["source_id"] == "arXiv:9999.00001"
    validated = ClaimAssessment(**sub_claim)
    assert validated.claim == sub_claim["claim"]
    assert validated.confidence == sub_claim["confidence"]

    report = ReflectionReport(
        claims=assessment["claims"],
        overall_confidence=assessment["overall_confidence"],
    )
    assert report.overall_confidence == assessment["overall_confidence"]


def test_overall_confidence_combines_sub_claim_and_reflection_scores():
    claims = [
        ClaimAssessment(claim="Supported claim", confidence=10.0),
        ClaimAssessment(claim="Unsupported claim", confidence=1.0),
    ]

    assert calculate_claim_confidence(
        {
            "status": "SUPPORTED",
            "supporting_evidence": [{}, {}, {}],
            "contradictory_evidence": [],
        }
    ) == 10.0
    assert compute_overall_confidence(
        claims,
        evidence_quality_score=10,
        plausibility_score=10,
    ) == 6.85