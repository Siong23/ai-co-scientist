"""Offline coverage for the Reflection agent's deep verification review."""

import json
from unittest.mock import patch

from app.agents_modules.ranking_helpers import format_reflection_report
from app.agents_modules.reflection import ReflectionAgent
from app.agents_modules.reflection_helpers import (
    call_llm_for_deep_verification,
    recommendation_after_deep_verification,
)
from app.config import config
from app.models import AssumptionVerdict, ContextMemory, Hypothesis, ReflectionReport, ResearchGoal


def _verification_payload(*assumptions):
    return json.dumps({"assumptions": list(assumptions), "summary": "Decomposition summary."})


def _assumption(status="UNCERTAIN", fundamental=False, text="Bandwidth scales linearly with slice count."):
    return {
        "assumption": text,
        "status": status,
        "fundamental": fundamental,
        "reasoning": "Judged independently of the hypothesis.",
    }


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_deep_verification_parses_assumption_verdicts():
    hypothesis = Hypothesis(hypothesis_id="H1", text="Hierarchical RL allocates slice bandwidth.")

    with patch(
        "app.agents_modules.reflection_helpers._call_llm",
        return_value=_verification_payload(
            _assumption(status="VALID", fundamental=True),
            _assumption(status="INVALID", text="Quantum processors eliminate packet loss."),
        ),
    ):
        result = call_llm_for_deep_verification(hypothesis, ResearchGoal("Allocate 5G bandwidth"))

    assert [item["status"] for item in result["assumptions"]] == ["VALID", "INVALID"]
    assert result["assumptions"][0]["fundamental"] is True
    assert result["deep_verification_summary"] == "Decomposition summary."


def test_deep_verification_bounds_the_assumption_count():
    hypothesis = Hypothesis(hypothesis_id="H1", text="Some hypothesis.")

    with (
        patch.dict(config["reflection"], {"max_assumptions": 2}),
        patch(
            "app.agents_modules.reflection_helpers._call_llm",
            return_value=_verification_payload(_assumption(), _assumption(), _assumption(), _assumption()),
        ),
    ):
        result = call_llm_for_deep_verification(hypothesis)

    assert len(result["assumptions"]) == 2


def test_unrecognized_status_is_treated_as_unsettled():
    """A status the model invented must not be read as a refutation."""

    hypothesis = Hypothesis(hypothesis_id="H1", text="Some hypothesis.")

    with patch(
        "app.agents_modules.reflection_helpers._call_llm",
        return_value=_verification_payload(_assumption(status="PROBABLY_WRONG")),
    ):
        result = call_llm_for_deep_verification(hypothesis)

    assert result["assumptions"][0]["status"] == "UNCERTAIN"


def test_deep_verification_returns_nothing_when_the_model_fails():
    hypothesis = Hypothesis(hypothesis_id="H1", text="Some hypothesis.")

    with patch(
        "app.agents_modules.reflection_helpers._call_llm",
        return_value="Error: LM Studio returned an empty response.",
    ):
        assert call_llm_for_deep_verification(hypothesis) == {}

    with patch("app.agents_modules.reflection_helpers._call_llm", return_value="not json at all"):
        assert call_llm_for_deep_verification(hypothesis) == {}


# ---------------------------------------------------------------------------
# Verdict gate
# ---------------------------------------------------------------------------


def test_fundamental_invalid_assumption_rejects_the_hypothesis():
    review = {"recommendation": "ACCEPT", "assumptions": [_assumption(status="INVALID", fundamental=True)]}
    assert recommendation_after_deep_verification(review) == "REJECT"


def test_peripheral_invalid_assumption_downgrades_to_revise():
    """A repairable error belongs to the next refinement pass, not the bin."""

    review = {"recommendation": "ACCEPT", "assumptions": [_assumption(status="INVALID", fundamental=False)]}
    assert recommendation_after_deep_verification(review) == "REVISE"


def test_uncertain_assumptions_leave_the_verdict_alone():
    review = {
        "recommendation": "ACCEPT",
        "assumptions": [_assumption(status="UNCERTAIN", fundamental=True), _assumption(status="VALID")],
    }
    assert recommendation_after_deep_verification(review) == "ACCEPT"


def test_the_gate_never_upgrades_an_existing_verdict():
    for recommendation in ("REJECT", "UNREVIEWED"):
        review = {"recommendation": recommendation, "assumptions": [_assumption(status="VALID")]}
        assert recommendation_after_deep_verification(review) == recommendation


def test_missing_assumptions_leave_the_verdict_alone():
    assert recommendation_after_deep_verification({"recommendation": "ACCEPT"}) == "ACCEPT"


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def _review(recommendation="ACCEPT", score=8):
    return {
        "novelty_review": "HIGH",
        "feasibility_review": "HIGH",
        "alignment_score": score,
        "novelty_score": score,
        "feasibility_score": score,
        "plausibility_score": score,
        "testability_score": score,
        "evidence_quality_score": score,
        "expected_research_value_score": score,
        "strengths": [],
        "weaknesses": [],
        "recommendation": recommendation,
        "comment": "Reviewed.",
        "references": [],
    }


def _run_reflection(verification, *, enabled=True):
    hypothesis = Hypothesis(hypothesis_id="H1", text="Hierarchical RL allocates slice bandwidth.")

    with (
        patch("app.agents_modules.reflection.call_llm_for_reflection", return_value=_review()),
        patch(
            "app.agents_modules.reflection.evaluate_claims",
            return_value={
                "claims": [{"claim": "c", "status": "SUPPORTED", "confidence": 8.0}],
                "overall_confidence": 8.0,
            },
        ),
        patch(
            "app.agents_modules.reflection.call_llm_for_deep_verification",
            return_value=verification,
        ) as verify,
        patch.dict(config["reflection"], {"deep_verification_enabled": enabled}),
    ):
        ReflectionAgent(max_workers=1).review_hypotheses(
            [hypothesis],
            ContextMemory(),
            ResearchGoal(description="Allocate 5G bandwidth", constraints=""),
        )

    return hypothesis, verify


def test_reflection_stores_assumptions_and_applies_the_gate():
    hypothesis, verify = _run_reflection(
        {
            "assumptions": [_assumption(status="INVALID", fundamental=True)],
            "deep_verification_summary": "The premise is contradicted.",
        }
    )

    assert verify.call_count == 1
    report = hypothesis.reflection_report
    assert report.recommendation == "REJECT"
    assert report.assumptions[0].status == "INVALID"
    assert report.deep_verification_summary == "The premise is contradicted."


def test_reflection_keeps_its_verdict_when_verification_is_unavailable():
    hypothesis, _ = _run_reflection({})

    assert hypothesis.reflection_report.recommendation == "ACCEPT"
    assert hypothesis.reflection_report.assumptions == []


def test_disabled_deep_verification_makes_no_call():
    verification = {"assumptions": [_assumption(status="INVALID", fundamental=True)]}
    hypothesis, verify = _run_reflection(verification, enabled=False)

    assert verify.call_count == 0
    assert hypothesis.reflection_report.recommendation == "ACCEPT"


# ---------------------------------------------------------------------------
# Hand-off to ranking
# ---------------------------------------------------------------------------


def test_ranking_sees_the_assumption_verdicts():
    report = ReflectionReport(
        recommendation="REVISE",
        assumptions=[
            AssumptionVerdict(
                assumption="Quantum processors eliminate packet loss.",
                status="INVALID",
                fundamental=False,
            )
        ],
        deep_verification_summary="One peripheral premise is contradicted.",
    )

    formatted = format_reflection_report(report)

    assert "Deep Verification of Assumptions:" in formatted
    assert "[INVALID, peripheral] Quantum processors eliminate packet loss." in formatted
    assert "One peripheral premise is contradicted." in formatted


def test_ranking_output_is_unchanged_without_deep_verification():
    assert "Deep Verification" not in format_reflection_report(ReflectionReport(recommendation="ACCEPT"))
