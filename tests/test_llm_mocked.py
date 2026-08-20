"""Offline tests of the LLM boundary: parsing and error propagation."""

import json
from unittest.mock import MagicMock, patch

import app.utils as utils
from app.agents import (
    ReflectionAgent,
    call_llm_for_generation,
    call_llm_for_hypothesis_revision,
    call_llm_for_reflection,
)
from app.models import ContextMemory, Hypothesis, ResearchGoal


def _native_response(content: str):
    response = MagicMock()
    response.json.return_value = {
        "output": [{"type": "message", "content": content}],
    }
    return response


def test_generation_happy_path_parses_hypotheses():
    payload = json.dumps(
        [
            {
                "title": "Hypothesis A",
                "hypothesis": "Perovskite tandem cells improve efficiency.",
                "rationale": "Retrieved evidence supports tandem designs.",
                "feasibility": "Compare tandem and baseline cells.",
                "source_ids": ["arXiv:1111.1111"],
            },
            {
                "title": "Hypothesis B",
                "hypothesis": "Bifacial panel coatings improve yield.",
                "rationale": "Retrieved evidence supports bifacial capture.",
                "feasibility": "Measure coated and uncoated panels.",
                "source_ids": ["arXiv:2222.2222"],
            },
        ]
    )
    with patch.object(utils.requests, "post", return_value=_native_response(payload)):
        result = call_llm_for_generation("test goal", num_hypotheses=2, temperature=0.7)

    assert [h["title"] for h in result] == ["Hypothesis A", "Hypothesis B"]
    assert all({"hypothesis", "rationale", "feasibility", "source_ids"}.issubset(h) for h in result)


def test_generation_handles_markdown_fenced_json():
    expected = {
        "title": "T",
        "hypothesis": "H",
        "rationale": "R",
        "feasibility": "F",
        "source_ids": ["arXiv:1234.5678"],
    }
    payload = f"```json\n{json.dumps([expected])}\n```"
    with patch.object(utils.requests, "post", return_value=_native_response(payload)):
        result = call_llm_for_generation("test goal")

    assert result == [expected]


def test_generation_accepts_wrapped_json_and_common_field_aliases():
    payload = json.dumps(
        {
            "hypotheses": [
                {
                    "Title": "T",
                    "Hypothesis": "H",
                    "Rationale": "R",
                    "Feasibility": "F",
                    "Source IDs": ["arXiv:1234.5678"],
                }
            ]
        }
    )
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = call_llm_for_generation("test goal", num_hypotheses=1)

    assert result == [
        {
            "title": "T",
            "hypothesis": "H",
            "rationale": "R",
            "feasibility": "F",
            "source_ids": ["arXiv:1234.5678"],
        }
    ]
    assert mock_call.call_count == 1


def test_generation_repairs_unparsable_output_once():
    repaired = json.dumps(
        [
            {
                "title": "T",
                "hypothesis": "H",
                "rationale": "R",
                "feasibility": "F",
                "source_ids": ["arXiv:1234.5678"],
            }
        ]
    )
    with patch(
        "app.agents.call_llm",
        side_effect=["Here are the hypotheses: not JSON", repaired],
    ) as mock_call:
        result = call_llm_for_generation(
            "test goal",
            num_hypotheses=1,
            temperature=0.7,
            model="selected-local-model",
        )

    assert result[0]["title"] == "T"
    assert mock_call.call_count == 2
    repair_call = mock_call.call_args_list[1]
    assert repair_call.kwargs == {
        "temperature": 0.0,
        "model": "selected-local-model",
        "max_tokens": 2048,
        "reasoning": "off",
    }
    assert "format" in repair_call.args[0].lower()


def test_generation_recovers_remaining_candidate_after_truncated_json():
    first = {
        "title": "Hypothesis A",
        "hypothesis": "Complete hypothesis A.",
        "rationale": "Complete rationale A.",
        "feasibility": "Complete feasibility A.",
        "source_ids": ["arXiv:1111.1111"],
    }
    second = {
        "title": "Hypothesis B",
        "hypothesis": "Complete hypothesis B.",
        "rationale": "Complete rationale B.",
        "feasibility": "Complete feasibility B.",
        "source_ids": ["arXiv:2222.2222"],
    }
    truncated = json.dumps([first])[:-1] + ', {"title": "Hypothesis B", "hypothesis": "cut off'

    with patch(
        "app.agents.call_llm",
        side_effect=[truncated, json.dumps([second])],
    ) as mock_call:
        result = call_llm_for_generation(
            "test goal with numbered strategies",
            num_hypotheses=2,
            model="selected-local-model",
        )

    assert result == [first, second]
    assert mock_call.call_count == 2
    recovery_call = mock_call.call_args_list[1]
    assert "candidate 2 of 2" in recovery_call.args[0]
    assert "test goal with numbered strategies" in recovery_call.args[0]
    assert recovery_call.kwargs == {
        "temperature": 0.2,
        "model": "selected-local-model",
        "max_tokens": 3072,
        "reasoning": "off",
    }


def test_generation_accepts_complete_candidates_when_only_closing_array_was_truncated():
    candidate = {
        "title": "Hypothesis A",
        "hypothesis": "Complete hypothesis.",
        "rationale": "Complete rationale.",
        "feasibility": "Complete feasibility.",
        "source_ids": ["arXiv:1111.1111"],
    }
    truncated = json.dumps([candidate])[:-1]

    with patch("app.agents.call_llm", return_value=truncated) as mock_call:
        result = call_llm_for_generation("test goal", num_hypotheses=1)

    assert result == [candidate]
    assert mock_call.call_count == 1


def test_generation_retries_when_model_reports_incomplete_candidate_response():
    candidate = {
        "title": "Recovered hypothesis",
        "hypothesis": "Complete hypothesis.",
        "rationale": "Complete rationale.",
        "feasibility": "Complete feasibility.",
        "source_ids": ["arXiv:1111.1111"],
    }
    incomplete_error = json.dumps(
        {
            "error": (
                "Incomplete candidate response: missing closing brackets and remaining fields."
            )
        }
    )

    with patch(
        "app.agents.call_llm",
        side_effect=[incomplete_error, json.dumps([candidate])],
    ) as mock_call:
        result = call_llm_for_generation("test goal", num_hypotheses=1)

    assert result == [candidate]
    assert mock_call.call_count == 2


def test_generation_recovers_when_format_repair_reports_missing_candidate_content():
    candidate = {
        "title": "Recovered hypothesis",
        "hypothesis": "Complete hypothesis.",
        "rationale": "Complete rationale.",
        "feasibility": "Complete feasibility.",
        "source_ids": ["arXiv:1111.1111"],
    }
    incomplete_error = json.dumps(
        {"error": "Incomplete candidate response: missing remaining fields."}
    )

    with patch(
        "app.agents.call_llm",
        side_effect=["not structured JSON", incomplete_error, json.dumps([candidate])],
    ) as mock_call:
        result = call_llm_for_generation("test goal", num_hypotheses=1)

    assert result == [candidate]
    assert mock_call.call_count == 3


def test_generation_parses_insufficient_context_error():
    payload = json.dumps({"error": ("The retrieved context is insufficient to generate grounded hypotheses.")})
    with patch("app.agents.call_llm", return_value=payload):
        result = call_llm_for_generation("test goal")

    assert result == [
        {
            "title": "Error",
            "text": ("The retrieved context is insufficient to generate grounded hypotheses."),
        }
    ]


def test_generation_passes_selected_model_to_llm_boundary():
    expected = {
        "title": "T",
        "hypothesis": "H",
        "rationale": "R",
        "feasibility": "F",
        "source_ids": ["arXiv:1234.5678"],
    }
    payload = json.dumps([expected])
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = call_llm_for_generation("test goal", model="selected-local-model")

    assert result == [expected]
    assert mock_call.call_args.kwargs["model"] == "selected-local-model"
    assert "do not return an error object" in mock_call.call_args.args[0]


def test_401_propagates_as_error_hypothesis():
    with patch.object(
        utils.requests,
        "post",
        side_effect=Exception("Error code: 401 - No auth credentials found"),
    ):
        result = call_llm_for_generation("test goal", num_hypotheses=2)

    assert len(result) == 1
    assert result[0]["title"] == "Error"
    assert "authentication failed" in result[0]["text"].lower()


def test_reflection_error_returns_not_reviewed():
    # call_llm is imported into app.agents' namespace, so patch it there.
    hypothesis = Hypothesis(text="some hypothesis", hypothesis_id="test-id-1")
    research_goal = ResearchGoal(description="test goal", constraints="")
    context = ContextMemory()
    
    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        review = call_llm_for_reflection(hypothesis, research_goal, context)
    
    assert review["novelty_review"] == "UNREVIEWED"
    assert review["feasibility_review"] == "UNREVIEWED"
    assert review["references"] == []
    assert review["recommendation"] == "UNREVIEWED"
    assert review["strengths"] == []
    assert review["weaknesses"] == []


def test_reflection_passes_selected_model_to_llm_boundary():
    payload = json.dumps(
        {
            "alignment_score": 8,
            "novelty_score": 9,
            "feasibility_score": 6,
            "plausibility_score": 8,
            "testability_score": 7,
            "evidence_quality_score": 5,
            "expected_research_value_score": 8,
            "strengths": ["Well grounded in retrieved evidence."],
            "weaknesses": ["Feasibility plan is vague."],
            "comment": "Looks plausible.",
            "references": [],
        }
    )
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        hypothesis = Hypothesis(text="some hypothesis", hypothesis_id="test-id-1")
        research_goal = ResearchGoal(description="test goal", constraints="")
        context = ContextMemory()
        review = call_llm_for_reflection(hypothesis, research_goal, context, model="selected-local-model")

    assert review["novelty_review"] == "HIGH"  # 9 converts to HIGH
    assert review["novelty_score"] == 9
    assert review["strengths"] == ["Well grounded in retrieved evidence."]
    assert review["weaknesses"] == ["Feasibility plan is vague."]
    assert review["recommendation"] == "ACCEPT"  # every score is 4 or above
    assert mock_call.call_args.kwargs["model"] == "selected-local-model"


def test_reflection_recommends_revise_when_any_score_below_four():
    payload = json.dumps(
        {
            "alignment_score": 8,
            "novelty_score": 9,
            "feasibility_score": 3,
            "plausibility_score": 8,
            "testability_score": 7,
            "evidence_quality_score": 5,
            "expected_research_value_score": 8,
            "comment": "Feasibility is weak.",
            "references": [],
        }
    )
    with patch("app.agents.call_llm", return_value=payload):
        hypothesis = Hypothesis(text="some hypothesis", hypothesis_id="test-id-low-score")
        research_goal = ResearchGoal(description="test goal", constraints="")
        context = ContextMemory()
        review = call_llm_for_reflection(hypothesis, research_goal, context)

    assert review["feasibility_score"] == 3
    assert review["recommendation"] == "REVISE"


def test_reflection_retries_invalid_review_values():
    first_payload = json.dumps(
        {
            "alignment_score": "invalid",
            "novelty_score": "not_a_number",
            "feasibility_score": 6,
            "plausibility_score": 7,
            "testability_score": 7,
            "evidence_quality_score": 5,
            "expected_research_value_score": 7,
            "comment": "Invalid values.",
            "references": [],
        }
    )
    second_payload = json.dumps(
        {
            "alignment_score": 5,
            "novelty_score": 2,
            "feasibility_score": 8,
            "plausibility_score": 7,
            "testability_score": 6,
            "evidence_quality_score": 5,
            "expected_research_value_score": 7,
            "comment": "Repaired review.",
            "references": [],
        }
    )
    with patch(
        "app.agents.call_llm",
        side_effect=[first_payload, second_payload],
    ) as mock_call:
        hypothesis = Hypothesis(text="some hypothesis", hypothesis_id="test-id-2")
        research_goal = ResearchGoal(description="test goal", constraints="")
        context = ContextMemory()
        review = call_llm_for_reflection(hypothesis, research_goal, context)

        assert review["novelty_review"] == "LOW"  # 2 converts to LOW
        assert review["feasibility_review"] == "HIGH"  # 8 converts to HIGH
        assert review["novelty_score"] == 2
        assert review["feasibility_score"] == 8
        assert review["recommendation"] == "REJECT"  # score < 3 is REJECT
        assert mock_call.call_count == 2


def test_reflection_three_tier_recommendations():
    from app.agents_modules.reflection_helpers import _recommendation_from_scores

    # All >= 5 -> ACCEPT
    assert _recommendation_from_scores({
        "alignment_score": 5, "novelty_score": 6, "feasibility_score": 7,
        "plausibility_score": 8, "testability_score": 5, "evidence_quality_score": 9,
        "expected_research_value_score": 10,
    }) == "ACCEPT"

    # Any in [3, 4] and none < 3 -> REVISE
    assert _recommendation_from_scores({
        "alignment_score": 5, "novelty_score": 4, "feasibility_score": 7,
        "plausibility_score": 8, "testability_score": 5, "evidence_quality_score": 9,
        "expected_research_value_score": 10,
    }) == "REVISE"

    assert _recommendation_from_scores({
        "alignment_score": 3, "novelty_score": 6, "feasibility_score": 7,
        "plausibility_score": 8, "testability_score": 5, "evidence_quality_score": 9,
        "expected_research_value_score": 10,
    }) == "REVISE"

    # Any < 3 -> REJECT
    assert _recommendation_from_scores({
        "alignment_score": 2, "novelty_score": 6, "feasibility_score": 7,
        "plausibility_score": 8, "testability_score": 5, "evidence_quality_score": 9,
        "expected_research_value_score": 10,
    }) == "REJECT"

    assert _recommendation_from_scores({
        "alignment_score": 1, "novelty_score": 1, "feasibility_score": 1,
        "plausibility_score": 1, "testability_score": 1, "evidence_quality_score": 1,
        "expected_research_value_score": 1,
    }) == "REJECT"


def test_reflection_rejects_model_references_when_no_verified_sources_exist():
    payload = json.dumps(
        {
            "alignment_score": 8,
            "novelty_score": 8,
            "feasibility_score": 8,
            "plausibility_score": 8,
            "testability_score": 8,
            "evidence_quality_score": 8,
            "expected_research_value_score": 8,
            "strengths": [],
            "weaknesses": [],
            "comment": "Looks plausible.",
            "references": ["invented:123"],
        }
    )

    with patch("app.agents.call_llm", return_value=payload):
        review = call_llm_for_reflection(
            Hypothesis(text="some hypothesis", hypothesis_id="test-no-sources"),
            ResearchGoal(description="test goal", constraints=""),
            ContextMemory(),
        )

    assert review["references"] == []


def test_reflection_uses_hypothesis_evidence_instead_of_latest_context_sources():
    payload = json.dumps(
        {
            "alignment_score": 8,
            "novelty_score": 8,
            "feasibility_score": 8,
            "plausibility_score": 8,
            "testability_score": 8,
            "evidence_quality_score": 8,
            "expected_research_value_score": 8,
            "strengths": [],
            "weaknesses": [],
            "comment": "Looks plausible.",
            "references": ["paper:attached"],
        }
    )
    hypothesis = Hypothesis(text="some hypothesis", hypothesis_id="test-own-evidence")
    hypothesis.evidence_sources = [
        {
            "source_id": "paper:attached",
            "title": "Attached evidence",
            "abstract": "Evidence attached to this hypothesis.",
        }
    ]
    context = ContextMemory()
    context.last_retrieved_sources = [
        {
            "source_id": "paper:unrelated",
            "title": "Unrelated latest evidence",
            "abstract": "Evidence from a later generation cycle.",
        }
    ]

    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        review = call_llm_for_reflection(
            hypothesis,
            ResearchGoal(description="test goal", constraints=""),
            context,
        )

    prompt = mock_call.call_args.args[0]
    assert "paper:attached" in prompt
    assert "paper:unrelated" not in prompt
    assert review["references"] == ["paper:attached"]


def test_reflection_agent_does_not_create_zero_score_report_after_llm_failure():
    hypothesis = Hypothesis(text="some hypothesis", hypothesis_id="test-review-failure")

    with patch(
        "app.agents_modules.reflection.call_llm_for_reflection",
        return_value={
            "novelty_review": "UNREVIEWED",
            "feasibility_review": "UNREVIEWED",
            "alignment_score": 0,
            "novelty_score": 0,
            "feasibility_score": 0,
            "plausibility_score": 0,
            "testability_score": 0,
            "evidence_quality_score": 0,
            "expected_research_value_score": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendation": "UNREVIEWED",
            "comment": "LLM review failed.",
            "references": [],
        },
    ):
        ReflectionAgent().review_hypotheses(
            [hypothesis],
            ContextMemory(),
            ResearchGoal(description="test goal", constraints=""),
        )

    assert hypothesis.reflection_report is None
    assert hypothesis.novelty_review == "UNREVIEWED"
    assert hypothesis.feasibility_review == "UNREVIEWED"


def test_reflection_agent_stores_sub_claim_assessments_on_report():
    hypothesis = Hypothesis(text="some hypothesis", hypothesis_id="test-sub-claims")
    review = {
        "novelty_review": "HIGH",
        "feasibility_review": "HIGH",
        "alignment_score": 8,
        "novelty_score": 8,
        "feasibility_score": 8,
        "plausibility_score": 8,
        "testability_score": 8,
        "evidence_quality_score": 8,
        "expected_research_value_score": 8,
        "strengths": [],
        "weaknesses": [],
        "recommendation": "ACCEPT",
        "comment": "Looks plausible.",
        "references": [],
    }
    claim_assessment = {
        "claims": [
            {
                "claim": "The method improves recall.",
                "status": "SUPPORTED",
                "confidence": 8.0,
                "supporting_evidence": [{"source_id": "paper-1"}],
                "contradictory_evidence": [],
            }
        ],
        "overall_confidence": 8.0,
    }

    with (
        patch("app.agents_modules.reflection.call_llm_for_reflection", return_value=review),
        patch("app.agents_modules.reflection.evaluate_claims", return_value=claim_assessment),
    ):
        ReflectionAgent().review_hypotheses(
            [hypothesis],
            ContextMemory(),
            ResearchGoal(description="test goal", constraints=""),
        )

    assert hypothesis.reflection_report is not None
    assert hypothesis.reflection_report.claims[0].claim == "The method improves recall."
    assert hypothesis.reflection_report.claims[0].confidence == 8.0
    assert hypothesis.reflection_report.overall_confidence == 8.0


def test_reflection_agent_does_not_rewrite_revise_hypothesis_in_place():
    hypothesis = Hypothesis(
        title="Original title",
        text="Original hypothesis text.",
        hypothesis_id="test-revise-provenance",
    )
    hypothesis.references = [{"source_id": "paper:structured"}]
    review = {
        "novelty_review": "LOW",
        "feasibility_review": "HIGH",
        "alignment_score": 8,
        "novelty_score": 3,
        "feasibility_score": 8,
        "plausibility_score": 8,
        "testability_score": 8,
        "evidence_quality_score": 8,
        "expected_research_value_score": 8,
        "strengths": ["Testable."],
        "weaknesses": ["Insufficient novelty."],
        "recommendation": "REVISE",
        "comment": "Create a distinct revised descendant.",
        "references": ["paper:review"],
    }

    with (
        patch("app.agents_modules.reflection.call_llm_for_reflection", return_value=review),
        patch("app.agents_modules.reflection.call_llm_for_hypothesis_revision") as mock_revision,
        patch(
            "app.agents_modules.reflection.evaluate_claims",
            return_value={"claims": [], "overall_confidence": 1.0},
        ),
    ):
        ReflectionAgent().review_hypotheses(
            [hypothesis],
            ContextMemory(),
            ResearchGoal(description="test goal", constraints=""),
        )

    assert hypothesis.title == "Original title"
    assert hypothesis.text == "Original hypothesis text."
    assert hypothesis.reflection_report.recommendation == "REVISE"
    assert hypothesis.review_reference_ids == ["paper:review"]
    assert hypothesis.references == [{"source_id": "paper:structured"}]
    mock_revision.assert_not_called()


def test_hypothesis_revision_returns_revised_fields():
    payload = json.dumps(
        [
            {
                "title": "Revised title",
                "hypothesis": "Revised hypothesis text.",
                "rationale": "Revised rationale grounded in the cited evidence.",
                "feasibility": "Revised, concrete feasibility plan.",
                "source_ids": ["arXiv:1111.1111"],
            }
        ]
    )
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        hypothesis = Hypothesis(text="Hypothesis: weak claim.", hypothesis_id="test-id-revise")
        hypothesis.evidence_source_ids = ["arXiv:1111.1111"]
        research_goal = ResearchGoal(description="test goal", constraints="")
        revised = call_llm_for_hypothesis_revision(hypothesis, research_goal, model="selected-local-model")

    assert revised["title"] == "Revised title"
    assert revised["hypothesis"] == "Revised hypothesis text."
    assert mock_call.call_args.kwargs["model"] == "selected-local-model"


def test_hypothesis_revision_returns_none_on_llm_error():
    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        hypothesis = Hypothesis(text="Hypothesis: weak claim.", hypothesis_id="test-id-revise-error")
        research_goal = ResearchGoal(description="test goal", constraints="")
        revised = call_llm_for_hypothesis_revision(hypothesis, research_goal)

    assert revised is None
