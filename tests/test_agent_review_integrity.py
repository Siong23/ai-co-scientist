"""Regression tests for immutable reviews and evidence-based feedback."""

import json
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from app.agents import assess_supervisor_state
from app.agents_modules.meta_review import MetaReviewAgent
from app.agents_modules.proximity import ProximityAgent
from app.agents_modules.ranking import RankingAgent
from app.agents_modules.reflection import ReflectionAgent
from app.agents_modules.supervisor import SupervisorAgent
from app.models import ContextMemory, Hypothesis, PairwiseDecision, ReflectionReport, ResearchGoal


def reviewed(identifier, recommendation="ACCEPT"):
    return Hypothesis(
        identifier,
        identifier,
        "A specific original mechanism with measurable predictions.",
        reflection_report=ReflectionReport(
            recommendation=recommendation,
            weaknesses=["Missing negative controls"],
        ),
    )


def test_supervisor_preserves_revise_version_for_evolution():
    hypothesis = reviewed("parent", "REVISE")
    hypothesis.elo_score = 1300
    context = ContextMemory()
    context.add_hypothesis(hypothesis)
    before = deepcopy(hypothesis.to_dict())
    supervisor = SupervisorAgent()
    supervisor.reflection_agent = Mock()
    routing = supervisor.step_reflection(ResearchGoal("goal"), context, Mock(), {})
    assert routing["revise"] == [hypothesis]
    assert hypothesis.to_dict() == before
    supervisor.reflection_agent.revise_hypotheses.assert_not_called()


def test_revision_creates_unreviewed_child_without_inheriting_scores():
    parent = reviewed("parent", "REVISE")
    parent.elo_score = 1400
    parent.audit_verdict = "accept"
    parent.evidence_source_ids = ["source-1"]
    before = deepcopy(parent.to_dict())
    with patch(
        "app.agents_modules.reflection.call_llm_for_hypothesis_revision",
        return_value={
            "title": "Child",
            "hypothesis": "Test an alternative pathway through selective inhibition and controls.",
        },
    ):
        (child,) = ReflectionAgent().revise_hypotheses([parent], ResearchGoal("goal"))
    assert parent.to_dict() == before
    assert child.hypothesis_id != parent.hypothesis_id
    assert child.parent_ids == [parent.hypothesis_id]
    assert child.reflection_report is None and child.audit_verdict is None
    assert child.elo_score == 1200


@pytest.mark.parametrize("response", [{"title": "Empty"}, {"hypothesis": " "}])
def test_invalid_revision_does_not_create_child(response):
    with patch("app.agents_modules.reflection.call_llm_for_hypothesis_revision", return_value=response):
        assert ReflectionAgent().revise_hypotheses([reviewed("parent", "REVISE")], ResearchGoal("goal")) == []


def test_abstained_pair_can_be_retried_and_remains_unranked_until_success():
    context = ContextMemory()
    hypotheses = [reviewed("A"), reviewed("B")]
    for h in hypotheses:
        context.add_hypothesis(h)
    goal = ResearchGoal("goal")
    agent = RankingAgent()
    decision = PairwiseDecision(
        hypothesis_a_id="A",
        hypothesis_b_id="B",
        outcome="ABSTAIN",
        scores_a={},
        scores_b={},
        confidence=1,
        reasoning="Transient model failure",
    )
    with patch("app.agents_modules.ranking.run_pairwise_debate", return_value=decision):
        agent.run_tournament(hypotheses, context, goal)
    state = assess_supervisor_state(context, goal)
    assert state["unranked_accepted_count"] == 2
    assert state["tournament_comparisons"] == 0
    assert [h.elo_score for h in hypotheses] == [1200, 1200]
    decision = decision.model_copy(update={"outcome": "A", "scores_a": {"novelty": 8}, "scores_b": {"novelty": 7}})
    with patch("app.agents_modules.ranking.run_pairwise_debate", return_value=decision) as debate:
        agent.run_tournament(hypotheses, context, goal)
        agent.run_tournament(hypotheses, context, goal)
    assert debate.call_count == 1
    assert [m["outcome"] for m in context.tournament_results] == ["ABSTAIN", "A"]
    assert sum(h.elo_score for h in hypotheses) == 2400
    assert assess_supervisor_state(context, goal)["unranked_accepted_count"] == 0


def test_missing_score_abstention_is_recorded():
    context = ContextMemory()
    decision = PairwiseDecision(
        hypothesis_a_id="A",
        hypothesis_b_id="B",
        outcome="A",
        scores_a={},
        scores_b={},
        confidence=1,
        reasoning="Missing scores",
    )
    with patch("app.agents_modules.ranking.run_pairwise_debate", return_value=decision):
        RankingAgent().run_tournament([reviewed("A"), reviewed("B")], context, ResearchGoal("goal"))
    assert context.tournament_results[0]["outcome"] == "ABSTAIN"


def test_abstentions_do_not_create_false_convergence_snapshots():
    context = ContextMemory()
    for h in [reviewed("A"), reviewed("B")]:
        context.add_hypothesis(h)
    supervisor = SupervisorAgent()
    supervisor.ranking_agent = Mock()
    supervisor.ranking_agent.run_tournament.side_effect = lambda *a, **kw: context.tournament_results.append(
        {"hypothesis_a": "A", "hypothesis_b": "B", "outcome": "ABSTAIN"}
    )
    supervisor.step_ranking(ResearchGoal("goal"), context, Mock(), {})
    assert context.supervisor_state["elo_snapshots"] == []


def test_meta_review_synthesizes_rejected_reviews_and_tournament_reasoning():
    context = ContextMemory()
    rejected = reviewed("rejected", "REJECT")
    rejected.is_active = False
    context.add_hypothesis(rejected)
    context.tournament_results.append(
        {"hypothesis_a": "rejected", "hypothesis_b": "other", "outcome": "B", "reasoning": "No falsification endpoint"}
    )
    with patch(
        "app.agents.call_llm",
        return_value=json.dumps(
            {
                "critiques": ["Control groups are consistently underspecified"],
                "next_steps": ["Specify negative controls and falsification endpoints"],
            }
        ),
    ) as llm:
        result = MetaReviewAgent().summarize_and_feedback(
            context,
            {},
            research_goal=ResearchGoal("goal", llm_model="chosen-model"),
        )
    prompt = llm.call_args.args[0]
    assert "Missing negative controls" in prompt and "No falsification endpoint" in prompt
    assert llm.call_args.kwargs["model"] == "chosen-model"
    assert result["synthesis_mode"] == "llm"
    assert result["research_overview"]["top_ranked_hypotheses"] == []
    assert result["research_overview"]["suggested_next_steps"][0].startswith("Specify negative")
    assert context.meta_review_feedback[-1] == result


@pytest.mark.parametrize(
    "response", ["Error: unavailable", "[]", '{"critiques": "wrong type"}', '{"critiques": [1], "next_steps": ["x"]}']
)
def test_meta_review_falls_back_on_invalid_model_output(response):
    context = ContextMemory()
    context.add_hypothesis(reviewed("A"))
    with patch("app.agents.call_llm", return_value=response):
        result = MetaReviewAgent().summarize_and_feedback(context, {}, research_goal=ResearchGoal("goal"))
    assert result["synthesis_mode"] == "heuristic"
    assert result["research_overview"]["suggested_next_steps"]
    assert len(context.meta_review_feedback) == 1


def test_meta_review_empty_context_never_calls_model():
    with patch("app.agents.call_llm") as llm:
        result = MetaReviewAgent().summarize_and_feedback(ContextMemory(), {}, research_goal=ResearchGoal("goal"))
    llm.assert_not_called()
    assert result["synthesis_mode"] == "heuristic"


def _capture_reflection_prompt(context):
    """Run one review against a stubbed model and return the prompt it received."""
    hypothesis = Hypothesis("H1", "Slice control", "Hypothesis: A specific mechanism with measurable predictions.")
    context.add_hypothesis(hypothesis)
    review = json.dumps(
        {
            "alignment_score": 8,
            "novelty_score": 7,
            "feasibility_score": 7,
            "plausibility_score": 7,
            "testability_score": 8,
            "evidence_quality_score": 6,
            "expected_research_value_score": 7,
            "strengths": ["Clear mechanism"],
            "weaknesses": ["No power analysis"],
            "sub_claims": [],
            "proposed_tests": ["Measure p99 latency against the baseline."],
            "comment": "Reasonable but under-specified.",
            "references": [],
        }
    )
    prompts = []

    def fake_llm(prompt, **_kwargs):
        prompts.append(prompt)
        return review

    with patch("app.agents_modules.reflection_helpers._call_llm", fake_llm):
        ReflectionAgent(max_workers=1).review_hypotheses([hypothesis], context, ResearchGoal("goal"))

    assert prompts
    return prompts[0]


def test_recurring_meta_review_critiques_reach_the_next_reflection_prompt():
    context = ContextMemory()
    context.meta_review_feedback.append(
        {
            "meta_review_critique": [
                "Reviews repeatedly miss that the baseline is never specified.",
                "Reviews accept unjustified latency thresholds.",
            ],
            "research_overview": {"suggested_next_steps": ["Evolve the top two hypotheses."]},
        }
    )

    prompt = _capture_reflection_prompt(context)

    assert "Reviews repeatedly miss that the baseline is never specified." in prompt
    assert "Reviews accept unjustified latency thresholds." in prompt
    # The critiques are a coverage checklist about earlier hypotheses, so the
    # reviewer must not read them as findings against the hypothesis at hand.
    assert "not this one" in prompt
    assert "do not lower a score for an issue this hypothesis avoids" in prompt
    # Next steps steer Generation and Evolution, not a peer review.
    assert "Evolve the top two hypotheses." not in prompt


def test_reflection_prompt_carries_no_meta_review_block_before_the_first_synthesis():
    prompt = _capture_reflection_prompt(ContextMemory())

    assert "Recurring critiques" not in prompt


def test_reflection_meta_review_block_is_bounded():
    context = ContextMemory()
    context.meta_review_feedback.append(
        {"meta_review_critique": [f"Critique {index} " + "x" * 900 for index in range(9)]}
    )

    prompt = _capture_reflection_prompt(context)

    assert "Critique 4" in prompt
    assert "Critique 5" not in prompt
    assert "x" * 600 not in prompt


def test_proximity_duplicate_keeps_higher_elo_candidate_without_unpack_error():
    context = ContextMemory()
    weaker, stronger = reviewed("A"), reviewed("B")
    stronger.elo_score = 1300
    context.add_hypothesis(weaker)
    context.add_hypothesis(stronger)
    agent = ProximityAgent()
    with patch.object(agent.scorer, "score", return_value=1.0):
        result = agent.get_proximity_analysis(context)
    assert weaker.is_active is False
    assert stronger.is_active is True
    assert result["near_duplicates"][0]["canonical_id"] == "B"


def test_meta_review_maps_research_areas_into_the_overview():
    """The overview is the map of covered ground the next cycle plans against."""

    context = ContextMemory()
    context.add_hypothesis(reviewed("A"))
    with patch(
        "app.agents.call_llm",
        return_value=json.dumps(
            {
                "critiques": ["Control groups are consistently underspecified"],
                "next_steps": ["Specify negative controls"],
                "research_areas": [
                    {
                        "area": "Transporter inhibition",
                        "rationale": "Every accepted hypothesis so far targets efflux.",
                        "example_experiments": ["Knock down the transporter and re-measure sensitivity."],
                    },
                    {"area": "No rationale given"},
                ],
            }
        ),
    ):
        result = MetaReviewAgent().summarize_and_feedback(context, {}, research_goal=ResearchGoal("goal"))

    areas = result["research_overview"]["research_areas"]
    assert [area["area"] for area in areas] == ["Transporter inhibition"]
    assert areas[0]["example_experiments"] == ["Knock down the transporter and re-measure sensitivity."]


def test_meta_review_keeps_its_critiques_when_the_areas_are_malformed():
    context = ContextMemory()
    context.add_hypothesis(reviewed("A"))
    with patch(
        "app.agents.call_llm",
        return_value=json.dumps(
            {
                "critiques": ["Control groups are consistently underspecified"],
                "next_steps": ["Specify negative controls"],
                "research_areas": "not a list",
            }
        ),
    ):
        result = MetaReviewAgent().summarize_and_feedback(context, {}, research_goal=ResearchGoal("goal"))

    assert result["synthesis_mode"] == "llm"
    assert result["research_overview"]["research_areas"] == []


def test_generation_prompt_carries_the_research_overview():
    from app.agents_modules.generation import GenerationAgent

    context = ContextMemory()
    context.meta_review_feedback.append(
        {
            "meta_review_critique": ["Control groups are underspecified"],
            "research_overview": {
                "suggested_next_steps": ["Specify negative controls"],
                "research_areas": [
                    {
                        "area": "Transporter inhibition",
                        "rationale": "Every accepted hypothesis so far targets efflux.",
                        "example_experiments": ["Knock down the transporter."],
                    }
                ],
            },
        }
    )

    formatted = GenerationAgent._format_meta_review_feedback(GenerationAgent.__new__(GenerationAgent), context)

    assert "Transporter inhibition: Every accepted hypothesis so far targets efflux." in formatted
    assert "Example experiment: Knock down the transporter." in formatted
    assert "do not restate a hypothesis that already covers one" in formatted
    assert "Control groups are underspecified" in formatted


def test_generation_prompt_is_unchanged_without_research_areas():
    from app.agents_modules.generation import GenerationAgent

    context = ContextMemory()
    context.meta_review_feedback.append(
        {
            "meta_review_critique": ["Control groups are underspecified"],
            "research_overview": {"suggested_next_steps": ["Specify negative controls"]},
        }
    )

    formatted = GenerationAgent._format_meta_review_feedback(GenerationAgent.__new__(GenerationAgent), context)

    assert "Research areas already covered" not in formatted
    assert "Control groups are underspecified" in formatted
