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
