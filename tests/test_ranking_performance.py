"""Offline regression tests for efficient ranking."""

from unittest.mock import patch

from app.agents_modules.ranking import RankingAgent
from app.agents_modules.ranking_helpers import RANKING_LLM_MODEL, run_pairwise_debate
from app.models import ContextMemory, Hypothesis, PairwiseDecision, ResearchGoal


def _hypothesis(hypothesis_id: str) -> Hypothesis:
    return Hypothesis(hypothesis_id=hypothesis_id, text=f"Hypothesis {hypothesis_id}")


def _decision(hypothesis_a: Hypothesis, hypothesis_b: Hypothesis) -> PairwiseDecision:
    return PairwiseDecision(
        hypothesis_a_id=hypothesis_a.hypothesis_id,
        hypothesis_b_id=hypothesis_b.hypothesis_id,
        outcome="A",
        confidence=0.8,
        reasoning="A better matches the goal.",
    )


def test_tournament_only_compares_pairs_with_new_hypotheses():
    old_hypotheses = [_hypothesis("old-1"), _hypothesis("old-2"), _hypothesis("old-3")]
    new_hypothesis = _hypothesis("new-1")
    hypotheses = old_hypotheses + [new_hypothesis]
    context = ContextMemory()
    goal = ResearchGoal(description="Test goal")

    with patch("app.agents.run_pairwise_debate", side_effect=_decision) as debate:
        RankingAgent().run_tournament(
            hypotheses,
            context,
            goal,
            new_hypotheses=[new_hypothesis],
        )

    assert debate.call_count == len(old_hypotheses)
    assert {
        frozenset((call.args[0].hypothesis_id, call.args[1].hypothesis_id))
        for call in debate.call_args_list
    } == {
        frozenset((old.hypothesis_id, new_hypothesis.hypothesis_id))
        for old in old_hypotheses
    }


def test_pairwise_ranking_uses_one_llm_adjudication():
    goal = ResearchGoal(description="Test goal", llm_model="fast-local-model")
    response = """Decision:\nA\n\nShort Justification:\nA is more feasible.\n\nDecisive Criteria:\n- Feasibility\n\nConfidence:\n0.8"""

    with patch("app.agents_modules.ranking_helpers._call_llm", return_value=response) as call_llm:
        decision = run_pairwise_debate(_hypothesis("a"), _hypothesis("b"), goal)

    assert decision.outcome == "A"
    assert call_llm.call_count == 1
    assert call_llm.call_args.kwargs["model"] == RANKING_LLM_MODEL
