"""Offline tests for Reflection-to-Supervisor routing."""

from unittest.mock import Mock

from app.agents import SupervisorAgent
from app.models import ContextMemory, Hypothesis, ReflectionReport, ResearchGoal


def _hypothesis(hypothesis_id: str, recommendation: str | None) -> Hypothesis:
    hypothesis = Hypothesis(hypothesis_id, hypothesis_id, f"Text for {hypothesis_id}")
    if recommendation is not None:
        hypothesis.reflection_report = ReflectionReport(
            alignment_score=8,
            novelty_score=8,
            feasibility_score=8,
            plausibility_score=8,
            testability_score=8,
            evidence_quality_score=8,
            expected_research_value_score=8,
            recommendation=recommendation,
        )
    return hypothesis


def test_supervisor_only_sends_reflection_accepted_hypotheses_to_ranking():
    accepted = _hypothesis("H-accept", "ACCEPT")
    revise = _hypothesis("H-revise", "REVISE")
    unreviewed = _hypothesis("H-unreviewed", None)
    evolved_accepted = _hypothesis("E-accept", "ACCEPT")
    evolved_revise = _hypothesis("E-revise", "REVISE")

    context = ContextMemory()
    for hypothesis in (accepted, revise, unreviewed):
        context.add_hypothesis(hypothesis)

    supervisor = SupervisorAgent()
    supervisor.generation_agent = Mock()
    supervisor.generation_agent.generate_new_hypotheses.return_value = ([], [])
    supervisor.generation_agent.rag_retriever.last_query_plan = None
    supervisor.generation_agent.rag_retriever.last_query_fidelity = []
    supervisor.generation_agent.rag_retriever.last_search_stats = []
    supervisor.reflection_agent = Mock()
    supervisor.ranking_agent = Mock()
    supervisor.evolution_agent = Mock()
    supervisor.evolution_agent.evolve_hypotheses.return_value = [
        evolved_accepted,
        evolved_revise,
    ]
    supervisor.proximity_agent = Mock()
    supervisor.proximity_agent.build_proximity_graph.return_value = {
        "adjacency_graph": {},
        "nodes": [],
        "edges": [],
    }
    supervisor.meta_review_agent = Mock()
    supervisor.meta_review_agent.summarize_and_feedback.return_value = {}

    details = supervisor.run_cycle(
        ResearchGoal(description="test goal", top_k_hypotheses=1),
        context,
    )

    first_call, second_call = supervisor.ranking_agent.run_tournament.call_args_list
    assert first_call.args[0] == [accepted]
    assert first_call.kwargs["new_hypotheses"] == []
    assert second_call.args[0] == [accepted, evolved_accepted]
    assert second_call.kwargs["new_hypotheses"] == [evolved_accepted]

    assert details["steps"]["reflection"]["routing"] == {
        "accepted": ["H-accept"],
        "revise": ["H-revise"],
        "unreviewed": ["H-unreviewed"],
    }
    assert details["steps"]["reflection_evolved"]["routing"] == {
        "accepted": ["E-accept"],
        "revise": ["E-revise"],
        "unreviewed": [],
    }
