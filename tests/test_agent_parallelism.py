"""Offline concurrency tests for independent agent work."""

from threading import Barrier
from unittest.mock import patch

from app.agents import EvolutionAgent, GenerationAgent, ReflectionAgent
from app.models import ContextMemory, Hypothesis, ResearchGoal
from app.rag_retriever import SearchQueryPlan


def _goal(*, top_k: int = 2) -> ResearchGoal:
    return ResearchGoal(
        description="Find a testable mechanism for treatment resistance.",
        constraints={},
        llm_model="offline-model",
        top_k_hypotheses=top_k,
    )


def _review_payload(hypothesis_id: str) -> dict:
    return {
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
        "comment": f"Reviewed {hypothesis_id}",
        "references": [],
        "sub_claims": [f"Claim for {hypothesis_id}"],
    }


def test_reflection_reviews_run_concurrently_and_commit_in_input_order():
    hypotheses = [
        Hypothesis("H1", "First", "First mechanism."),
        Hypothesis("H2", "Second", "Second mechanism."),
    ]
    rendezvous = Barrier(2, timeout=2)

    def review(*, hypothesis, **_kwargs):
        rendezvous.wait()
        return _review_payload(hypothesis.hypothesis_id)

    with (
        patch("app.agents_modules.reflection.call_llm_for_reflection", side_effect=review),
        patch(
            "app.agents_modules.reflection.evaluate_claims",
            return_value={"claims": [], "overall_confidence": 8.0},
        ) as assess,
    ):
        ReflectionAgent(max_workers=2).review_hypotheses(hypotheses, ContextMemory(), _goal())

    assert [hypothesis.review_comments[-1] for hypothesis in hypotheses] == ["Reviewed H1", "Reviewed H2"]
    assert all(hypothesis.reflection_report is not None for hypothesis in hypotheses)
    assert sorted(call.kwargs["claims"] for call in assess.call_args_list) == [
        ["Claim for H1"],
        ["Claim for H2"],
    ]


def test_generation_evidence_judgments_run_concurrently():
    rendezvous = Barrier(2, timeout=2)
    coverage = object()

    def relevance(*_args, **_kwargs):
        rendezvous.wait()
        return ["source-1"], None

    def coverage_grade(*_args, **_kwargs):
        rendezvous.wait()
        return coverage, None

    agent = GenerationAgent(
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    agent.grading_workers = 2
    with (
        patch("app.agents_modules.generation.call_llm_for_relevance_filter", side_effect=relevance),
        patch("app.agents_modules.generation.call_llm_for_evidence_coverage", side_effect=coverage_grade),
    ):
        result = agent._grade_candidate_evidence(
            _goal(),
            SearchQueryPlan(queries=(), required_terms=()),
            "candidate evidence",
            {"source-1"},
        )

    assert result == (["source-1"], None, coverage, None)


def test_reflection_revisions_run_concurrently_and_commit_in_input_order():
    hypotheses = [
        Hypothesis("H1", "First", "First mechanism."),
        Hypothesis("H2", "Second", "Second mechanism."),
    ]
    rendezvous = Barrier(2, timeout=2)

    def revise(hypothesis, *_args, **_kwargs):
        rendezvous.wait()
        return {
            "title": f"Revised {hypothesis.hypothesis_id}",
            "hypothesis": f"Revised text for {hypothesis.hypothesis_id}.",
        }

    with patch("app.agents_modules.reflection.call_llm_for_hypothesis_revision", side_effect=revise):
        revised = ReflectionAgent(max_workers=2).revise_hypotheses(hypotheses, _goal())

    assert [child.parent_ids for child in revised] == [["H1"], ["H2"]]
    assert [child.title for child in revised] == ["Revised H1", "Revised H2"]
    assert all(child.reflection_report is None and child.elo_score == 1200 for child in revised)
    assert [hypothesis.title for hypothesis in hypotheses] == ["First", "Second"]


def test_evolution_strategies_run_concurrently_and_preserve_strategy_order():
    context = ContextMemory()
    first = Hypothesis("H1", "First", "First mechanism.")
    second = Hypothesis("H2", "Second", "Second mechanism.")
    first.elo_score = 1300
    second.elo_score = 1200
    context.add_hypothesis(first)
    context.add_hypothesis(second)
    rendezvous = Barrier(2, timeout=2)

    def evolve(strategy, parents, _research_goal, *, diagnostics, **_kwargs):
        rendezvous.wait()
        diagnostics.append(
            {
                "strategy": strategy,
                "parent_ids": [parent.hypothesis_id for parent in parents],
                "status": "accepted",
                "reason": "accepted",
            }
        )
        return {"title": f"{strategy} child", "text": f"A distinct {strategy} mechanism."}

    agent = EvolutionAgent(
        strategies=("combination", "feasibility"),
        max_candidates_per_cycle=2,
        quality_repair_attempts=0,
        max_workers=2,
    )
    with patch("app.agents_modules.evolution.call_llm_for_evolution", side_effect=evolve):
        evolved = agent.evolve_hypotheses(context, _goal())

    assert [hypothesis.evolution_strategy for hypothesis in evolved] == ["combination", "feasibility"]
    assert [attempt["strategy"] for attempt in context.last_evolution_attempts] == ["combination", "feasibility"]
