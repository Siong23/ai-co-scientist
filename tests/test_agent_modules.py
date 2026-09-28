"""Tests for the modular agent layout and compatibility façade."""

from app.agents import (
    EvolutionAgent,
    GenerationAgent,
    MetaReviewAgent,
    ProximityAgent,
    RankingAgent,
    ReflectionAgent,
    SupervisorAgent,
    call_llm_for_generation,
    combine_hypotheses,
    run_pairwise_debate,
)
from app.agents import call_llm_for_reflection as facade_call_llm_for_reflection
from app.agents_modules.evolution import EvolutionAgent as ModularEvolutionAgent
from app.agents_modules.generation import GenerationAgent as ModularGenerationAgent
from app.agents_modules.meta_review import MetaReviewAgent as ModularMetaReviewAgent
from app.agents_modules.proximity import ProximityAgent as ModularProximityAgent
from app.agents_modules.ranking import RankingAgent as ModularRankingAgent
from app.agents_modules.reflection import ReflectionAgent as ModularReflectionAgent
from app.agents_modules.reflection_helpers import call_llm_for_reflection as modular_call_llm_for_reflection
from app.agents_modules.supervisor import SupervisorAgent as ModularSupervisorAgent
from app.models import ContextMemory, Hypothesis, ReflectionReport


def test_agents_are_reexported_from_individual_modules():
    agent_pairs = (
        (GenerationAgent, ModularGenerationAgent),
        (ReflectionAgent, ModularReflectionAgent),
        (RankingAgent, ModularRankingAgent),
        (EvolutionAgent, ModularEvolutionAgent),
        (ProximityAgent, ModularProximityAgent),
        (MetaReviewAgent, ModularMetaReviewAgent),
        (SupervisorAgent, ModularSupervisorAgent),
    )

    for facade_class, modular_class in agent_pairs:
        assert facade_class is modular_class
        assert facade_class.__module__.startswith("app.agents_modules.")


def test_agent_helpers_are_implemented_outside_the_compatibility_facade():
    helper_functions = (
        call_llm_for_generation,
        run_pairwise_debate,
        facade_call_llm_for_reflection,
        modular_call_llm_for_reflection,
        combine_hypotheses,
    )

    for helper in helper_functions:
        assert helper.__module__.startswith("app.agents_modules.")


def test_meta_review_recommends_cluster_representatives():
    context = ContextMemory()
    for hypothesis_id, elo_score in (("H1", 1300), ("H2", 1250), ("H3", 1200), ("H4", 1100)):
        context.add_hypothesis(Hypothesis(hypothesis_id=hypothesis_id, title=hypothesis_id, elo_score=elo_score))

    overview = ModularMetaReviewAgent().summarize_and_feedback(
        context,
        {},
        proximity_data={
            "clusters": {"H1": 0, "H2": 0, "H3": 1, "H4": 1},
            "cluster_members": {0: ["H1", "H2"], 1: ["H3", "H4"]},
            "connectivity": {"H1": 2, "H2": 0, "H3": 1, "H4": 0},
            "highly_connected": ["H1"],
            "isolated": [],
        },
    )

    steps = overview["research_overview"]["suggested_next_steps"]
    assert any("H1, H3" in step for step in steps)


def _reviewed(hypothesis_id: str, recommendation: str, elo_score: float = 1200.0) -> Hypothesis:
    hypothesis = Hypothesis(hypothesis_id=hypothesis_id, title=hypothesis_id, elo_score=elo_score)
    hypothesis.reflection_report = ReflectionReport(recommendation=recommendation)
    return hypothesis


def test_ranked_ids_count_only_decided_matches():
    context = ContextMemory()
    context.tournament_results = [
        {"hypothesis_a": "A", "hypothesis_b": "B", "outcome": "A"},
        {"hypothesis_a": "C", "hypothesis_b": "D", "outcome": "ABSTAIN"},
        {"hypothesis_a": "E", "hypothesis_b": "F", "outcome": "TIE"},
    ]

    assert context.ranked_hypothesis_ids() == {"A", "B", "E", "F"}


def test_meta_review_top_list_holds_only_hypotheses_that_played_a_match():
    """Two unranked REVISE hypotheses at the default 1200 once outranked the runner-up at 1199.97."""

    context = ContextMemory()
    for hypothesis in (
        _reviewed("E7447", "ACCEPT", 1231.3),
        _reviewed("E4071", "ACCEPT", 1199.97),
        _reviewed("G1653", "REVISE"),
        _reviewed("G7708", "REVISE"),
    ):
        context.add_hypothesis(hypothesis)
    context.tournament_results = [{"hypothesis_a": "E4071", "hypothesis_b": "E7447", "outcome": "B"}]

    overview = ModularMetaReviewAgent().summarize_and_feedback(context, {})

    top = overview["research_overview"]["top_ranked_hypotheses"]
    assert [hypothesis["id"] for hypothesis in top] == ["E7447", "E4071"]


def test_meta_review_top_list_falls_back_to_accepted_hypotheses_before_any_match():
    context = ContextMemory()
    context.add_hypothesis(_reviewed("A", "ACCEPT"))
    context.add_hypothesis(_reviewed("R", "REVISE"))

    overview = ModularMetaReviewAgent().summarize_and_feedback(context, {})

    top = overview["research_overview"]["top_ranked_hypotheses"]
    assert [hypothesis["id"] for hypothesis in top] == ["A"]


def test_meta_review_treats_isolated_hypotheses_as_validation_targets():
    context = ContextMemory()
    for hypothesis_id in ("H1", "H2", "H3"):
        context.add_hypothesis(Hypothesis(hypothesis_id=hypothesis_id, title=hypothesis_id))

    overview = ModularMetaReviewAgent().summarize_and_feedback(
        context,
        {},
        proximity_data={"clusters": {}, "cluster_members": {}, "isolated": ["H1", "H2", "H3"]},
    )

    steps = overview["research_overview"]["suggested_next_steps"]
    assert any("not by itself evidence" in step for step in steps)
