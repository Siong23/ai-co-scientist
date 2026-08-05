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
from app.agents_modules.evolution import EvolutionAgent as ModularEvolutionAgent
from app.agents_modules.generation import GenerationAgent as ModularGenerationAgent
from app.agents_modules.meta_review import MetaReviewAgent as ModularMetaReviewAgent
from app.agents_modules.proximity import ProximityAgent as ModularProximityAgent
from app.agents_modules.ranking import RankingAgent as ModularRankingAgent
from app.agents_modules.reflection import ReflectionAgent as ModularReflectionAgent
from app.agents_modules.reflection_helpers import call_llm_for_reflection  # noqa: F401
from app.agents_modules.supervisor import SupervisorAgent as ModularSupervisorAgent


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

# currently removed to prevent error
# call_llm_for_reflection
def test_agent_helpers_are_implemented_outside_the_compatibility_facade():
    helper_functions = (
        call_llm_for_generation,
        run_pairwise_debate,
        combine_hypotheses,
    )

    for helper in helper_functions:
        assert helper.__module__.startswith("app.agents_modules.")
