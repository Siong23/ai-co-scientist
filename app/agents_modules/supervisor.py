"""Workflow supervisor agent."""

from __future__ import annotations

from typing import Dict

from ..models import ContextMemory, ResearchGoal
from ._compat import _legacy
from .evolution import EvolutionAgent
from .generation import GenerationAgent
from .meta_review import MetaReviewAgent
from .proximity import ProximityAgent
from .ranking import RankingAgent
from .reflection import ReflectionAgent


class SupervisorAgent:
    """Orchestrates the Open AI Co-Scientist workflow."""

    def __init__(self):
        self.generation_agent = GenerationAgent()
        self.reflection_agent = ReflectionAgent()
        self.ranking_agent = RankingAgent()
        self.evolution_agent = EvolutionAgent()
        self.proximity_agent = ProximityAgent()
        self.meta_review_agent = MetaReviewAgent()

    def run_cycle(self, research_goal: ResearchGoal, context: ContextMemory) -> Dict:
        """Runs a single cycle of hypothesis generation and refinement."""
        _legacy.logger.info("--- Starting Cycle %d ---", context.iteration_number + 1)
        cycle_details = {"iteration": context.iteration_number + 1, "steps": {}, "meta_review": {}}

        # 1. Generation
        _legacy.logger.info("Step 1: Generation")
        new_hypotheses, generation_errors = self.generation_agent.generate_new_hypotheses(research_goal, context)
        for nh in new_hypotheses:
            context.add_hypothesis(nh)  # Add to central context
        cycle_details["steps"]["generation"] = {
            "hypotheses": [h.to_dict() for h in new_hypotheses],
            "sources": list(context.last_retrieved_sources),
        }

        # Propagate LLM errors to top-level errors field for frontend display, so a
        # generation failure surfaces its real cause instead of an empty ranking.
        if generation_errors:
            cycle_details["errors"] = generation_errors

        # Get all active hypotheses for subsequent steps
        active_hypos = context.get_active_hypotheses()

        # 2. Reflection
        _legacy.logger.info("Step 2: Reflection")
        self.reflection_agent.review_hypotheses(active_hypos, context, research_goal)  # Pass research_goal
        cycle_details["steps"]["reflection"] = {"hypotheses": [h.to_dict() for h in active_hypos]}

        # 3. Ranking (Tournament 1)
        _legacy.logger.info("Step 3: Ranking 1")
        self.ranking_agent.run_tournament(active_hypos, context, research_goal)  # Pass research_goal
        cycle_details["steps"]["ranking1"] = {"hypotheses": [h.to_dict() for h in active_hypos]}

        # 4. Evolution
        _legacy.logger.info("Step 4: Evolution")
        evolved_hypotheses = self.evolution_agent.evolve_hypotheses(context, research_goal)  # Pass research_goal
        if evolved_hypotheses:
            for eh in evolved_hypotheses:
                context.add_hypothesis(eh)
            _legacy.logger.info("Step 4a: Reviewing Evolved Hypotheses")
            self.reflection_agent.review_hypotheses(evolved_hypotheses, context, research_goal)  # Pass research_goal
            active_hypos = context.get_active_hypotheses()  # Update active list
            cycle_details["steps"]["evolution"] = {"hypotheses": [h.to_dict() for h in evolved_hypotheses]}
            # Add explicit step for reviewing evolved hypotheses AFTER evolution
            cycle_details["steps"]["reflection_evolved"] = {"hypotheses": [h.to_dict() for h in evolved_hypotheses]}
        else:
            cycle_details["steps"]["evolution"] = {"hypotheses": []}

        # 5. Ranking (Tournament 2 - includes evolved)
        _legacy.logger.info("Step 5: Ranking 2")
        self.ranking_agent.run_tournament(active_hypos, context, research_goal)  # Pass research_goal
        cycle_details["steps"]["ranking2"] = {"hypotheses": [h.to_dict() for h in active_hypos]}

        # 6. Proximity Analysis
        _legacy.logger.info("Step 6: Proximity Analysis")
        proximity_result = self.proximity_agent.build_proximity_graph(context)  # Pass context
        cycle_details["steps"]["proximity"] = {
            "adjacency_graph": proximity_result["adjacency_graph"],
            "nodes": proximity_result["nodes"],
            "edges": proximity_result["edges"],
        }

        # 7. Meta-review
        _legacy.logger.info("Step 7: Meta-Review")
        overview = self.meta_review_agent.summarize_and_feedback(context, proximity_result["adjacency_graph"])
        cycle_details["meta_review"] = overview
        # Add meta-review to steps for consistency
        cycle_details["steps"]["meta_review"] = overview

        # Increment iteration number at the end of the cycle
        context.iteration_number += 1
        _legacy.logger.info("--- Cycle %d Complete ---", context.iteration_number)
        return cycle_details
