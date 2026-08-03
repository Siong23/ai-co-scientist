"""Hypothesis evolution agent."""

from __future__ import annotations

from typing import List

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ._compat import _legacy


class EvolutionAgent:
    def evolve_hypotheses(self, context: ContextMemory, research_goal: ResearchGoal) -> List[Hypothesis]:
        """Evolves hypotheses by combining top candidates, using research_goal settings."""
        # Use top_k from research_goal
        top_k = research_goal.top_k_hypotheses
        active = context.get_active_hypotheses()
        if len(active) < 2:
            _legacy.logger.info("Not enough active hypotheses to perform evolution.")
            return []

        sorted_by_elo = sorted(active, key=lambda h: h.elo_score, reverse=True)
        top_candidates = sorted_by_elo[:top_k]

        new_hypotheses = []
        # Combine the top two for now, could be extended
        if len(top_candidates) >= 2:
            # Optional: Add check to prevent combining very similar hypotheses
            # sim = similarity_score(top_candidates[0].text, top_candidates[1].text)
            # if sim < 0.8: # Example threshold
            new_h = _legacy.combine_hypotheses(top_candidates[0], top_candidates[1])
            _legacy.logger.info("Evolved hypothesis created: %s from parents %s", new_h.hypothesis_id, new_h.parent_ids)
            new_hypotheses.append(new_h)
            # else:
            #     _legacy.logger.info("Skipping evolution: Top 2 hypotheses are too similar (score: %.2f)", sim)

        return new_hypotheses
