"""Hypothesis ranking agent."""

from __future__ import annotations

import random
from typing import List

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ._compat import _legacy


class RankingAgent:
    def run_tournament(self, hypotheses: List[Hypothesis], context: ContextMemory, research_goal: ResearchGoal) -> None:
        """Runs a pairwise tournament to rank hypotheses, using research_goal settings."""
        # Use k_factor from research_goal
        k_factor = research_goal.elo_k_factor

        if len(hypotheses) < 2:
            _legacy.logger.info("Not enough hypotheses to run a tournament.")
            return

        active_hypotheses = [h for h in hypotheses if h.is_active]
        if len(active_hypotheses) < 2:
            _legacy.logger.info("Not enough *active* hypotheses to run a tournament.")
            return

        random.shuffle(active_hypotheses)  # Shuffle only active ones

        # Simple round-robin: each active hypothesis debates every other active one once
        pairs = []
        for i in range(len(active_hypotheses)):
            for j in range(i + 1, len(active_hypotheses)):
                pairs.append((active_hypotheses[i], active_hypotheses[j]))

        _legacy.logger.info(f"Running tournament with {len(pairs)} pairs.")
        for hA, hB in pairs:
            # winner = _legacy.run_pairwise_debate(hA, hB)
            decision = _legacy.run_pairwise_debate(
                hA,
                hB,
                research_goal,
            )
            # outcome = decision.outcome

            # loser = hB if winner == hA else hA
            # # Pass the specific k_factor
            # _legacy.update_elo(winner, loser, k_factor=k_factor)

            # Elo Update
            if decision.outcome == "A":
                _legacy.update_elo(hA, hB, k_factor = k_factor)
            elif decision.outcome == "B":
                _legacy.update_elo(hB, hA, k_factor = k_factor)
            elif decision.outcome == "TIE":
                # Handle tie: no Elo update, but log it
                _legacy.logger.info(f"Tie between {hA.hypothesis_id} and {hB.hypothesis_id}. No Elo update.")
                _legacy.update_elo_tie(hA, hB, k_factor = k_factor)
            elif decision.outcome == "ABSTAIN":
                # Handle abstain: no Elo update, but log it
                _legacy.logger.info(f"Abstain between {hA.hypothesis_id} and {hB.hypothesis_id}. No Elo update.")

            # Record result in context (consider if this needs iteration info)
            context.tournament_results.append(
                {
                    "iteration": context.iteration_number,  # Add iteration number
                    "hypothesis_a": hA.hypothesis_id,
                    "hypothesis_b": hB.hypothesis_id,
                    "outcome": decision.outcome,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "elo_a_after": hA.elo_score,
                    "elo_b_after": hB.elo_score,
                    "scores_a": decision.scores_a,
                    "scores_b": decision.scores_b,
                    "criteria": decision.decisive_criteria,
                }
            )