"""Hypothesis ranking agent."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterable, List

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..utils import logger
from .ranking_helpers import run_pairwise_debate, update_elo, update_elo_tie


class RankingAgent:
    def run_tournament(
        self,
        hypotheses: List[Hypothesis],
        context: ContextMemory,
        research_goal: ResearchGoal,
        new_hypotheses: Iterable[Hypothesis] | None = None,
    ) -> None:
        """Rank active hypotheses, optionally comparing only newly introduced ones."""
        # Use k_factor from research_goal
        k_factor = research_goal.elo_k_factor

        if len(hypotheses) < 2:
            logger.info("Not enough hypotheses to run a tournament.")
            return

        active_hypotheses = [h for h in hypotheses if h.is_active]
        if len(active_hypotheses) < 2:
            logger.info("Not enough *active* hypotheses to run a tournament.")
            return

        random.shuffle(active_hypotheses)  # Shuffle only active ones

        new_hypothesis_ids = (
            {hypothesis.hypothesis_id for hypothesis in new_hypotheses}
            if new_hypotheses is not None
            else None
        )

        # Compare every pair for the first tournament. In later tournaments,
        # only compare pairs containing a newly generated or evolved hypothesis:
        # old-vs-old outcomes are already represented in their Elo scores.
        pairs = []
        for i in range(len(active_hypotheses)):
            for j in range(i + 1, len(active_hypotheses)):
                h_a, h_b = active_hypotheses[i], active_hypotheses[j]
                if new_hypothesis_ids is None or (
                    h_a.hypothesis_id in new_hypothesis_ids
                    or h_b.hypothesis_id in new_hypothesis_ids
                ):
                    pairs.append((h_a, h_b))

        if not pairs:
            logger.info("No new hypotheses require ranking comparisons.")
            return

        for h in active_hypotheses:
            logger.info(
                "Ranking input: %s | reflection_report=%s",
                h.hypothesis_id,
                h.reflection_report is not None,
            )

        logger.info(f"Running tournament with {len(pairs)} pairs.")

        # ---- Parallel LLM Debates ----
        def run_match(pair):
            hA, hB = pair
            try:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] START {hA.hypothesis_id} vs {hB.hypothesis_id}"
                )
                decision = run_pairwise_debate(
                    hA,
                    hB,
                    research_goal
                )
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] END {hA.hypothesis_id} vs {hB.hypothesis_id}"
                )
                return hA, hB, decision

            except Exception as e:
                logger.error(
                    f"Ranking failed for {hA.hypothesis_id} vs {hB.hypothesis_id}: {e}"
                )
                return None

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(
                executor.map(run_match, pairs)
            )

        # ---- Sequential Elo Update + Save Results ----
        for result in results:
            if result is None:
                continue

            hA, hB, decision = result

            # ------------------------------------------------------------
            # Safety gate: never update Elo without valid Reflection scores
            # ------------------------------------------------------------
            if decision.outcome in {"A", "B", "TIE"}:
                if not decision.scores_a or not decision.scores_b:
                    logger.warning(
                        "Skipping Elo update for %s vs %s because ranking "
                        "scores are missing.",
                        hA.hypothesis_id,
                        hB.hypothesis_id,
                    )

                    decision.outcome = "ABSTAIN"

                    if not decision.reasoning:
                        decision.reasoning = (
                            "Elo update skipped because one or both hypotheses "
                            "lack valid Reflection-based ranking scores."
                        )

                    continue

            # ------------------------------------------------------------
            # Elo update
            # ------------------------------------------------------------
            if decision.outcome == "A":
                update_elo(
                    hA,
                    hB,
                    k_factor=k_factor
                )
            elif decision.outcome == "B":
                update_elo(
                    hB,
                    hA,
                    k_factor=k_factor
                )
            elif decision.outcome == "TIE":
                update_elo_tie(
                    hA,
                    hB,
                    k_factor=k_factor
                )
            elif decision.outcome == "ABSTAIN":
                logger.info(
                    f"Judge abstained: no clear winner determined between "
                    f"{hA.hypothesis_id} and {hB.hypothesis_id}."
                )
                if not decision.reasoning:
                    decision.reasoning = (
                        "The judge could not determine a clear winner "
                        "after evaluating both hypotheses."
                    )

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
