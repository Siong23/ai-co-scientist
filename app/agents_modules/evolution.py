"""Hypothesis evolution agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from ..config import config
from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..utils import execution_cancelled, logger
from .evolution_helpers import (
    EVOLUTION_STRATEGIES,
    EvolutionStrategy,
    call_llm_for_evolution,
    create_evolved_hypothesis,
    resolve_parent_evidence,
)

# Reflection's ACCEPT rubric passes a hypothesis at 5 and above, so a score
# below that is the gap an evolution strategy has to close.
_ACCEPT_THRESHOLD = 5.0

# Strategies that draw on every selected parent; the rest refine one parent.
_MULTI_PARENT_STRATEGIES = frozenset({"combination", "inspiration", "out_of_box"})

# Which strategy repairs which measured weakness.  Plausibility and feasibility
# share a strategy because its instruction covers both: repair invalid
# assumptions, and describe an implementable validation path.
_WEAKNESS_STRATEGIES: dict[str, EvolutionStrategy] = {
    "feasibility_score": "feasibility",
    "plausibility_score": "feasibility",
    "evidence_quality_score": "grounding",
    "testability_score": "simplification",
    "novelty_score": "out_of_box",
}


class EvolutionAgent:
    """Iteratively create new tournament candidates from top-ranked parents."""

    def __init__(
        self,
        strategies: tuple[EvolutionStrategy, ...] | None = None,
        max_candidates_per_cycle: int | None = None,
        quality_repair_attempts: int | None = None,
        transport_retry_attempts: int | None = None,
        max_workers: int | None = None,
    ):
        evolution_config = config.get("evolution", {})
        configured = tuple(evolution_config.get("strategies", EVOLUTION_STRATEGIES))
        valid_strategies = tuple(strategy for strategy in configured if strategy in EVOLUTION_STRATEGIES)
        self.strategies = strategies or valid_strategies or EVOLUTION_STRATEGIES
        configured_limit = evolution_config.get("max_candidates_per_cycle", 3)
        self.max_candidates_per_cycle = max(1, int(max_candidates_per_cycle or configured_limit))
        configured_repairs = evolution_config.get("quality_repair_attempts", 1)
        self.quality_repair_attempts = max(
            0,
            int(configured_repairs if quality_repair_attempts is None else quality_repair_attempts),
        )
        configured_transport_retries = evolution_config.get("transport_retry_attempts", 2)
        self.transport_retry_attempts = max(
            0,
            int(configured_transport_retries if transport_retry_attempts is None else transport_retry_attempts),
        )
        self.max_tokens = int(config.get("llm_max_tokens", {}).get("evolution", 2048))
        self.max_workers = max(
            1,
            int(
                max_workers
                if max_workers is not None
                else evolution_config.get(
                    "max_workers", config.get("agent_parallelism", {}).get("evolution_workers", 3)
                )
            ),
        )

    @staticmethod
    def _strategy_deficits(parents: Sequence[Hypothesis]) -> dict[EvolutionStrategy, float]:
        """Score how badly each strategy is needed by the parents under review.

        Rotating blindly through the library means the strategy that addresses
        a parent's actual weakness may not come round for several cycles, so a
        hypothesis can be evolved repeatedly without its worst dimension ever
        being worked on.  Reflection already measured those dimensions; this
        turns them into selection pressure.

        Each dimension contributes how far the worst parent falls below the
        review rubric's ACCEPT threshold, so a strategy is prioritized in
        proportion to the size of the gap it addresses.
        """
        deficits: dict[EvolutionStrategy, float] = {}
        for parent in parents:
            report = getattr(parent, "reflection_report", None)
            if report is None:
                continue
            for field, strategy in _WEAKNESS_STRATEGIES.items():
                score = getattr(report, field, None)
                if not isinstance(score, (int, float)):
                    continue
                gap = max(0.0, _ACCEPT_THRESHOLD - float(score))
                if gap > 0:
                    deficits[strategy] = max(deficits.get(strategy, 0.0), gap)

            # A peripheral assumption the deep verification review contradicted
            # is exactly what the feasibility strategy exists to repair.
            if any(assumption.status == "INVALID" for assumption in getattr(report, "assumptions", []) or []):
                deficits["feasibility"] = max(deficits.get("feasibility", 0.0), _ACCEPT_THRESHOLD)
        return deficits

    def _strategies_for_cycle(self, context: ContextMemory, parents: Sequence[Hypothesis]) -> list[EvolutionStrategy]:
        """Order the strategy library by what the parents actually need.

        Rotation still sets the baseline order, so a cycle whose parents carry
        no reviews behaves exactly as before and the whole library keeps being
        explored over time.  Measured weaknesses only reorder that baseline.
        """
        if not self.strategies:
            return []
        parent_count = len(parents)
        start = (context.iteration_number * self.max_candidates_per_cycle) % len(self.strategies)
        ordered = list(self.strategies[start:] + self.strategies[:start])

        deficits = self._strategy_deficits(parents)
        if deficits:
            ordered.sort(key=lambda strategy: -deficits.get(strategy, 0.0))

        selected = []
        for strategy in ordered:
            if parent_count < 2 and strategy in _MULTI_PARENT_STRATEGIES:
                continue
            selected.append(strategy)
            if len(selected) >= self.max_candidates_per_cycle:
                break

        # The paper's Evolution agent ends every pass with an out-of-the-box
        # idea. Rotation only reaches it on a second Cycle, so a single Cycle's
        # children all refined one idea and landed in one cluster. With three
        # slots the divergent strategy takes the last one; with fewer it would
        # displace the repair the parents' reviews asked for.
        if (
            parent_count >= 2
            and self.max_candidates_per_cycle >= 3
            and "out_of_box" in self.strategies
            and "out_of_box" not in selected
        ):
            selected[-1] = "out_of_box"
        return selected

    def _assign_parents(
        self,
        strategies: Sequence[EvolutionStrategy],
        parents: Sequence[Hypothesis],
    ) -> dict[EvolutionStrategy, list[Hypothesis]]:
        """Map each strategy to the parents it evolves.

        Multi-parent strategies see every parent. Two refinements of the same
        parent converge on the same fix - one run's feasibility and
        simplification children of one parent were near-duplicates - so each
        single-parent strategy takes a parent of its own while one is free:
        the free parent whose review shows the largest gap that strategy
        repairs, then the higher-ranked one.
        """
        assignments: dict[EvolutionStrategy, list[Hypothesis]] = {}
        free = list(parents)
        for strategy in strategies:
            if strategy in _MULTI_PARENT_STRATEGIES:
                assignments[strategy] = list(parents)
                continue
            pool = free or list(parents)
            chosen = max(pool, key=lambda parent: self._strategy_deficits([parent]).get(strategy, 0.0))
            if chosen in free:
                free.remove(chosen)
            assignments[strategy] = [chosen]
        return assignments

    def evolve_hypotheses(self, context: ContextMemory, research_goal: ResearchGoal) -> List[Hypothesis]:
        """Create independently reviewable children without replacing their parents."""
        context.last_evolution_attempts = []
        active = context.get_active_hypotheses()
        if not active:
            logger.info("No active hypotheses to evolve.")
            return []

        parent_count = max(1, int(research_goal.top_k_hypotheses))
        top_candidates = self._select_parents(
            active,
            parent_count,
            getattr(context, "proximity_analysis", None),
            # Before any decided match every Elo is equal, so there is nothing to rank by.
            ranked_ids=context.ranked_hypothesis_ids() or None,
        )
        strategies = self._strategies_for_cycle(context, top_candidates)
        strategy_parents = self._assign_parents(strategies, top_candidates)

        def evolve_one(strategy: EvolutionStrategy) -> tuple[Hypothesis | None, list[dict]]:
            diagnostics: list[dict] = []
            if execution_cancelled():
                return None, diagnostics
            parents = strategy_parents[strategy]
            evidence_sources = resolve_parent_evidence(
                parents,
                context.last_retrieved_sources,
            )
            candidate = call_llm_for_evolution(
                strategy,
                parents,
                research_goal,
                max_tokens=self.max_tokens,
                evidence_sources=evidence_sources,
                diagnostics=diagnostics,
                quality_repair_attempts=self.quality_repair_attempts,
                transport_retry_attempts=self.transport_retry_attempts,
                meta_review_feedback=getattr(context, "meta_review_feedback", None),
            )
            if candidate is None:
                return None, diagnostics
            evolved = create_evolved_hypothesis(
                candidate,
                parents,
                strategy,
                evidence_sources=evidence_sources,
            )
            logger.debug(
                "Evolved hypothesis %s created with strategy %s from parents %s",
                evolved.hypothesis_id,
                strategy,
                evolved.parent_ids,
            )
            return evolved, diagnostics

        max_workers = min(self.max_workers, len(strategies))
        if max_workers <= 1:
            results = [evolve_one(strategy) for strategy in strategies]
        else:
            logger.debug("Evolving %d candidates with %d workers.", len(strategies), max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(evolve_one, strategies))
        for _, diagnostics in results:
            context.last_evolution_attempts.extend(diagnostics)
        new_hypotheses = [candidate for candidate, _ in results if candidate is not None]

        # A stitched fallback is not a new scientific hypothesis and can receive an
        # artificial ranking advantage from its length. Keep the parents unchanged
        # when every strategy fails instead of adding a misleading tournament entry.
        if not new_hypotheses and len(top_candidates) >= 2:
            logger.warning(
                "All Evolution strategies failed; no evolved hypothesis was created from parents %s.",
                [parent.hypothesis_id for parent in top_candidates],
            )

        return new_hypotheses

    @staticmethod
    def _select_parents(
        active: List[Hypothesis],
        parent_count: int,
        proximity_data: dict | None,
        ranked_ids: set | None = None,
    ) -> List[Hypothesis]:
        """Prefer one strong exemplar per cluster, then fill by Elo.

        Until a tournament match is played every Elo is the initial 1200, and
        sorting on the ID alone picked the same Generation parents on every
        pass, sending an identical Evolution prompt again.  Ties therefore go
        first to hypotheses no earlier pass has evolved, then to the stronger
        Reflection verdict.

        When ``ranked_ids`` names the hypotheses that have played a match, those
        come first by Elo: an unranked hypothesis still holds the default 1200,
        which would otherwise outrank a ranked one that lost a close match.
        """
        by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in active}
        selected = []
        exemplar_ids = (proximity_data or {}).get("exemplar_ids", [])
        for hypothesis_id in exemplar_ids:
            hypothesis = by_id.get(hypothesis_id)
            if hypothesis is not None and hypothesis not in selected:
                selected.append(hypothesis)
            if len(selected) >= parent_count:
                return selected

        evolved_ids = {
            parent_id for hypothesis in active for parent_id in (getattr(hypothesis, "parent_ids", None) or [])
        }

        def rank(hypothesis: Hypothesis) -> tuple:
            report = getattr(hypothesis, "reflection_report", None)
            ranked = ranked_ids is None or hypothesis.hypothesis_id in ranked_ids
            return (
                ranked,
                hypothesis.elo_score if ranked else 0.0,
                hypothesis.hypothesis_id not in evolved_ids,
                str(getattr(report, "recommendation", "")).upper() == "ACCEPT",
                float(getattr(report, "alignment_score", 0) or 0),
                float(getattr(report, "overall_confidence", 0) or 0),
                hypothesis.hypothesis_id,
            )

        for hypothesis in sorted(active, key=rank, reverse=True):
            if hypothesis not in selected:
                selected.append(hypothesis)
            if len(selected) >= parent_count:
                break
        return selected
