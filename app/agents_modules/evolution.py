"""Hypothesis evolution agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

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


class EvolutionAgent:
    """Iteratively create new tournament candidates from top-ranked parents."""

    def __init__(
        self,
        strategies: tuple[EvolutionStrategy, ...] | None = None,
        max_candidates_per_cycle: int | None = None,
        quality_repair_attempts: int | None = None,
        transport_retry_attempts: int | None = None,
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
            int(
                configured_transport_retries
                if transport_retry_attempts is None
                else transport_retry_attempts
            ),
        )
        self.max_tokens = int(config.get("llm_max_tokens", {}).get("evolution", 2048))
        self.max_workers = max(1, int(evolution_config.get("max_workers", 3)))

    def _strategies_for_cycle(self, context: ContextMemory, parent_count: int) -> list[EvolutionStrategy]:
        """Rotate through the strategy library while respecting parent-count requirements."""
        if not self.strategies:
            return []
        start = (context.iteration_number * self.max_candidates_per_cycle) % len(self.strategies)
        ordered = self.strategies[start:] + self.strategies[:start]
        selected = []
        for strategy in ordered:
            if parent_count < 2 and strategy in {"combination", "inspiration", "out_of_box"}:
                continue
            selected.append(strategy)
            if len(selected) >= self.max_candidates_per_cycle:
                break
        return selected

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
        )
        strategies = self._strategies_for_cycle(context, len(top_candidates))
        def evolve_one(strategy: EvolutionStrategy) -> tuple[Hypothesis | None, list[dict]]:
            diagnostics: list[dict] = []
            if execution_cancelled():
                return None, diagnostics
            parents = top_candidates if strategy in {"combination", "inspiration", "out_of_box"} else top_candidates[:1]
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
            logger.info(
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
            logger.info("Evolving %d candidates with %d workers.", len(strategies), max_workers)
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
    ) -> List[Hypothesis]:
        """Prefer one strong exemplar per cluster, then fill by Elo."""
        by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in active}
        selected = []
        exemplar_ids = (proximity_data or {}).get("exemplar_ids", [])
        for hypothesis_id in exemplar_ids:
            hypothesis = by_id.get(hypothesis_id)
            if hypothesis is not None and hypothesis not in selected:
                selected.append(hypothesis)
            if len(selected) >= parent_count:
                return selected

        for hypothesis in sorted(active, key=lambda item: (item.elo_score, item.hypothesis_id), reverse=True):
            if hypothesis not in selected:
                selected.append(hypothesis)
            if len(selected) >= parent_count:
                break
        return selected
