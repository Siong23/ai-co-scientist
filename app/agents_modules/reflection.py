"""Hypothesis reflection agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..config import config
from ..models import ContextMemory, Hypothesis, ReflectionReport, ResearchGoal
from ..utils import execution_cancelled, logger, redact_secrets
from .reflection_helpers import (
    call_llm_for_hypothesis_revision,
    call_llm_for_reflection,
    evaluate_claims,
    recommendation_after_claim_assessment,
)


def _build_reflection_report(result: Dict) -> Optional[ReflectionReport]:
    # A transport or parse failure is not a scientific review.  Keeping the
    # report absent ensures downstream ranking can abstain instead of treating
    # synthetic zero scores as evidence about the hypothesis's quality.
    if result.get("recommendation") == "UNREVIEWED":
        return None

    return ReflectionReport(
        alignment_score=result.get("alignment_score", 0),
        novelty_score=result.get("novelty_score", 0),
        feasibility_score=result.get("feasibility_score", 0),
        plausibility_score=result.get("plausibility_score", 0),
        testability_score=result.get("testability_score", 0),
        evidence_quality_score=result.get("evidence_quality_score", 0),
        expected_research_value_score=result.get("expected_research_value_score", 0),
        strengths=result.get("strengths", []),
        weaknesses=result.get("weaknesses", []),
        recommendation=result.get("recommendation", "UNREVIEWED"),
        claims=result.get("claims", []),
        overall_confidence=result.get("overall_confidence", 1.0),
        review_comments=[result["comment"]] if result.get("comment") else [],
    )


class ReflectionAgent:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def review_hypotheses(
        self, hypotheses: List[Hypothesis], context: ContextMemory, research_goal: ResearchGoal
    ) -> None:
        """Reviews hypotheses using LLM, based on research_goal settings."""
        reflect_temp = research_goal.reflection_temperature
        pending = [hypothesis for hypothesis in hypotheses if hypothesis.reflection_report is None]
        if not pending or execution_cancelled():
            return
        configured_workers = int(self.max_workers if self.max_workers is not None else config.get("reflection", {}).get("max_workers", config.get("agent_parallelism", {}).get("reflection_workers", 3)))
        max_workers = max(1, min(configured_workers, len(pending)))

        def review_one(h: Hypothesis) -> None:
            result = call_llm_for_reflection(
                hypothesis=h,
                research_goal=research_goal,
                context=context,
                temperature=reflect_temp,
                model=research_goal.llm_model,
            )
            h.novelty_review = result["novelty_review"]
            h.feasibility_review = result["feasibility_review"]
            if result["comment"] != "Could not parse LLM response.":
                h.review_comments.append(result["comment"])
            # This field describes the current review, so replace stale IDs
            # instead of accumulating duplicates across repeated cycles.
            # Hypothesis.references remains reserved for source dictionaries.
            h.review_reference_ids = list(result["references"])

            reflection_report = _build_reflection_report(result)
            if reflection_report is not None:
                result.update(
                    evaluate_claims(
                        h,
                        evidence_quality_score=result["evidence_quality_score"],
                        plausibility_score=result["plausibility_score"],
                        model=research_goal.llm_model,
                        claims=result.get("sub_claims"),
                    )
                )
                result["recommendation"] = recommendation_after_claim_assessment(result)
                reflection_report = _build_reflection_report(result)
            h.reflection_report = reflection_report

            # Reflection owns review artifacts only.  It must not rewrite a
            # hypothesis under the same ID because that would invalidate its
            # evidence, review history, and tournament provenance.  A REVISE
            # recommendation is consumed by Evolution, which creates a child.

            logger.info(
                "Reviewed hypothesis: %s, Novelty: %s, Feasibility: %s",
                h.hypothesis_id,
                h.novelty_review,
                h.feasibility_review,
            )

        if max_workers == 1:
            for hypothesis in pending:
                review_one(hypothesis)
            return

        logger.info("Reviewing %d hypotheses with %d workers.", len(pending), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(review_one, pending))

    def revise_hypotheses(
        self, hypotheses: List[Hypothesis], research_goal: ResearchGoal
    ) -> List[Hypothesis]:
        """Revise REVISE-flagged hypotheses using LLM revision helper."""
        if not hypotheses or execution_cancelled():
            return []

        def revise_one(hypo: Hypothesis) -> Hypothesis | None:
            try:
                revised = call_llm_for_hypothesis_revision(
                    hypo,
                    research_goal,
                    temperature=research_goal.generation_temperature,
                    model=research_goal.llm_model,
                )
                if revised and isinstance(revised, dict):
                    if revised.get("title"):
                        hypo.title = revised["title"]
                    new_text = revised.get("hypothesis") or revised.get("text")
                    if new_text:
                        hypo.text = new_text
                    logger.info(
                        "Revised hypothesis %s after REVISE verdict.", hypo.hypothesis_id
                    )
                    return hypo
            except Exception as exc:
                logger.warning(
                    "Hypothesis revision failed for %s: %s",
                    hypo.hypothesis_id,
                    redact_secrets(str(exc)),
                )
            return None

        configured_workers = int(self.max_workers if self.max_workers is not None else config.get("reflection", {}).get("max_workers", config.get("agent_parallelism", {}).get("reflection_workers", 3)))
        max_workers = max(1, min(configured_workers, len(hypotheses)))
        if max_workers == 1:
            results = [revise_one(hypothesis) for hypothesis in hypotheses]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(revise_one, hypotheses))
        return [hypothesis for hypothesis in results if hypothesis is not None]
