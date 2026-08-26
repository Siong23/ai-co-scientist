"""Hypothesis reflection agent."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..config import config
from ..models import ContextMemory, Hypothesis, ReflectionReport, ResearchGoal
from ..utils import logger, redact_secrets
from .reflection_helpers import (
    call_llm_for_hypothesis_revision,
    call_llm_for_reflection,
    evaluate_claims,
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
        configured_workers = config.get("agent_parallelism", {}).get("reflection_workers", 3)
        self.max_workers = max(1, int(configured_workers if max_workers is None else max_workers))

    @staticmethod
    def _review_one(
        hypothesis: Hypothesis,
        context: ContextMemory,
        research_goal: ResearchGoal,
        temperature: float,
    ) -> Dict:
        """Compute one independent review without mutating workflow state."""
        result = dict(
            call_llm_for_reflection(
                hypothesis=hypothesis,
                research_goal=research_goal,
                context=context,
                temperature=temperature,
                model=research_goal.llm_model,
            )
        )
        if _build_reflection_report(result) is not None:
            result.update(
                evaluate_claims(
                    hypothesis,
                    evidence_quality_score=result["evidence_quality_score"],
                    plausibility_score=result["plausibility_score"],
                    model=research_goal.llm_model,
                    sub_claims=result.get("sub_claims"),
                )
            )
        return result

    @staticmethod
    def _apply_review(hypothesis: Hypothesis, result: Dict) -> None:
        """Apply a completed review on the coordinator thread."""
        hypothesis.novelty_review = result["novelty_review"]
        hypothesis.feasibility_review = result["feasibility_review"]
        if result["comment"] != "Could not parse LLM response.":
            hypothesis.review_comments.append(result["comment"])
        hypothesis.review_reference_ids = list(result["references"])
        hypothesis.reflection_report = _build_reflection_report(result)

    def review_hypotheses(
        self, hypotheses: List[Hypothesis], context: ContextMemory, research_goal: ResearchGoal
    ) -> None:
        """Reviews hypotheses using LLM, based on research_goal settings."""
        # Use reflection temperature from research_goal
        reflect_temp = research_goal.reflection_temperature

        if not hypotheses:
            return

        def review(hypothesis: Hypothesis) -> Dict:
            return self._review_one(hypothesis, context, research_goal, reflect_temp)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(hypotheses))) as executor:
            # map preserves input order; shared model objects are mutated only
            # after every worker has completed.
            results = list(executor.map(review, hypotheses))

        for h, result in zip(hypotheses, results):
            self._apply_review(h, result)

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

    def revise_hypotheses(self, hypotheses: List[Hypothesis], research_goal: ResearchGoal) -> List[Hypothesis]:
        """Revise REVISE-flagged hypotheses using LLM revision helper."""
        if not hypotheses:
            return []

        def revise(hypothesis: Hypothesis) -> Optional[Dict]:
            try:
                return call_llm_for_hypothesis_revision(
                    hypothesis,
                    research_goal,
                    temperature=research_goal.generation_temperature,
                    model=research_goal.llm_model,
                )
            except Exception as exc:
                logger.warning(
                    "Hypothesis revision failed for %s: %s",
                    hypothesis.hypothesis_id,
                    redact_secrets(str(exc)),
                )
                return None

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(hypotheses))) as executor:
            revisions = list(executor.map(revise, hypotheses))

        revised_list = []
        for hypo, revised in zip(hypotheses, revisions):
            if revised and isinstance(revised, dict):
                if revised.get("title"):
                    hypo.title = revised["title"]
                new_text = revised.get("hypothesis") or revised.get("text")
                if new_text:
                    hypo.text = new_text
                logger.info("Revised hypothesis %s after REVISE verdict.", hypo.hypothesis_id)
                revised_list.append(hypo)
        return revised_list
