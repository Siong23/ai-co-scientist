"""Hypothesis reflection agent."""

from __future__ import annotations

from typing import List

from ..models import ContextMemory, Hypothesis, ReflectionReport, ResearchGoal
from ._compat import _legacy


class ReflectionAgent:
    def review_hypotheses(
        self, hypotheses: List[Hypothesis], context: ContextMemory, research_goal: ResearchGoal
    ) -> None:
        """Reviews hypotheses using LLM, based on research_goal settings."""
        # Use reflection temperature from research_goal
        reflect_temp = research_goal.reflection_temperature

        for h in hypotheses:
            # Avoid re-reviewing if already reviewed (optional optimization)
            # if h.novelty_review is not None and h.feasibility_review is not None:
            #    continue
            # Pass the specific temperature
            result = _legacy.call_llm_for_reflection(hypothesis=h, research_goal=research_goal, context=context, temperature=reflect_temp, model=research_goal.llm_model,)
            h.novelty_review = result["novelty_review"]
            h.feasibility_review = result["feasibility_review"]
            if result["comment"] != "Could not parse LLM response.":
                h.review_comments.append(result["comment"])
            if result["references"]:
                h.references.extend(result["references"])
            
            # Create and store ReflectionReport with all scores
            reflection_report = ReflectionReport(
                alignment_score=result.get("alignment_score", 0),
                novelty_score=result.get("novelty_score", 0),
                feasibility_score=result.get("feasibility_score", 0),
                plausibility_score=result.get("plausibility_score", 0),
                testability_score=result.get("testability_score", 0),
                evidence_quality_score=result.get("evidence_quality_score", 0),
                expected_research_value_score=result.get("expected_research_value_score", 0),
                review_comments=[result["comment"]] if result.get("comment") else [],
            )
            h.reflection_report = reflection_report
            
            _legacy.logger.info(
                "Reviewed hypothesis: %s, Novelty: %s, Feasibility: %s",
                h.hypothesis_id,
                h.novelty_review,
                h.feasibility_review,
            )
