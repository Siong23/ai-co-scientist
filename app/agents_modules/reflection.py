"""Hypothesis reflection agent."""

from __future__ import annotations

from typing import List

from ..models import ContextMemory, Hypothesis, ResearchGoal
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
            result = _legacy.call_llm_for_reflection(h.text, temperature=reflect_temp, model=research_goal.llm_model)
            h.novelty_review = result["novelty_review"]
            h.feasibility_review = result["feasibility_review"]
            # Append comment only if it's not the default error message
            if result["comment"] != "Could not parse LLM response.":
                h.review_comments.append(result["comment"])
            # Only extend references if the list is not empty
            if result["references"]:
                h.references.extend(result["references"])
            _legacy.logger.info(
                "Reviewed hypothesis: %s, Novelty: %s, Feasibility: %s",
                h.hypothesis_id,
                h.novelty_review,
                h.feasibility_review,
            )
