"""Hypothesis reflection agent."""

from __future__ import annotations

from typing import Dict, List

from ..models import ContextMemory, Hypothesis, ReflectionReport, ResearchGoal
from ._compat import _legacy

# Bounds the REVISE-fix loop so a stubborn low-scoring hypothesis cannot stall the pipeline.
_MAX_REVISION_ATTEMPTS = 3


def _build_reflection_report(result: Dict) -> ReflectionReport:
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
        review_comments=[result["comment"]] if result.get("comment") else [],
    )


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

            reflection_report = _build_reflection_report(result)
            h.reflection_report = reflection_report

            # A criterion scored below 4/10: attempt bounded LLM-driven revision,
            # re-reviewing after each attempt until it clears the bar or attempts run out.
            attempts = 0
            while reflection_report.recommendation == "REVISE" and attempts < _MAX_REVISION_ATTEMPTS:
                revised = _legacy.call_llm_for_hypothesis_revision(
                    hypothesis=h,
                    research_goal=research_goal,
                    temperature=reflect_temp,
                    model=research_goal.llm_model,
                )
                if revised is None:
                    break

                h.title = str(revised["title"]).strip()
                h.text = (
                    f"Hypothesis: {str(revised['hypothesis']).strip()}\n\n"
                    f"Rationale: {str(revised['rationale']).strip()}\n\n"
                    f"Feasibility: {str(revised['feasibility']).strip()}"
                )

                result = _legacy.call_llm_for_reflection(
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
                if result["references"]:
                    h.references.extend(result["references"])

                reflection_report = _build_reflection_report(result)
                h.reflection_report = reflection_report
                attempts += 1

            _legacy.logger.info(
                "Reviewed hypothesis: %s, Novelty: %s, Feasibility: %s",
                h.hypothesis_id,
                h.novelty_review,
                h.feasibility_review,
            )

