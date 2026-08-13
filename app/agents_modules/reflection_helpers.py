"""Reflection-agent LLM helpers."""

from __future__ import annotations

import json
from typing import Dict, List

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..utils import logger
from .generation_helpers import _call_llm


def _strip_fenced_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _convert_score_to_review(score: int) -> str:
    """Convert a 1-10 numeric score to HIGH/MEDIUM/LOW review format.
    
    LOW: 1-4
    MEDIUM: 5-7
    HIGH: 8-10
    """
    if 1 <= score <= 4:
        return "LOW"
    elif 5 <= score <= 7:
        return "MEDIUM"
    elif 8 <= score <= 10:
        return "HIGH"
    else:
        return "UNREVIEWED"


def _parse_reflection_response(response: str, retrieved_sources: List[dict]) -> dict | None:
    try:
        cleaned_response = _strip_fenced_json(response)
        parsed_data = json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Error parsing LLM reflection response: %s", response, exc_info=True)
        return None

    # Parse all 7 numeric scores (1-10, no decimals)
    score_fields = [
        "alignment_score",
        "novelty_score",
        "feasibility_score",
        "plausibility_score",
        "testability_score",
        "evidence_quality_score",
        "expected_research_value_score",
    ]
    
    scores = {}
    for field in score_fields:
        try:
            score = int(parsed_data.get(field, 0))
            if not (1 <= score <= 10):
                logger.warning(
                    "Invalid score received for %s: %s (must be 1-10)",
                    field,
                    score,
                )
                return None
            scores[field] = score
        except (ValueError, TypeError):
            logger.warning(
                "Could not parse %s as integer: %s",
                field,
                parsed_data.get(field),
            )
            return None

    # Convert novelty_score and feasibility_score to HIGH/MEDIUM/LOW format
    novelty_review = _convert_score_to_review(scores["novelty_score"])
    feasibility_review = _convert_score_to_review(scores["feasibility_score"])

    review_data = {
        "novelty_review": novelty_review,
        "feasibility_review": feasibility_review,
        "alignment_score": scores["alignment_score"],
        "novelty_score": scores["novelty_score"],
        "feasibility_score": scores["feasibility_score"],
        "plausibility_score": scores["plausibility_score"],
        "testability_score": scores["testability_score"],
        "evidence_quality_score": scores["evidence_quality_score"],
        "expected_research_value_score": scores["expected_research_value_score"],
        # Other fields
        "comment": str(parsed_data.get("comment", "No comment provided.")),
        "references": [],
    }

    raw_refs = parsed_data.get("references", [])
    if isinstance(raw_refs, list):
        valid_source_ids = {
            str(src.get("source_id")) for src in retrieved_sources if isinstance(src, dict) and "source_id" in src
        }
        review_data["references"] = [
            ref for ref in raw_refs if isinstance(ref, str) and (ref in valid_source_ids or not valid_source_ids)
        ]
    else:
        logger.warning("Invalid references format received: %s", raw_refs)

    return review_data


def call_llm_for_reflection(
    hypothesis: Hypothesis,
    research_goal: ResearchGoal | None = None,
    context: ContextMemory | None = None,
    temperature: float = 0.3,
    model: str | None = None,
) -> Dict:
    """Evaluates a hypothesis against strictly provided retrieved sources to prevent hallucinated references."""
    logger.info("LLM reflection called for hypothesis %s", hypothesis.hypothesis_id)

    # Format the verified retrieved articles from context memory
    retrieved_sources = getattr(context, "last_retrieved_sources", [])
    if retrieved_sources:
        formatted_sources = "\n\n".join(
            f"Source ID: {src.get('source_id', 'Unknown')}\nTitle: {src.get('title', 'Untitled')}\nAbstract: {src.get('abstract', 'No abstract')}"
            for src in retrieved_sources if isinstance(src, dict)
        )
    else:
        formatted_sources = "No verified literature sources currently available in context memory."

    prompt = (
        "You are a rigorous scientific peer reviewer evaluating a candidate hypothesis.\n\n"
        "Research Goal:\n"
        f"{research_goal.description}\n\n"
        "Constraints:\n"
        f"{research_goal.constraints}\n\n"
        "Hypothesis to Review:\n"
        f"{hypothesis.text}\n\n"
        "Verified Retrieved Sources Available in Memory:\n"
        f"{formatted_sources}\n\n"
        "Review the hypothesis thoroughly and rate it on the following criteria using integer scores from 1 to 10 (no decimals):\n\n"
        "1. alignment_score (1-10): How well does this hypothesis align with the research goal and constraints?\n"
        "2. novelty_score (1-10): How original is this idea relative to existing literature? (1=No novelty, 10=Highly novel)\n"
        "3. feasibility_score (1-10): Can this be experimentally tested with current techniques? (1=Infeasible, 10=Highly feasible)\n"
        "4. plausibility_score (1-10): How theoretically sound and plausible is this hypothesis?\n"
        "5. testability_score (1-10): How clearly testable are the claims in this hypothesis?\n"
        "6. evidence_quality_score (1-10): How well supported is this hypothesis by the provided sources?\n"
        "7. expected_research_value_score (1-10): What is the potential impact and value of research on this hypothesis?\n\n"
        "Additionally, provide:\n"
        "- comment: Concise summary critique explaining the ratings and suggestions.\n"
        "- references: Array of exact Source IDs from the provided sources that support this hypothesis.\n\n"
        "STRICT CITATION RULE: In the 'references' array, return ONLY exact Source IDs from the 'Verified Retrieved Sources' list above. "
        "DO NOT invent, recall, or introduce any external paper titles, arXiv IDs, DOIs, or PMIDs from outside the provided text. "
        "If no provided sources are relevant, return an empty array [].\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "alignment_score": 1-10,\n'
        '  "novelty_score": 1-10,\n'
        '  "feasibility_score": 1-10,\n'
        '  "plausibility_score": 1-10,\n'
        '  "testability_score": 1-10,\n'
        '  "evidence_quality_score": 1-10,\n'
        '  "expected_research_value_score": 1-10,\n'
        '  "comment": "Concise summary critique explaining the ratings and suggestions.",\n'
        '  "references": ["exact Source ID from the provided list above"]\n'
        "}"
    )

    response = _call_llm(
        prompt,
        temperature=temperature,
        model=model,
        reasoning="off",
    )
    logger.info("LLM reflection response for hypothesis: %s", response)

    if response.startswith("Error:"):
        logger.error("LLM reflection call failed: %s", response)
        return {
            "novelty_review": "UNREVIEWED",
            "feasibility_review": "UNREVIEWED",
            "alignment_score": 0,
            "novelty_score": 0,
            "feasibility_score": 0,
            "plausibility_score": 0,
            "testability_score": 0,
            "evidence_quality_score": 0,
            "expected_research_value_score": 0,
            "comment": f"LLM review failed: {response}",
            "references": [],
        }

    review_data = _parse_reflection_response(response, retrieved_sources)
    if review_data is not None:
        logger.info("Parsed reflection data: %s", review_data)
        return review_data

    logger.warning(
        "Reflection review response did not validate; retrying with a format-only repair prompt."
    )
    schema_instruction = (
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "alignment_score": 1-10,\n'
        '  "novelty_score": 1-10,\n'
        '  "feasibility_score": 1-10,\n'
        '  "plausibility_score": 1-10,\n'
        '  "testability_score": 1-10,\n'
        '  "evidence_quality_score": 1-10,\n'
        '  "expected_research_value_score": 1-10,\n'
        '  "comment": "Concise summary critique explaining the ratings and suggestions.",\n'
        '  "references": ["exact Source ID from the provided list above"]\n'
        "}"
    )
    repair_prompt = (
        "The previous response did not satisfy the required JSON schema or contained invalid score values (must be integers 1-10). "
        "Reformat the candidate response below to valid JSON only, preserving the original scientific meaning. "
        "Do not add Markdown, commentary, or new information.\n\n"
        f"{schema_instruction}\n\n"
        f"Candidate response:\n{response}"
    )
    repaired_response = _call_llm(
        repair_prompt,
        temperature=0.0,
        model=model,
        reasoning="off",
    )
    logger.info("LLM reflection repair response for hypothesis: %s", repaired_response)

    if repaired_response.startswith("Error:"):
        logger.error("LLM reflection repair call failed: %s", repaired_response)
        return {
            "novelty_review": "UNREVIEWED",
            "feasibility_review": "UNREVIEWED",
            "alignment_score": 0,
            "novelty_score": 0,
            "feasibility_score": 0,
            "plausibility_score": 0,
            "testability_score": 0,
            "evidence_quality_score": 0,
            "expected_research_value_score": 0,
            "comment": f"LLM review failed during repair: {repaired_response}",
            "references": [],
        }

    review_data = _parse_reflection_response(repaired_response, retrieved_sources)
    if review_data is not None:
        logger.info("Parsed reflection data after repair: %s", review_data)
        return review_data

    logger.error(
        "Could not parse repaired LLM reflection response as valid review JSON: %s",
        repaired_response,
    )
    return {
        "novelty_review": "UNREVIEWED",
        "feasibility_review": "UNREVIEWED",
        "alignment_score": 0,
        "novelty_score": 0,
        "feasibility_score": 0,
        "plausibility_score": 0,
        "testability_score": 0,
        "evidence_quality_score": 0,
        "expected_research_value_score": 0,
        "comment": "Could not parse LLM response after format repair.",
        "references": [],
    }
