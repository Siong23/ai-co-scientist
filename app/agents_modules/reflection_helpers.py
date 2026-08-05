"""Reflection-agent LLM helpers."""

from __future__ import annotations

import json
from typing import Dict

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..utils import logger
from .generation_helpers import _call_llm


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
        "Review the hypothesis thoroughly. Evaluate:\n"
        "1. Novelty (HIGH, MEDIUM, LOW): How original is this idea relative to existing literature?\n"
        "2. Feasibility (HIGH, MEDIUM, LOW): Can this be experimentally tested with current techniques?\n"
        "3. Strengths & Weaknesses: Specific scientific feedback and potential failure modes.\n\n"
        "STRICT CITATION RULE: In the 'references' array, return ONLY exact Source IDs from the 'Verified Retrieved Sources' list above. "
        "DO NOT invent, recall, or introduce any external paper titles, arXiv IDs, DOIs, or PMIDs from outside the provided text. "
        "If no provided sources are relevant, return an empty array [].\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "novelty_review": "HIGH | MEDIUM | LOW",\n'
        '  "feasibility_review": "HIGH | MEDIUM | LOW",\n'
        '  "comment": "Concise summary critique explaining the ratings and suggestions.",\n'
        '  "references": ["exact Source ID from the provided list above"]\n'
        "}"
    )

    response = _call_llm(prompt, temperature=temperature, model=model)
    logger.info("LLM reflection response for hypothesis: %s", response)

    if response.startswith("Error:"):
        logger.error("LLM reflection call failed: %s", response)
        return {
            "novelty_review": "UNREVIEWED",
            "feasibility_review": "UNREVIEWED",
            "comment": f"LLM review failed: {response}",
            "references": [],
        }

    review_data = {
        "novelty_review": "UNREVIEWED",
        "feasibility_review": "UNREVIEWED",
        "comment": "Could not parse LLM response.",
        "references": [],
    }

    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        parsed_data = json.loads(cleaned_response)

        novelty = parsed_data.get("novelty_review", "UNREVIEWED").upper()
        if novelty in ["HIGH", "MEDIUM", "LOW"]:
            review_data["novelty_review"] = novelty
        else:
            logger.warning("Invalid novelty review value received: %s", novelty)

        feasibility = parsed_data.get("feasibility_review", "UNREVIEWED").upper()
        if feasibility in ["HIGH", "MEDIUM", "LOW"]:
            review_data["feasibility_review"] = feasibility
        else:
            logger.warning("Invalid feasibility review value received: %s", feasibility)

        review_data["comment"] = parsed_data.get("comment", "No comment provided.")
        
        # Hard validation: Filter model references against verified IDs in retrieved_sources
        raw_refs = parsed_data.get("references", [])
        if isinstance(raw_refs, list):
            valid_source_ids = {
                str(src.get("source_id")) for src in retrieved_sources if isinstance(src, dict) and "source_id" in src
            }
            # Only allow references that exist in the retrieved sources memory
            review_data["references"] = [
                ref for ref in raw_refs if isinstance(ref, str) and (ref in valid_source_ids or not valid_source_ids)
            ]
        else:
            logger.warning("Invalid references format received: %s", raw_refs)

    except (json.JSONDecodeError, AttributeError, KeyError) as exc:
        logger.warning("Error parsing LLM reflection response: %s", response, exc_info=True)
        review_data["comment"] = f"Could not parse LLM response: {exc}"

    logger.info("Parsed reflection data: %s", review_data)
    return review_data
