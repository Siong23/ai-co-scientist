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


def _parse_reflection_response(response: str, retrieved_sources: List[dict]) -> dict | None:
    try:
        cleaned_response = _strip_fenced_json(response)
        parsed_data = json.loads(cleaned_response)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Error parsing LLM reflection response: %s", response, exc_info=True)
        return None

    novelty = str(parsed_data.get("novelty_review", "UNREVIEWED")).upper()
    feasibility = str(parsed_data.get("feasibility_review", "UNREVIEWED")).upper()
    if novelty not in ["HIGH", "MEDIUM", "LOW"] or feasibility not in ["HIGH", "MEDIUM", "LOW"]:
        logger.warning(
            "Invalid reflection review values received: novelty=%s, feasibility=%s",
            novelty,
            feasibility,
        )
        return None

    review_data = {
        "novelty_review": novelty,
        "feasibility_review": feasibility,
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
        '  "novelty_review": "HIGH | MEDIUM | LOW",\n'
        '  "feasibility_review": "HIGH | MEDIUM | LOW",\n'
        '  "comment": "Concise summary critique explaining the ratings and suggestions.",\n'
        '  "references": ["exact Source ID from the provided list above"]\n'
        "}"
    )
    repair_prompt = (
        "The previous response did not satisfy the required JSON schema or contained invalid novelty/feasibility values. "
        "Reformat the candidate response below to valid JSON only, preserving the original scientific meaning. "
        "Do not add Markdown, commentary, or new information.\n\n"
        f"{schema_instruction}\n\n"
        f"Candidate response:\n{response}"
    )
    repaired_response = _call_llm(repair_prompt, temperature=0.0, model=model)
    logger.info("LLM reflection repair response for hypothesis: %s", repaired_response)

    if repaired_response.startswith("Error:"):
        logger.error("LLM reflection repair call failed: %s", repaired_response)
        return {
            "novelty_review": "UNREVIEWED",
            "feasibility_review": "UNREVIEWED",
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
        "comment": "Could not parse LLM response after format repair.",
        "references": [],
    }
