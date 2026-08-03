"""Reflection-agent LLM helpers."""

from __future__ import annotations

import json
from typing import Dict

from ..utils import logger
from .generation_helpers import _call_llm


def call_llm_for_reflection(hypothesis_text: str, temperature: float = 0.5, model: str | None = None) -> Dict:
    """Calls LLM for reviewing a hypothesis, handling JSON parsing."""
    logger.info("LLM reflection called with temperature: %.2f", temperature)
    prompt = (
        f"Review the following hypothesis and provide a novelty assessment (HIGH, MEDIUM, or LOW), "
        f"a feasibility assessment (HIGH, MEDIUM, or LOW), a comment, and a list of relevant references in JSON format:\n\n"
        f"Hypothesis: {hypothesis_text}\n\n"
        f"For references, provide arXiv IDs (e.g., '2301.12345'), DOIs, or paper titles with venues that are relevant to this hypothesis. "
        f"Do not provide PubMed IDs (PMIDs) unless this is specifically a biomedical/life sciences hypothesis.\n\n"
        f"Return the response as a JSON object with the following keys: 'novelty_review', 'feasibility_review', 'comment', 'references'."
    )
    # Pass the received temperature down to the actual LLM call
    response = _call_llm(prompt, temperature=temperature, model=model)
    logger.info("LLM reflection response for hypothesis: %s", response)

    if response.startswith("Error:"):
        logger.error(f"LLM reflection call failed: {response}")
        return {
            "novelty_review": "Not reviewed",
            "feasibility_review": "Not reviewed",
            "comment": f"LLM review failed: {response}",
            "references": [],
        }

    # Default values
    review_data = {
        "novelty_review": "MEDIUM",
        "feasibility_review": "MEDIUM",
        "comment": "Could not parse LLM response.",
        "references": [],
    }

    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        parsed_data = json.loads(response)

        # Update defaults with parsed data, performing basic validation
        novelty = parsed_data.get("novelty_review", "MEDIUM").upper()
        if novelty in ["HIGH", "MEDIUM", "LOW"]:
            review_data["novelty_review"] = novelty
        else:
            logger.warning("Invalid novelty review value received: %s", novelty)

        feasibility = parsed_data.get("feasibility_review", "MEDIUM").upper()
        if feasibility in ["HIGH", "MEDIUM", "LOW"]:
            review_data["feasibility_review"] = feasibility
        else:
            logger.warning("Invalid feasibility review value received: %s", feasibility)

        review_data["comment"] = parsed_data.get("comment", "No comment provided.")
        # review_data["references"] = parsed_data.get("references", [])
        # if not isinstance(review_data["references"], list):
        #     logger.warning("Invalid references format received: %s", review_data["references"])
        #     review_data["references"] = []
        references = parsed_data.get("references", [])
        if isinstance(references, list):
            review_data["references"] = references
        else:
            logger.warning("Invalid references format received: %s", review_data["references"])
            review_data["references"] = []

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        logger.warning("Error parsing LLM reflection response: %s", response, exc_info=True)
        review_data["comment"] = f"Could not parse LLM response: {e}"  # Update comment with error

    logger.info("Parsed reflection data: %s", review_data)
    return review_data
