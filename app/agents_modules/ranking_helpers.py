"""Ranking-agent debate and Elo helpers."""

from __future__ import annotations

import math
import random
import re

from ..models import Hypothesis, ResearchGoal
from ..utils import logger
from .generation_helpers import _call_llm


def format_references(references):
    if not references:
        return "No references provided."

    formatted = []

    for ref in references:
        if isinstance(ref, dict):
            title = ref.get("title", "Unknown title")
            authors = ref.get("authors", "")
            year = ref.get("year", "")

            formatted.append(f"{title} ({authors}, {year})")
        else:
            formatted.append(str(ref))

    return "\n".join(formatted)


def run_pairwise_debate(hypoA: Hypothesis, hypoB: Hypothesis, research_goal: ResearchGoal) -> tuple[Hypothesis, str]:
    """Compares two hypotheses based on novelty and feasibility scores."""

    # def score(h: Hypothesis) -> int:
    #     mapping = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, None: 0, "ERROR": 0}  # Handle ERROR case
    #     score_novelty = mapping.get(h.novelty_review, 0) if isinstance(h.novelty_review, str) else 0
    #     score_feasibility = mapping.get(h.feasibility_review, 0) if isinstance(h.feasibility_review, str) else 0
    #     return score_novelty + score_feasibility

    # scoreA = score(hypoA)
    # scoreB = score(hypoB)

    # if scoreA > scoreB:
    #     winner = hypoA
    # elif scoreB > scoreA:
    #     winner = hypoB
    # else:
    #     winner = random.choice([hypoA, hypoB])  # Tie-breaker

    # logger.info(
    #     "Debate: %s (score %d) vs %s (score %d) => Winner: %s",
    #     hypoA.hypothesis_id,
    #     scoreA,
    #     hypoB.hypothesis_id,
    #     scoreB,
    #     winner.hypothesis_id,
    # )
    # return winner

    reviewA = f"""
    Novelty Review:
    {hypoA.novelty_review}

    Feasibility Review:
    {hypoA.feasibility_review}

    Reviewer Comments:
    {chr(10).join(map(str, hypoA.review_comments))}

    References:
    {format_references(hypoA.references)}
    """

    reviewB = f"""
    Novelty Review:
    {hypoB.novelty_review}

    Feasibility Review:
    {hypoB.feasibility_review}

    Reviewer Comments:
    {chr(10).join(map(str, hypoB.review_comments))}

    References:
    {format_references(hypoB.references)}
    """

    # Format constraints into readable bullet points
    considerations = (
        "\n".join(f"- {k}: {v}" for k, v in research_goal.constraints.items()) if research_goal.constraints else "None"
    )

    prompt = f"""
    You are an expert evaluator tasked with comparing two hypotheses.

    Evaluate the two provided hypotheses (Hypothesis 1 and Hypothesis 2) and
    determine which one is superior.

    Evaluate the hypotheses based on:
    - Alignment with the research goal
    - {research_goal.idea_attributes}
    - The Evaluation Criteria provided below
    - Quality and relevance of the supporting evidence
    - Reviewer comments
    - Overall scientific merit

    The listed Evidence Sources indicate which retrieved literature supports each
    hypothesis. Consider the strength and relevance of this supporting evidence
    during your evaluation, but do not judge solely by the number of supporting
    sources.

    Provide your reasoning first, then end your response with exactly one of the following:

    better hypothesis: 1

    or

    better hypothesis: 2

    Goal:
    {research_goal.description}

    Evaluation Criteria:
    {research_goal.preferences}

    Considerations:
    {considerations}

    Each hypothesis includes an independent review containing novelty and
    feasibility assessments, reviewer comments, and supporting references.

    Do not determine the winner solely based on the novelty or feasibility
    ratings. Instead, use the review comments, supporting evidence, and
    overall scientific quality to make a balanced comparative judgement.

    Hypothesis 1:
    {hypoA.text}

    Evidence Sources:
    {chr(10).join(hypoA.evidence_source_ids) if hypoA.evidence_source_ids else "No evidence sources."}

    Review of Hypothesis 1:
    {reviewA}

    Hypothesis 2:
    {hypoB.text}

    Evidence Sources:
    {chr(10).join(hypoB.evidence_source_ids) if hypoB.evidence_source_ids else "No evidence sources."}

    Review of Hypothesis 2:
    {reviewB}

    """

    response = _call_llm(
        prompt,
        temperature=0.2,
        model=research_goal.llm_model,
    )

    logger.info("Pairwise ranking response:\n%s", response)

    try:
        winner_index = parse_pairwise_result(response)

        winner = hypoA if winner_index == 1 else hypoB

    except ValueError:
        logger.warning(
            "Could not parse LLM ranking response.\n%s",
            response,
        )

        winner = random.choice([hypoA, hypoB])

    return winner, response

    # match = re.search(
    #     r"better\s+(?:idea|hypothesis)\s*:\s*([12])",
    #     response,
    #     re.IGNORECASE,
    # )

    # if match:
    #     winner = hypoA if match.group(1) == "1" else hypoB
    # else:
    #     logger.warning("Could not parse LLM ranking. Using random winner.")
    #     winner = random.choice([hypoA, hypoB])

    # return winner, response


def parse_pairwise_result(response: str) -> int:
    """
    Parse the LLM response and return the winning hypothesis index (1 or 2).
    Raises:
        ValueError: if no winner can be identified.
    """
    patterns = [
        r"better\s+(?:idea|hypothesis)\s*:\s*([12])",
        r"winner\s*:\s*([12])",
        r"better\s+(?:idea|hypothesis)\s+is\s+([12])",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))

    raise ValueError("Could not determine winner.")


def update_elo(winner: Hypothesis, loser: Hypothesis, k_factor: int):
    """Updates Elo scores after a comparison, using provided k_factor."""
    # k_factor is now passed as an argument
    ratingA = winner.elo_score
    ratingB = loser.elo_score
    expectedA = 1 / (1 + math.pow(10, (ratingB - ratingA) / 400))
    expectedB = 1 - expectedA  # Or 1 / (1 + math.pow(10, (ratingA - ratingB) / 400))
    winner.elo_score = ratingA + k_factor * (1 - expectedA)
    loser.elo_score = ratingB + k_factor * (0 - expectedB)  # Loser's score update
    logger.info(
        "Updated Elo: Winner %s -> %.2f, Loser %s -> %.2f",
        winner.hypothesis_id,
        winner.elo_score,
        loser.hypothesis_id,
        loser.elo_score,
    )
