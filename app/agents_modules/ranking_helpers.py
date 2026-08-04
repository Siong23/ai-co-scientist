"""Ranking-agent debate and Elo helpers."""

from __future__ import annotations

import math
# import random
import re
# import logging

from typing import Dict, List
from ..models import Hypothesis, ResearchGoal, PairwiseDecision
from ..utils import logger
from .generation_helpers import _call_llm

# logger=logging.getLogger(__name__)

def score_hypothesis(
    hypothesis: Hypothesis,
    research_goal: ResearchGoal,
) -> Dict[str, float]:

    report = hypothesis.reflection_report

    if report is None:
        return {}

    alignment = (
        report.novelty_score +
        report.feasibility_score +
        report.plausibility_score
    ) / 3

    return {
        "research_goal_alignment": alignment,
        "novelty": report.novelty_score,
        "feasibility": report.feasibility_score,
        "scientific_plausibility": report.plausibility_score,
        "testability": report.testability_score,
        "evidence_quality": report.evidence_quality_score,
        "expected_research_value": report.expected_research_value_score,
    }

def parse_decisive_criteria(response: str) -> List[str]:

    match = re.search(
        r"Decisive Criteria:\s*(.*?)(?:Confidence:|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return []

    criteria = []

    for line in match.group(1).splitlines():
        line = line.strip("- ").strip()

        if line:
            criteria.append(line)

    return criteria

def format_evidence_sources(hypothesis):
    
    if not hypothesis.evidence_sources:
        return "No evidence provided."


    output=[]


    for idx, source in enumerate(
        hypothesis.evidence_sources,
        start=1
    ):

        output.append(
            f"""
            Evidence {idx}

            Title:
            {source.get("title")}

            Finding:
            {source.get("finding")}

            Limitation:
            {source.get("limitation")}

            """
        )

    return "\n".join(output)

def format_references(hypothesis_or_sources):
    """
    Backwards-compatible wrapper expected by app.agents.
    Accepts either a Hypothesis (with .evidence_sources) or a list of source dicts,
    and returns the same formatted string as format_evidence_sources.
    """
    # If passed a Hypothesis-like object with evidence_sources attribute
    if hypothesis_or_sources is None:
        return "No evidence provided."

    sources = None
    if hasattr(hypothesis_or_sources, "evidence_sources"):
        sources = hypothesis_or_sources.evidence_sources
    elif isinstance(hypothesis_or_sources, list):
        sources = hypothesis_or_sources
    else:
        # Fallback: try to treat it as a single source dict
        try:
            if isinstance(hypothesis_or_sources, dict):
                sources = [hypothesis_or_sources]
            else:
                return str(hypothesis_or_sources)
        except Exception:
            return "No evidence provided."

    if not sources:
        return "No evidence provided."

    output = []
    for idx, source in enumerate(sources, start=1):
        title = source.get("title", "No title")
        finding = source.get("finding", "")
        limitation = source.get("limitation", "")
        url = source.get("url", "")
        entry_lines = [
            f"Evidence {idx}",
            "",
            f"Title:\n{title}",
            "",
            f"Finding:\n{finding}",
            "",
            f"Limitation:\n{limitation}",
        ]
        if url:
            entry_lines.extend(["", f"URL:\n{url}"])
        output.append("\n".join(entry_lines))

    return "\n\n".join(output)

def format_reflection_report(report):

    if report is None:
        return "No reflection report available."


    return f"""
    Novelty Score:
    {report.novelty_score}/10


    Feasibility Score:
    {report.feasibility_score}/10


    Plausibility Score:
    {report.plausibility_score}/10


    Testability Score:
    {report.testability_score}/10


    Evidence Quality:
    {report.evidence_quality_score}/10


    Strengths:

    {chr(10).join(report.strengths)}


    Weaknesses:

    {chr(10).join(report.weaknesses)}


    Contradictions:

    {chr(10).join(report.contradictions)}


    Recommendation:

    {report.recommendation}


    Confidence:

    {report.confidence}

    """

def generate_debate_argument(
    candidate: Hypothesis,
    opponent: Hypothesis,
    review: str,
    opponent_review: str,
    research_goal: ResearchGoal,
) -> str:
    """
    Generate an argument defending one hypothesis while critiquing the opponent.
    """

    considerations = (
        "\n".join(f"- {k}: {v}" for k, v in research_goal.constraints.items())
        if research_goal.constraints
        else "None"
    )

    prompt = f"""
    You are a domain expert participating in a scientific debate.

    Your task is to defend YOUR hypothesis while critically evaluating the competing hypothesis.

    Research Goal:
    {research_goal.description}

    Evaluation Criteria:
    {research_goal.preferences}

    Idea Attributes:
    {research_goal.idea_attributes}

    Considerations:
    {considerations}

    YOUR hypothesis:

    {candidate.text}

    Reflection Report:
    {review}

    Supporting Evidence Sources:
    {format_evidence_sources(candidate)}

    --------------------------------------

    Competing hypothesis:

    {opponent.text}

    Reflection Report:
    {opponent_review}

    Supporting Evidence Sources:
    {format_evidence_sources(opponent)}

    --------------------------------------

    Produce:

    1. Strengths of YOUR hypothesis
    2. Weaknesses of the competing hypothesis
    3. Why YOUR hypothesis better satisfies the research goal

    Keep your response concise.
    """

    return _call_llm(
        prompt,
        temperature=0.3,
        model=research_goal.llm_model,
    )

def judge_debate(
    hypoA: Hypothesis,
    hypoB: Hypothesis,
    debateA: str,
    debateB: str,
    research_goal: ResearchGoal,
) -> str:
    """
    Judge the debate and determine the winning hypothesis.
    """

    prompt = f"""
    You are the final judge in a scientific debate.

    Research Goal:
    {research_goal.description}

    Evaluation Criteria:
    {research_goal.preferences}

    Idea Attributes:
    {research_goal.idea_attributes}

    Hypothesis 1

    {hypoA.text}

    Hypothesis 2

    {hypoB.text}

    ----------------------------------

    Argument defending Hypothesis 1

    {debateA}

    ----------------------------------

    Argument defending Hypothesis 2

    {debateB}

    ----------------------------------

    Review both arguments objectively.

    Do NOT simply count the number of claims.

    Consider:

    - scientific novelty
    - feasibility
    - reviewer comments
    - supporting evidence
    - overall scientific merit
    - alignment with the research goal

    Explain your reasoning.

    Finish your response exactly:

    Decision:
    A/B/TIE/ABSTAIN

    Decisive Criteria:

    - Evidence Quality
    - Feasibility

    Confidence:
    0.82
    """

    return _call_llm(
        prompt,
        temperature=0.2,
        model=research_goal.llm_model,
    )

def run_pairwise_debate(hypoA: Hypothesis, hypoB: Hypothesis, research_goal: ResearchGoal) -> PairwiseDecision:
    """Compares two hypotheses based on novelty and feasibility scores."""

    reviewA = format_reflection_report(
        hypoA.reflection_report
    )

    reviewB = format_reflection_report(
        hypoB.reflection_report
    )

    scores_a = score_hypothesis(
        hypoA,
        research_goal,
    )

    scores_b = score_hypothesis(
        hypoB,
        research_goal,
    )

    debateA = generate_debate_argument(
        hypoA,
        hypoB,
        reviewA,
        reviewB,
        research_goal,
    )

    debateB = generate_debate_argument(
        hypoB,
        hypoA,
        reviewB,
        reviewA,
        research_goal,
    )

    # First judgment
    response1 = judge_debate(
        hypoA,
        hypoB,
        debateA,
        debateB,
        research_goal,
    )

    try:
        winner1_index = parse_pairwise_result(response1)
    except ValueError:
        winner1_index = "ABSTAIN"

    # Second judgment with hypotheses swapped
    response2 = judge_debate(
        hypoB,
        hypoA,
        debateB,
        debateA,
        research_goal,
    )

    try:
        winner2_index = parse_pairwise_result(response2)
    except ValueError:
        winner2_index = "ABSTAIN"

    if winner1_index == "A" and winner2_index == "A":
        # A wins both orderings
        # winner = hypoA
        response = response1

    elif winner1_index == "B" and winner2_index == "B":
        # B wins both orderings
        # winner = hypoB
        response = response2

    else:
        logger.warning("Possible position bias detected. Running tie-breaker.")

        response = judge_debate(
            hypoA,
            hypoB,
            debateA,
            debateB,
            research_goal,
        )

    try:
        outcome = parse_pairwise_result(response)
    except ValueError:
        logger.warning("Could not parse final debate decision.")
        outcome = "ABSTAIN"

    confidence = parse_confidence(response)

    criteria = parse_decisive_criteria(response)

    reasoning = f"""
    === Debate for Hypothesis 1 ===

    {debateA}

    ===============================

    === Debate for Hypothesis 2 ===

    {debateB}

    ===============================

    === Judge Round 1 ===

    {response1}

    === Judge Round 2 (Swapped)  ===

    {response2}

    ===============================
    
    === Final Decision ===

    {response}
    """

    logger.info("Pairwise ranking response:\n%s", response)

    return PairwiseDecision(
        hypothesis_a_id = hypoA.hypothesis_id,
        hypothesis_b_id = hypoB.hypothesis_id,
        outcome = outcome,
        scores_a = scores_a,
        scores_b = scores_b,
        confidence = confidence,
        reasoning = reasoning,
        decisive_criteria = criteria,
    )

def update_elo_tie(hypoA, hypoB, k_factor):

    ratingA=hypoA.elo_score
    ratingB=hypoB.elo_score


    expectedA = 1/(1+math.pow(
        10,
        (ratingB-ratingA)/400
    ))

    expectedB = 1-expectedA


    hypoA.elo_score += k_factor*(0.5-expectedA)

    hypoB.elo_score += k_factor*(0.5-expectedB)

def parse_confidence(response):

    match = re.search(
        r"confidence\s*:\s*(0?\.\d+|1\.0)",
        response,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return 0.0

def parse_pairwise_result(response: str) -> str:
    """
    Parse ranking decision from LLM response.

    Returns:
        "A"
        "B"
        "TIE"
        "ABSTAIN"
    """

    patterns = {
        "A": [
            r"decision\s*:\s*A",
            r"winner\s*:\s*A",
        ],

        "B": [
            r"decision\s*:\s*B",
            r"winner\s*:\s*B",
        ],

        "TIE": [
            r"decision\s*:\s*tie",
            r"result\s*:\s*tie",
        ],

        "ABSTAIN": [
            r"decision\s*:\s*abstain",
            r"result\s*:\s*abstain",
            r"insufficient\s+evidence",
        ]
    }


    for label, regexes in patterns.items():
        for pattern in regexes:
            if re.search(pattern, response, re.IGNORECASE):
                return label


    raise ValueError(
        "Could not determine ranking decision."
    )


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