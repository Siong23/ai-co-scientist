"""Ranking-agent debate and Elo helpers."""

from __future__ import annotations

import math

# import random
import re

# import logging
from typing import Dict, List

from ..models import Hypothesis, PairwiseDecision, ReflectionReport, ResearchGoal
from ..utils import logger
from .generation_helpers import _call_llm

# logger=logging.getLogger(__name__)

# Ranking uses a faster, dedicated local model so repeated pairwise decisions do
# not inherit the much larger generation model's latency.
RANKING_LLM_MODEL = "qwen/qwen3.6-35b-a3b"

def clean_markdown(text):
    if not text:
        return ""

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("**", "")
    return text.strip()

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

def parse_confidence(response):

    match = re.search(
        r"confidence\s*:\s*(0?\.\d+|1\.0|[0-9]+%)",
        response,
        re.IGNORECASE
    )

    if match:
        value = match.group(1)

        if "%" in value:
            return float(value.replace("%","")) / 100

        return float(value)

    return 0.0

def parse_decisive_criteria(response: str) -> List[str]:

    match = re.search(
        r"Decisive Criteria:\s*(.*?)(?:Conclusion:|Decision:|Confidence:|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return []

    criteria = []

    for line in match.group(1).splitlines():
        line = line.strip("- ").strip()

        if line and line != "**":
            criteria.append(clean_markdown(line))

    return criteria


def format_evidence_sources(hypothesis: Hypothesis) -> str:
    """
    Format only the evidence sources attached to this hypothesis.

    Supports the source fields commonly used by the project:
    content, abstract, summary, title, source_id, and url.
    """
    sources = getattr(hypothesis, "evidence_sources", None)

    if not sources:
        return "No evidence provided."

    output = []

    for idx, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue

        source_id = source.get("source_id", "Unknown source ID")
        title = source.get("title", "No title")
        content = source.get("content", "")
        abstract = source.get("abstract", "")
        summary = source.get("summary", "")
        url = source.get("url", "")

        # Prefer the most useful available text representation.
        evidence_text = (
            content
            or abstract
            or summary
            or source.get("finding", "")
            or "No source content available."
        )

        entry = [
            f"Evidence {idx}",
            f"Source ID: {source_id}",
            f"Title: {title}",
            f"Evidence Content:\n{evidence_text}",
        ]

        if url:
            entry.append(f"URL: {url}")

        output.append("\n".join(entry))

    if not output:
        return "No valid evidence provided."

    return "\n\n".join(output)


def format_references(hypothesis_or_sources):
    """
    Backwards-compatible evidence formatter.

    Accepts:
    - Hypothesis
    - list of source dictionaries
    - single source dictionary
    """
    if hypothesis_or_sources is None:
        return "No evidence provided."

    if isinstance(hypothesis_or_sources, Hypothesis):
        return format_evidence_sources(hypothesis_or_sources)

    if isinstance(hypothesis_or_sources, dict):
        sources = [hypothesis_or_sources]

    elif isinstance(hypothesis_or_sources, list):
        sources = hypothesis_or_sources

    else:
        return "No evidence provided."

    if not sources:
        return "No evidence provided."

    output = []

    for idx, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue

        source_id = source.get("source_id", "Unknown source ID")
        title = source.get("title", "No title")

        evidence_text = (
            source.get("content")
            or source.get("abstract")
            or source.get("summary")
            or source.get("finding")
            or "No source content available."
        )

        url = source.get("url", "")

        entry = [
            f"Evidence {idx}",
            f"Source ID: {source_id}",
            f"Title: {title}",
            f"Evidence Content:\n{evidence_text}",
        ]

        if url:
            entry.append(f"URL: {url}")

        output.append("\n".join(entry))

    return "\n\n".join(output) if output else "No valid evidence provided."


def format_reflection_report(
    report: ReflectionReport | None,
) -> str:
    """
    Format the structured ReflectionReport for the Ranking Agent.

    The ReflectionReport is the authoritative scientific review.
    """

    if report is None:
        return "No reflection report available."

    output = [
        "=== Structured Reflection Report ===",
        "",
        f"Novelty Score: {report.novelty_score}/10",
        f"Feasibility Score: {report.feasibility_score}/10",
        f"Plausibility Score: {report.plausibility_score}/10",
        f"Testability Score: {report.testability_score}/10",
        f"Evidence Quality: {report.evidence_quality_score}/10",
        f"Expected Research Value: "
        f"{report.expected_research_value_score}/10",
        "",
        "Claims and Evidence Assessment:",
    ]

    claims = getattr(report, "claims", None) or []

    if claims:
        for idx, claim in enumerate(claims, start=1):
            claim_text = getattr(claim, "claim", "")
            status = getattr(claim, "status", "UNVERIFIED")

            supporting_ids = getattr(
                claim,
                "supporting_source_ids",
                [],
            ) or []

            contradictory_ids = getattr(
                claim,
                "contradictory_source_ids",
                [],
            ) or []

            output.extend(
                [
                    "",
                    f"Claim {idx}: {claim_text}",
                    f"Status: {status}",
                    "Supporting Source IDs: "
                    + (
                        ", ".join(supporting_ids)
                        if supporting_ids
                        else "None"
                    ),
                    "Contradictory Source IDs: "
                    + (
                        ", ".join(contradictory_ids)
                        if contradictory_ids
                        else "None"
                    ),
                ]
            )
    else:
        output.append("No claim assessments provided.")

    output.extend(
        [
            "",
            "Strengths:",
        ]
    )

    strengths = getattr(report, "strengths", None) or []

    if strengths:
        output.extend(f"- {item}" for item in strengths)
    else:
        output.append("- None provided.")

    output.append("")
    output.append("Weaknesses:")

    weaknesses = getattr(report, "weaknesses", None) or []

    if weaknesses:
        output.extend(f"- {item}" for item in weaknesses)
    else:
        output.append("- None provided.")

    output.append("")
    output.append("Contradictions:")

    contradictions = getattr(report, "contradictions", None) or []

    if contradictions:
        output.extend(f"- {item}" for item in contradictions)
    else:
        output.append("- None provided.")

    output.extend(
        [
            "",
            f"Recommendation: {report.recommendation}",
            f"Reflection Confidence: {report.confidence}",
        ]
    )

    return "\n".join(output)

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
        model=RANKING_LLM_MODEL,
        reasoning="off",
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

    Hypothesis A

    {hypoA.text}

    Hypothesis B

    {hypoB.text}

    ----------------------------------

    Argument defending Hypothesis A

    {debateA}

    ----------------------------------

    Argument defending Hypothesis B

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

    Finish your response EXACTLY in the following format:

    Decision:
    A/B/TIE/ABSTAIN

    Short Justification:
    One or two sentences explaining why the winning hypothesis is better.

    Decisive Criteria:
    - Criterion 1
    - Criterion 2

    If the decision is TIE, list the shared strengths or equally balanced evaluation factors as the decisive criteria.

    If the decision is ABSTAIN, list the reasons that prevented a confident comparison as the decisive criteria.

    Confidence:
    Provide your confidence as a decimal number between 0 and 1.
    """

    return _call_llm(
        prompt,
        temperature=0.2,
        model=RANKING_LLM_MODEL,
        reasoning="off",
    )


def judge_hypotheses(
    hypo_a: Hypothesis,
    hypo_b: Hypothesis,
    review_a: str,
    review_b: str,
    research_goal: ResearchGoal,
) -> str:
    """Make one structured ranking decision with scientific and implementation auditing."""

    evidence_a = format_evidence_sources(hypo_a)
    evidence_b = format_evidence_sources(hypo_b)

    prompt = f"""
    You are the final ranking judge for two competing scientific hypotheses.

    Your task is to determine which hypothesis provides the stronger research
    direction while checking BOTH scientific quality and implementation risk.

    IMPORTANT:
    The Reflection Report is the structured scientific review produced by the
    Reflection Agent. Treat it as authoritative review evidence.

    Do not invent sources, citations, experimental results, or numerical
    evidence that are not provided below.

    Research Goal:
    {research_goal.description}

    Evaluation Criteria:
    {research_goal.preferences}

    Idea Attributes:
    {research_goal.idea_attributes}

    ================================================================
    HYPOTHESIS A
    ================================================================

    {hypo_a.text}

    ---------------- Reflection Report A ----------------

    {review_a}

    ---------------- Evidence Sources A ----------------

    {evidence_a}

    ================================================================
    HYPOTHESIS B
    ================================================================

    {hypo_b.text}

    ---------------- Reflection Report B ----------------

    {review_b}

    ---------------- Evidence Sources B ----------------

    {evidence_b}

    ================================================================
    EVALUATION REQUIREMENTS
    ================================================================

    Evaluate both hypotheses using:

    1. Scientific novelty
    2. Feasibility
    3. Scientific plausibility
    4. Testability
    5. Evidence quality
    6. Expected research value
    7. Alignment with the research goal

    Pay particular attention to claim assessment status:

    - SUPPORTED: the supplied evidence supports the claim.
    - CONTRADICTED: the supplied evidence conflicts with the claim.
    - MIXED: the supplied evidence contains both support and contradiction.
    - NOT_FOUND: the supplied evidence does not establish the claim.
    - UNVERIFIED: the claim cannot currently be established from the supplied evidence.

    Do not treat NOT_FOUND or UNVERIFIED as proof that a proposed mechanism
    is scientifically false. A novel mechanism may be legitimate if it is
    plausible, testable, and falsifiable.

    However, unsupported exact numerical claims must not be treated as
    established results.

    Do not select a hypothesis simply because it reports a higher metric,
    score, or expected performance.

    ================================================================
    IMPLEMENTATION AUDIT
    ================================================================

    Carefully inspect the hypothesis descriptions and reflection reports for
    possible metric-winning implementation problems.

    Check specifically for:

    1. SILENT MODEL SCALING
       Did a hypothesis improve results mainly by secretly increasing model
       capacity?

    2. PARAMETER OR COMPUTATIONAL BLOATING
       Does the proposed improvement depend on excessive model size,
       computation, memory, or latency?

    3. DATA OR EVALUATION LEAKAGE
       Does the approach appear to use test information, future information,
       duplicated samples, validation information, or other information that
       should not be available during evaluation?

    4. METRIC-WINNING WITHOUT SCIENTIFIC VALUE
       Does the hypothesis optimize a metric while weakening the actual
       research objective, generalization, robustness, or scientific validity?

    5. CORRUPTED OR UNMAINTAINABLE LOGIC
       Does the proposed improvement introduce fragile, unnecessarily complex,
       inconsistent, or difficult-to-maintain implementation logic?

    6. UNFAIR COMPARISON
       Does one hypothesis receive additional resources, information,
       preprocessing, or experimental advantages that are not justified?

    IMPORTANT:
    Do not assume that a problem exists simply because it is possible.
    Penalize an issue only when there is evidence in the provided hypothesis,
    reflection report, or evidence sources.

    A hypothesis with a slightly lower reported metric may be preferable if
    it is scientifically sound, reproducible, fair, and implementation-safe.

    If the available information is insufficient to confidently determine
    which hypothesis is better, use ABSTAIN.

    If both hypotheses are genuinely equivalent, use TIE.

    ================================================================
    OUTPUT FORMAT
    ================================================================

    Finish your response EXACTLY in this format:

    Decision:
    A/B/TIE/ABSTAIN

    Short Justification:
    Provide 1-2 concise sentences explaining the decision.

    Decisive Criteria:
    - Criterion 1
    - Criterion 2

    Confidence:
    A decimal number between 0 and 1.
    """

    return _call_llm(
        prompt,
        temperature=0.2,
        model=RANKING_LLM_MODEL,
        reasoning="off",
    )


def parse_short_justification(response: str) -> str:

    match = re.search(
        r"Short Justification:\s*(.*?)(?:Decisive Criteria:|Confidence:|$)",
        response,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        return clean_markdown(match.group(1).strip())

    return ""


def run_pairwise_debate(
    hypoA: Hypothesis,
    hypoB: Hypothesis,
    research_goal: ResearchGoal,
) -> PairwiseDecision:
    """
    Compare two hypotheses using their structured ReflectionReports.

    Ranking must not perform an evidence-based comparison when either
    hypothesis has not been successfully reviewed.
    """

    report_a = hypoA.reflection_report
    report_b = hypoB.reflection_report

    # ------------------------------------------------------------
    # Reflection prerequisite
    # ------------------------------------------------------------

    if report_a is None or report_b is None:
        missing = []

        if report_a is None:
            missing.append(f"A ({hypoA.hypothesis_id})")

        if report_b is None:
            missing.append(f"B ({hypoB.hypothesis_id})")

        reason = (
            "Ranking abstained because the required ReflectionReport "
            "is missing for: "
            + ", ".join(missing)
        )

        logger.warning(reason)

        return PairwiseDecision(
            hypothesis_a_id=hypoA.hypothesis_id,
            hypothesis_b_id=hypoB.hypothesis_id,
            outcome="ABSTAIN",
            scores_a={},
            scores_b={},
            confidence=0.0,
            reasoning=reason,
            decisive_criteria=[
                "Required ReflectionReport is missing."
            ],
        )

    # ------------------------------------------------------------
    # Structured reflection reports
    # ------------------------------------------------------------

    reviewA = format_reflection_report(report_a)
    reviewB = format_reflection_report(report_b)

    # ------------------------------------------------------------
    # Deterministic scores derived from ReflectionReport
    # ------------------------------------------------------------

    scores_a = score_hypothesis(
        hypoA,
        research_goal,
    )

    scores_b = score_hypothesis(
        hypoB,
        research_goal,
    )

    # Defensive check
    if not scores_a or not scores_b:
        reason = (
            "Ranking abstained because one or both ReflectionReports "
            "could not produce valid ranking scores."
        )

        logger.warning(reason)

        return PairwiseDecision(
            hypothesis_a_id=hypoA.hypothesis_id,
            hypothesis_b_id=hypoB.hypothesis_id,
            outcome="ABSTAIN",
            scores_a=scores_a,
            scores_b=scores_b,
            confidence=0.0,
            reasoning=reason,
            decisive_criteria=[
                "Invalid or incomplete ReflectionReport scores."
            ],
        )

    # ------------------------------------------------------------
    # LLM ranking judge
    # ------------------------------------------------------------

    response = judge_hypotheses(
        hypoA,
        hypoB,
        reviewA,
        reviewB,
        research_goal,
    )

    try:
        outcome = parse_pairwise_result(response)
    except ValueError:
        logger.warning("Could not parse final debate decision.")
        outcome = "ABSTAIN"

    confidence = parse_confidence(response)
    criteria = parse_decisive_criteria(response)
    reasoning = clean_markdown(
        parse_short_justification(response)
    )

    logger.info(
        "Pairwise ranking response:\n%s",
        response,
    )

    return PairwiseDecision(
        hypothesis_a_id=hypoA.hypothesis_id,
        hypothesis_b_id=hypoB.hypothesis_id,
        outcome=outcome,
        scores_a=scores_a,
        scores_b=scores_b,
        confidence=confidence,
        reasoning=reasoning,
        decisive_criteria=criteria,
    )


def update_elo_tie(hypoA: Hypothesis, hypoB: Hypothesis, k_factor: int):

    ratingA=hypoA.elo_score
    ratingB=hypoB.elo_score


    expectedA = 1/(1+math.pow(
        10,
        (ratingB-ratingA)/400
    ))

    expectedB = 1-expectedA


    hypoA.elo_score += k_factor*(0.5-expectedA)

    hypoB.elo_score += k_factor*(0.5-expectedB)


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
            r"decision\s*:\s*A(?:\s|$|[^\w])",  # A followed by whitespace, end, or non-word char
            r"winner\s*:\s*A(?:\s|$|[^\w])",
        ],
        "B": [
            r"decision\s*:\s*B(?:\s|$|[^\w])",
            r"winner\s*:\s*B(?:\s|$|[^\w])",
        ],
        "TIE": [
            r"decision\s*:\s*TIE\b",
        ],
        "ABSTAIN": [
            r"decision\s*:\s*ABSTAIN\b",
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
