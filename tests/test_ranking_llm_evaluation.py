"""
LLM-based evaluation experiment for the Ranking Agent.

This experiment evaluates the QUALITY of Ranking Agent decisions using
an independent LLM evaluator.

The existing test_ranking_performance.py evaluates:
    - correctness
    - efficiency
    - consistency
    - robustness
    - regression behaviour

This experiment evaluates:
    - research-goal alignment
    - scientific validity
    - novelty
    - feasibility
    - testability
    - evidence quality
    - reasoning quality
    - decision appropriateness

IMPORTANT:
The evaluator model should preferably be different from the Ranking Agent
model to reduce evaluator bias.

Run with:

    pytest tests/test_ranking_llm_evaluation.py -v -s -m integration
"""

import json
import os
import re
from statistics import mean

import pytest

from app.agents_modules.generation_helpers import _call_llm
from app.agents_modules.ranking_helpers import (
    RANKING_LLM_MODEL,
    run_pairwise_debate,
)
from app.models import ClaimAssessment, Hypothesis, ReflectionReport, ResearchGoal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use a different model from the Ranking Agent where possible.
#
# You can override this from the terminal:
#
# PowerShell:
# $env:RANKING_EVALUATOR_MODEL="google/gemini-2.5-flash"
#
# CMD:
# set RANKING_EVALUATOR_MODEL=google/gemini-2.5-flash
#
EVALUATOR_LLM_MODEL = os.getenv(
    "RANKING_EVALUATOR_MODEL",
    "qwen/qwen3.5-9b",
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def _reflection_report(
    novelty=7.0,
    feasibility=7.0,
    plausibility=7.0,
    testability=7.0,
    evidence_quality=7.0,
    research_value=7.0,
    strengths=None,
    weaknesses=None,
):
    """Create a reflection report for evaluation experiments."""

    return ReflectionReport(
        novelty_score=novelty,
        feasibility_score=feasibility,
        plausibility_score=plausibility,
        testability_score=testability,
        evidence_quality_score=evidence_quality,
        expected_research_value_score=research_value,
        strengths=strengths or [],
        weaknesses=weaknesses or [],
        recommendation="ACCEPT",
        overall_confidence=8.0,
        claims=[
        ClaimAssessment(
            claim="...",
            status="SUPPORTED",
            confidence=8.0,
            supporting_source_ids=["S1"],
            contradictory_source_ids=[],
        )
    ],
    )


def _hypothesis(
    hypothesis_id,
    text,
    reflection_report=None,
    evidence_sources=None,
):
    """Create an evaluation hypothesis."""

    hypothesis = Hypothesis(
        hypothesis_id=hypothesis_id,
        text=text,
    )

    hypothesis.reflection_report = reflection_report

    hypothesis.evidence_sources = evidence_sources or []

    return hypothesis


# ---------------------------------------------------------------------------
# Evaluation cases
# ---------------------------------------------------------------------------

EVALUATION_CASES = [
    {
        "case_id": "case_01",
        "description": "Clear feasibility advantage",
        "goal": (
            "Develop a practical intrusion detection approach for "
            "telecommunication network traffic that can be evaluated "
            "using publicly available datasets."
        ),
        "hypothesis_a": {
            "text": (
                "Use a lightweight Random Forest model with engineered "
                "network traffic features and evaluate it on two public "
                "telecommunication intrusion datasets."
            ),
            "reflection": _reflection_report(
                novelty=7,
                feasibility=9,
                plausibility=9,
                testability=9,
                evidence_quality=8,
                research_value=9,
            ),
        },
        "hypothesis_b": {
            "text": (
                "Develop a highly complex proprietary neural architecture "
                "requiring a large private dataset that is not currently "
                "available for experimentation."
            ),
            "reflection": _reflection_report(
                novelty=9,
                feasibility=2,
                plausibility=6,
                testability=2,
                evidence_quality=4,
                research_value=6,
            ),
        },
    },

    {
        "case_id": "case_02",
        "description": "Clear scientific validity advantage",
        "goal": (
            "Investigate whether data poisoning can reduce machine learning "
            "model performance under realistic attack assumptions."
        ),
        "hypothesis_a": {
            "text": (
                "Inject mislabeled training samples into the training set "
                "while keeping the test set completely clean and evaluate "
                "the resulting change in model performance."
            ),
            "reflection": _reflection_report(
                novelty=7,
                feasibility=9,
                plausibility=9,
                testability=10,
                evidence_quality=9,
                research_value=9,
            ),
        },
        "hypothesis_b": {
            "text": (
                "Inject poisoned samples into both training and test sets "
                "and report the resulting accuracy decrease as evidence "
                "that poisoning is effective."
            ),
            "reflection": _reflection_report(
                novelty=6,
                feasibility=8,
                plausibility=3,
                testability=5,
                evidence_quality=3,
                research_value=3,
            ),
        },
    },

    {
        "case_id": "case_03",
        "description": "Potential implementation risk",
        "goal": (
            "Improve anomaly detection while maintaining a fair comparison "
            "between baseline and proposed approaches."
        ),
        "hypothesis_a": {
            "text": (
                "Improve the anomaly detector by increasing the model size "
                "by 20 times while keeping all other experimental settings "
                "unchanged."
            ),
            "reflection": _reflection_report(
                novelty=5,
                feasibility=5,
                plausibility=7,
                testability=7,
                evidence_quality=6,
                research_value=5,
                weaknesses=[
                    "The improvement may primarily result from increased "
                    "model capacity rather than the proposed algorithm."
                ],
            ),
        },
        "hypothesis_b": {
            "text": (
                "Improve anomaly detection using a new feature selection "
                "method while keeping model architecture, training budget, "
                "and evaluation protocol unchanged."
            ),
            "reflection": _reflection_report(
                novelty=7,
                feasibility=9,
                plausibility=8,
                testability=9,
                evidence_quality=8,
                research_value=8,
            ),
        },
    },

    {
        "case_id": "case_04",
        "description": "Strong evidence advantage",
        "goal": (
            "Identify a robust method for detecting malicious network "
            "traffic."
        ),
        "hypothesis_a": {
            "text": (
                "Use an established intrusion detection method supported by "
                "results from three independent public datasets and compare "
                "against multiple baseline models."
            ),
            "reflection": _reflection_report(
                novelty=7,
                feasibility=9,
                plausibility=9,
                testability=9,
                evidence_quality=10,
                research_value=9,
            ),
        },
        "hypothesis_b": {
            "text": (
                "Use a newly proposed detection method evaluated only on "
                "one synthetic dataset with no baseline comparison."
            ),
            "reflection": _reflection_report(
                novelty=9,
                feasibility=7,
                plausibility=5,
                testability=5,
                evidence_quality=2,
                research_value=5,
            ),
        },
    },
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_hypothesis(hypothesis):
    """Format a hypothesis for the evaluator."""

    report = hypothesis.reflection_report

    if report is None:
        reflection = "No reflection report available."
    else:
        reflection = f"""
Novelty: {report.novelty_score}/10
Feasibility: {report.feasibility_score}/10
Scientific Plausibility: {report.plausibility_score}/10
Testability: {report.testability_score}/10
Evidence Quality: {report.evidence_quality_score}/10
Expected Research Value: {report.expected_research_value_score}/10

Strengths:
{chr(10).join(report.strengths) if report.strengths else "None"}

Weaknesses:
{chr(10).join(report.weaknesses) if report.weaknesses else "None"}
"""


    return f"""
Hypothesis ID:
{hypothesis.hypothesis_id}

Hypothesis:
{hypothesis.text}

Reflection Report:
{reflection}
"""


# ---------------------------------------------------------------------------
# LLM evaluator
# ---------------------------------------------------------------------------

def evaluate_ranking_decision(
    research_goal,
    hypothesis_a,
    hypothesis_b,
    ranking_decision,
):
    """
    Ask an independent LLM to evaluate the Ranking Agent decision.

    The evaluator does NOT generate the original ranking.
    It evaluates whether the existing ranking decision is justified.
    """

    prompt = f"""
You are an independent evaluator assessing an AI scientific Ranking Agent.

You must evaluate whether the Ranking Agent made a scientifically
reasonable decision.

Do NOT simply agree with the Ranking Agent.

Research Goal:
{research_goal}

========================================
HYPOTHESIS A
========================================

{_format_hypothesis(hypothesis_a)}

========================================
HYPOTHESIS B
========================================

{_format_hypothesis(hypothesis_b)}

========================================
RANKING AGENT DECISION
========================================

Decision:
{ranking_decision.outcome}

Confidence:
{ranking_decision.confidence}

Reasoning:
{ranking_decision.reasoning}

Decisive Criteria:
{ranking_decision.decisive_criteria}

========================================
EVALUATION CRITERIA
========================================

Score each criterion from 1 to 5.

1 = Very Poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Evaluate:

1. research_goal_alignment
   Does the decision appropriately consider the research goal?

2. scientific_validity
   Is the selected decision scientifically sound?

3. novelty
   Does the decision appropriately consider meaningful novelty?

4. feasibility
   Does the decision consider whether the approach can realistically
   be implemented?

5. testability
   Does the decision favour hypotheses that can be properly tested?

6. evidence_quality
   Does the decision appropriately consider the quality of supporting
   evidence?

7. reasoning_quality
   Is the Ranking Agent's reasoning logically justified?

8. decision_appropriateness
   Is A, B, TIE, or ABSTAIN an appropriate decision based on the
   available information?

Also determine whether you agree with the Ranking Agent's final decision.

Return ONLY valid JSON:

{{
    "evaluator_decision": "A",
    "evaluator_decision": "B",
    "decision_agreement": true,
    "research_goal_alignment": 1,
    "scientific_validity": 1,
    "novelty": 1,
    "feasibility": 1,
    "testability": 1,
    "evidence_quality": 1,
    "reasoning_quality": 1,
    "decision_appropriateness": 1,
    "justification": "Brief explanation."
}}

The evaluator_decision must be exactly one of:

A
B
TIE
ABSTAIN
"""

    response = _call_llm(
        prompt,
        temperature=0.0,
        model=EVALUATOR_LLM_MODEL,
        reasoning="off",
    )

    evaluation = _parse_evaluator_response(response)

    evaluation["decision_agreement"] = (
        evaluation["evaluator_decision"]
        == ranking_decision.outcome
    )

    return evaluation


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_evaluator_response(response):
    """Extract evaluator JSON from the LLM response."""

    response = response.strip()

    # Remove markdown JSON fences if the model adds them.
    response = re.sub(
        r"^```json\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )

    response = re.sub(
        r"\s*```$",
        "",
        response,
        flags=re.IGNORECASE,
    )

    try:
        result = json.loads(response)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Evaluator did not return valid JSON:\n{response}"
        ) from exc

    required_fields = [
        "evaluator_decision",
        "research_goal_alignment",
        "scientific_validity",
        "novelty",
        "feasibility",
        "testability",
        "evidence_quality",
        "reasoning_quality",
        "decision_appropriateness",
        "justification",
    ]

    for field in required_fields:
        assert field in result, (
            f"Evaluator response missing required field: {field}"
        )

    assert result["evaluator_decision"] in {
        "A",
        "B",
        "TIE",
        "ABSTAIN",
    }

    score_fields = [
        "research_goal_alignment",
        "scientific_validity",
        "novelty",
        "feasibility",
        "testability",
        "evidence_quality",
        "reasoning_quality",
        "decision_appropriateness",
    ]

    for field in score_fields:
        score = float(result[field])

        assert 1 <= score <= 5, (
            f"{field} must be between 1 and 5, got {score}"
        )

    return result


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def _calculate_overall_score(result):
    """Calculate the average quality score independently."""

    criteria = [
        result["research_goal_alignment"],
        result["scientific_validity"],
        result["novelty"],
        result["feasibility"],
        result["testability"],
        result["evidence_quality"],
        result["reasoning_quality"],
        result["decision_appropriateness"],
    ]

    return mean(criteria)


# ---------------------------------------------------------------------------
# Main LLM evaluation experiment
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ranking_agent_llm_evaluation():
    """
    Evaluate Ranking Agent decisions using an independent LLM.

    This is the main experiment requested by the evaluation framework.
    """

    results = []

    print("\n")
    print("=" * 70)
    print("RANKING AGENT LLM-BASED EVALUATION")
    print("=" * 70)

    print(f"Ranking model:   {RANKING_LLM_MODEL}")
    print(f"Evaluator model: {EVALUATOR_LLM_MODEL}")

    for case in EVALUATION_CASES:

        print("\n" + "-" * 70)
        print(f"Case: {case['case_id']}")
        print(f"Description: {case['description']}")

        research_goal = ResearchGoal(
            description=case["goal"],
        )

        hypothesis_a = _hypothesis(
            "A",
            case["hypothesis_a"]["text"],
            case["hypothesis_a"]["reflection"],
        )

        hypothesis_b = _hypothesis(
            "B",
            case["hypothesis_b"]["text"],
            case["hypothesis_b"]["reflection"],
        )

        # ---------------------------------------------------------------
        # Run the actual Ranking Agent
        # ---------------------------------------------------------------

        ranking_decision = run_pairwise_debate(
            hypothesis_a,
            hypothesis_b,
            research_goal,
        )

        print("\nRanking Agent Decision:")
        print(f"  Decision:   {ranking_decision.outcome}")
        print(f"  Confidence: {ranking_decision.confidence}")
        print(f"  Reasoning:  {ranking_decision.reasoning}")

        # ---------------------------------------------------------------
        # Independent LLM evaluation
        # ---------------------------------------------------------------

        evaluation = evaluate_ranking_decision(
            research_goal=research_goal.description,
            hypothesis_a=hypothesis_a,
            hypothesis_b=hypothesis_b,
            ranking_decision=ranking_decision,
        )

        calculated_score = _calculate_overall_score(evaluation)

        evaluation["calculated_overall_score"] = calculated_score
        evaluation["ranking_decision"] = ranking_decision.outcome
        evaluation["case_id"] = case["case_id"]

        results.append(evaluation)

        print("\nIndependent LLM Evaluation:")
        print(
            f"  Evaluator decision: "
            f"{evaluation['evaluator_decision']}"
        )

        print(
            f"  Agreement: "
            f"{evaluation['decision_agreement']}"
        )

        print(
            f"  Overall quality: "
            f"{calculated_score:.2f}/5.00"
        )

        print(
            f"  Quality percentage: "
            f"{(calculated_score / 5) * 100:.2f}%"
        )

        print(
            f"  Justification: "
            f"{evaluation['justification']}"
        )

    # -------------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------------

    agreement_count = sum(
        1
        for result in results
        if result["decision_agreement"]
    )

    agreement_rate = (
        agreement_count / len(results)
        if results
        else 0
    )

    average_quality = mean(
        result["calculated_overall_score"]
        for result in results
    )

    quality_percentage = (
        average_quality / 5
    ) * 100

    print("\n")
    print("=" * 70)
    print("FINAL LLM EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Total evaluation cases: {len(results)}"
    )

    print(
        f"Evaluator agreement: "
        f"{agreement_count}/{len(results)} "
        f"({agreement_rate * 100:.2f}%)"
    )

    print(
        f"Average quality score: "
        f"{average_quality:.2f}/5.00"
    )

    print(
        f"Average quality percentage: "
        f"{quality_percentage:.2f}%"
    )

    print("=" * 70)

    # Basic experiment validity checks.
    assert len(results) == len(EVALUATION_CASES)

    assert 0 <= agreement_rate <= 1

    assert 1 <= average_quality <= 5