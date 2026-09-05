"""Opt-in DeepEval metrics for persisted Co-Scientist hypotheses."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


class LLMEvaluationError(RuntimeError):
    """Raised when an LLM-backed evaluation cannot be completed."""


METRIC_DEFINITIONS = (
    {
        "name": "Goal alignment",
        "criteria": (
            "Judge whether the proposed research hypothesis directly addresses the "
            "research goal. Reward a clear mechanism and outcome connected to the goal; "
            "penalize tangential or generic proposals."
        ),
        "evaluation_params": [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        "evaluation_steps": [
            "Identify the central objective and constraints in the research goal.",
            "Identify the intervention, mechanism, and intended outcome in the hypothesis.",
            "Score how directly and completely the hypothesis addresses the goal.",
        ],
    },
    {
        "name": "Scientific testability",
        "criteria": (
            "Judge whether the hypothesis is falsifiable and can be tested by a concrete "
            "experiment. Reward explicit variables, measurable outcomes, a plausible "
            "comparison or control, and a predicted direction of effect."
        ),
        "evaluation_params": [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        "evaluation_steps": [
            "Extract the proposed intervention or independent variable.",
            "Extract measurable outcomes, comparison conditions, and predicted effects.",
            "Score whether a researcher could design a falsifying experiment from the text.",
        ],
    },
    {
        "name": "Evidence support",
        "criteria": (
            "Judge whether the supplied evidence sources support the factual and mechanistic "
            "claims made by the hypothesis. Use only the supplied retrieval context; do not "
            "assume that a citation supports claims not represented in that context."
        ),
        "evaluation_params": [
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        "evaluation_steps": [
            "Identify the hypothesis's factual and mechanistic claims.",
            "For each material claim, locate support or contradiction in the supplied evidence.",
            "Score the coverage and strength of support, penalizing unsupported extrapolation.",
        ],
        "requires_evidence": True,
    },
)


def hypothesis_as_text(hypothesis: Mapping[str, Any]) -> str:
    """Return the persisted hypothesis fields that the judge should assess."""
    title = hypothesis.get("title")
    body = hypothesis.get("text")
    if not isinstance(body, str) or not body.strip():
        raise LLMEvaluationError("selected hypothesis text must be a non-empty string")
    if isinstance(title, str) and title.strip():
        return f"Title: {title.strip()}\n\nHypothesis: {body.strip()}"
    return body.strip()


def evidence_as_context(sources: Any) -> list[str]:
    """Serialize evidence records without inventing content absent from the artifact."""
    if not isinstance(sources, list) or not all(isinstance(item, Mapping) for item in sources):
        raise LLMEvaluationError("evidence_sources must be a list of objects")
    return [json.dumps(dict(source), ensure_ascii=False, sort_keys=True) for source in sources]


def build_test_case(parsed_run: Mapping[str, Any]) -> LLMTestCase:
    """Map a deterministic parser result to a DeepEval single-turn test case."""
    goal = parsed_run.get("research_goal")
    hypothesis = parsed_run.get("selected_hypothesis")
    if not isinstance(goal, str) or not goal.strip():
        raise LLMEvaluationError("research_goal must be a non-empty string")
    if not isinstance(hypothesis, Mapping):
        raise LLMEvaluationError("selected_hypothesis must be an object")

    return LLMTestCase(
        input=goal.strip(),
        actual_output=hypothesis_as_text(hypothesis),
        retrieval_context=evidence_as_context(parsed_run.get("evidence_sources", [])),
        metadata={
            "run_id": parsed_run.get("run_id"),
            "hypothesis_source_step": parsed_run.get("hypothesis_source_step"),
            "hypothesis_id": hypothesis.get("id"),
        },
    )


def evaluate_parsed_run(
    parsed_run: Mapping[str, Any],
    *,
    threshold: float = 0.7,
    model: Any = None,
    metric_factory: Callable[..., Any] = GEval,
) -> dict[str, Any]:
    """Measure the selected hypothesis and return a JSON-serializable report."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise LLMEvaluationError("metric threshold must be numeric")
    if not 0 <= float(threshold) <= 1:
        raise LLMEvaluationError("metric threshold must be between 0 and 1")

    test_case = build_test_case(parsed_run)
    results: list[dict[str, Any]] = []

    for definition in METRIC_DEFINITIONS:
        if definition.get("requires_evidence") and not test_case.retrieval_context:
            results.append(
                {
                    "name": definition["name"],
                    "status": "skipped",
                    "reason": "the selected hypothesis has no persisted evidence sources",
                }
            )
            continue

        kwargs = {
            key: value
            for key, value in definition.items()
            if key in {"name", "criteria", "evaluation_params", "evaluation_steps"}
        }
        try:
            metric = metric_factory(
                **kwargs,
                threshold=float(threshold),
                model=model,
                async_mode=False,
            )
            metric.measure(test_case)
            score = getattr(metric, "score", None)
            passed = bool(metric.is_successful())
        except Exception as exc:
            raise LLMEvaluationError(
                f"DeepEval metric {definition['name']!r} failed: {exc}"
            ) from exc

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise LLMEvaluationError(
                f"DeepEval metric {definition['name']!r} returned no numeric score"
            )
        results.append(
            {
                "name": definition["name"],
                "status": "completed",
                "score": float(score),
                "threshold": float(threshold),
                "passed": passed,
                "reason": getattr(metric, "reason", None),
            }
        )

    completed = [result for result in results if result["status"] == "completed"]
    return {
        "passed": bool(completed) and all(result["passed"] for result in completed),
        "metrics": results,
    }
