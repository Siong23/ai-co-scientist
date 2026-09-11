"""Opt-in DeepEval metric suites for persisted Co-Scientist hypotheses.

Metric definitions are declarative (:data:`METRIC_DEFINITIONS`) and execution is
centralized in :func:`evaluate_parsed_run`, so adding a metric never means
adding another branch to the runner.

Custom scientific metrics use ``GEval`` with explicit ``evaluation_steps`` and
never ``criteria`` as well: DeepEval 4.2.2 ignores ``criteria`` once steps are
supplied, and fixed steps judge more reproducibly than a generated rubric.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    DAGMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from rubrics.errors import LLMEvaluationError
from rubrics.experimental_readiness import (
    EVALUATION_PARAMS as READINESS_PARAMS,
)
from rubrics.experimental_readiness import (
    build_experimental_readiness_dag,
)
from rubrics.retrieval_context import RetrievalContext, extract_retrieval_context
from rubrics.suites import (
    DEFAULT_METRIC_SUITE,
    METRIC_SUITES,
    SUITE_ALL,
    SUITE_HYPOTHESIS,
    SUITE_LEGACY,
    SUITE_RAG,
)

KIND_GEVAL = "geval"
KIND_DAG = "dag"

KIND_ANSWER_RELEVANCY = "answer_relevancy"
KIND_FAITHFULNESS = "faithfulness"
KIND_CONTEXTUAL_RELEVANCY = "contextual_relevancy"

#: Metric classes per kind; tests substitute fakes so the suite stays offline.
DEFAULT_METRIC_FACTORIES: Mapping[str, Callable[..., Any]] = {
    KIND_GEVAL: GEval,
    KIND_DAG: DAGMetric,
    KIND_ANSWER_RELEVANCY: AnswerRelevancyMetric,
    KIND_FAITHFULNESS: FaithfulnessMetric,
    KIND_CONTEXTUAL_RELEVANCY: ContextualRelevancyMetric,
}


@dataclass(frozen=True)
class MetricSpec:
    """A metric, the suite it belongs to, and the artifact fields it needs."""

    name: str
    suite: str
    kind: str
    evaluation_params: tuple[SingleTurnParams, ...]
    evaluation_steps: tuple[str, ...] = ()
    #: Builds this metric's decision tree. DAG metrics only, and called once per
    #: construction because nodes cache their verdict across a ``measure`` call.
    dag_builder: Callable[[], Any] | None = None

    @property
    def requires_retrieval_context(self) -> bool:
        """Whether this metric may only run against substantive source text."""
        return SingleTurnParams.RETRIEVAL_CONTEXT in self.evaluation_params

    def build_kwargs(self, *, threshold: float, model: Any) -> dict[str, Any]:
        """Return the constructor arguments for this metric's factory."""
        kwargs: dict[str, Any] = {
            "threshold": threshold,
            "model": model,
            "async_mode": False,
        }
        if self.kind == KIND_GEVAL:
            # Only GEval takes a rubric. The built-in metrics declare their own
            # required params and reject unexpected keyword arguments.
            kwargs["name"] = self.name
            kwargs["evaluation_params"] = list(self.evaluation_params)
            kwargs["evaluation_steps"] = list(self.evaluation_steps)
        elif self.kind == KIND_DAG:
            # A DAG carries its rubric in the graph: each node holds its own
            # criteria and evaluation_params, so the metric takes neither.
            if self.dag_builder is None:
                raise LLMEvaluationError(f"metric {self.name!r} is a DAG metric but declares no dag_builder")
            kwargs["name"] = self.name
            kwargs["dag"] = self.dag_builder()
        return kwargs


METRIC_DEFINITIONS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="Goal alignment",
        suite=SUITE_HYPOTHESIS,
        kind=KIND_GEVAL,
        evaluation_params=(SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT),
        evaluation_steps=(
            "Identify the central objective and every explicit constraint stated in the research goal.",
            "Identify the intervention, mechanism, and intended outcome proposed by the hypothesis.",
            "Check each explicit goal constraint against the hypothesis and note any it ignores or contradicts.",
            "Score how directly and completely the hypothesis addresses the stated goal, penalizing "
            "tangential, generic, or only partially responsive proposals.",
        ),
    ),
    # Placed directly after goal alignment because the two answer the pipeline's
    # first two questions: is this hypothesis about the right thing, and could
    # anyone actually run it. A DAG rather than a GEval so that each lost point
    # names the slot the hypothesis left empty; see experimental_readiness.
    MetricSpec(
        name="Experimental readiness",
        suite=SUITE_HYPOTHESIS,
        kind=KIND_DAG,
        evaluation_params=tuple(READINESS_PARAMS),
        dag_builder=build_experimental_readiness_dag,
    ),
    MetricSpec(
        name="Scientific testability",
        suite=SUITE_HYPOTHESIS,
        kind=KIND_GEVAL,
        evaluation_params=(SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT),
        evaluation_steps=(
            "Identify the intervention or independent variable the hypothesis proposes to manipulate.",
            "Identify the dependent variable or outcome, and whether the text states how it would be measured.",
            "Identify any comparison, baseline, or control condition, and whether one is needed for this claim.",
            "Identify the predicted direction or size of the effect.",
            "Decide whether a researcher could design a concrete experiment whose result would falsify the "
            "hypothesis as written.",
            "Score only on the elements above. Award no credit for scientific vocabulary, hedging, or a "
            "confident tone that is not backed by a stated variable, measurement, comparison, or prediction.",
        ),
    ),
    MetricSpec(
        name="Feasibility",
        suite=SUITE_HYPOTHESIS,
        kind=KIND_GEVAL,
        evaluation_params=(SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT),
        evaluation_steps=(
            "List the data, equipment, instrumentation, and measurements the proposed work would require, "
            "using only what the research goal and the hypothesis state.",
            "Judge whether the text establishes that those resources are obtainable, or whether it silently "
            "depends on proprietary datasets, unavailable hardware, or access it never mentions.",
            "Assess experimental complexity and implementation burden: scale, duration, expertise, and the "
            "number of steps that must succeed together.",
            "Note any dependency that is unrealistic or impossible as stated.",
            "Score how realistically this research could be executed on the supplied information. Do not "
            "assume access to resources the text does not mention.",
        ),
    ),
    MetricSpec(
        name="Scientific plausibility",
        suite=SUITE_HYPOTHESIS,
        kind=KIND_GEVAL,
        evaluation_params=(SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT),
        evaluation_steps=(
            "State, step by step, the causal or mechanistic account the hypothesis proposes.",
            "Check whether each step follows from the previous one, or whether the argument jumps from "
            "correlation, analogy, or restatement to a causal claim.",
            "Check the mechanism for internal contradictions and for assumptions that are physically, "
            "biologically, or computationally impossible.",
            "Check whether the predicted outcome actually follows from the stated mechanism.",
            "Score the internal coherence and scientific reasoning of the mechanism. Judge only the reasoning "
            "in the supplied text; do not claim to have verified any statement against external literature, "
            "and do not credit or penalize the hypothesis on sources that were not supplied.",
        ),
    ),
    MetricSpec(
        name="Novelty vs retrieved prior art",
        suite=SUITE_HYPOTHESIS,
        kind=KIND_GEVAL,
        evaluation_params=(
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ),
        evaluation_steps=(
            "Summarize the approaches, mechanisms, and findings that the retrieval context actually describes.",
            "State what the hypothesis proposes for the research goal.",
            "Identify the specific respects in which the hypothesis differs from the retrieved work - a "
            "different mechanism, combination, setting, or measurement - and the respects in which it "
            "restates that work.",
            "Score how meaningfully the hypothesis differs from the approaches present in the retrieval "
            "context. Treat that context as the only record of prior art available: do not substitute your "
            "own background knowledge for it, and do not assert novelty with respect to the wider literature.",
        ),
    ),
    MetricSpec(
        name="Answer relevancy",
        suite=SUITE_RAG,
        kind=KIND_ANSWER_RELEVANCY,
        evaluation_params=(SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT),
    ),
    MetricSpec(
        name="Faithfulness",
        suite=SUITE_RAG,
        kind=KIND_FAITHFULNESS,
        evaluation_params=(
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ),
    ),
    # DeepEval 4.2.2 scores the retrieval context against the input alone here;
    # ACTUAL_OUTPUT is deliberately absent because the metric never reads it.
    MetricSpec(
        name="Contextual relevancy",
        suite=SUITE_RAG,
        kind=KIND_CONTEXTUAL_RELEVANCY,
        evaluation_params=(
            SingleTurnParams.INPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ),
    ),
    # Superseded by Faithfulness, which measures the same failure mode with
    # DeepEval's purpose-built claim/truth decomposition. Kept out of every
    # default suite so that the two are never double-counted.
    MetricSpec(
        name="Evidence support",
        suite=SUITE_LEGACY,
        kind=KIND_GEVAL,
        evaluation_params=(
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ),
        evaluation_steps=(
            "Identify the hypothesis's factual and mechanistic claims.",
            "For each material claim, locate support or contradiction in the supplied evidence.",
            "Score the coverage and strength of support, penalizing unsupported extrapolation.",
        ),
    ),
)


def resolve_suite(suite: str) -> tuple[str, ...]:
    """Expand a requested suite name into the concrete suites it selects."""
    if suite not in METRIC_SUITES:
        raise LLMEvaluationError(f"unknown metric suite {suite!r}; choose from {', '.join(METRIC_SUITES)}")
    if suite == SUITE_ALL:
        return (SUITE_HYPOTHESIS, SUITE_RAG)
    return (suite,)


def select_metric_specs(suite: str) -> tuple[MetricSpec, ...]:
    """Return the metric definitions belonging to a requested suite."""
    selected = resolve_suite(suite)
    return tuple(spec for spec in METRIC_DEFINITIONS if spec.suite in selected)


def hypothesis_as_text(hypothesis: Mapping[str, Any]) -> str:
    """Return the persisted hypothesis fields that the judge should assess."""
    title = hypothesis.get("title")
    body = hypothesis.get("text")
    if not isinstance(body, str) or not body.strip():
        raise LLMEvaluationError("selected hypothesis text must be a non-empty string")
    if isinstance(title, str) and title.strip():
        return f"Title: {title.strip()}\n\nHypothesis: {body.strip()}"
    return body.strip()


def build_test_case(parsed_run: Mapping[str, Any]) -> LLMTestCase:
    """Map a deterministic parser result to a DeepEval single-turn test case."""
    goal = parsed_run.get("research_goal")
    hypothesis = parsed_run.get("selected_hypothesis")
    if not isinstance(goal, str) or not goal.strip():
        raise LLMEvaluationError("research_goal must be a non-empty string")
    if not isinstance(hypothesis, Mapping):
        raise LLMEvaluationError("selected_hypothesis must be an object")

    context = extract_retrieval_context(parsed_run.get("evidence_sources", []))
    return LLMTestCase(
        input=goal.strip(),
        actual_output=hypothesis_as_text(hypothesis),
        retrieval_context=context.as_list(),
        metadata={
            "run_id": parsed_run.get("run_id"),
            "hypothesis_source_step": parsed_run.get("hypothesis_source_step"),
            "hypothesis_id": hypothesis.get("id"),
        },
    )


def _validate_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise LLMEvaluationError("metric threshold must be numeric")
    if not 0 <= float(threshold) <= 1:
        raise LLMEvaluationError("metric threshold must be between 0 and 1")
    return float(threshold)


def _run_metric(
    spec: MetricSpec,
    test_case: LLMTestCase,
    *,
    threshold: float,
    model: Any,
    factory: Callable[..., Any],
) -> dict[str, Any]:
    """Construct, measure, and serialize a single metric."""
    try:
        metric = factory(**spec.build_kwargs(threshold=threshold, model=model))
        metric.measure(test_case)
        score = getattr(metric, "score", None)
        passed = bool(metric.is_successful())
    except Exception as exc:
        raise LLMEvaluationError(f"DeepEval metric {spec.name!r} failed: {exc}") from exc

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise LLMEvaluationError(f"DeepEval metric {spec.name!r} returned no numeric score")

    return {
        "name": spec.name,
        "status": "completed",
        "score": float(score),
        "threshold": threshold,
        "passed": passed,
        "reason": getattr(metric, "reason", None),
    }


def evaluate_parsed_run(
    parsed_run: Mapping[str, Any],
    *,
    threshold: float = 0.7,
    model: Any = None,
    suite: str = DEFAULT_METRIC_SUITE,
    metric_factories: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    """Measure the selected hypothesis and return a JSON-serializable report."""
    threshold = _validate_threshold(threshold)
    specs = select_metric_specs(suite)
    factories = {**DEFAULT_METRIC_FACTORIES, **(metric_factories or {})}

    test_case = build_test_case(parsed_run)
    context: RetrievalContext = extract_retrieval_context(parsed_run.get("evidence_sources", []))

    results: list[dict[str, Any]] = []
    for spec in specs:
        if spec.requires_retrieval_context and not context.is_substantive:
            results.append({"name": spec.name, "status": "skipped", "reason": context.reason})
            continue
        results.append(
            _run_metric(
                spec,
                test_case,
                threshold=threshold,
                model=model,
                factory=factories[spec.kind],
            )
        )

    completed = [result for result in results if result["status"] == "completed"]
    report: dict[str, Any] = {
        "suite": suite,
        "status": "completed" if completed else "no_metrics_completed",
        "passed": bool(completed) and all(result["passed"] for result in completed),
        "metrics": results,
        "retrieval_context": context.summary(),
    }
    if not completed:
        report["reason"] = (
            f"every metric in the {suite!r} suite was skipped for this artifact"
            if results
            else f"the {suite!r} suite selected no metrics"
        )
    return report


__all__ = [
    "DEFAULT_METRIC_FACTORIES",
    "DEFAULT_METRIC_SUITE",
    "METRIC_DEFINITIONS",
    "METRIC_SUITES",
    "SUITE_ALL",
    "SUITE_HYPOTHESIS",
    "SUITE_LEGACY",
    "SUITE_RAG",
    "LLMEvaluationError",
    "MetricSpec",
    "build_test_case",
    "evaluate_parsed_run",
    "hypothesis_as_text",
    "resolve_suite",
    "select_metric_specs",
]
