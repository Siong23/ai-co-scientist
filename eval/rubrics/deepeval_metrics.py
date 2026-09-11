"""Opt-in DeepEval metric suites for persisted Co-Scientist hypotheses.

Metric definitions are declarative (:data:`METRIC_DEFINITIONS`) and execution is
centralized in :func:`evaluate_parsed_run`, so adding a metric never means
adding another branch to the runner.

Custom scientific metrics use ``GEval`` with explicit ``evaluation_steps`` and
never ``criteria`` as well: DeepEval 4.2.2 ignores ``criteria`` once steps are
supplied, and fixed steps judge more reproducibly than a generated rubric.

Measurement itself belongs to DeepEval. :func:`evaluate_parsed_run` decides
which metrics this artifact can support, hands them to ``deepeval.evaluate()``
in a single call -- which is also what renders DeepEval's native result table
on the terminal -- and translates the ``MetricData`` it returns back into this
project's report schema. No code here calls ``metric.measure()``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from deepeval import evaluate as deepeval_evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    DAGMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.test_run.test_run import TestRunResultDisplay

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

#: Truths per evidence document for Faithfulness. Left unset, DeepEval asks for
#: every undisputed truth in the retrieval context, and on a real run that meant
#: 141 statements -- author names, affiliations, and e-mail addresses among them
#: -- which a hypothesis is then checked against. Those cost 6+ minutes to
#: generate, and they re-enter the verdict prompt in full, where the judge ran
#: out of room and returned unparseable JSON. Capping the extraction keeps the
#: truths to the ones a claim could actually contradict.
FAITHFULNESS_TRUTHS_LIMIT = 5
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

#: DeepEval's own evaluator, which both measures the metrics and prints the
#: native result table. Tests inject a double here so the suite stays offline.
DEFAULT_EVALUATION_RUNNER: Callable[..., Any] = deepeval_evaluate


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
    #: Constructor arguments particular to one metric class, such as a built-in
    #: metric's own tuning knobs.
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

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
        kwargs.update(self.extra_kwargs)
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
        extra_kwargs={"truths_extraction_limit": FAITHFULNESS_TRUTHS_LIMIT},
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


def _build_metric(
    spec: MetricSpec,
    *,
    threshold: float,
    model: Any,
    factory: Callable[..., Any],
) -> Any:
    """Construct one metric, naming it when its factory rejects our arguments."""
    try:
        return factory(**spec.build_kwargs(threshold=threshold, model=model))
    except LLMEvaluationError:
        raise
    except Exception as exc:
        raise LLMEvaluationError(f"DeepEval metric {spec.name!r} could not be constructed: {exc}") from exc


def _deepeval_metric_name(metric: Any) -> str | None:
    """The name DeepEval reports for a metric, or ``None`` for a test double."""
    name = getattr(metric, "__name__", None)
    return name if isinstance(name, str) else None


def metric_data_to_report_entry(metric_data: Any, *, name: str, threshold: float) -> dict[str, Any]:
    """Translate one DeepEval ``MetricData`` into this project's report entry.

    Only the documented fields cross over, so no DeepEval object is ever
    serialized into the report. ``name`` is this project's metric name rather
    than DeepEval's, which appends a ``[GEval]`` or ``[DAG]`` suffix of its own.
    """
    error = getattr(metric_data, "error", None)
    if error:
        raise LLMEvaluationError(f"DeepEval metric {name!r} failed: {error}")

    score = getattr(metric_data, "score", None)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise LLMEvaluationError(f"DeepEval metric {name!r} returned no numeric score")

    reported_threshold = getattr(metric_data, "threshold", None)
    if isinstance(reported_threshold, bool) or not isinstance(reported_threshold, (int, float)):
        reported_threshold = threshold

    reason = getattr(metric_data, "reason", None)
    return {
        "name": name,
        "status": "completed",
        "score": float(score),
        "threshold": float(reported_threshold),
        "passed": bool(getattr(metric_data, "success", False)),
        "reason": reason if isinstance(reason, str) else None,
    }


def _run_native_evaluation(
    test_case: LLMTestCase,
    prepared: Sequence[tuple[MetricSpec, Any]],
    *,
    identifier: str | None,
    results_folder: str | None,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    """Run DeepEval's own evaluator and return its ``MetricData`` per spec.

    ``run_async=False`` is deliberate: the judge is a single local LM Studio
    server, and issuing several 27B judging calls at once makes every one of
    them slower. ``print_results`` is what puts DeepEval's native result table
    on the terminal, which is this harness's entire metric display.

    The score cache is written off as well. This harness never reads it -- a
    stale judgement is exactly what an audit must not reuse -- so writing it
    buys nothing, and on Windows it actively breaks: DeepEval takes a *shared*
    lock to read the cache back, which needs pywin32, and without it the read
    returns ``None`` that ``cache_test_case`` then dereferences. The first run
    in a fresh directory survives because there is no cache file to read; every
    run after it dies at the end with ``'NoneType' object has no attribute
    'test_cases_lookup_map'`` after the judge has already done all the work.
    """
    try:
        evaluation_result = runner(
            test_cases=[test_case],
            metrics=[metric for _, metric in prepared],
            identifier=identifier,
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(
                print_results=True,
                show_indicator=True,
                display_option=TestRunResultDisplay.ALL,
                results_folder=results_folder,
                # This is a batch command: never hold a finished run open on an
                # "open the inspect TUI?" prompt that nobody is there to answer.
                inspect_after_run=False,
            ),
            cache_config=CacheConfig(write_cache=False),
        )
    except LLMEvaluationError:
        raise
    except Exception as exc:
        raise LLMEvaluationError(f"DeepEval evaluation failed: {exc}") from exc

    test_results = getattr(evaluation_result, "test_results", None) or []
    if len(test_results) != 1:
        raise LLMEvaluationError(f"DeepEval returned {len(test_results)} test results for one test case")

    metrics_data = getattr(test_results[0], "metrics_data", None) or []
    if len(metrics_data) != len(prepared):
        raise LLMEvaluationError(f"DeepEval returned {len(metrics_data)} metric results for {len(prepared)} metrics")

    # DeepEval reports one MetricData per metric, in the order the metrics were
    # passed. Pair them back up by position and check that the names agree, so a
    # future reordering surfaces as an error rather than as mislabeled scores.
    measured: dict[str, Any] = {}
    for (spec, metric), metric_data in zip(prepared, metrics_data):
        expected = _deepeval_metric_name(metric)
        returned = getattr(metric_data, "name", None)
        if expected is not None and returned is not None and returned != expected:
            raise LLMEvaluationError(
                f"DeepEval returned metric results out of order: expected {expected!r}, found {returned!r}"
            )
        measured[spec.name] = metric_data
    return measured


def _run_identifier(parsed_run: Mapping[str, Any]) -> str | None:
    """Label the DeepEval test run with the run id it scored, when there is one."""
    run_id = parsed_run.get("run_id")
    return run_id.strip() if isinstance(run_id, str) and run_id.strip() else None


def evaluate_parsed_run(
    parsed_run: Mapping[str, Any],
    *,
    threshold: float = 0.7,
    model: Any = None,
    suite: str = DEFAULT_METRIC_SUITE,
    metric_factories: Mapping[str, Callable[..., Any]] | None = None,
    evaluation_runner: Callable[..., Any] | None = None,
    results_folder: str | None = None,
) -> dict[str, Any]:
    """Measure the selected hypothesis and return a JSON-serializable report.

    A metric needing source text the artifact does not carry is filtered out
    *before* DeepEval sees it -- passing it anyway would trade a documented skip
    for an error -- and reappears in the report as a skipped entry in its
    original position.
    """
    threshold = _validate_threshold(threshold)
    specs = select_metric_specs(suite)
    factories = {**DEFAULT_METRIC_FACTORIES, **(metric_factories or {})}

    test_case = build_test_case(parsed_run)
    context: RetrievalContext = extract_retrieval_context(parsed_run.get("evidence_sources", []))

    prepared: list[tuple[MetricSpec, Any]] = []
    skipped: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.requires_retrieval_context and not context.is_substantive:
            skipped[spec.name] = {"name": spec.name, "status": "skipped", "reason": context.reason}
            continue
        prepared.append((spec, _build_metric(spec, threshold=threshold, model=model, factory=factories[spec.kind])))

    measured: dict[str, Any] = {}
    if prepared:
        measured = _run_native_evaluation(
            test_case,
            prepared,
            identifier=_run_identifier(parsed_run),
            results_folder=results_folder,
            runner=evaluation_runner or DEFAULT_EVALUATION_RUNNER,
        )

    results = [
        skipped[spec.name]
        if spec.name in skipped
        else metric_data_to_report_entry(measured[spec.name], name=spec.name, threshold=threshold)
        for spec in specs
    ]

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
    "DEFAULT_EVALUATION_RUNNER",
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
    "metric_data_to_report_entry",
    "resolve_suite",
    "select_metric_specs",
]
