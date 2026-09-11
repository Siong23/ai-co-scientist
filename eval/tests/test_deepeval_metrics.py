"""Offline coverage for the DeepEval metric suites.

Every test here substitutes a metric factory, an evaluation runner, or a stub
judge, so the suite never contacts LM Studio, OpenAI, Confident AI, or any other
endpoint. The one test that does call DeepEval's real runner drives it with
metrics that score without a judge.
"""

import json
from functools import partial
from pathlib import Path

import pytest
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.test_run import MetricData
from deepeval.test_run.test_run import TestRunResultDisplay as ResultDisplay
from rubrics.deepeval_metrics import (
    DEFAULT_EVALUATION_RUNNER,
    DEFAULT_METRIC_FACTORIES,
    FAITHFULNESS_TRUTHS_LIMIT,
    KIND_ANSWER_RELEVANCY,
    KIND_CONTEXTUAL_RELEVANCY,
    KIND_DAG,
    KIND_FAITHFULNESS,
    KIND_GEVAL,
    METRIC_DEFINITIONS,
    LLMEvaluationError,
    build_test_case,
    evaluate_parsed_run,
    metric_data_to_report_entry,
    resolve_suite,
    select_metric_specs,
)
from tests.test_retrieval_context import (
    ABSTRACT,
    PASSAGE_A,
    PASSAGE_B,
    metadata_only_source,
    sourced_evidence,
)

GOAL = "Improve perovskite humidity stability."

HYPOTHESIS_METRICS = [
    "Goal alignment",
    "Experimental readiness",
    "Scientific testability",
    "Feasibility",
    "Scientific plausibility",
    "Novelty vs retrieved prior art",
]
RAG_METRICS = ["Answer relevancy", "Faithfulness", "Contextual relevancy"]
LEGACY_METRICS = ["Evidence support"]

#: How DeepEval names each metric kind in its own results: GEval and DAG append
#: a suffix to the name they were built with, and the built-in metrics carry
#: fixed display names. The doubles below reproduce that, so every test drives
#: the adapter with names that are deliberately not this project's metric names.
DEEPEVAL_DISPLAY_NAMES = {
    KIND_GEVAL: "{name} [GEval]",
    KIND_DAG: "{name} [DAG]",
    KIND_ANSWER_RELEVANCY: "Answer Relevancy",
    KIND_FAITHFULNESS: "Faithfulness",
    KIND_CONTEXTUAL_RELEVANCY: "Contextual Relevancy",
}


def parsed_run(*, evidence="sourced"):
    """Build a parser result whose evidence quality the caller chooses."""
    sources = {
        "sourced": [sourced_evidence()],
        "metadata": [metadata_only_source()],
        "none": [],
    }[evidence]
    return {
        "run_id": "run-test",
        "research_goal": GOAL,
        "hypothesis_source_step": "ranking_final",
        "selected_hypothesis": {
            "id": "H1",
            "title": "Hydrophobic barrier",
            "text": "Adding a hydrophobic barrier will increase stability after 1,000 hours.",
        },
        "evidence_sources": sources,
    }


class FakeMetric:
    """A metric double that records its construction and refuses to be measured.

    Measuring is DeepEval's job now: any call to ``measure`` here means project
    code went back to driving metrics itself.
    """

    built: list[dict] = []

    def __init__(self, kind, **kwargs):
        self.kind = kind
        self.kwargs = kwargs
        type(self).built.append(kwargs)
        self.threshold = kwargs["threshold"]

    @property
    def __name__(self):
        return DEEPEVAL_DISPLAY_NAMES[self.kind].format(name=self.kwargs.get("name"))

    def measure(self, test_case):
        raise AssertionError(f"{self.__name__} was measured by project code")

    def is_successful(self):
        raise AssertionError(f"{self.__name__} was read back by project code")


class RecordingRunner:
    """A stand-in for ``deepeval.evaluate`` that records how it was called.

    It answers with the real ``EvaluationResult``/``MetricData`` types so the
    report adapter is exercised against DeepEval's own objects rather than a
    second guess at their shape.
    """

    def __init__(self, scores=None, default_score=0.8):
        # Keyed case-insensitively so tests can write this project's metric
        # names while the runner only ever sees DeepEval's.
        self.scores = {name.lower(): score for name, score in (scores or {}).items()}
        self.default_score = default_score
        self.calls: list[dict] = []

    def score_for(self, deepeval_name):
        base = deepeval_name.removesuffix(" [GEval]").removesuffix(" [DAG]").strip().lower()
        return self.scores.get(base, self.default_score)

    def results_for(self, metrics):
        return [
            MetricData(
                name=metric.__name__,
                score=self.score_for(metric.__name__),
                threshold=metric.threshold,
                success=self.score_for(metric.__name__) >= metric.threshold,
                reason=f"fake judgement for {metric.__name__}",
            )
            for metric in metrics
        ]

    def __call__(self, *, test_cases, metrics, identifier, async_config, display_config, cache_config):
        self.calls.append(
            {
                "test_cases": test_cases,
                "metrics": metrics,
                "identifier": identifier,
                "async_config": async_config,
                "display_config": display_config,
                "cache_config": cache_config,
            }
        )
        metrics_data = self.results_for(metrics)
        return EvaluationResult(
            test_results=[
                TestResult(
                    name="test_case_0",
                    success=all(data.success for data in metrics_data),
                    metrics_data=metrics_data,
                    conversational=False,
                )
            ],
            confident_link=None,
            test_run_id=None,
        )


class StubJudge(DeepEvalBaseLLM):
    """A DeepEvalBaseLLM that lets metrics construct without any provider."""

    def __init__(self):
        super().__init__("stub-judge")

    def load_model(self):
        return self

    def generate(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("offline tests must not generate")

    async def a_generate(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("offline tests must not generate")

    def get_model_name(self):
        return "stub-judge"


class OfflineMetric(BaseMetric):
    """A real ``BaseMetric`` that scores without a judge.

    Used to drive DeepEval's actual runner in the compatibility test below.
    """

    def __init__(self, name, score, threshold=0.7):
        self.name = name
        self.fixed_score = score
        self.threshold = threshold
        self.async_mode = False
        self.strict_mode = False
        self.verbose_mode = False
        self.evaluation_model = "offline-stub"

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case, *args, **kwargs):
        self.score = self.fixed_score
        self.reason = f"offline {self.name}"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):  # pragma: no cover - sync only
        return self.measure(test_case)

    def is_successful(self):
        return self.success


@pytest.fixture
def fake_factories():
    """Route every metric kind through FakeMetric and reset its recordings."""
    FakeMetric.built = []
    yield {kind: partial(FakeMetric, kind) for kind in DEFAULT_METRIC_FACTORIES}
    FakeMetric.built = []


def run_evaluation(parsed, factories, *, runner=None, **kwargs):
    """Evaluate a parsed run with the offline doubles wired in."""
    return evaluate_parsed_run(
        parsed,
        metric_factories=factories,
        evaluation_runner=runner if runner is not None else RecordingRunner(),
        **kwargs,
    )


def names(report):
    return [metric["name"] for metric in report["metrics"]]


def by_name(report, name):
    return next(metric for metric in report["metrics"] if metric["name"] == name)


# --- suite selection ---------------------------------------------------------


def test_resolve_suite_expands_all_to_hypothesis_and_rag():
    assert resolve_suite("all") == ("hypothesis", "rag")
    assert resolve_suite("hypothesis") == ("hypothesis",)
    assert resolve_suite("rag") == ("rag",)
    assert resolve_suite("legacy") == ("legacy",)


def test_unknown_suite_is_rejected():
    with pytest.raises(LLMEvaluationError, match="unknown metric suite"):
        resolve_suite("everything")


@pytest.mark.parametrize(
    ("suite", "expected"),
    [
        ("hypothesis", HYPOTHESIS_METRICS),
        ("rag", RAG_METRICS),
        ("legacy", LEGACY_METRICS),
        ("all", HYPOTHESIS_METRICS + RAG_METRICS),
    ],
)
def test_metric_suite_selects_exactly_its_metrics(suite, expected):
    assert [spec.name for spec in select_metric_specs(suite)] == expected


def test_all_suite_never_double_counts_evidence_support():
    """Faithfulness supersedes Evidence support, so 'all' must not run both."""
    all_names = [spec.name for spec in select_metric_specs("all")]
    assert "Faithfulness" in all_names
    assert "Evidence support" not in all_names


# --- native DeepEval execution -----------------------------------------------


def test_metrics_are_handed_to_deepeval_instead_of_being_measured_here(fake_factories):
    """The doubles raise if measured, so a passing report proves DeepEval ran them."""
    runner = RecordingRunner()

    report = run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=runner)

    assert len(runner.calls) == 1
    assert [metric["status"] for metric in report["metrics"]] == ["completed"] * len(HYPOTHESIS_METRICS)


def test_one_test_case_carrying_the_selected_hypothesis_is_evaluated(fake_factories):
    runner = RecordingRunner()

    run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=runner)
    test_cases = runner.calls[0]["test_cases"]

    assert len(test_cases) == 1
    assert isinstance(test_cases[0], LLMTestCase)
    assert test_cases[0].input == GOAL
    assert "Hydrophobic barrier" in test_cases[0].actual_output
    # The DeepEval test run is labeled with the run it scored.
    assert runner.calls[0]["identifier"] == "run-test"


@pytest.mark.parametrize(
    ("suite", "expected"),
    [
        (
            "hypothesis",
            [
                "Goal alignment [GEval]",
                "Experimental readiness [DAG]",
                "Scientific testability [GEval]",
                "Feasibility [GEval]",
                "Scientific plausibility [GEval]",
                "Novelty vs retrieved prior art [GEval]",
            ],
        ),
        ("rag", ["Answer Relevancy", "Faithfulness", "Contextual Relevancy"]),
        ("legacy", ["Evidence support [GEval]"]),
    ],
)
def test_the_requested_suite_builds_exactly_its_metrics(fake_factories, suite, expected):
    runner = RecordingRunner()

    run_evaluation(parsed_run(), fake_factories, suite=suite, runner=runner)

    assert [metric.__name__ for metric in runner.calls[0]["metrics"]] == expected


def test_evaluation_runs_sequentially_against_the_single_local_judge(fake_factories):
    """Concurrent 27B judging calls to one LM Studio server only slow each other."""
    runner = RecordingRunner()

    run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=runner)
    async_config = runner.calls[0]["async_config"]

    assert isinstance(async_config, AsyncConfig)
    assert async_config.run_async is False


def test_deepeval_is_asked_for_its_own_terminal_results(fake_factories):
    runner = RecordingRunner()

    run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=runner)
    display_config = runner.calls[0]["display_config"]

    assert isinstance(display_config, DisplayConfig)
    assert display_config.print_results is True
    assert display_config.show_indicator is True
    assert display_config.display_option is ResultDisplay.ALL
    # One run, one report: DeepEval's own result folder stays off by default.
    assert display_config.results_folder is None
    # A batch command must never stop on the interactive inspect prompt.
    assert display_config.inspect_after_run is False


def test_the_score_cache_is_never_written(fake_factories):
    """Regression: a written cache is read back under a Windows shared lock.

    Without pywin32 that read returns None, which ``cache_test_case`` then
    dereferences -- so every run after the first died with ``'NoneType' object
    has no attribute 'test_cases_lookup_map'`` once the judge had already done
    all the work. An audit must not reuse a stale judgement anyway.
    """
    runner = RecordingRunner()

    run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=runner)
    cache_config = runner.calls[0]["cache_config"]

    assert isinstance(cache_config, CacheConfig)
    assert cache_config.write_cache is False
    assert cache_config.use_cache is False


def test_deepevals_own_result_folder_is_opt_in(fake_factories, tmp_path):
    runner = RecordingRunner()

    run_evaluation(parsed_run(), fake_factories, suite="rag", runner=runner, results_folder=str(tmp_path))

    assert runner.calls[0]["display_config"].results_folder == str(tmp_path)


def test_the_configured_judge_and_sequential_mode_reach_every_metric(fake_factories):
    judge = StubJudge()

    run_evaluation(parsed_run(), fake_factories, suite="all", model=judge)

    assert len(FakeMetric.built) == len(HYPOTHESIS_METRICS + RAG_METRICS)
    assert all(built["model"] is judge for built in FakeMetric.built)
    assert all(built["async_mode"] is False for built in FakeMetric.built)


# --- translating DeepEval results into the project report --------------------


def test_metric_data_becomes_a_report_entry_under_this_projects_metric_name():
    entry = metric_data_to_report_entry(
        MetricData(name="Goal alignment [GEval]", score=0.84, threshold=0.7, success=True, reason="because"),
        name="Goal alignment",
        threshold=0.7,
    )

    assert entry == {
        "name": "Goal alignment",
        "status": "completed",
        "score": 0.84,
        "threshold": 0.7,
        "passed": True,
        "reason": "because",
    }


def test_the_threshold_and_verdict_come_from_deepeval_not_from_the_request():
    entry = metric_data_to_report_entry(
        MetricData(name="Faithfulness", score=0.3, threshold=0.9, success=False, reason=None),
        name="Faithfulness",
        threshold=0.7,
    )

    assert entry["score"] == 0.3
    assert entry["threshold"] == 0.9
    assert entry["passed"] is False
    assert entry["reason"] is None


def test_a_metric_deepeval_could_not_score_is_an_evaluation_error():
    with pytest.raises(LLMEvaluationError, match="returned no numeric score"):
        metric_data_to_report_entry(
            MetricData(name="Faithfulness", score=None, threshold=0.7, success=False),
            name="Faithfulness",
            threshold=0.7,
        )
    with pytest.raises(LLMEvaluationError, match="judge timed out"):
        metric_data_to_report_entry(
            MetricData(name="Faithfulness", score=None, threshold=0.7, success=False, error="judge timed out"),
            name="Faithfulness",
            threshold=0.7,
        )


def test_results_returned_out_of_order_are_rejected(fake_factories):
    """Positional pairing is only safe while the names still line up."""

    class ShufflingRunner(RecordingRunner):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.test_results[0].metrics_data.reverse()
            return result

    with pytest.raises(LLMEvaluationError, match="out of order"):
        run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=ShufflingRunner())


def test_a_missing_metric_result_is_rejected_rather_than_silently_dropped(fake_factories):
    class ForgetfulRunner(RecordingRunner):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.test_results[0].metrics_data.pop()
            return result

    with pytest.raises(LLMEvaluationError, match="5 metric results for 6 metrics"):
        run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=ForgetfulRunner())


def test_a_failing_runner_is_reported_as_an_evaluation_error(fake_factories):
    def broken_runner(**kwargs):
        raise RuntimeError("judge unreachable")

    with pytest.raises(LLMEvaluationError, match="DeepEval evaluation failed: judge unreachable"):
        run_evaluation(parsed_run(), fake_factories, suite="hypothesis", runner=broken_runner)


# --- hypothesis suite --------------------------------------------------------


def test_hypothesis_suite_scores_every_hypothesis_metric(fake_factories):
    runner = RecordingRunner(
        scores={
            "Goal alignment": 0.9,
            "Experimental readiness": 0.8,
            "Scientific testability": 0.6,
            "Feasibility": 0.75,
            "Scientific plausibility": 0.82,
            "Novelty vs retrieved prior art": 0.71,
        }
    )
    report = run_evaluation(parsed_run(), fake_factories, threshold=0.7, suite="hypothesis", runner=runner)

    assert names(report) == HYPOTHESIS_METRICS
    assert [metric["status"] for metric in report["metrics"]] == ["completed"] * len(HYPOTHESIS_METRICS)
    assert by_name(report, "Goal alignment")["score"] == 0.9
    assert by_name(report, "Experimental readiness")["score"] == 0.8
    assert by_name(report, "Feasibility")["score"] == 0.75
    assert by_name(report, "Scientific plausibility")["score"] == 0.82
    assert by_name(report, "Goal alignment")["reason"] == "fake judgement for Goal alignment [GEval]"
    # One metric below threshold fails the run.
    assert by_name(report, "Scientific testability")["passed"] is False
    assert report["passed"] is False


def test_threshold_decides_pass_and_fail(fake_factories):
    runner = RecordingRunner(default_score=0.72)

    assert run_evaluation(parsed_run(), fake_factories, threshold=0.7, suite="hypothesis", runner=runner)["passed"]
    assert not run_evaluation(parsed_run(), fake_factories, threshold=0.8, suite="hypothesis", runner=runner)["passed"]


def test_novelty_is_skipped_without_substantive_prior_art(fake_factories):
    runner = RecordingRunner()
    report = run_evaluation(parsed_run(evidence="metadata"), fake_factories, suite="hypothesis", runner=runner)
    novelty = by_name(report, "Novelty vs retrieved prior art")

    assert novelty["status"] == "skipped"
    assert "only citation metadata" in novelty["reason"]
    assert "score" not in novelty
    # A metric that cannot be grounded never reaches DeepEval at all.
    assert "Novelty vs retrieved prior art [GEval]" not in [metric.__name__ for metric in runner.calls[0]["metrics"]]
    # The goal-only metrics still run.
    assert by_name(report, "Goal alignment")["status"] == "completed"
    assert by_name(report, "Feasibility")["status"] == "completed"


def test_novelty_runs_against_real_prior_art_text(fake_factories):
    report = run_evaluation(parsed_run(), fake_factories, suite="hypothesis")

    assert by_name(report, "Novelty vs retrieved prior art")["status"] == "completed"


# --- RAG suite ---------------------------------------------------------------


def test_answer_relevancy_runs_without_any_retrieval_context(fake_factories):
    report = run_evaluation(parsed_run(evidence="none"), fake_factories, suite="rag")

    assert by_name(report, "Answer relevancy")["status"] == "completed"
    assert by_name(report, "Answer relevancy")["score"] == 0.8


@pytest.mark.parametrize("evidence", ["metadata", "none"])
@pytest.mark.parametrize("metric", ["Faithfulness", "Contextual relevancy"])
def test_grounded_rag_metrics_skip_without_substantive_evidence(fake_factories, evidence, metric):
    report = run_evaluation(parsed_run(evidence=evidence), fake_factories, suite="rag")
    entry = by_name(report, metric)

    assert entry["status"] == "skipped"
    assert entry["reason"]
    assert "score" not in entry


def test_grounded_rag_metrics_run_on_real_persisted_evidence_text(fake_factories):
    report = run_evaluation(parsed_run(), fake_factories, suite="rag")

    assert [metric["status"] for metric in report["metrics"]] == ["completed"] * 3
    assert report["retrieval_context"] == {
        "substantive": True,
        "evidence_source_count": 1,
        "sources_with_text": 1,
        "passage_count": 3,
        "reason": None,
    }


def test_legacy_suite_still_exposes_evidence_support(fake_factories):
    report = run_evaluation(parsed_run(), fake_factories, suite="legacy")

    assert names(report) == LEGACY_METRICS
    assert report["metrics"][0]["status"] == "completed"


# --- report shape and all-skipped handling -----------------------------------


def test_report_is_json_serializable_with_the_documented_shape(fake_factories):
    report = run_evaluation(parsed_run(evidence="metadata"), fake_factories, threshold=0.7, suite="all")
    restored = json.loads(json.dumps(report, ensure_ascii=False))

    assert restored["suite"] == "all"
    assert restored["status"] == "completed"
    assert set(by_name(restored, "Goal alignment")) == {
        "name",
        "status",
        "score",
        "threshold",
        "passed",
        "reason",
    }
    assert set(by_name(restored, "Faithfulness")) == {"name", "status", "reason"}
    assert by_name(restored, "Faithfulness")["status"] == "skipped"
    assert by_name(restored, "Goal alignment")["threshold"] == 0.7


def test_skipped_metrics_keep_their_place_in_the_declared_order(fake_factories):
    report = run_evaluation(parsed_run(evidence="metadata"), fake_factories, suite="all")

    assert names(report) == HYPOTHESIS_METRICS + RAG_METRICS
    assert [metric["status"] for metric in report["metrics"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
        "skipped",
        "completed",
        "skipped",
        "skipped",
    ]


def test_all_skipped_suite_is_reported_explicitly_and_does_not_pass(fake_factories):
    runner = RecordingRunner()
    report = run_evaluation(parsed_run(evidence="none"), fake_factories, suite="legacy", runner=runner)

    assert [metric["status"] for metric in report["metrics"]] == ["skipped"]
    assert report["status"] == "no_metrics_completed"
    assert report["passed"] is False
    assert "every metric" in report["reason"]
    # With nothing to measure there is no reason to start a DeepEval run.
    assert runner.calls == []


def test_skipped_metrics_do_not_fail_a_run_whose_other_metrics_pass(fake_factories):
    report = run_evaluation(parsed_run(evidence="metadata"), fake_factories, threshold=0.7, suite="all")

    assert any(metric["status"] == "skipped" for metric in report["metrics"])
    assert report["passed"] is True


def test_scores_are_reported_on_deepevals_zero_to_one_scale(fake_factories):
    report = run_evaluation(parsed_run(), fake_factories, suite="all")

    for metric in report["metrics"]:
        if metric["status"] == "completed":
            assert 0.0 <= metric["score"] <= 1.0


# --- test-case construction and validation -----------------------------------


def test_build_test_case_maps_saved_artifact_fields():
    case = build_test_case(parsed_run())

    assert case.input == GOAL
    assert "Hydrophobic barrier" in case.actual_output
    assert "1,000 hours" in case.actual_output
    assert case.retrieval_context == [PASSAGE_A.strip(), PASSAGE_B.strip(), ABSTRACT.strip()]
    assert case.metadata["hypothesis_id"] == "H1"


def test_build_test_case_leaves_retrieval_context_unset_for_metadata_only_evidence():
    assert build_test_case(parsed_run(evidence="metadata")).retrieval_context is None


def test_llm_evaluation_validates_hypothesis_text_and_threshold(fake_factories):
    run = parsed_run()
    run["selected_hypothesis"]["text"] = " "

    with pytest.raises(LLMEvaluationError, match="text must be a non-empty"):
        build_test_case(run)
    with pytest.raises(LLMEvaluationError, match="between 0 and 1"):
        run_evaluation(parsed_run(), fake_factories, threshold=1.1)


# --- DeepEval 4.2.2 compatibility --------------------------------------------


def test_the_installed_deepeval_runner_returns_the_metric_data_the_adapter_reads(monkeypatch, tmp_path):
    """Pin the 4.2.2 result contract by running DeepEval's real evaluator.

    The metrics score without a judge, telemetry is opted out, and the run is
    written inside ``tmp_path``, so nothing leaves the machine. The native
    terminal display is asserted through the recorded ``DisplayConfig``
    elsewhere; rendering it here would only make the suite noisy.
    """
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    monkeypatch.chdir(tmp_path)

    result = DEFAULT_EVALUATION_RUNNER(
        test_cases=[build_test_case(parsed_run())],
        metrics=[OfflineMetric("Goal alignment [GEval]", 0.9), OfflineMetric("Faithfulness", 0.5)],
        identifier="run-test",
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(
            print_results=False,
            show_indicator=False,
            display_option=ResultDisplay.ALL,
            inspect_after_run=False,
        ),
    )
    metrics_data = result.test_results[0].metrics_data

    assert len(result.test_results) == 1
    # Results come back one per metric, in the order the metrics were passed.
    assert [data.name for data in metrics_data] == ["Goal alignment [GEval]", "Faithfulness"]
    assert metric_data_to_report_entry(metrics_data[0], name="Goal alignment", threshold=0.7) == {
        "name": "Goal alignment",
        "status": "completed",
        "score": 0.9,
        "threshold": 0.7,
        "passed": True,
        "reason": "offline Goal alignment [GEval]",
    }
    assert metric_data_to_report_entry(metrics_data[1], name="Faithfulness", threshold=0.7)["passed"] is False


def test_geval_specs_never_pass_both_criteria_and_evaluation_steps():
    """Regression: GEval takes one rubric source, and ours is evaluation_steps."""
    for spec in METRIC_DEFINITIONS:
        if spec.kind != KIND_GEVAL:
            continue
        kwargs = spec.build_kwargs(threshold=0.7, model=None)
        assert "criteria" not in kwargs, spec.name
        assert kwargs["evaluation_steps"], spec.name


def test_geval_metrics_construct_against_the_installed_deepeval():
    for spec in METRIC_DEFINITIONS:
        if spec.kind != KIND_GEVAL:
            continue
        metric = GEval(**spec.build_kwargs(threshold=0.7, model=StubJudge()))

        assert metric.criteria is None
        assert metric.evaluation_steps == list(spec.evaluation_steps)
        assert metric.evaluation_params == list(spec.evaluation_params)
        assert metric.threshold == 0.7
        assert metric.async_mode is False


def test_the_display_names_the_doubles_use_are_the_ones_deepeval_produces():
    """The name-ordering guard is only meaningful against DeepEval's real names."""
    for spec in METRIC_DEFINITIONS:
        if spec.kind not in (KIND_GEVAL, KIND_DAG):
            continue
        factory = DEFAULT_METRIC_FACTORIES[spec.kind]
        metric = factory(**spec.build_kwargs(threshold=0.7, model=StubJudge()))

        assert metric.__name__ == DEEPEVAL_DISPLAY_NAMES[spec.kind].format(name=spec.name)


@pytest.mark.parametrize(
    ("kind", "metric_class"),
    [
        (KIND_ANSWER_RELEVANCY, AnswerRelevancyMetric),
        (KIND_FAITHFULNESS, FaithfulnessMetric),
        (KIND_CONTEXTUAL_RELEVANCY, ContextualRelevancyMetric),
    ],
)
def test_builtin_specs_declare_exactly_the_params_deepeval_requires(kind, metric_class):
    spec = next(spec for spec in METRIC_DEFINITIONS if spec.kind == kind)

    assert DEFAULT_METRIC_FACTORIES[kind] is metric_class
    assert list(spec.evaluation_params) == list(metric_class._required_params)
    # Built-in metrics reject GEval-only rubric arguments.
    kwargs = spec.build_kwargs(threshold=0.7, model=None)
    assert {"criteria", "evaluation_steps", "evaluation_params", "name", "dag"}.isdisjoint(kwargs)
    assert {"threshold", "model", "async_mode"} <= set(kwargs)
    metric = metric_class(**spec.build_kwargs(threshold=0.7, model=StubJudge()))
    assert metric.__name__ == DEEPEVAL_DISPLAY_NAMES[kind]


def test_faithfulness_caps_how_many_truths_it_extracts():
    """Uncapped, one real run yielded 141 truths -- author e-mails included.

    They cost minutes to generate and re-enter the verdict prompt in full, where
    the judge ran out of room and returned unparseable JSON.
    """
    spec = next(spec for spec in METRIC_DEFINITIONS if spec.kind == KIND_FAITHFULNESS)

    kwargs = spec.build_kwargs(threshold=0.7, model=None)

    assert kwargs["truths_extraction_limit"] == FAITHFULNESS_TRUTHS_LIMIT
    assert FaithfulnessMetric(**spec.build_kwargs(threshold=0.7, model=StubJudge())).truths_extraction_limit == (
        FAITHFULNESS_TRUTHS_LIMIT
    )


def test_no_other_metric_carries_faithfulness_only_tuning():
    for spec in METRIC_DEFINITIONS:
        if spec.kind == KIND_FAITHFULNESS:
            continue
        assert "truths_extraction_limit" not in spec.build_kwargs(threshold=0.7, model=None), spec.name


def test_every_metric_kind_has_a_factory():
    assert {spec.kind for spec in METRIC_DEFINITIONS} <= set(DEFAULT_METRIC_FACTORIES)


# --- CLI ---------------------------------------------------------------------


def test_cli_rejects_an_unknown_metric_suite(capsys):
    from scripts import evaluate_run

    with pytest.raises(SystemExit) as excinfo:
        evaluate_run.build_parser().parse_args(["run.json", "--metric-suite", "everything"])

    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("suite", ["hypothesis", "rag", "legacy", "all"])
def test_cli_accepts_every_documented_metric_suite(suite):
    from scripts import evaluate_run

    args = evaluate_run.build_parser().parse_args(["run.json", "--metric-suite", suite])

    assert args.metric_suite == suite


def test_cli_defaults_to_the_all_suite():
    from scripts import evaluate_run

    assert evaluate_run.build_parser().parse_args(["run.json"]).metric_suite == "all"


@pytest.mark.parametrize(
    ("suite", "expected"),
    [
        ("hypothesis", "hyp-run-test.json"),
        ("rag", "rag-run-test.json"),
        ("legacy", "legacy-run-test.json"),
        ("all", "all-run-test.json"),
    ],
)
def test_report_filename_is_derived_from_the_suite_and_run_id(monkeypatch, tmp_path, suite, expected):
    from scripts import evaluate_run

    monkeypatch.setattr(evaluate_run, "REPORTS_DIR", tmp_path)

    assert evaluate_run.default_report_path(suite, "run-test", Path("run.json")) == tmp_path / expected


def test_derived_report_name_falls_back_to_the_run_filename(monkeypatch, tmp_path):
    from scripts import evaluate_run

    monkeypatch.setattr(evaluate_run, "REPORTS_DIR", tmp_path)

    assert evaluate_run.default_report_path("hypothesis", None, Path("../results/runs/run-abc.json")) == (
        tmp_path / "hyp-run-abc.json"
    )
    # A run id that is not a usable filename never escapes the reports folder.
    assert evaluate_run.default_report_path("rag", "../../etc/passwd", Path("run.json")) == (
        tmp_path / "rag-etc-passwd.json"
    )


def patch_cli_evaluation(monkeypatch, tmp_path, *, parsed, factories, runner=None):
    """Point the CLI at a temporary reports folder and an offline evaluation."""
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "offline-placeholder")
    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed)
    monkeypatch.setattr(evaluate_run, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr("deepeval.models.LocalModel", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(
        deepeval_metrics,
        "evaluate_parsed_run",
        lambda parsed_run_arg, **kwargs: evaluate_parsed_run(
            parsed_run_arg,
            metric_factories=factories,
            evaluation_runner=runner if runner is not None else RecordingRunner(),
            **{key: value for key, value in kwargs.items() if key != "model"},
        ),
    )
    return evaluate_run


JUDGE_ARGS = [
    "--judge-model",
    "test-local-judge",
    "--judge-base-url",
    "http://localhost:1234/v1/",
]


def test_cli_leaves_the_metric_display_to_deepeval_and_files_the_report(monkeypatch, capsys, tmp_path, fake_factories):
    evaluate_run = patch_cli_evaluation(monkeypatch, tmp_path, parsed=parsed_run(), factories=fake_factories)

    exit_code = evaluate_run.main(["unused.json", "--llm-metrics", "--metric-suite", "hypothesis", *JUDGE_ARGS])
    captured = capsys.readouterr()
    report_path = tmp_path / "hyp-run-test.json"

    assert exit_code == 0
    # No audit JSON on the terminal: DeepEval's own table is the display.
    assert "{" not in captured.out
    assert captured.out.splitlines()[0] == "AI Co-Scientist report written to:"
    assert report_path.name in captured.out
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["llm_evaluation"]["suite"] == "hypothesis"
    assert written["llm_evaluation"]["metrics"][0]["reason"]


def test_cli_names_the_metrics_that_never_reached_deepeval(monkeypatch, capsys, tmp_path, fake_factories):
    evaluate_run = patch_cli_evaluation(
        monkeypatch, tmp_path, parsed=parsed_run(evidence="metadata"), factories=fake_factories
    )

    exit_code = evaluate_run.main(["unused.json", "--llm-metrics", "--metric-suite", "rag", *JUDGE_ARGS])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Skipped metrics:" in captured.out
    assert "- Faithfulness: the persisted evidence sources carry only citation metadata" in captured.out
    assert "- Contextual relevancy:" in captured.out


def test_cli_report_option_overrides_the_derived_path(monkeypatch, capsys, tmp_path, fake_factories):
    evaluate_run = patch_cli_evaluation(monkeypatch, tmp_path, parsed=parsed_run(), factories=fake_factories)
    chosen = tmp_path / "chosen" / "report.json"

    exit_code = evaluate_run.main(
        ["unused.json", "--llm-metrics", "--metric-suite", "rag", "--report", str(chosen), *JUDGE_ARGS]
    )

    assert exit_code == 0
    assert chosen.exists()
    assert not (tmp_path / "rag-run-test.json").exists()
    assert chosen.name in capsys.readouterr().out


def test_cli_passes_explicit_local_judge_and_suite_after_deepeval_import(monkeypatch, tmp_path, fake_factories):
    from deepeval.models import LocalModel
    from deepeval.models.llms import local_model as local_module
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "offline-placeholder")
    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed_run())
    monkeypatch.setattr(evaluate_run, "REPORTS_DIR", tmp_path)
    clients = []

    def fake_client(**kwargs):
        clients.append(kwargs)
        return object()

    monkeypatch.setattr(local_module, "OpenAI", fake_client)

    def evaluate(parsed, *, threshold, model, suite, results_folder):
        assert isinstance(model, LocalModel)
        assert model.name == "test-local-judge"
        assert model.temperature == 0
        assert suite == "hypothesis"
        assert results_folder is None
        model.load_model()
        return evaluate_parsed_run(
            parsed,
            threshold=threshold,
            model=model,
            suite=suite,
            metric_factories=fake_factories,
            evaluation_runner=RecordingRunner(scores={"Scientific testability": 0.6}),
        )

    monkeypatch.setattr(deepeval_metrics, "evaluate_parsed_run", evaluate)
    exit_code = evaluate_run.main(["unused.json", "--llm-metrics", "--metric-suite", "hypothesis", *JUDGE_ARGS])

    assert exit_code == 1
    assert clients
    assert all(client["base_url"] == "http://localhost:1234/v1" for client in clients)
    assert all(client["api_key"] == "offline-placeholder" for client in clients)
    assert all(built["model"] is not None for built in FakeMetric.built)


def test_cli_reports_an_all_skipped_suite_instead_of_passing(monkeypatch, capsys, tmp_path, fake_factories):
    evaluate_run = patch_cli_evaluation(
        monkeypatch, tmp_path, parsed=parsed_run(evidence="none"), factories=fake_factories
    )

    exit_code = evaluate_run.main(["unused.json", "--llm-metrics", "--metric-suite", "legacy", *JUDGE_ARGS])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No metric completed" in captured.err
    written = json.loads((tmp_path / "legacy-run-test.json").read_text(encoding="utf-8"))
    assert written["llm_evaluation"]["passed"] is False


def test_parser_only_mode_still_prints_the_json_and_never_evaluates(monkeypatch, capsys, tmp_path):
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed_run())
    monkeypatch.setattr(evaluate_run, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        deepeval_metrics,
        "evaluate_parsed_run",
        lambda *args, **kwargs: pytest.fail("parser-only mode must not evaluate"),
    )
    monkeypatch.setattr(
        deepeval_metrics,
        "DEFAULT_EVALUATION_RUNNER",
        lambda **kwargs: pytest.fail("parser-only mode must not call DeepEval"),
    )

    exit_code = evaluate_run.main(["unused.json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.startswith("AI Co-Scientist evaluation report")
    assert json.loads(captured.out.split("\n", 1)[1])["run_id"] == "run-test"
    # Without --llm-metrics there is no report to file.
    assert list(tmp_path.iterdir()) == []


def test_reports_and_errors_never_leak_environment_secrets(monkeypatch, capsys, tmp_path, fake_factories):
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    secret = "sk-super-secret-judge-key"
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", secret)
    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed_run())
    monkeypatch.setattr(evaluate_run, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        deepeval_metrics,
        "evaluate_parsed_run",
        lambda parsed, **kwargs: (_ for _ in ()).throw(
            deepeval_metrics.LLMEvaluationError(f"judge refused key {secret}")
        ),
    )
    monkeypatch.setattr("deepeval.models.LocalModel", lambda **kwargs: None, raising=False)

    report_path = tmp_path / "report.json"
    exit_code = evaluate_run.main(["unused.json", "--llm-metrics", *JUDGE_ARGS, "--report", str(report_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert secret not in captured.out
    assert secret not in captured.err
    assert "[REDACTED]" in captured.err
    assert not report_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_report_file_records_the_suite_and_never_contains_the_key(monkeypatch, tmp_path, fake_factories):
    secret = "sk-super-secret-judge-key"
    evaluate_run = patch_cli_evaluation(monkeypatch, tmp_path, parsed=parsed_run(), factories=fake_factories)
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", secret)

    report_path = tmp_path / "report.json"
    evaluate_run.main(
        ["unused.json", "--llm-metrics", "--metric-suite", "rag", *JUDGE_ARGS, "--report", str(report_path)]
    )
    written = report_path.read_text(encoding="utf-8")

    assert secret not in written
    assert json.loads(written)["llm_evaluation"]["suite"] == "rag"
