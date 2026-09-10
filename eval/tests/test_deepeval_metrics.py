"""Offline coverage for the DeepEval metric suites.

Every test here substitutes fake metric factories or stub judges, so the suite
never contacts LM Studio, OpenAI, or any other endpoint.
"""

import json

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from rubrics.deepeval_metrics import (
    DEFAULT_METRIC_FACTORIES,
    KIND_ANSWER_RELEVANCY,
    KIND_CONTEXTUAL_RELEVANCY,
    KIND_FAITHFULNESS,
    KIND_GEVAL,
    METRIC_DEFINITIONS,
    LLMEvaluationError,
    build_test_case,
    evaluate_parsed_run,
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
    "Scientific testability",
    "Feasibility",
    "Scientific plausibility",
    "Novelty vs retrieved prior art",
]
RAG_METRICS = ["Answer relevancy", "Faithfulness", "Contextual relevancy"]
LEGACY_METRICS = ["Evidence support"]


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
    """Records its construction and returns a name-dependent fixed score."""

    scores: dict[str, float] = {}
    built: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).built.append(kwargs)
        self.threshold = kwargs["threshold"]
        self.score = None
        self.reason = None

    def measure(self, test_case):
        assert test_case.input == GOAL
        self.score = type(self).scores.get(self.kwargs.get("name"), 0.8)
        self.reason = "fake judgement"
        return self.score

    def is_successful(self):
        return self.score >= self.threshold


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


@pytest.fixture
def fake_factories():
    """Route every metric kind through FakeMetric and reset its recordings."""
    FakeMetric.scores = {}
    FakeMetric.built = []
    yield {kind: FakeMetric for kind in DEFAULT_METRIC_FACTORIES}
    FakeMetric.scores = {}
    FakeMetric.built = []


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


# --- hypothesis suite --------------------------------------------------------


def test_hypothesis_suite_scores_every_hypothesis_metric(fake_factories):
    FakeMetric.scores = {
        "Goal alignment": 0.9,
        "Scientific testability": 0.6,
        "Feasibility": 0.75,
        "Scientific plausibility": 0.82,
        "Novelty vs retrieved prior art": 0.71,
    }
    report = evaluate_parsed_run(parsed_run(), threshold=0.7, suite="hypothesis", metric_factories=fake_factories)

    assert names(report) == HYPOTHESIS_METRICS
    assert [metric["status"] for metric in report["metrics"]] == ["completed"] * 5
    assert by_name(report, "Goal alignment")["score"] == 0.9
    assert by_name(report, "Feasibility")["score"] == 0.75
    assert by_name(report, "Scientific plausibility")["score"] == 0.82
    # One metric below threshold fails the run.
    assert by_name(report, "Scientific testability")["passed"] is False
    assert report["passed"] is False


def test_threshold_decides_pass_and_fail(fake_factories):
    FakeMetric.scores = {name: 0.72 for name in HYPOTHESIS_METRICS}

    assert evaluate_parsed_run(parsed_run(), threshold=0.7, suite="hypothesis", metric_factories=fake_factories)[
        "passed"
    ]
    assert not evaluate_parsed_run(parsed_run(), threshold=0.8, suite="hypothesis", metric_factories=fake_factories)[
        "passed"
    ]


def test_novelty_is_skipped_without_substantive_prior_art(fake_factories):
    report = evaluate_parsed_run(parsed_run(evidence="metadata"), suite="hypothesis", metric_factories=fake_factories)
    novelty = by_name(report, "Novelty vs retrieved prior art")

    assert novelty["status"] == "skipped"
    assert "only citation metadata" in novelty["reason"]
    assert "score" not in novelty
    # The goal-only metrics still run.
    assert by_name(report, "Goal alignment")["status"] == "completed"
    assert by_name(report, "Feasibility")["status"] == "completed"


def test_novelty_runs_against_real_prior_art_text(fake_factories):
    report = evaluate_parsed_run(parsed_run(), suite="hypothesis", metric_factories=fake_factories)

    assert by_name(report, "Novelty vs retrieved prior art")["status"] == "completed"


# --- RAG suite ---------------------------------------------------------------


def test_answer_relevancy_runs_without_any_retrieval_context(fake_factories):
    report = evaluate_parsed_run(parsed_run(evidence="none"), suite="rag", metric_factories=fake_factories)

    assert by_name(report, "Answer relevancy")["status"] == "completed"
    assert by_name(report, "Answer relevancy")["score"] == 0.8


@pytest.mark.parametrize("evidence", ["metadata", "none"])
@pytest.mark.parametrize("metric", ["Faithfulness", "Contextual relevancy"])
def test_grounded_rag_metrics_skip_without_substantive_evidence(fake_factories, evidence, metric):
    report = evaluate_parsed_run(parsed_run(evidence=evidence), suite="rag", metric_factories=fake_factories)
    entry = by_name(report, metric)

    assert entry["status"] == "skipped"
    assert entry["reason"]
    assert "score" not in entry


def test_grounded_rag_metrics_run_on_real_persisted_evidence_text(fake_factories):
    report = evaluate_parsed_run(parsed_run(), suite="rag", metric_factories=fake_factories)

    assert [metric["status"] for metric in report["metrics"]] == ["completed"] * 3
    assert report["retrieval_context"] == {
        "substantive": True,
        "evidence_source_count": 1,
        "sources_with_text": 1,
        "passage_count": 3,
        "reason": None,
    }


def test_legacy_suite_still_exposes_evidence_support(fake_factories):
    report = evaluate_parsed_run(parsed_run(), suite="legacy", metric_factories=fake_factories)

    assert names(report) == LEGACY_METRICS
    assert report["metrics"][0]["status"] == "completed"


# --- report shape and all-skipped handling -----------------------------------


def test_report_is_json_serializable_with_the_documented_shape(fake_factories):
    report = evaluate_parsed_run(
        parsed_run(evidence="metadata"), threshold=0.7, suite="all", metric_factories=fake_factories
    )
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


def test_all_skipped_suite_is_reported_explicitly_and_does_not_pass(fake_factories):
    report = evaluate_parsed_run(parsed_run(evidence="none"), suite="legacy", metric_factories=fake_factories)

    assert [metric["status"] for metric in report["metrics"]] == ["skipped"]
    assert report["status"] == "no_metrics_completed"
    assert report["passed"] is False
    assert "every metric" in report["reason"]


def test_skipped_metrics_do_not_fail_a_run_whose_other_metrics_pass(fake_factories):
    report = evaluate_parsed_run(
        parsed_run(evidence="metadata"), threshold=0.7, suite="all", metric_factories=fake_factories
    )

    assert any(metric["status"] == "skipped" for metric in report["metrics"])
    assert report["passed"] is True


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
        evaluate_parsed_run(parsed_run(), threshold=1.1, metric_factories=fake_factories)


# --- DeepEval 4.2.2 compatibility --------------------------------------------


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
    assert set(spec.build_kwargs(threshold=0.7, model=None)) == {"threshold", "model", "async_mode"}
    metric_class(**spec.build_kwargs(threshold=0.7, model=StubJudge()))


def test_every_metric_kind_has_a_factory():
    assert {spec.kind for spec in METRIC_DEFINITIONS} <= set(DEFAULT_METRIC_FACTORIES)


def test_scores_are_reported_on_deepevals_zero_to_one_scale(fake_factories):
    report = evaluate_parsed_run(parsed_run(), suite="all", metric_factories=fake_factories)

    for metric in report["metrics"]:
        if metric["status"] == "completed":
            assert 0.0 <= metric["score"] <= 1.0


def test_the_configured_judge_reaches_every_metric(fake_factories):
    judge = StubJudge()
    evaluate_parsed_run(parsed_run(), suite="all", model=judge, metric_factories=fake_factories)

    assert len(FakeMetric.built) == len(HYPOTHESIS_METRICS + RAG_METRICS)
    assert all(built["model"] is judge for built in FakeMetric.built)
    assert all(built["async_mode"] is False for built in FakeMetric.built)


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


def test_cli_passes_explicit_local_judge_and_suite_after_deepeval_import(monkeypatch, fake_factories):
    from deepeval.models import LocalModel
    from deepeval.models.llms import local_model as local_module
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "offline-placeholder")
    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed_run())
    FakeMetric.scores = {"Scientific testability": 0.6}
    clients = []

    def fake_client(**kwargs):
        clients.append(kwargs)
        return object()

    monkeypatch.setattr(local_module, "OpenAI", fake_client)

    def evaluate(parsed, *, threshold, model, suite):
        assert isinstance(model, LocalModel)
        assert model.name == "test-local-judge"
        assert model.temperature == 0
        assert suite == "hypothesis"
        model.load_model()
        return evaluate_parsed_run(
            parsed, threshold=threshold, model=model, suite=suite, metric_factories=fake_factories
        )

    monkeypatch.setattr(deepeval_metrics, "evaluate_parsed_run", evaluate)
    exit_code = evaluate_run.main(
        [
            "unused.json",
            "--llm-metrics",
            "--metric-suite",
            "hypothesis",
            "--judge-model",
            "test-local-judge",
            "--judge-base-url",
            "http://localhost:1234/v1/",
        ]
    )

    assert exit_code == 1
    assert clients
    assert all(client["base_url"] == "http://localhost:1234/v1" for client in clients)
    assert all(client["api_key"] == "offline-placeholder" for client in clients)
    assert all(built["model"] is not None for built in FakeMetric.built)


def test_cli_reports_an_all_skipped_suite_instead_of_passing(monkeypatch, capsys, fake_factories):
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "offline-placeholder")
    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed_run(evidence="none"))
    monkeypatch.setattr(
        deepeval_metrics,
        "evaluate_parsed_run",
        lambda parsed, *, threshold, model, suite: evaluate_parsed_run(
            parsed, threshold=threshold, model=None, suite=suite, metric_factories=fake_factories
        ),
    )
    monkeypatch.setattr("deepeval.models.LocalModel", lambda **kwargs: None, raising=False)

    exit_code = evaluate_run.main(
        [
            "unused.json",
            "--llm-metrics",
            "--metric-suite",
            "legacy",
            "--judge-model",
            "test-local-judge",
            "--judge-base-url",
            "http://localhost:1234/v1/",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No metric completed" in captured.err
    assert json.loads(captured.out.split("\n", 1)[1])["llm_evaluation"]["passed"] is False


def test_reports_and_errors_never_leak_environment_secrets(monkeypatch, capsys, tmp_path, fake_factories):
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    secret = "sk-super-secret-judge-key"
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", secret)
    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed_run())
    monkeypatch.setattr(
        deepeval_metrics,
        "evaluate_parsed_run",
        lambda parsed, *, threshold, model, suite: (_ for _ in ()).throw(
            deepeval_metrics.LLMEvaluationError(f"judge refused key {secret}")
        ),
    )
    monkeypatch.setattr("deepeval.models.LocalModel", lambda **kwargs: None, raising=False)

    report_path = tmp_path / "report.json"
    exit_code = evaluate_run.main(
        [
            "unused.json",
            "--llm-metrics",
            "--judge-model",
            "test-local-judge",
            "--judge-base-url",
            "http://localhost:1234/v1/",
            "--report",
            str(report_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert secret not in captured.out
    assert secret not in captured.err
    assert "[REDACTED]" in captured.err
    assert not report_path.exists()


def test_report_file_records_the_suite_and_never_contains_the_key(monkeypatch, tmp_path, fake_factories):
    from rubrics import deepeval_metrics

    from scripts import evaluate_run

    secret = "sk-super-secret-judge-key"
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", secret)
    monkeypatch.setattr(evaluate_run, "parse_run", lambda *args: parsed_run())
    monkeypatch.setattr(
        deepeval_metrics,
        "evaluate_parsed_run",
        lambda parsed, *, threshold, model, suite: evaluate_parsed_run(
            parsed, threshold=threshold, model=None, suite=suite, metric_factories=fake_factories
        ),
    )
    monkeypatch.setattr("deepeval.models.LocalModel", lambda **kwargs: None, raising=False)

    report_path = tmp_path / "report.json"
    evaluate_run.main(
        [
            "unused.json",
            "--llm-metrics",
            "--metric-suite",
            "rag",
            "--judge-model",
            "test-local-judge",
            "--judge-base-url",
            "http://localhost:1234/v1/",
            "--report",
            str(report_path),
        ]
    )
    written = report_path.read_text(encoding="utf-8")

    assert secret not in written
    assert json.loads(written)["llm_evaluation"]["suite"] == "rag"
