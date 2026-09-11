"""Offline coverage for the Experimental readiness DAG.

These tests never generate: they walk the decision tree structurally and
construct the metric against a stub judge. The point of a DAG metric is that
its scoring is fixed by the graph rather than by a judge's mood, so the graph
itself is what there is to regression-test.
"""

import warnings

import pytest
from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import BinaryJudgementNode, VerdictNode
from deepeval.metrics.dag.schema import BinaryJudgementVerdict, MetricScoreReason
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from rubrics.deepeval_metrics import (
    DEFAULT_METRIC_FACTORIES,
    KIND_DAG,
    METRIC_DEFINITIONS,
    LLMEvaluationError,
    MetricSpec,
)
from rubrics.experimental_readiness import (
    COMPARISON_CRITERIA,
    INTERVENTION_CRITERIA,
    MEASUREMENT_CRITERIA,
    PREDICTION_CRITERIA,
    SCORE_NO_COMPARISON,
    SCORE_NO_INTERVENTION,
    SCORE_NO_MEASUREMENT,
    SCORE_NO_PREDICTION,
    SCORE_READY,
    build_experimental_readiness_dag,
)
from rubrics.suites import SUITE_HYPOTHESIS
from tests.test_deepeval_metrics import (  # noqa: F401
    StubJudge,
    fake_factories,
    parsed_run,
    run_evaluation,
)

METRIC_NAME = "Experimental readiness"

#: Lets the scripted judge below work out which rung it is being asked about.
RUNG_CRITERIA = {
    "intervention": INTERVENTION_CRITERIA,
    "measurement": MEASUREMENT_CRITERIA,
    "comparison": COMPARISON_CRITERIA,
    "prediction": PREDICTION_CRITERIA,
}

#: The ladder, top to bottom: each rung's label and the score for failing it.
LADDER = [
    ("intervention", SCORE_NO_INTERVENTION),
    ("measurement", SCORE_NO_MEASUREMENT),
    ("comparison", SCORE_NO_COMPARISON),
    ("prediction", SCORE_NO_PREDICTION),
]


def readiness_spec():
    return next(spec for spec in METRIC_DEFINITIONS if spec.name == METRIC_NAME)


def verdict(node, value):
    return next(child for child in node.children if child.verdict is value)


def walk():
    """Yield each rung as (node, failing VerdictNode, passing VerdictNode)."""
    node = build_experimental_readiness_dag().root_nodes[0]
    while node is not None:
        no, yes = verdict(node, False), verdict(node, True)
        yield node, no, yes
        node = yes.child


# --- the ladder --------------------------------------------------------------


def test_the_ladder_asks_the_four_questions_in_dependency_order():
    assert [node.label for node, _, _ in walk()] == [label for label, _ in LADDER]


def test_failing_a_rung_scores_exactly_that_rung():
    assert [(node.label, no.score) for node, no, _ in walk()] == LADDER


def test_only_a_hypothesis_that_clears_every_rung_is_fully_ready():
    rungs = list(walk())
    *intermediate, (_, _, final_yes) = rungs

    # Every passing verdict except the last hands off to another question.
    assert all(yes.child is not None and yes.score is None for _, _, yes in intermediate)
    assert final_yes.score == SCORE_READY
    assert final_yes.child is None


def test_scores_rise_monotonically_down_the_ladder():
    """A later failure must never score worse than an earlier one."""
    scores = [no.score for _, no, _ in walk()] + [SCORE_READY]

    assert scores == sorted(scores)
    assert scores == sorted(set(scores)), "two rungs share a score, so the report cannot tell them apart"


def test_every_rung_is_a_binary_judgement_with_one_yes_and_one_no():
    for node, _, _ in walk():
        assert isinstance(node, BinaryJudgementNode), node.label
        assert len(node.children) == 2, node.label
        assert all(isinstance(child, VerdictNode) for child in node.children), node.label
        assert sorted(child.verdict for child in node.children) == [False, True], node.label


# --- what the judge is allowed to see ----------------------------------------


def test_no_rung_may_consult_retrieval_context():
    """Readiness is a property of the hypothesis text, not of the evidence."""
    for node, _, _ in walk():
        assert node.evaluation_params == [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT], node.label


def test_every_rung_forbids_the_judge_from_filling_the_gap_itself():
    """The failure mode this metric exists to catch is a helpful judge."""
    for node, _, _ in walk():
        assert "your own domain knowledge" in node.criteria, node.label


def test_readiness_scores_a_hypothesis_with_no_evidence_at_all(fake_factories):  # noqa: F811
    report = run_evaluation(parsed_run(evidence="none"), fake_factories, suite=SUITE_HYPOTHESIS)
    metric = next(entry for entry in report["metrics"] if entry["name"] == METRIC_NAME)

    assert metric["status"] == "completed"


# --- DeepEval 4.2.2 compatibility --------------------------------------------


def test_the_readiness_spec_builds_a_dag_metric_not_a_geval():
    spec = readiness_spec()

    assert spec.kind == KIND_DAG
    assert DEFAULT_METRIC_FACTORIES[KIND_DAG] is DAGMetric


def test_a_dag_metric_takes_a_graph_and_never_a_geval_rubric():
    kwargs = readiness_spec().build_kwargs(threshold=0.7, model=None)

    assert set(kwargs) == {"threshold", "model", "async_mode", "name", "dag"}
    assert "criteria" not in kwargs
    assert "evaluation_steps" not in kwargs
    assert "evaluation_params" not in kwargs


def test_each_construction_gets_its_own_graph():
    """Nodes cache their verdict, so a shared graph would leak between runs."""
    spec = readiness_spec()

    first = spec.build_kwargs(threshold=0.7, model=None)["dag"]
    second = spec.build_kwargs(threshold=0.7, model=None)["dag"]

    assert first is not second
    assert first.root_nodes[0] is not second.root_nodes[0]


def test_the_readiness_metric_constructs_against_the_installed_deepeval():
    metric = DAGMetric(**readiness_spec().build_kwargs(threshold=0.7, model=StubJudge()))

    assert metric.threshold == 0.7
    assert metric.async_mode is False


def test_building_the_dag_uses_the_supported_top_down_api():
    """Passing `children=` to a node is the deprecated bottom-up API in 4.2.2."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        build_experimental_readiness_dag()


def test_a_dag_spec_without_a_builder_is_rejected_rather_than_silently_skipped():
    spec = MetricSpec(
        name="Broken",
        suite=SUITE_HYPOTHESIS,
        kind=KIND_DAG,
        evaluation_params=(SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT),
    )

    with pytest.raises(LLMEvaluationError, match="dag_builder"):
        spec.build_kwargs(threshold=0.7, model=None)


# --- the ladder end to end ---------------------------------------------------


class ScriptedJudge(DeepEvalBaseLLM):
    """Answers each rung from a fixed script instead of calling a model.

    The graph decides the score, so scripting the four verdicts pins down what
    the metric actually reports for a hypothesis with a given set of gaps --
    which the structural tests above cannot show on their own.
    """

    def __init__(self, **answers: bool):
        self.answers = answers
        self.asked: list[str] = []
        super().__init__("scripted-judge")

    def load_model(self):
        return self

    def generate_with_schema(self, prompt, schema=None, **_kwargs):
        if schema is MetricScoreReason:
            return MetricScoreReason(reason="scripted")
        for label, criteria in RUNG_CRITERIA.items():
            if criteria in prompt:
                self.asked.append(label)
                return BinaryJudgementVerdict(verdict=self.answers[label], reason="scripted")
        raise AssertionError("the judge was asked something outside the ladder")

    def generate(self, *args, **kwargs):  # pragma: no cover - schema path only
        raise AssertionError("DAG nodes always generate against a schema")

    async def a_generate(self, *args, **kwargs):  # pragma: no cover - sync only
        raise AssertionError("the suite runs with async_mode=False")

    def get_model_name(self):
        return "scripted-judge"


def measure_with(**answers):
    """Score the readiness ladder against a scripted set of verdicts."""
    judge = ScriptedJudge(**answers)
    metric = DAGMetric(**readiness_spec().build_kwargs(threshold=0.7, model=judge))
    metric.measure(LLMTestCase(input="Improve humidity stability.", actual_output="Some hypothesis."))
    return metric, judge


COMPLETE = {"intervention": True, "measurement": True, "comparison": True, "prediction": True}


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (None, SCORE_READY / 10),
        ("prediction", SCORE_NO_PREDICTION / 10),
        ("comparison", SCORE_NO_COMPARISON / 10),
        ("measurement", SCORE_NO_MEASUREMENT / 10),
        ("intervention", SCORE_NO_INTERVENTION / 10),
    ],
)
def test_the_first_missing_slot_fixes_the_reported_score(gap, expected):
    metric, _ = measure_with(**{**COMPLETE, gap: False} if gap else COMPLETE)

    assert metric.score == expected


def test_a_hypothesis_with_every_slot_filled_passes_the_threshold():
    metric, _ = measure_with(**COMPLETE)

    assert metric.is_successful()


def test_a_hypothesis_missing_its_comparator_fails_the_threshold():
    """0.6 against a 0.7 threshold, and the reason names the empty slot."""
    metric, _ = measure_with(**{**COMPLETE, "comparison": False})

    assert metric.score == 0.6
    assert not metric.is_successful()


def test_the_ladder_stops_at_the_first_gap_instead_of_asking_the_rest():
    _, judge = measure_with(**{**COMPLETE, "measurement": False})

    assert judge.asked == ["intervention", "measurement"]


def test_scores_stay_on_deepevals_zero_to_one_scale():
    for gap in [None, *COMPLETE]:
        metric, _ = measure_with(**{**COMPLETE, gap: False} if gap else COMPLETE)
        assert 0.0 <= metric.score <= 1.0
