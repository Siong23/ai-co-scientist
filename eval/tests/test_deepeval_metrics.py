import pytest

from rubrics.deepeval_metrics import (
    LLMEvaluationError,
    build_test_case,
    evaluate_parsed_run,
)


def parsed_run(*, evidence=True):
    sources = (
        [{"source_id": "doi:example", "title": "Humidity barrier study"}]
        if evidence
        else []
    )
    return {
        "run_id": "run-test",
        "research_goal": "Improve perovskite humidity stability.",
        "hypothesis_source_step": "ranking_final",
        "selected_hypothesis": {
            "id": "H1",
            "title": "Hydrophobic barrier",
            "text": "Adding a hydrophobic barrier will increase stability after 1,000 hours.",
        },
        "evidence_sources": sources,
    }


class FakeMetric:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.threshold = kwargs["threshold"]
        self.score = None
        self.reason = None

    def measure(self, test_case):
        assert test_case.input == "Improve perovskite humidity stability."
        self.score = 0.8 if self.name != "Scientific testability" else 0.6
        self.reason = f"Reason for {self.name}"
        return self.score

    def is_successful(self):
        return self.score >= self.threshold


def test_build_test_case_maps_saved_artifact_fields():
    case = build_test_case(parsed_run())

    assert "Hydrophobic barrier" in case.actual_output
    assert "1,000 hours" in case.actual_output
    assert case.retrieval_context == [
        '{"source_id": "doi:example", "title": "Humidity barrier study"}'
    ]
    assert case.metadata["hypothesis_id"] == "H1"


def test_evaluation_reports_scores_and_threshold_failures_without_network():
    report = evaluate_parsed_run(parsed_run(), threshold=0.7, metric_factory=FakeMetric)

    assert report["passed"] is False
    assert [metric["status"] for metric in report["metrics"]] == [
        "completed",
        "completed",
        "completed",
    ]
    assert report["metrics"][1]["score"] == 0.6
    assert report["metrics"][1]["passed"] is False


def test_evidence_metric_is_skipped_when_artifact_has_no_sources():
    report = evaluate_parsed_run(parsed_run(evidence=False), metric_factory=FakeMetric)

    assert report["metrics"][2]["name"] == "Evidence support"
    assert report["metrics"][2]["status"] == "skipped"
    assert "no persisted evidence" in report["metrics"][2]["reason"]


def test_llm_evaluation_validates_hypothesis_text_and_threshold():
    run = parsed_run()
    run["selected_hypothesis"]["text"] = " "

    with pytest.raises(LLMEvaluationError, match="text must be a non-empty"):
        build_test_case(run)
    with pytest.raises(LLMEvaluationError, match="between 0 and 1"):
        evaluate_parsed_run(parsed_run(), threshold=1.1, metric_factory=FakeMetric)
