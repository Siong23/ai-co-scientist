"""
Tests for ExperimentComparator.

These tests verify the paper-vs-automated-experiment comparison
without running an actual deep-learning experiment or making
real LLM calls.
"""


import pytest

from app.experiments.experiment_comparator import ExperimentComparator
from unittest.mock import MagicMock


# ============================================================
# Test Data
# ============================================================


def create_experiment_result():
    """
    Create a realistic ExperimentRunner result.

    ExperimentComparator expects metrics under:

        experiment_result
            -> outputs
                -> metrics
    """

    return {
        "success": True,
        "status": "completed",
        "run_directory": (
            "app/experiments/results/runs/"
            "E2634_20260909_123319_881448"
        ),
        "outputs": {
            "metrics": {
                "accuracy": 0.9384,
                "precision_weighted": 0.9271,
                "recall_weighted": 0.9384,
                "f1_weighted": 0.9321,
                "confusion_matrix": [
                    [950, 30],
                    [45, 975],
                ],
                "training_seconds": 120.5,
                "evaluation_seconds": 5.2,
                "total_execution_seconds": 125.7,
            },
            "metrics_path": (
                "app/experiments/results/runs/"
                "E2634_20260909_123319_881448/metrics.json"
            ),
            "experiment_summary": {
                "model": "BiLSTM",
                "dataset": "5G-NIDD",
            },
            "training_history": {},
        },
    }


def create_paper_result():
    """
    Create the paper result that would normally be extracted
    by the LLM from the hypothesis evidence.
    """
    return {
        "success": True,
        "status": "completed",
        "model_name": "BiLSTM",
        "dataset": "5G-NIDD",
        "dataset_version": None,
        "task": "5G intrusion detection",
        "evaluation_protocol": "Test set evaluation",
        "split": "Test set",
        "metrics": {
            "accuracy": 0.9520,
            "precision_weighted": 0.9480,
            "recall_weighted": 0.9510,
            "f1_weighted": 0.9490,
        },
        "source_id": "paper-bitad",
        "evidence_quote": (
            "The BiLSTM model achieved an accuracy of 95.20%, "
            "weighted precision of 94.80%, weighted recall of "
            "95.10%, and weighted F1-score of 94.90%."
        ),
    }


def create_hypothesis():
    """
    Create a Rank #1 hypothesis with attached evidence.

    The actual comparator retrieves evidence from the hypothesis
    itself using _get_hypothesis_evidence().
    """

    return {
        "hypothesis_id": "H001",
        "title": "BiLSTM with temporal attention for 5G intrusion detection",
        "text": (
            "A BiLSTM model with temporal attention can improve "
            "5G network intrusion detection performance."
        ),
        "elo_score": 1245.6,
        "evidence_sources": [
            {
                "source_id": "paper-bitad",
                "title": (
                    "BiTAD: An Interpretable Temporal "
                    "Anomaly Detector for 5G Networks"
                ),
                "url": "https://example.com/paper",
                "content": (
                    "The paper evaluates BiLSTM-based models "
                    "for 5G intrusion detection."
                ),
            }
        ],
    }


# ============================================================
# Experiment Result Extraction
# ============================================================


def test_extract_experiment_results():
    """
    Test that the comparator correctly extracts the four
    comparable metrics from ExperimentRunner output.
    """

    comparator = ExperimentComparator()

    experiment_result = create_experiment_result()

    result = comparator.extract_experiment_results(
        experiment_result
    )

    assert result["success"] is True
    assert result["status"] == "completed"

    metrics = result["metrics"]

    assert metrics["accuracy"] == 0.9384
    assert metrics["precision_weighted"] == 0.9271
    assert metrics["recall_weighted"] == 0.9384
    assert metrics["f1_weighted"] == 0.9321

    # Non-comparable metrics should not appear in normalized metrics.
    assert "confusion_matrix" not in metrics
    assert "training_seconds" not in metrics

    assert (
        result["run_directory"]
        == (
            "app/experiments/results/runs/"
            "E2634_20260909_123319_881448"
        )
    )


# ============================================================
# Metric Comparison
# ============================================================


def test_compare_metrics():
    """
    Test numerical comparison between paper and automated
    experiment metrics.
    """

    comparator = ExperimentComparator()

    paper_result = create_paper_result()

    experiment_result = {
        "success": True,
        "status": "completed",
        "metrics": {
            "accuracy": 0.9384,
            "precision_weighted": 0.9271,
            "recall_weighted": 0.9384,
            "f1_weighted": 0.9321,
        },
    }

    result = comparator.compare_metrics(
        paper_result,
        experiment_result,
    )

    assert result["success"] is True
    assert result["status"] == "compared"

    metrics = result["metrics"]

    assert "accuracy" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "f1_weighted" in metrics

    # Difference = experiment - paper

    assert round(
        metrics["accuracy"]["difference"], 4
    ) == -0.0136

    assert round(
        metrics["precision_weighted"]["difference"], 4
    ) == -0.0209

    assert round(
        metrics["recall_weighted"]["difference"], 4
    ) == -0.0126

    assert round(
        metrics["f1_weighted"]["difference"], 4
    ) == -0.0169

    # Percentage-point differences

    assert round(
        metrics["accuracy"]["difference_percentage_points"], 2
    ) == -1.36

    assert round(
        metrics["precision_weighted"]["difference_percentage_points"],
        2,
    ) == -2.09

    assert round(
        metrics["recall_weighted"]["difference_percentage_points"],
        2,
    ) == -1.26

    assert round(
        metrics["f1_weighted"]["difference_percentage_points"],
        2,
    ) == -1.69

    # All automated metrics are lower than the paper.
    assert set(result["worse_metrics"]) == {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    }

    assert result["improved_metrics"] == []
    assert result["unchanged_metrics"] == []

    assert round(
        result["average_difference"], 4
    ) == -0.016


# ============================================================
# Comparability
# ============================================================


def test_check_comparability():
    """
    Test that paper and experiment results are considered
    comparable when they share the required metrics.
    """

    comparator = ExperimentComparator()

    paper_result = create_paper_result()

    experiment_result = {
        "success": True,
        "status": "completed",
        "metrics": {
            "accuracy": 0.9384,
            "precision_weighted": 0.9271,
            "recall_weighted": 0.9384,
            "f1_weighted": 0.9321,
        },
        "experiment_summary": {
            "model": "BiLSTM",
            "dataset": "5G-NIDD",
        },
    }

    hypothesis = create_hypothesis()

    result = comparator.check_comparability(
        paper_result,
        experiment_result,
        hypothesis,
    )

    assert result["comparable"] is True

    assert set(result["common_metrics"]) == {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    }

    assert result["paper_model"] == "BiLSTM"
    assert result["experiment_model"] == "BiLSTM"
    assert result["paper_dataset"] == "5G-NIDD"
    assert result["comparison_level"] == "direct"
    assert result["dataset_warning"] is None


def test_check_comparability_with_different_dataset():
    """
    Test that a paper using a different dataset is still allowed
    to produce a comparison, but receives a dataset warning and
    is classified as a partial comparison.
    """
    comparator = ExperimentComparator()

    paper_result = create_paper_result()
    paper_result["dataset"] = "UNSW-NB15"

    experiment_result = {
        "success": True,
        "status": "completed",
        "metrics": {
            "accuracy": 0.9384,
            "precision_weighted": 0.9271,
            "recall_weighted": 0.9384,
            "f1_weighted": 0.9321,
        },
        "experiment_summary": {
            "model": "BiLSTM",
            "dataset": "5G-NIDD",
        },
    }

    hypothesis = create_hypothesis()

    result = comparator.check_comparability(
        paper_result,
        experiment_result,
        hypothesis,
    )

    assert result["comparable"] is True
    assert result["comparison_level"] == "partial"
    assert result["dataset_warning"] is not None
    assert "offline 5G-NIDD" in result["dataset_warning"]


def test_check_comparability_without_common_metrics():
    """
    Test that comparability is rejected when the paper and
    experiment have no common numerical evaluation metrics.
    """
    comparator = ExperimentComparator()

    paper_result = {
        "success": True,
        "metrics": {},
        "model_name": "BiLSTM",
        "dataset": "5G-NIDD",
    }

    experiment_result = {
        "success": True,
        "metrics": {
            "some_other_metric": 0.90,
        },
    }

    hypothesis = create_hypothesis()

    result = comparator.check_comparability(
        paper_result,
        experiment_result,
        hypothesis,
    )

    assert result["comparable"] is False
    assert result["comparison_level"] == "none"
    assert result["common_metrics"] == []


def test_clamp_metric():
    comparator = ExperimentComparator()

    assert comparator._clamp_metric(0.952) == pytest.approx(0.952)
    assert comparator._clamp_metric(95.2) == pytest.approx(0.952)
    assert comparator._clamp_metric(100.0) == pytest.approx(1.0)


# ============================================================
# Paper Evidence Retrieval
# ============================================================


def test_load_paper_evidence_with_mocked_reader():
    """
    Test that paper evidence can be loaded from an evidence URL
    without making a real HTTP request.

    PaperReader.read_paper() is mocked so the test remains fast
    and deterministic.
    """

    comparator = ExperimentComparator()

    hypothesis = create_hypothesis()

    mocked_paper_text = """
    4. Results

    The BiLSTM model achieved an accuracy of 95.20%,
    weighted precision of 94.80%, weighted recall of 95.10%,
    and weighted F1-score of 94.90% on the 5G-NIDD dataset.
    """

    # Mock PaperReader.read_paper().
    comparator.paper_reader.read_paper = MagicMock(
        return_value=mocked_paper_text
    )

    evidence_sources = comparator._get_hypothesis_evidence(
        hypothesis
    )

    loaded_sources = comparator._load_paper_evidence(
        evidence_sources
    )

    assert len(loaded_sources) == 1

    source = loaded_sources[0]

    assert source["paper_retrieved"] is True

    assert source["content"] == mocked_paper_text

    assert source["source_id"] == "paper-bitad"

    # Verify that the reader was called using the evidence URL.
    comparator.paper_reader.read_paper.assert_called_once_with(
        "https://example.com/paper"
    )


def test_get_paper_url_converts_arxiv_html_to_pdf():
    comparator = ExperimentComparator()

    source = {
        "url": "https://arxiv.org/html/2603.11006v1"
    }

    url = comparator._get_paper_url(source)

    assert url == "https://arxiv.org/pdf/2603.11006v1"


def test_get_paper_url_preserves_pdf_url():
    comparator = ExperimentComparator()

    source = {
        "pdf_url": "https://arxiv.org/pdf/2603.11006v1"
    }

    url = comparator._get_paper_url(source)

    assert url == "https://arxiv.org/pdf/2603.11006v1"


def test_load_paper_evidence_handles_reader_failure():
    """
    Test that PaperReader failures do not crash the comparator.

    The original evidence source is preserved and the retrieval
    error is recorded.
    """

    comparator = ExperimentComparator()

    hypothesis = create_hypothesis()

    comparator.paper_reader.read_paper = MagicMock(
        side_effect=Exception("Unable to download paper")
    )

    evidence_sources = comparator._get_hypothesis_evidence(
        hypothesis
    )

    loaded_sources = comparator._load_paper_evidence(
        evidence_sources
    )

    assert len(loaded_sources) == 1

    source = loaded_sources[0]

    assert source["paper_retrieved"] is False

    assert (
        source["paper_retrieval_error"]
        == "Unable to download paper"
    )

    # Existing evidence should remain available.
    assert (
        source["content"]
        == "The paper evaluates BiLSTM-based models "
        "for 5G intrusion detection."
    )

    comparator.paper_reader.read_paper.assert_called_once_with(
        "https://example.com/paper"
    )


def test_debug_hypothesis_evidence_sources():
    comparator = ExperimentComparator()
    hypothesis = create_hypothesis()

    result = comparator.extract_paper_results(hypothesis)

    print("\n===== PAPER EXTRACTION RESULT =====")
    print(result)
    print("===================================\n")


# ============================================================
# Complete Comparison
# ============================================================


def test_compare_success(monkeypatch):
    """
    Test the complete comparison workflow.

    The paper extraction and LLM explanation are mocked so that
    this test does not make a real LLM/API call.
    """

    comparator = ExperimentComparator()

    hypothesis = create_hypothesis()

    experiment_result = create_experiment_result()

    # Mock paper extraction.
    monkeypatch.setattr(
        comparator,
        "extract_paper_results",
        lambda hypothesis: create_paper_result(),
    )

    # Mock scientific explanation.
    monkeypatch.setattr(
        comparator,
        "explain_difference",
        lambda *args, **kwargs: {
            "success": True,
            "status": "completed",
            "overall_assessment": (
                "The automated experiment achieved lower "
                "performance than the published result."
            ),
            "reproduction_level": "lower",
            "confirmed_observations": [
                "All four comparable metrics were lower."
            ],
            "possible_explanations": [
                "Differences in preprocessing or hyperparameters."
            ],
            "limitations": [
                "The exact paper training configuration is unavailable."
            ],
            "recommendation": (
                "Review preprocessing, split configuration, "
                "and hyperparameters."
            ),
        },
    )

    result = comparator.compare(
        hypothesis,
        experiment_result,
    )

    assert result["success"] is True
    assert result["status"] == "completed"

    assert result["hypothesis_id"] == "H001"

    assert (
        result["hypothesis_title"]
        == "BiLSTM with temporal attention for 5G intrusion detection"
    )

    assert result["paper_result"] is not None
    assert result["experiment_result"] is not None
    assert result["comparability"] is not None
    assert result["metric_comparison"] is not None
    assert result["explanation"] is not None

    assert (
        result["paper_result"]["metrics"]["accuracy"]
        == 0.9520
    )

    assert (
        result["experiment_result"]["metrics"]["accuracy"]
        == 0.9384
    )

    assert round(
        result["metric_comparison"]["metrics"]["accuracy"]["difference"],
        4,
    ) == -0.0136

    assert (
        result["explanation"]["reproduction_level"]
        == "lower"
    )


# ============================================================
# Missing Metrics
# ============================================================


def test_compare_without_experiment_metrics(monkeypatch):
    """
    Test that comparison fails gracefully when the automated
    experiment contains no comparable metrics.
    """

    comparator = ExperimentComparator()

    hypothesis = create_hypothesis()

    experiment_result = {
        "success": True,
        "status": "completed",
        "outputs": {
            "metrics": {}
        },
    }

    monkeypatch.setattr(
        comparator,
        "extract_paper_results",
        lambda hypothesis: create_paper_result(),
    )

    result = comparator.compare(
        hypothesis,
        experiment_result,
    )

    assert result["success"] is False
    assert result["status"] == "experiment_result_unavailable"

    assert result["experiment_result"]["success"] is False
    assert result["experiment_result"]["status"] == "no_valid_metrics"
    assert result["experiment_result"]["metrics"] == {}

    assert result["metric_comparison"] is None


# ============================================================
# Failed Experiment
# ============================================================


def test_extract_experiment_results_with_failed_experiment():
    """
    Test that a failed ExperimentRunner result is rejected.
    """

    comparator = ExperimentComparator()

    result = comparator.extract_experiment_results(
        {
            "success": False,
            "status": "experiment_failed",
            "errors": [
                "Training process timed out."
            ],
        }
    )

    assert result["success"] is False
    assert result["status"] == "experiment_failed"
    assert result["metrics"] == {}


# ============================================================
# Missing Outputs
# ============================================================


def test_extract_experiment_results_with_missing_outputs():
    """
    Test that the comparator detects missing experiment outputs.
    """

    comparator = ExperimentComparator()

    result = comparator.extract_experiment_results(
        {
            "success": True,
            "status": "completed",
        }
    )

    assert result["success"] is False
    assert result["status"] == "missing_outputs"
    assert result["metrics"] == {}


# ============================================================
# Missing Metrics
# ============================================================


def test_extract_experiment_results_with_missing_metrics():
    """
    Test that the comparator detects missing metrics.json output.
    """

    comparator = ExperimentComparator()

    result = comparator.extract_experiment_results(
        {
            "success": True,
            "status": "completed",
            "outputs": {
                "metrics": None,
            },
        }
    )

    assert result["success"] is False
    assert result["status"] == "missing_metrics"
    assert result["metrics"] == {}


# ============================================================
# Metric Alias / Percentage Handling
# ============================================================


def test_metric_alias_and_percentage_normalization():
    """
    Test that percentage values and alternative metric names
    are normalized correctly.
    """

    comparator = ExperimentComparator()

    experiment_result = {
        "success": True,
        "status": "completed",
        "outputs": {
            "metrics": {
                "acc": 95.2,
                "weighted_precision": 94.8,
                "weighted recall": 95.1,
                "f1-score": 94.9,
            }
        },
    }

    result = comparator.extract_experiment_results(
        experiment_result
    )

    assert result["success"] is True

    assert round(
        result["metrics"]["accuracy"],
        3,
    ) == 0.952
    assert round(
        result["metrics"]["precision_weighted"],
        3,
    ) == 0.948

    assert round(
        result["metrics"]["recall_weighted"],
        3,
    ) == 0.951

    assert round(
        result["metrics"]["f1_weighted"],
        3,
    ) == 0.949


# ============================================================
# Human-Readable Formatting
# ============================================================


def test_format_comparison():
    """
    Test the human-readable comparison formatter.
    """

    comparator = ExperimentComparator()

    comparison_result = {
        "success": True,
        "status": "completed",
        "hypothesis_title": (
            "BiLSTM with temporal attention for 5G intrusion detection"
        ),
        "metric_comparison": {
            "metrics": {
                "accuracy": {
                    "paper": 0.9520,
                    "experiment": 0.9384,
                    "difference_percentage_points": -1.36,
                },
                "f1_weighted": {
                    "paper": 0.9490,
                    "experiment": 0.9321,
                    "difference_percentage_points": -1.69,
                },
            }
        },
        "explanation": {
            "overall_assessment": (
                "The automated experiment performed below "
                "the published result."
            ),
            "recommendation": (
                "Review preprocessing and hyperparameters."
            ),
        },
    }

    formatted = comparator.format_comparison(
        comparison_result
    )

    assert "Experiment Comparison" in formatted

    assert (
        "Rank #1 Hypothesis: "
        "BiLSTM with temporal attention for 5G intrusion detection"
        in formatted
    )

    assert "accuracy" in formatted
    assert "Paper=95.20%" in formatted
    assert "Experiment=93.84%" in formatted
    assert "Difference=-1.36 pp" in formatted

    assert "f1_weighted" in formatted
    assert "Paper=94.90%" in formatted
    assert "Experiment=93.21%" in formatted
    assert "Difference=-1.69 pp" in formatted

    assert "Assessment:" in formatted
    assert "Recommendation:" in formatted
