"""
Real Integration Test for the Automated Experiment Pipeline.

This test is intended to be run manually because it executes a real
experiment and may download/read research papers.

It is NOT intended to be part of the automated pytest test suite.

Run with:

    python test_real_comparison.py

Pipeline:

    Rank #1 Hypothesis
            |
            v
    Evidence Sources
            |
            v
    PaperReader / PaperLibrary
            |
            v
    Evidence-Derived Evaluation Metrics
            |
            v
    CodeGenerationAgent
            |
            v
    ExperimentRunner
            |
            v
    ExperimentComparator
            |
            v
    Paper vs Experiment
"""

import json
from pathlib import Path
from typing import Any, Dict


from app.experiments.experiment_orchestrator import ExperimentOrchestrator


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = Path(
    "app/experiments/results/runs/"
    "E7546_20260929_105937_551410/"
    "experiment_config.json"
)

DATASET_NAME = "5G-NIDD"
DATASET_PATH = "data/5g_nidd/5g_nidd.csv"
DEVICE = "cpu"


# ============================================================
# Utility Functions
# ============================================================

def print_json(
    value: Any,
    title: str = "",
) -> None:
    """Pretty-print a JSON-compatible value."""

    if title:
        print(title)

    print(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def print_list(
    values: Any,
    empty_message: str = "None",
) -> None:
    """Print a list in a readable format."""

    if not isinstance(values, list) or not values:
        print(empty_message)
        return

    for value in values:
        if isinstance(value, dict):
            print(
                "-",
                value.get("name")
                or value.get("metric")
                or value.get("title")
                or str(value),
            )
        else:
            print("-", value)


def get_nested_dict(
    value: Any,
    key: str,
) -> Dict[str, Any]:
    """Safely retrieve a nested dictionary."""

    if not isinstance(value, dict):
        return {}

    nested = value.get(key)

    if isinstance(nested, dict):
        return nested

    return {}


def extract_reference_metric_guidance(
    reference_experiment: Any,
) -> Dict[str, Any]:
    """
    Extract evidence-derived evaluation metrics from the reference
    experiment.

    This mirrors the revised evidence-driven architecture and is
    intentionally generic. It does not assume classification metrics.
    """

    guidance = {
        "metrics": [],
        "metric_definitions": {},
        "reference_metrics": {},
    }

    if not isinstance(reference_experiment, dict):
        return guidance

    sources = reference_experiment.get(
        "sources",
        [],
    )

    if not isinstance(sources, list):
        return guidance

    seen_metrics = set()

    for source in sources:
        if not isinstance(source, dict):
            continue

        details = source.get(
            "experiment_details",
            {},
        )

        if not isinstance(details, dict):
            continue

        metrics = details.get(
            "metrics",
            [],
        )

        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, str):
                    continue

                metric_name = metric.strip()

                if not metric_name:
                    continue

                metric_key = metric_name.lower()

                if metric_key not in seen_metrics:
                    guidance["metrics"].append(
                        metric_name
                    )
                    seen_metrics.add(metric_key)

        definitions = details.get(
            "metric_definitions",
            {},
        )

        if isinstance(definitions, dict):
            for name, definition in definitions.items():
                metric_name = str(name)

                if metric_name not in guidance[
                    "metric_definitions"
                ]:
                    guidance[
                        "metric_definitions"
                    ][metric_name] = definition

        reference_metrics = details.get(
            "reference_metrics",
            {},
        )

        if isinstance(reference_metrics, dict):
            for name, value in reference_metrics.items():
                metric_name = str(name)

                if metric_name not in guidance[
                    "reference_metrics"
                ]:
                    guidance[
                        "reference_metrics"
                    ][metric_name] = value

    return guidance


def extract_experiment_metrics(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract the metrics produced by the automated experiment.

    This function intentionally does not assume accuracy, precision,
    recall, F1, confusion matrix, latency, throughput, or any other
    specific metric.
    """

    execution = result.get(
        "execution",
        {}
    )

    if not isinstance(execution, dict):
        return {}

    metrics = execution.get(
        "metrics"
    )

    if isinstance(metrics, dict):
        return metrics

    # Some runner implementations may place metrics inside an
    # experiment_result field.
    experiment_result = execution.get(
        "experiment_result",
        {}
    )

    if isinstance(experiment_result, dict):
        metrics = experiment_result.get(
            "metrics"
        )

        if isinstance(metrics, dict):
            return metrics

    return {}


def extract_comparison_section(
    comparison: Any,
    key: str,
) -> Any:
    """Safely retrieve a field from the comparison result."""

    if not isinstance(comparison, dict):
        return None

    value = comparison.get(key)

    if value is not None:
        return value

    metric_comparison = comparison.get(
        "metric_comparison"
    )

    if isinstance(metric_comparison, dict):
        return metric_comparison.get(key)

    return None


# ============================================================
# Load Existing Rank #1 Hypothesis
# ============================================================

print("=" * 70)
print("REAL RANK #1 EXPERIMENT TEST")
print("=" * 70)

print("\nConfiguration path:")
print(CONFIG_PATH)

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Experiment configuration not found: {CONFIG_PATH}"
    )

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    config = json.load(f)

if not isinstance(config, dict):
    raise ValueError(
        "experiment_config.json must contain a JSON object."
    )

specification = config.get(
    "specification",
    {}
)

if not isinstance(specification, dict):
    raise ValueError(
        "The configuration does not contain a valid "
        "'specification' object."
    )

hypothesis = specification.get(
    "selected_hypothesis",
    {}
)

if not isinstance(hypothesis, dict):
    raise ValueError(
        "The specification does not contain a valid "
        "'selected_hypothesis'."
    )

research_goal = specification.get(
    "research_goal"
)

print("\nRank #1 Hypothesis:")
print(
    hypothesis.get("title")
    or hypothesis.get("text")
    or hypothesis.get("hypothesis")
    or "Unknown"
)

print("\nResearch Goal:")
print(
    research_goal
    if research_goal
    else "Not provided"
)

print("\nEvidence Sources:")

evidence_sources = hypothesis.get(
    "evidence_sources",
    []
)

if not isinstance(evidence_sources, list):
    evidence_sources = []

if evidence_sources:
    for index, source in enumerate(
        evidence_sources,
        start=1,
    ):
        if not isinstance(source, dict):
            print(f"- Source #{index}: {source}")
            continue

        title = (
            source.get("title")
            or source.get("name")
            or "Untitled source"
        )

        url = (
            source.get("canonical_url")
            or source.get("url")
            or source.get("arxiv_url")
            or source.get("source_url")
            or "No URL"
        )

        print(
            f"- Source #{index}: {title}"
        )
        print(
            f"  URL: {url}"
        )
else:
    print("None")


# ============================================================
# Display Original Specification
# ============================================================

print("\n" + "=" * 70)
print("LOADED EXPERIMENT SPECIFICATION")
print("=" * 70)

print_json(
    specification
)


# ============================================================
# Create Orchestrator
# ============================================================

print("\n" + "=" * 70)
print("CREATING EXPERIMENT ORCHESTRATOR")
print("=" * 70)

orchestrator = ExperimentOrchestrator(
    dataset_name=DATASET_NAME,
    dataset_path=DATASET_PATH,
    device=DEVICE,
)

print(
    "ExperimentOrchestrator created successfully."
)

print(
    "Shared PaperLibrary:",
    type(orchestrator.paper_library).__name__,
)

print(
    "PaperReader:",
    type(orchestrator.paper_reader).__name__,
)

print(
    "ExperimentComparator:",
    type(orchestrator.experiment_comparator).__name__,
)


# ============================================================
# Run Complete Experiment Pipeline
# ============================================================

print("\n" + "=" * 70)
print("RUNNING COMPLETE EXPERIMENT PIPELINE")
print("=" * 70)

print(
    "\nFlow:"
    "\nRank #1 Hypothesis"
    "\n    -> Evidence Sources"
    "\n    -> PaperReader / PaperLibrary"
    "\n    -> Evidence-Derived Evaluation Metrics"
    "\n    -> CodeGenerationAgent"
    "\n    -> ExperimentRunner"
    "\n    -> ExperimentComparator"
    "\n    -> Paper vs Experiment"
)

result = orchestrator.run_experiment(
    context=specification,
    research_goal=research_goal,
    hypothesis=hypothesis,
    execute_generated_code=True,
)

if not isinstance(result, dict):
    raise ValueError(
        "ExperimentOrchestrator.run_experiment() "
        "did not return a dictionary."
    )


# ============================================================
# Display Pipeline Result
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE RESULT")
print("=" * 70)

print(
    "Success:",
    result.get("success"),
)

print(
    "Status:",
    result.get("status"),
)

print("\nErrors:")

errors = result.get(
    "errors",
    []
)

if isinstance(errors, list) and errors:
    for error in errors:
        print("-", error)
else:
    print("None")


# ============================================================
# Experiment Preparation
# ============================================================

preparation = result.get(
    "experiment_preparation",
    {}
)

if not isinstance(preparation, dict):
    preparation = {}


# ============================================================
# Reference Experiment Extracted from Evidence
# ============================================================

reference_experiment = preparation.get(
    "reference_experiment"
)

print("\n" + "=" * 70)
print("REFERENCE EXPERIMENT FROM EVIDENCE")
print("=" * 70)

if reference_experiment:
    print(
        "Reference experiment available: YES"
    )

    print(
        "Reference extraction status:",
        reference_experiment.get(
            "status",
            "unknown",
        ),
    )

    print(
        "Reference source count:",
        reference_experiment.get(
            "source_count",
            len(
                reference_experiment.get(
                    "sources",
                    [],
                )
            ),
        ),
    )

    print("\nReference experiment data:")

    print_json(
        reference_experiment
    )

    sources = reference_experiment.get(
        "sources",
        [],
    )

    if not isinstance(sources, list):
        sources = []

    if sources:
        print("\nReference sources:")

        for index, source in enumerate(
            sources,
            start=1,
        ):
            print(
                f"\nReference Source #{index}:"
            )

            if not isinstance(source, dict):
                print(
                    "Source data:",
                    source,
                )
                continue

            print(
                "Title:",
                source.get(
                    "title",
                    "Unknown",
                ),
            )

            print(
                "URL:",
                source.get(
                    "source_url"
                )
                or source.get(
                    "url"
                )
                or source.get(
                    "canonical_url"
                ),
            )

            print(
                "Source type:",
                source.get(
                    "source_type"
                ),
            )

            print(
                "Source ID:",
                source.get(
                    "source_id"
                ),
            )

            print(
                "Indexed:",
                source.get(
                    "indexed"
                ),
            )

            experiment_details = source.get(
                "experiment_details"
            )

            if isinstance(
                experiment_details,
                dict,
            ):
                print(
                    "Experiment details extracted: YES"
                )

                print(
                    "Metrics:",
                )

                print_list(
                    experiment_details.get(
                        "metrics",
                        [],
                    )
                )

                print(
                    "Metric definitions:"
                )

                metric_definitions = (
                    experiment_details.get(
                        "metric_definitions",
                        {},
                    )
                )

                if isinstance(
                    metric_definitions,
                    dict,
                ):
                    if metric_definitions:
                        print_json(
                            metric_definitions
                        )
                    else:
                        print("None")
                else:
                    print("None")

                print(
                    "Reference metrics:"
                )

                reference_metrics = (
                    experiment_details.get(
                        "reference_metrics",
                        {},
                    )
                )

                if isinstance(
                    reference_metrics,
                    dict,
                ):
                    if reference_metrics:
                        print_json(
                            reference_metrics
                        )
                    else:
                        print("None")
                else:
                    print("None")

                print(
                    "Results text available:",
                    bool(
                        source.get(
                            "results_text"
                        )
                    ),
                )

            else:
                print(
                    "Experiment details extracted: NO"
                )

else:
    print(
        "Reference experiment available: NO"
    )


# ============================================================
# Evidence-Derived Evaluation Guidance
# ============================================================

print("\n" + "=" * 70)
print("EVIDENCE-DERIVED EVALUATION GUIDANCE")
print("=" * 70)

evaluation_guidance = preparation.get(
    "evaluation_guidance"
)

if not isinstance(
    evaluation_guidance,
    dict,
):
    evaluation_guidance = specification.get(
        "evaluation_guidance",
        {}
    )

if not isinstance(
    evaluation_guidance,
    dict,
):
    evaluation_guidance = {}

print("\nHypothesis metrics:")

hypothesis_metrics = (
    evaluation_guidance.get(
        "hypothesis_metrics",
        []
    )
)

print_list(
    hypothesis_metrics
)

print("\nEvidence metrics:")

evidence_metrics = (
    evaluation_guidance.get(
        "evidence_metrics",
        evaluation_guidance.get(
            "metrics",
            []
        ),
    )
)

print_list(
    evidence_metrics
)

print("\nMetric definitions:")

metric_definitions = (
    evaluation_guidance.get(
        "metric_definitions",
        {}
    )
)

if isinstance(
    metric_definitions,
    dict
) and metric_definitions:
    print_json(
        metric_definitions
    )
else:
    print("None")

print("\nPaper/reference metric values:")

reference_metrics = (
    evaluation_guidance.get(
        "reference_metrics",
        {}
    )
)

if isinstance(
    reference_metrics,
    dict
) and reference_metrics:
    print_json(
        reference_metrics
    )
else:
    print("None")

print(
    "\nImportant:"
    "\nReference metric values are paper results only."
    "\nThey must NOT be copied into the generated experiment."
)


# ============================================================
# Evaluation Metrics in Final Specification
# ============================================================

print("\n" + "=" * 70)
print("FINAL EXPERIMENT EVALUATION METRICS")
print("=" * 70)

prepared_specification = preparation.get(
    "experiment_specification",
    {}
)

if not isinstance(prepared_specification, dict):
    prepared_specification = {}

final_evaluation_metrics = prepared_specification.get(
    "evaluation_metrics",
    []
)

if not final_evaluation_metrics:
    final_evaluation_metrics = specification.get(
        "evaluation_metrics",
        []
    )

evaluation_metric_definitions = prepared_specification.get(
    "evaluation_metric_definitions",
    prepared_specification.get(
        "metric_definitions",
        {}
    ),
)

if not evaluation_metric_definitions:
    evaluation_metric_definitions = specification.get(
        "evaluation_metric_definitions",
        specification.get(
            "metric_definitions",
            {}
        ),
    )

print(
    "Final metrics selected/guided for the automated experiment:"
)

print_list(
    final_evaluation_metrics
)

print(
    "\nEvaluation metric definitions:"
)

if (
    isinstance(
        evaluation_metric_definitions,
        dict,
    )
    and evaluation_metric_definitions
):
    print_json(
        evaluation_metric_definitions
    )
else:
    print("None")


# ============================================================
# Final Experiment Design
# ============================================================

print("\n" + "=" * 70)
print("FINAL EXPERIMENT DESIGN")
print("=" * 70)

experiment_design = prepared_specification.get(
    "experiment_design",
    {}
)

if isinstance(experiment_design, dict):
    print_json(experiment_design)
else:
    print("No experiment design available.")

print("\nExperiment type:")
print(
    prepared_specification.get(
        "experiment_type",
        "Unknown",
    )
)


# ============================================================
# Code Generation Requirements
# ============================================================

print("\n" + "=" * 70)
print("CODE GENERATION REQUIREMENTS")
print("=" * 70)

code_generation_requirements = prepared_specification.get(
    "code_generation_requirements",
    {}
)

if isinstance(code_generation_requirements, dict):
    print_json(code_generation_requirements)
else:
    print("No code generation requirements available.")


# ============================================================
# Automated Experiment
# ============================================================

execution = result.get(
    "execution"
)

print("\n" + "=" * 70)
print("AUTOMATED EXPERIMENT")
print("=" * 70)

if isinstance(
    execution,
    dict,
):
    print(
        "Automated experiment result available: YES"
    )

    print_json(
        execution
    )
else:
    print(
        "Automated experiment result available: NO"
    )


# ============================================================
# Automated Experiment Metrics
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATED EXPERIMENT METRICS")
print("=" * 70)

experiment_metrics = extract_experiment_metrics(
    result
)

if experiment_metrics:
    print_json(
        experiment_metrics
    )
else:
    print(
        "No experiment metrics were found."
    )

print(
    "\nNote:"
    "\nThese values must be independently produced by the"
    "\nautomated experiment and must not come from paper"
    "\nreference values."
)


# ============================================================
# Paper vs Experiment Comparison
# ============================================================

comparison = result.get(
    "comparison"
)

print("\n" + "=" * 70)
print("PAPER VS AUTOMATED EXPERIMENT")
print("=" * 70)

if isinstance(
    comparison,
    dict,
):
    print_json(
        comparison
    )

    print("\nComparison status:")

    print(
        comparison.get(
            "status",
            "unknown",
        )
    )

    comparability_result = comparison.get(
        "comparability"
    )

    if isinstance(
        comparability_result,
        dict,
    ):
        comparable = comparability_result.get(
            "comparable"
        )
    else:
        comparable = comparison.get(
            "comparable"
        )

    print(
        "Comparable:",
        comparable,
    )

    print(
        "Comparison success:",
        comparison.get(
            "success"
        ),
    )

    # --------------------------------------------------------
    # Common Metrics
    # --------------------------------------------------------

    common_metrics = extract_comparison_section(
        comparison,
        "common_metrics",
    )

    if common_metrics is None:
        common_metrics = (
            extract_comparison_section(
                comparison,
                "comparable_metrics",
            )
        )

    print(
        "\nCommon metrics:"
    )

    print_list(
        common_metrics
    )

    # --------------------------------------------------------
    # Paper-only Metrics
    # --------------------------------------------------------

    paper_only_metrics = extract_comparison_section(
        comparison,
        "paper_only_metrics",
    )

    print(
        "\nPaper-only metrics:"
    )

    print_list(
        paper_only_metrics
    )

    # --------------------------------------------------------
    # Experiment-only Metrics
    # --------------------------------------------------------

    experiment_only_metrics = extract_comparison_section(
        comparison,
        "experiment_only_metrics",
    )

    print(
        "\nExperiment-only metrics:"
    )

    print_list(
        experiment_only_metrics
    )

    # --------------------------------------------------------
    # Unit Mismatches
    # --------------------------------------------------------

    unit_mismatches = extract_comparison_section(
        comparison,
        "unit_mismatches",
    )

    print(
        "\nUnit mismatches:"
    )

    print_list(
        unit_mismatches
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = comparison.get(
        "warnings",
        []
    )

    print(
        "\nComparison warnings:"
    )

    print_list(
        warnings
    )

    # --------------------------------------------------------
    # Metric Comparison
    # --------------------------------------------------------

    metric_comparison = comparison.get(
        "metric_comparison",
        {},
    )

    print(
        "\nMetric comparison:"
    )

    if isinstance(
        metric_comparison,
        dict,
    ):
        metrics = metric_comparison.get(
            "metrics",
            {}
        )

        if isinstance(
            metrics,
            dict
        ) and metrics:
            print_json(
                metrics
            )
        else:
            print(
                "No directly comparable metric values."
            )
    else:
        print(
            "No metric comparison data."
        )

    # --------------------------------------------------------
    # Dataset Compatibility
    # --------------------------------------------------------

    dataset_compatibility = extract_comparison_section(
        comparison,
        "dataset_compatibility",
    )

    print(
        "\nDataset compatibility:"
    )

    if isinstance(
        dataset_compatibility,
        dict,
    ):
        print_json(
            dataset_compatibility
        )
    elif dataset_compatibility is not None:
        print(
            dataset_compatibility
        )
    else:
        print(
            "Not reported."
        )

    # --------------------------------------------------------
    # Comparison Errors
    # --------------------------------------------------------

    comparison_errors = comparison.get(
        "errors",
        []
    )

    if comparison_errors:
        print(
            "\nComparison errors:"
        )

        if isinstance(
            comparison_errors,
            list,
        ):
            for error in comparison_errors:
                print(
                    "-",
                    error,
                )
        else:
            print(
                comparison_errors
            )

else:
    print(
        "No comparison result."
    )


# ============================================================
# Scientific Fidelity / Comparison Interpretation
# ============================================================

print("\n" + "=" * 70)
print("SCIENTIFIC COMPARISON INTERPRETATION")
print("=" * 70)

if isinstance(
    comparison,
    dict,
):
    comparability_result = comparison.get(
        "comparability",
        {}
    )

    if isinstance(
        comparability_result,
        dict,
    ):
        comparable = comparability_result.get(
            "comparable"
        )

        if comparable is True:
            print(
                "The comparator found compatible metrics/data"
                "\nthat can be compared between the paper and"
                "\nthe automated experiment."
            )

        elif comparable is False:
            print(
                "The experiment and paper are not directly"
                "\ncomparable based on the available evidence."
            )

            print(
                "\nThis does NOT necessarily mean the"
                "\nautomated experiment failed."
            )

            print(
                "It means the measured quantities, units,"
                "\ndataset/methodology, or available evidence"
                "\ndo not support a direct comparison."
            )

        else:
            print(
                "Comparability could not be determined."
            )
    else:
        print(
            "No structured comparability result available."
        )

else:
    print(
        "No comparison was available for interpretation."
    )


# ============================================================
# Final Test Summary
# ============================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

reference_available = bool(
    reference_experiment
)

execution_available = isinstance(
    execution,
    dict,
)

execution_success = bool(
    execution_available
    and execution.get(
        "success"
    )
)

comparison_available = isinstance(
    comparison,
    dict,
)

comparison_success = bool(
    comparison_available
    and comparison.get(
        "success"
    )
)

comparison_status = (
    comparison.get(
        "status"
    )
    if comparison_available
    else None
)

comparability = None

if comparison_available:
    comparability_result = comparison.get(
        "comparability"
    )

    if isinstance(
        comparability_result,
        dict,
    ):
        comparability = (
            comparability_result.get(
                "comparable"
            )
        )
    else:
        comparability = comparison.get(
            "comparable"
        )

print(
    "Pipeline success:",
    result.get(
        "success"
    ),
)

print(
    "Pipeline status:",
    result.get(
        "status"
    ),
)

print(
    "Reference experiment extracted:",
    reference_available,
)

print(
    "Evidence metrics extracted:",
    bool(evidence_metrics),
)

print(
    "Experiment evaluation metrics defined:",
    bool(final_evaluation_metrics),
)

print(
    "Automated execution available:",
    execution_available,
)

print(
    "Automated execution successful:",
    execution_success,
)

print(
    "Automated experiment metrics available:",
    bool(experiment_metrics),
)

print(
    "Comparison result available:",
    comparison_available,
)

print(
    "Comparison successful:",
    comparison_success,
)

print(
    "Comparison status:",
    comparison_status,
)

print(
    "Results comparable:",
    comparability,
)


# ============================================================
# Indexed Paper Sources
# ============================================================

if reference_experiment:
    sources = reference_experiment.get(
        "sources",
        [],
    )

    if isinstance(
        sources,
        list,
    ):
        indexed_sources = [
            source
            for source in sources
            if isinstance(source, dict)
            and source.get("indexed") is True
        ]

        print(
            "Indexed paper sources:",
            len(indexed_sources),
        )


# ============================================================
# Final Diagnostic
# ============================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSTIC")
print("=" * 70)

if not reference_available:
    print(
        "WARNING: No reference experiment was extracted "
        "from the evidence sources."
    )

if not evidence_metrics:
    print(
        "WARNING: No evidence-derived evaluation metrics "
        "were extracted."
    )

if not final_evaluation_metrics:
    print(
        "WARNING: No explicit evaluation metrics were "
        "present in the final experiment specification."
    )

if not execution_available:
    print(
        "WARNING: No automated experiment execution result "
        "was returned."
    )
elif not execution_success:
    print(
        "WARNING: Automated experiment execution failed."
    )
else:
    print(
        "OK: Automated experiment executed successfully."
    )

if not experiment_metrics:
    print(
        "WARNING: No independently generated experiment "
        "metrics were found."
    )
else:
    print(
        "OK: Automated experiment produced metrics."
    )

if not comparison_available:
    print(
        "WARNING: No paper-vs-experiment comparison result "
        "was returned."
    )
elif not comparison_success:
    print(
        "WARNING: Comparator did not complete successfully."
    )
elif comparability is False:
    print(
        "INFO: Comparison completed, but the results are "
        "not directly comparable."
    )
elif comparability is True:
    print(
        "OK: Comparison completed with comparable results."
    )
else:
    print(
        "INFO: Comparison completed, but comparability "
        "could not be determined."
    )


# ============================================================
# End
# ============================================================

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)