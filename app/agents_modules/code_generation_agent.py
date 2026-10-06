"""
Code Generation Agent.

This module connects the AI Co-Scientist hypothesis workflow with the
automated experiment pipeline.

Workflow:

    Selected Rank #1 Hypothesis
            +
    Research Goal
            +
    Reflection Report
            +
    Evidence Sources
            +
    PaperReader Experimental Details
            |
            v
    CodeGenerationAgent
            |
            +--> Model / Approach Recommendation
            +--> Experiment Plan
            +--> Evidence-derived Evaluation Metrics
            +--> Executable Python / PyTorch Code
            |
            v
    ExperimentRunner

The agent is responsible for CODE GENERATION only.
It does not execute the generated experiment.

Scientific priority:

    Rank #1 Hypothesis
        >
    Research Goal
        >
    Experiment Specification
        >
    Dataset / Environment
        >
    Evidence-derived Reference Experiment

Evidence sources guide methodology and evaluation metrics, but they do
not silently replace the selected hypothesis.
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import config
from ..data.dataset_manager import DatasetManager
from ..utils import logger

# ============================================================
# LLM Boundary
# ============================================================


def _call_llm(*args, **kwargs):
    """
    Use the existing application LLM façade.

    This follows the same pattern used by generation_helpers.py so that
    the project's existing LLM configuration and mocks remain effective.
    """
    from .. import agents as facade

    return facade.call_llm(*args, **kwargs)


# ============================================================
# Configuration Helpers
# ============================================================


def _output_token_limit(
    task: str,
    default: int,
) -> int:
    """
    Retrieve the configured output-token budget.

    Falls back to the supplied default when the configuration does not
    contain a valid value.
    """
    configured = config.get("llm_max_tokens", {})

    if not isinstance(configured, dict):
        return default

    try:
        return max(
            1,
            int(configured.get(task, default)),
        )
    except (TypeError, ValueError):
        return default


# ============================================================
# Code Generation Agent
# ============================================================


class CodeGenerationAgent:
    """
    Generates an executable experiment from a selected AI Co-Scientist
    hypothesis.

    The agent does not execute generated code.

    Responsibilities:
        1. Validate the experiment specification.
        2. Inspect the actual dataset schema when required by the experiment.
        3. Extract evidence-derived metric guidance.
        4. Build a detailed code-generation prompt.
        5. Ask the configured LLM for a structured experiment plan.
        6. Extract the generated Python/PyTorch code.
        7. Validate the generated response.
        8. Recover incomplete generations.
        9. Repair failed generated code when requested.
        10. Save generated code/results when requested.
    """

    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 5000
    REPAIR_MAX_TOKENS = 5000

    MAX_REPAIR_SOURCE_CHARS = 30000
    MAX_REPAIR_LOG_CHARS = 5000
    MAX_NOOP_REPAIR_RETRIES = 1

    # A complete experiment can exceed a single response budget.
    # The agent continues the unfinished file instead of regenerating it.
    MAX_CONTINUATION_ATTEMPTS = 3
    MAX_CONTINUATION_REWIND_LINES = 400
    MAX_CONTINUATION_SOURCE_CHARS = 24000
    MAX_CONTINUATION_TAIL_CHARS = 12000
    MIN_TRUNCATION_LINES = 40
    CONTINUATION_MAX_TOKENS = 8000

    TRUNCATION_SYNTAX_MARKERS = (
        "unterminated string literal",
        "unterminated triple-quoted string literal",
        "was never closed",
        "unexpected eof",
    )

    CONTINUATION_SYSTEM_PROMPT = (
        "You continue a partially written Python experiment file.\n"
        "You must preserve the existing experiment design, methodology, "
        "dataset, target, model, and evaluation approach.\n"
        "The selected Rank #1 hypothesis is authoritative for the "
        "implemented model/algorithm.\n"
        "The supporting paper is reference-only and must NOT replace the "
        "Rank #1 model/algorithm.\n"
        "Preserve the distinction between the reference model and the "
        "implemented Rank #1 model.\n"
        "Preserve scientifically compatible evidence-derived metrics.\n"
        "Calculate experiment metrics independently.\n"
        "NEVER copy, average, sample, simulate, or derive experiment "
        "results from paper reference values.\n"
        "If a required metric cannot be measured, record it as unavailable "
        "or not_directly_comparable rather than fabricating a value.\n"
        "If experiment_type is measurement_benchmark and training_required "
        "is false, do NOT introduce training, epochs, batches, optimizers, "
        "training loops, checkpoints, or training_history.json.\n"
        "Do NOT redesign or replace the experiment.\n"
        # ---------------------------------------------------------
        # IMPORTANT CONTINUATION RULES
        # ---------------------------------------------------------

        "Continue ONLY from the exact end of the supplied source code.\n"
        "Do NOT repeat, rewrite, or modify any code that already exists.\n"
        "Do NOT repeat existing imports, functions, classes, statements, "
        "artifact-writing code, or experiment logic.\n"
        "Return ONLY the missing suffix of the Python source.\n"
        "The returned text will be appended directly to the existing "
        "source code.\n"
        "Preserve the exact indentation level required by the preceding "
        "line and surrounding Python block.\n"
        "Do NOT start a new top-level program.\n"
        "Do NOT restart the experiment from the beginning.\n"
        "Do NOT add duplicate definitions of existing functions or variables.\n"
        "Do NOT repeat existing code merely to provide context.\n"
        "If the previous source already contains an artifact-writing block, "
        "do not generate that block again.\n"
        "The continuation must form valid Python when appended directly "
        "to the existing source.\n"
        "Do NOT use backslash-based line continuation; use parentheses "
        "for multi-line expressions.\n"
        "Do NOT add Markdown fences, JSON, explanations, commentary, or "
        "natural-language text.\n"
        "The completed program must remain compatible with the "
        "ExperimentRunner artifact contract.\n"
        "You return raw Python source code only."
    )

    REPAIR_SYSTEM_PROMPT = (
        "You repair an automatically generated Python experiment file.\n"
        "Preserve the existing scientific objective, experiment design, "
        "methodology, dataset, target, model, and evaluation approach.\n"
        "The selected Rank #1 hypothesis is authoritative for the "
        "implemented model/algorithm.\n"
        "The supporting paper is reference-only and must NOT replace "
        "the Rank #1 model/algorithm.\n"
        "Preserve the distinction between the reference model and the "
        "implemented Rank #1 model.\n"
        "Do NOT redesign the experiment.\n"
        "Do NOT replace the experiment with the methodology from an "
        "evidence paper unless explicitly required by the specification.\n"
        "Preserve scientifically compatible evidence-derived metrics.\n"
        "Calculate experiment metrics independently.\n"
        "NEVER copy, average, sample, simulate, or derive experiment "
        "results from paper reference values.\n"
        "NEVER assign a paper result directly to an experiment metric.\n"
        "If a required metric cannot be measured, record it as "
        "unavailable or not_directly_comparable.\n"
        "A proxy must be explicitly labelled as a proxy.\n"
        "Do NOT replace experiment-specific metrics with generic "
        "classification metrics merely to make the experiment run.\n"
        "Do NOT invent paper results or experiment results.\n"
        "If experiment_type is measurement_benchmark and training_required "
        "is false, do NOT introduce epochs, batches, optimizers, training "
        "loops, model training, checkpoints, or training_history.json.\n"
        "Do not fabricate measurements using random sampling.\n"
        "Only fix syntax errors, incomplete code, missing required "
        "sections, or genuine implementation errors.\n"
        "Do NOT use backslash-based line continuation; use parentheses "
        "for multi-line expressions.\n"
        "The repaired program must remain compatible with the "
        "ExperimentRunner artifact contract.\n"
        "Return the complete corrected Python source code only.\n"
        "Do not return JSON, Markdown fences, explanations, or commentary."
    )

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        output_directory: Optional[str | Path] = None,
    ):
        """
        Initialize the Code Generation Agent.

        Parameters
        ----------
        model:
            LLM model used for code generation.

        temperature:
            LLM generation temperature.

        output_directory:
            Optional directory where generated Python files are saved.
        """
        self.model = model or config.get(
            "code_generation_model",
            config.get("llm_model", None),
        )

        self.temperature = (
            self.DEFAULT_TEMPERATURE
            if temperature is None
            else temperature
        )

        self.output_directory = (
            Path(output_directory)
            if output_directory
            else None
        )

        if self.output_directory:
            self.output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ========================================================
    # Serialization
    # ========================================================

    @staticmethod
    def _to_serializable(
        value: Any,
    ) -> Any:
        """
        Convert common project objects into JSON-compatible data.
        """
        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): CodeGenerationAgent._to_serializable(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                CodeGenerationAgent._to_serializable(
                    item
                )
                for item in value
            ]

        if hasattr(value, "model_dump"):
            try:
                return CodeGenerationAgent._to_serializable(
                    value.model_dump()
                )
            except Exception:
                pass

        if hasattr(value, "to_dict"):
            try:
                return CodeGenerationAgent._to_serializable(
                    value.to_dict()
                )
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            return CodeGenerationAgent._to_serializable(
                vars(value)
            )

        return str(value)

    # ========================================================
    # Hypothesis Serialization
    # ========================================================

    def serialize_hypothesis(
        self,
        hypothesis: Any,
    ) -> Dict[str, Any]:
        """
        Serialize the selected Hypothesis using the actual fields used
        by app/models.py.
        """
        if hypothesis is None:
            return {}

        reflection_report = getattr(
            hypothesis,
            "reflection_report",
            None,
        )

        if reflection_report is not None:
            if hasattr(
                reflection_report,
                "model_dump",
            ):
                try:
                    reflection_data = (
                        reflection_report.model_dump()
                    )
                except Exception:
                    reflection_data = {}
            else:
                reflection_data = self._to_serializable(
                    reflection_report
                )
        else:
            reflection_data = None

        return {
            "hypothesis_id": getattr(
                hypothesis,
                "hypothesis_id",
                None,
            ),
            "title": getattr(
                hypothesis,
                "title",
                None,
            ),
            "text": getattr(
                hypothesis,
                "text",
                None,
            ),
            "elo_score": getattr(
                hypothesis,
                "elo_score",
                None,
            ),
            "is_active": getattr(
                hypothesis,
                "is_active",
                None,
            ),
            "parent_ids": getattr(
                hypothesis,
                "parent_ids",
                [],
            ),
            "evolution_strategy": getattr(
                hypothesis,
                "evolution_strategy",
                None,
            ),
            "reflection_report": reflection_data,
            "evidence_source_ids": getattr(
                hypothesis,
                "evidence_source_ids",
                [],
            ),
            "evidence_sources": getattr(
                hypothesis,
                "evidence_sources",
                [],
            ),
            "audit_score": getattr(
                hypothesis,
                "audit_score",
                None,
            ),
            "audit_verdict": getattr(
                hypothesis,
                "audit_verdict",
                None,
            ),
        }

    # ========================================================
    # Research Goal Serialization
    # ========================================================

    def serialize_research_goal(
        self,
        research_goal: Any,
    ) -> Dict[str, Any]:
        """
        Serialize ResearchGoal using the actual fields in app/models.py.
        """
        if research_goal is None:
            return {}

        return {
            "description": getattr(
                research_goal,
                "description",
                "",
            ),
            "preferences": getattr(
                research_goal,
                "preferences",
                None,
            ),
            "idea_attributes": getattr(
                research_goal,
                "idea_attributes",
                None,
            ),
            "constraints": getattr(
                research_goal,
                "constraints",
                {},
            ),
            "llm_model": getattr(
                research_goal,
                "llm_model",
                None,
            ),
            "num_hypotheses": getattr(
                research_goal,
                "num_hypotheses",
                None,
            ),
            "generation_temperature": getattr(
                research_goal,
                "generation_temperature",
                None,
            ),
            "reflection_temperature": getattr(
                research_goal,
                "reflection_temperature",
                None,
            ),
            "elo_k_factor": getattr(
                research_goal,
                "elo_k_factor",
                None,
            ),
            "top_k_hypotheses": getattr(
                research_goal,
                "top_k_hypotheses",
                None,
            ),
        }

    # ========================================================
    # Specification Validation
    # ========================================================

    def validate_specification(
        self,
        specification: Dict[str, Any],
    ) -> None:
        """
        Validate the minimum experiment specification required
        for code generation.
        """
        if not isinstance(
            specification,
            dict,
        ):
            raise TypeError(
                "Experiment specification must be a dictionary."
            )

        required_sections = [
            "dataset",
            "selected_hypothesis",
            "code_generation_requirements",
            "evaluation_metrics",
        ]

        missing = [
            section
            for section in required_sections
            if section not in specification
        ]

        if missing:
            raise ValueError(
                "Experiment specification is missing required "
                f"sections: {', '.join(missing)}"
            )

        dataset = specification.get(
            "dataset",
            {},
        )

        if not isinstance(dataset, dict):
            raise ValueError(
                "'dataset' must be a dictionary."
            )

        dataset_name = str(
            dataset.get(
                "name",
                "",
            )
        ).strip()

        if not dataset_name:
            raise ValueError(
                "Experiment specification must contain a dataset name."
            )

        hypothesis = specification.get(
            "selected_hypothesis",
            {},
        )

        if not isinstance(
            hypothesis,
            dict,
        ):
            raise ValueError(
                "'selected_hypothesis' must be a dictionary."
            )

        hypothesis_text = str(
            hypothesis.get(
                "text",
                "",
            )
        ).strip()

        if not hypothesis_text:
            raise ValueError(
                "Selected hypothesis does not contain usable text."
            )

        reference_experiment = specification.get(
            "reference_experiment",
            {},
        )

        if reference_experiment is None:
            reference_experiment = {}

        if not isinstance(
            reference_experiment,
            dict,
        ):
            raise ValueError(
                "'reference_experiment' must be a dictionary."
            )

        evaluation_metrics = specification.get(
            "evaluation_metrics",
            [],
        )

        if evaluation_metrics is None:
            specification["evaluation_metrics"] = []
        elif not isinstance(
            evaluation_metrics,
            list,
        ):
            raise ValueError(
                "'evaluation_metrics' must be a list."
            )

    # ========================================================
    # Evidence Metric Extraction
    # ========================================================

    @staticmethod
    def _extract_reference_metric_requirements(
        reference_experiment: Any,
    ) -> Dict[str, Any]:
        """
        Extract metric requirements from PaperReader output.

        The returned values are scientific guidance only.

        Important:
            - `metrics` describes metrics evaluated/discussed by the paper.
            - `metric_definitions` describes meaning, units, and direction.
            - `reference_metrics` contains explicitly extracted paper results together with their semantic value types.
            - `reference_conditions` contains experimental configuration/setup information and is not itself an evaluation result.
            - No reference result is copied into generated experiment results.
        """
        result: Dict[str, Any] = {
            "metrics": [],
            "metric_definitions": {},
            "reference_metrics": {},
            "reference_conditions": {},
        }

        if not isinstance(
            reference_experiment,
            dict,
        ):
            return result

        sources = reference_experiment.get(
            "sources",
            [],
        )

        if not isinstance(
            sources,
            list,
        ):
            return result

        seen_metrics = set()

        for source in sources:
            if not isinstance(
                source,
                dict,
            ):
                continue

            details = source.get(
                "experiment_details",
                {},
            )

            if not isinstance(
                details,
                dict,
            ):
                continue

            metrics = details.get(
                "metrics",
                [],
            )

            if isinstance(
                metrics,
                list,
            ):
                for metric in metrics:
                    if isinstance(
                        metric,
                        str,
                    ):
                        metric_name = metric.strip()

                        if not metric_name:
                            continue

                        normalized_name = (
                            metric_name.lower()
                        )

                        if normalized_name not in seen_metrics:
                            result["metrics"].append(
                                metric_name
                            )
                            seen_metrics.add(
                                normalized_name
                            )

            definitions = details.get(
                "metric_definitions",
                {},
            )

            if isinstance(
                definitions,
                dict,
            ):
                for name, definition in definitions.items():
                    if not isinstance(
                        name,
                        str,
                    ):
                        continue

                    if name not in result[
                        "metric_definitions"
                    ]:
                        result[
                            "metric_definitions"
                        ][name] = definition

            reference_metrics = details.get(
                "reference_metrics",
                {},
            )

            if isinstance(
                reference_metrics,
                dict,
            ):
                for name, value in reference_metrics.items():
                    if not isinstance(
                        name,
                        str,
                    ):
                        continue

                    if name not in result[
                        "reference_metrics"
                    ]:
                        result[
                            "reference_metrics"
                        ][name] = value

            reference_conditions = details.get(
                "reference_conditions",
                {},
            )

            if isinstance(reference_conditions, dict):
                for name, value in reference_conditions.items():
                    if not isinstance(name, str):
                        continue

                    if name not in result["reference_conditions"]:
                        result["reference_conditions"][name] = value

        return result

    @staticmethod
    def _select_prompt_metric_guidance(
        guidance: Dict[str, Any],
        required_metrics: Any,
    ) -> Dict[str, Any]:
        """
        Keep only evidence metrics relevant to the metrics explicitly
        requested by the current experiment specification.

        Paper reference values remain comparison-only.
        """
        if not isinstance(guidance, dict):
            return {}

        requested = set()
        if isinstance(required_metrics, list):
            for metric in required_metrics:
                if isinstance(metric, str) and metric.strip():
                    requested.add(metric.strip().lower())

        if not requested:
            return {
                "metrics": [],
                "metric_definitions": {},
                "reference_metrics": {},
                "reference_conditions": {},
            }

        def matches(name: str) -> bool:
            normalized = name.strip().lower()
            normalized_compact = normalized.replace(" ", "_")
            return any(
                normalized == item
                or normalized_compact == item.replace(" ", "_")
                for item in requested
            )

        metrics = [
            metric
            for metric in guidance.get("metrics", [])
            if isinstance(metric, str) and matches(metric)
        ]

        definitions = {
            name: value
            for name, value in (
                guidance.get("metric_definitions", {}) or {}
            ).items()
            if isinstance(name, str) and matches(name)
        }

        reference_metrics = {
            name: value
            for name, value in (
                guidance.get("reference_metrics", {}) or {}
            ).items()
            if isinstance(name, str) and matches(name)
        }

        return {
            "metrics": metrics,
            "metric_definitions": definitions,
            "reference_metrics": reference_metrics,
            "reference_conditions": {},
        }

    # ========================================================
    # Prompt Construction
    # ========================================================

    def build_system_prompt(self) -> str:
        """
        Build the system prompt used for experiment code generation.
        """
        return """
You are the Code Generation Agent in an AI Co-Scientist system.

Your task is to convert a scientifically evaluated Rank #1 hypothesis
into a complete, reproducible, executable experiment.

The generated Python program will be saved and executed automatically
by ExperimentRunner on a remote CPU/GPU server.

============================================================
1. EXPERIMENT PRIORITY
============================================================

Follow this priority order:

1. Selected Rank #1 hypothesis
2. Research goal
3. Experiment specification
4. Available dataset and dataset schema
5. Reference experiment extracted from evidence sources

The reference experiment is supporting scientific evidence.

It MUST NOT redefine, replace, or override the selected hypothesis
or experiment specification.

Never replace the selected experiment with the experiment performed
in an evidence paper merely because the paper provides a more detailed
methodology.

If the reference paper uses a different model, dataset, task, or
experimental setup, adapt only the relevant scientific ideas while
preserving the selected experiment.

============================================================
2. SCIENTIFIC FIDELITY
============================================================

Implement the selected hypothesis faithfully.

Do not silently change:

- research objective
- experiment objective
- dataset
- target variable
- model or algorithm
- experimental methodology
- evaluation procedure
- required metrics

If the hypothesis specifies a particular architecture or algorithm,
implement that architecture or algorithm.

If an implementation detail is missing, make the smallest reasonable
scientific assumption and record it in the experiment metadata.

If the original hypothesis cannot be directly tested with the available
dataset or environment, implement a clearly identified:

- scientific adaptation; or
- proxy experiment.

Do not claim that an adaptation or proxy experiment fully validates
the original hypothesis.

Record:

- original research objective
- implemented experiment objective
- reason for the adaptation/proxy
- assumptions
- adaptations
- limitations

============================================================
3. REFERENCE PAPER AND COMPARISON ROLE
============================================================

The supporting evidence sources serve TWO distinct purposes:

A. REFERENCE EXPERIMENT
B. EVALUATION / COMPARISON GUIDANCE

The selected Rank #1 hypothesis remains authoritative for OUR
automated experiment.

The reference paper remains authoritative for describing the
REFERENCE experiment, including:

- reference model
- reference algorithm
- reference methodology
- reference dataset
- reference experimental setup
- reference evaluation metrics
- metric definitions
- reported reference results

Do NOT replace the Rank #1 hypothesis model with the paper model.

The generated experiment represents the approach proposed by the
Rank #1 hypothesis.

The paper model/approach is retained as the REFERENCE MODEL for
downstream comparison.

============================================================
REFERENCE MODEL VS OUR MODEL
============================================================

The generated experiment must clearly identify:

1. Reference paper model/approach:
   The model or algorithm used by the supporting paper.

2. Rank #1 hypothesis model/approach:
   The model or algorithm proposed by the selected hypothesis and
   implemented by this experiment.

These are separate entities.

Do not report the paper model as the model implemented by the
automated experiment.

Do not silently replace the Rank #1 hypothesis model with the
reference paper model.

The experiment metadata and experiment_summary.json should record
the implemented Rank #1 model/approach.

The reference model/approach should be preserved as reference
metadata for downstream ExperimentComparator.

============================================================
REFERENCE METRICS VS REFERENCE CONDITIONS
============================================================

The reference experiment may contain two different types of
information.

1. REFERENCE METRICS

`reference_metrics` contains quantities reported by the supporting
paper that may be used for downstream scientific comparison.

Reference metrics may represent:

- measured_value
- upper_bound
- lower_bound
- range
- qualitative_result
- unknown

Reference metrics MUST preserve their semantic structure, including
when available:

- value
- unit
- value_type
- relation
- source_text

Do not flatten a structured reference metric into a plain number.

For example:

{
    "value": 80,
    "unit": "ms",
    "value_type": "upper_bound",
    "relation": "less_than"
}

means:

    latency < 80 ms

It does NOT mean:

    latency = 80 ms


2. REFERENCE CONDITIONS

`reference_conditions` describes the conditions under which the
reference experiment was performed.

Examples include:

- number of users
- number of User Equipment (UEs)
- batch size
- number of epochs
- hardware
- traffic load
- network configuration
- testbed configuration
- dataset size
- deployment configuration

Reference conditions are NOT automatically evaluation metrics.

For example:

{
    "value": 500,
    "unit": "UEs",
    "value_type": "configuration"
}

means that the reference experiment was conducted with 500 UEs.

It does NOT mean that:

    num_user_equipment = 500

is an automated-experiment performance measurement.

Reference conditions may be used to understand or reproduce the
reference setup when the local environment supports them.

Do not fabricate unavailable infrastructure, network load, users,
hardware, or measurements merely to reproduce a reference condition.

Reference metrics and reference conditions MUST remain separate from
the automated experiment's measured results.

============================================================
4. COMPARABLE EVALUATION METRICS
============================================================

The evaluation metrics should be selected to support a scientifically
meaningful comparison between the reference paper and the Rank #1
hypothesis experiment where such comparison is possible.

The selected Rank #1 hypothesis remains authoritative for the
automated experiment.

When the supporting paper reports explicit evaluation metrics:

1. Use the paper's `metrics`, `metric_definitions`, and structured
   `reference_metrics` from the evidence-derived reference experiment.

2. Use `evidence_metric_guidance` as additional guidance when
   determining which metrics are scientifically compatible with the
   Rank #1 hypothesis.

3. Prefer the SAME metrics for the Rank #1 experiment when they are
   scientifically compatible with the Rank #1 hypothesis.

4. Preserve the original metric definition.

5. Preserve the original unit.

6. Preserve the direction of improvement when known.

7. Preserve the semantic type of every reference value.

   A reference value may be:

   - measured_value
   - upper_bound
   - lower_bound
   - range
   - qualitative_result
   - unknown

8. Preserve the relation associated with a reference value when known.

   Possible relations include:

   - exact
   - less_than
   - less_than_or_equal
   - greater_than
   - greater_than_or_equal
   - range
   - none

9. A reference upper or lower bound is a comparison constraint,
   NOT an exact experiment result.

   Example:

       paper:
       latency < 80 ms

       means:

       value = 80
       unit = ms
       value_type = upper_bound
       relation = less_than

       It does NOT mean:

       experiment latency = 80 ms

10. A reference range must remain a range.

    Do not replace a range with its midpoint, minimum, maximum,
    or another invented single value.

11. A qualitative reference result must remain qualitative.

    Do not convert qualitative statements into invented numerical
    measurements.

12. Calculate every automated-experiment metric independently from
    the actual experiment execution.

13. NEVER copy a reference-paper numerical result into the automated
    experiment result.

14. NEVER use a reference-paper numerical result as:

    - simulated input
    - calibration value
    - seed value
    - target value
    - hard-coded measurement
    - generated measurement
    - random sampling boundary

15. NEVER estimate an automated-experiment result from a paper result.

16. Every reported automated-experiment measurement must originate
    from an operation actually performed by the generated experiment.

17. If the local environment cannot independently perform the
    measurement required for a reference metric, report that metric
    as:

    unavailable

    or:

    not_directly_comparable

    together with the reason.

18. A proxy may be used only when scientifically justified and MUST
    be explicitly labelled as a proxy.

19. A proxy MUST NOT be presented as equivalent to the original
    reference measurement.

20. Reference metrics are comparison guidance only. They must remain
    separate from the automated experiment's measured results.

Do not force accuracy, precision, recall, or F1 merely because the
available dataset is a classification dataset.

============================================================
5. EXPERIMENT TYPE
============================================================

Determine the experiment type from the selected hypothesis and
experiment specification.

Possible experiment types include:

- classification
- regression
- deep learning
- optimization
- orchestration
- control
- simulation
- networking
- security evaluation
- performance evaluation
- cryptographic evaluation
- proxy/adaptation experiment
- other scientifically specified experiments

Do NOT force every experiment into a supervised classification workflow.

Only perform operations that are appropriate for the selected experiment.

Examples:

- classification -> train/evaluate a classifier
- regression -> train/evaluate a regression model
- optimization/control -> execute the specified optimization/control
  procedure and evaluate its required objectives
- simulation -> execute the specified simulation and record its results
- networking/security -> measure the specified security/performance
  metrics
- cryptographic/performance -> measure latency, overhead, size,
  throughput, or other explicitly required measurements

============================================================
5A. MEASUREMENT BENCHMARK EXPERIMENTS
============================================================

If `experiment_type` is `measurement_benchmark`, the generated
experiment MUST NOT create a machine-learning training workflow
unless the experiment specification explicitly requires training.

Do NOT create:

- epochs
- batches
- optimizers
- training loops
- train/validation/test splits
- model training
- checkpoints
- training_history.json

unless explicitly required by the experiment specification.

A measurement benchmark must measure the actual experimental
quantity required by the Rank #1 hypothesis.

For example, if the evaluation metric is:

    IPsec tunnel setup latency

the experiment must measure the actual relevant tunnel setup process
when that infrastructure is available.

It MUST NOT:

- generate a random latency;
- sample latency from the paper's reported range;
- calculate the paper average;
- copy the paper result;
- derive an experiment result from the paper result.

If the required infrastructure is unavailable, record the metric as
unavailable/not_directly_comparable rather than fabricating a result.

The presence of a dataset does NOT mean that the dataset must be used
for training.

A dataset may instead be:

- supporting data;
- workload data;
- traffic data;
- reference data;
- contextual data;
- or unnecessary for a particular measurement.

Use it only when required by the selected hypothesis and experiment
design.  

============================================================
6. DATASET AND TARGET VALIDATION
============================================================

Use the dataset specified by the experiment specification when the
experiment design requires a dataset.

A dataset marked as supporting_or_reference_dataset must not
automatically be used for training or measurement.

Prefer the DATASET_PATH environment variable only when it is
explicitly provided by the experiment specification or execution
environment.

Never replace the specified dataset with another dataset.

Validate that the dataset exists before loading it.

Use the provided dataset schema to determine columns, data types,
candidate targets, and observed values.

Never assume that labels such as:

- Normal
- Benign
- Attack
- Malicious

exist unless they are actually present in the dataset.

For classification experiments:

1. Identify the target from the experiment specification.
2. Validate the target against the actual dataset schema.
3. Inspect the target distribution.
4. Verify that at least two classes are present.
5. If the target is derived from another column, validate the source
   column and observed values before transformation.
6. Verify that the final encoded class count matches the model output.
7. Convert encoded targets to an explicit NumPy integer array before
   creating PyTorch tensors.

If target validation fails, stop with a clear error.

Do not silently invent or change the target definition.

============================================================
7. DATA PREPROCESSING AND LEAKAGE
============================================================

Use preprocessing appropriate to the experiment.

For machine-learning experiments:

- fit preprocessing only on training data;
- apply fitted preprocessing to validation/test data;
- handle missing values explicitly;
- handle numerical and categorical features appropriately;
- never use the target as an input feature;
- prevent target leakage.

Exclude a feature when it clearly:

- directly represents the target;
- derives from the target;
- contains post-event information unavailable at prediction time.

Record excluded features and the reason for exclusion.

Do not remove features merely because they are correlated with
the target.

For large datasets, if sampling is necessary for computational
constraints, sample from the specified dataset rather than replacing it
with another dataset, and record the original and sampled row counts.

============================================================
8. REPRODUCIBILITY
============================================================

Use deterministic random seeds where appropriate.

Record the random seed in the experiment summary.

Use portable paths and environment variables.

Do not require interactive user input.

Do not use input().

Do not require a graphical desktop.

============================================================
9. PYTORCH AND HARDWARE
============================================================

If the selected experiment involves PyTorch model training:

- use the architecture specified by the hypothesis;
- use torch.nn.Module where appropriate;
- use an appropriate loss and optimizer;
- use model.train() during training;
- use model.eval() during evaluation;
- use torch.no_grad() when gradients are unnecessary;
- save the best checkpoint when model training requires one.

If the selected experiment does NOT involve model training:

- do not introduce a loss function;
- do not introduce an optimizer;
- do not introduce model.train();
- do not introduce training loops;
- do not introduce epochs or batches;
- do not introduce checkpoints;
- do not introduce training_history.json.

PyTorch being the required framework does NOT imply that model
training is required.

Use:

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

The program MUST remain executable on CPU when applicable.

Do not artificially increase model size, batch size, or training duration
to increase GPU utilization.

Use computationally reasonable settings for automated execution.

Maximum epochs should normally be 15 when model training is actually
required, unless the experiment specification explicitly requires
otherwise.

Use early stopping when scientifically appropriate.

Do not use unsupported PyTorch arguments.

In particular, do not pass `verbose` to
torch.optim.lr_scheduler.ReduceLROnPlateau.

============================================================
10. EXECUTION CONTRACT
============================================================

The generated program will be executed automatically by ExperimentRunner.

A successful process exit code alone does NOT mean that the experiment
succeeded.

The program MUST save machine-readable experiment results.

All experiment artifacts MUST be written inside:

    EXPERIMENT_OUTPUT_DIR

Create the directory if necessary.

The program MUST produce:

    metrics.json
    experiment_summary.json

For experiments involving model training, the program SHOULD produce:

    training_history.json

when training history is scientifically applicable.

If the experiment trains a model, also produce:

    best_model.pt

when a checkpoint is scientifically applicable.

If visualizations are relevant to the experiment, save them inside
EXPERIMENT_OUTPUT_DIR.

Console output is NOT a substitute for experiment artifacts.

For example, printing:

    Reward: ...
    Latency: ...
    Accuracy: ...

is insufficient.

The corresponding values must be written into the appropriate JSON
artifact.

============================================================
11. ARTIFACT CONTENTS
============================================================

The artifact structure must match the actual experiment type.

Do NOT invent classification metrics merely to satisfy an artifact
contract.

Examples:

Classification:
- accuracy
- precision
- recall
- F1
- confusion matrix

Regression:
- MAE
- RMSE
- R²

Optimization/control:
- reward
- cost
- constraint violations
- latency
- risk
- convergence
- other metrics required by the hypothesis

Security/networking:
- security metrics
- latency
- overhead
- throughput
- reliability
- other hypothesis-specific measurements

Cryptographic/performance:
- handshake latency
- encryption/decryption latency
- key generation time
- certificate/key size
- protocol overhead
- throughput
- memory usage
- other metrics required by the hypothesis

Use the metrics explicitly required by the selected experiment and the
scientifically compatible evidence-derived metrics.

Do not invent a metric merely because it appears in another experiment.

============================================================
12. METRIC DEFINITIONS AND UNITS
============================================================

When reporting a metric, preserve its scientific meaning.

If the metric has a unit, record the unit where practical.

Examples:

    handshake_latency_ms
    certificate_size_bytes
    throughput_mbps
    memory_mb

Do not silently convert milliseconds to seconds, bytes to kilobytes,
or percentages to proportions unless the experiment specification
requires that conversion.

If a conversion is performed, document it.

Do not compare values with incompatible units.

============================================================
13. TIMING
============================================================

Measure timing appropriate to the experiment.

For training experiments, record training time when applicable.

For evaluation experiments, record evaluation time when applicable.

For experiments with an overall execution phase, record total execution
time when applicable.

Use:

    time.perf_counter()

Do not fabricate timing values for phases that did not occur.

ExperimentRunner may record its own execution timing metadata.

============================================================
14. EXPERIMENT SUMMARY
============================================================

experiment_summary.json should contain, when applicable:

- experiment name
- research hypothesis
- experiment type
- dataset
- target variable
- sample count
- feature count
- class count
- model/algorithm
- configuration
- metrics
- metric definitions
- assumptions
- adaptations
- limitations
- training time
- evaluation time
- total execution time
- device
- GPU name
- random seed

For a proxy or adaptation experiment, explicitly identify it as such.

============================================================
15. VISUALIZATION
============================================================

Generate visualizations only when relevant to the selected experiment.

For training experiments, training-history plots may be generated.

For classification experiments, a confusion matrix may be generated.

For performance experiments, latency/throughput/overhead plots may be
generated when useful.

For other experiment types, generate visualizations appropriate to the
actual experiment.

Do not display plots interactively.

Use a non-interactive matplotlib backend.

Save visualization files inside EXPERIMENT_OUTPUT_DIR.
For classification and model-training experiments, ALWAYS save these four PNG files
inside EXPERIMENT_OUTPUT_DIR using exactly these file names (the runner validates them):
loss_visualization.png - training/validation loss per epoch; if the model has no epochs,
a bar chart of the final loss or error value.
accuracy_visualization.png - accuracy per epoch if tracked; otherwise a bar of final accuracy.
confusion_matrix_visualization.png - confusion matrix of the test predictions.
performance_metrics_visualization.png - bar chart of the main evaluation metrics.
Plot only values the experiment really measured. Close every figure after saving it.
============================================================
15b. TWO-MODEL PROTOCOL FOR UNSEEN ATTACKS
============================================================
Apply this section ONLY when the research goal or the selected hypothesis is about detecting
attack types that an existing model fails to recognise (unseen, novel, zero-day or unknown
attack types) or about detecting ALL attack types of the dataset.
The experiment must build and compare TWO models on the SAME data split:
1. UNSEEN ATTACK. Use the environment variable UNSEEN_ATTACK when it is set (match
case-insensitively). Otherwise use the attack type named ICMPFlood when the dataset has it;
if not, use the non-benign attack type with the fewest rows that still has at least 1000 rows.
Record it as unseen_attack in experiment_summary.json and in metrics.json.
2. SHARED SPLIT. Make ONE stratified train/test split of ALL rows (fixed seed, stratified by the
attack type column, test size about 30 percent). Very large classes may be down-sampled for
speed, but keep every row of the small classes, including the unseen attack. Fit scalers and
encoders on training rows only. Never tune anything on the test split.
3. MODEL 1 - EXISTING MODEL (baseline). Train on the training rows WITHOUT the unseen attack,
so it can detect every other attack but cannot know the unseen one. Use a standard, fast
classifier (for example a Random Forest or a small MLP). Evaluate it on the FULL test split,
including the unseen attack rows.
4. MODEL 2 - PROPOSED MODEL (from the Rank #1 hypothesis). Train on the training rows INCLUDING
the unseen attack so that it learns to detect all attack types. Handle the extreme class
imbalance explicitly (class weights, oversampling or a suitable loss) because the unseen attack
has very few rows. Evaluate it on the same full test split.
5. METRICS in metrics.json. The headline names accuracy, precision_weighted, recall_weighted,
f1_weighted, f1_macro and false_alarm_rate (false positives over false positives plus true
negatives, benign as the negative class) must be the PROPOSED model's values on the full test
split. Add the same metrics for Model 1 with the prefix baseline_ (for example
baseline_f1_weighted). Also add: unseen_attack_recall and baseline_unseen_attack_recall (recall
on the unseen attack rows), unseen_attack_missed_as_benign_rate and
baseline_unseen_attack_missed_as_benign_rate, and the dictionaries per_class_recall_proposed and
per_class_recall_baseline (attack type name to recall).
6. EXPLANATION. In model_recommendation fill reason_for_selection with a concrete explanation of
why this proposed model suits the class imbalance and the unseen attack, and fill
relationship_to_rank1_hypothesis. In experiment_summary.json also write baseline_model (name and
the classes it was trained on) and proposed_model (name and the classes it was trained on).

============================================================
16. CODE QUALITY
============================================================

Generate complete executable Python code.

Do NOT generate:

- pseudocode
- TODO placeholders
- incomplete functions
- `pass` instead of required functionality
- interactive input
- unnecessary dependencies
- unused imports

Use clear functions and meaningful variable names.

Keep the implementation modular and readable.

============================================================
17. FINAL SCIENTIFIC CHECK
============================================================

Before returning the code, verify that:

1. The implementation matches the selected hypothesis.
2. The implementation matches the experiment specification.
3. The selected dataset is actually used when required.
4. The target definition is validated when applicable.
5. No target leakage is introduced.
6. The reference paper has not silently replaced the selected experiment.
7. Evidence-derived metrics are used when scientifically compatible.
8. The experiment calculates its own metric values.
9. No paper result is copied into the generated experiment.
10. No unavailable metric is fabricated.
11. Required artifacts are written to EXPERIMENT_OUTPUT_DIR.
12. The program can run unattended.
13. CPU execution remains possible when applicable.
14. The experiment records assumptions and limitations.
15. The generated code is complete and executable.

============================================================
18. OUTPUT FORMAT
============================================================

Return exactly ONE valid JSON object with these keys:

{
    "model_recommendation": ...,
    "experiment_plan": ...,
    "assumptions": ...,
    "dependencies": ...,
    "pytorch_code": ...
}

The `pytorch_code` field MUST contain the complete executable Python
source code.

Do NOT return Markdown fences.

Do NOT return explanations outside the JSON object.
""".strip()

    @staticmethod
    def _compact_reference_experiment(
        reference_experiment: Any,
    ) -> Dict[str, Any]:
        """
        Reduce PaperReader output to the small, structured subset needed
        for code generation.

        Large extracted text, raw results prose, and provenance blobs are
        deliberately excluded. Reference results remain comparison-only.
        """
        if not isinstance(reference_experiment, dict):
            return {}

        compact: Dict[str, Any] = {
            "available": reference_experiment.get("available", True),
            "source_count": reference_experiment.get("source_count", 0),
            "sources": [],
        }

        sources = reference_experiment.get("sources", [])
        if not isinstance(sources, list):
            return compact

        allowed_detail_keys = (
            "experiment_type",
            "task",
            "models",
            "datasets",
            "metrics",
            "metric_definitions",
            "reference_metrics",
            "reference_conditions",
            "hyperparameters",
            "training_details",
            "experimental_setup",
            "methodology",
            "evaluation",
            "hardware",
            "deployment",
        )

        for source in sources:
            if not isinstance(source, dict):
                continue

            details = source.get("experiment_details", {})
            if not isinstance(details, dict):
                details = {}

            compact_details: Dict[str, Any] = {}
            for key in allowed_detail_keys:
                value = details.get(key)
                if value not in (None, "", [], {}):
                    compact_details[key] = value

            compact_source = {
                "source_url": source.get("source_url"),
                "source_id": source.get("source_id"),
                "source_type": source.get("source_type"),
                "experiment_details": compact_details,
                "evaluation_guidance": {
                    "metrics": compact_details.get("metrics", []),
                    "metric_definitions": compact_details.get(
                        "metric_definitions", {}
                    ),
                    "reference_metrics": compact_details.get(
                        "reference_metrics", {}
                    ),
                },
            }

            compact["sources"].append(compact_source)

        return compact

    def build_user_prompt(
        self,
        specification: Dict[str, Any],
    ) -> str:
        """
        Build a compact user prompt containing only the information
        required for scientifically faithful code generation.

        Large provenance and duplicated evidence content are excluded
        to reduce LLM context size and generation latency.
        """
        selected_hypothesis = specification.get(
            "selected_hypothesis",
            {},
        )

        if not isinstance(
            selected_hypothesis,
            dict,
        ):
            selected_hypothesis = {}

        evidence_sources = selected_hypothesis.get(
            "evidence_sources",
            [],
        )

        compact_evidence_sources = []

        if isinstance(
            evidence_sources,
            list,
        ):
            for source in evidence_sources:
                if not isinstance(
                    source,
                    dict,
                ):
                    continue

                compact_source = {}

                for key in (
                    "source_id",
                    "id",
                    "url",
                    "title",
                    "name",
                    "summary",
                    "description",
                    "source_type",
                ):
                    value = source.get(key)

                    if value not in (
                        None,
                        "",
                        [],
                        {},
                    ):
                        compact_source[key] = value

                compact_evidence_sources.append(
                    compact_source
                )

        compact_hypothesis = {
            "hypothesis_id": selected_hypothesis.get(
                "hypothesis_id"
            ),
            "title": selected_hypothesis.get(
                "title"
            ),
            "text": selected_hypothesis.get(
                "text"
            ),
            "evidence_source_ids": selected_hypothesis.get(
                "evidence_source_ids",
                [],
            ),
            "evidence_sources": compact_evidence_sources,
        }

        compact_reference_experiment = (
            self._compact_reference_experiment(
                specification.get(
                    "reference_experiment",
                    {},
                )
            )
        )

        reference_metric_guidance = (
            self._extract_reference_metric_requirements(
                specification.get(
                    "reference_experiment",
                    {},
                )
            )
        )

        prompt_metric_guidance = self._select_prompt_metric_guidance(
            reference_metric_guidance,
            specification.get("evaluation_metrics", []),
        )

        compact_specification = {
            "dataset": specification.get(
                "dataset",
                {},
            ),
            "dataset_schema": specification.get(
                "dataset_schema",
                {},
            ),
            "research_goal": specification.get(
                "research_goal",
                {},
            ),
            "selected_hypothesis": compact_hypothesis,
            "reference_experiment": compact_reference_experiment,
            "evidence_metric_guidance": prompt_metric_guidance,
            "experiment_design": specification.get(
                "experiment_design",
                {},
            ),
            "code_generation_requirements": specification.get(
                "code_generation_requirements",
                {},
            ),
            "evaluation_metrics": specification.get(
                "evaluation_metrics",
                [],
            ),
            "expected_artifacts": specification.get(
                "expected_artifacts",
                {},
            ),
        }

        logger.info(
            "dataset_schema size: %d",
            len(
                json.dumps(
                    self._to_serializable(
                        specification.get(
                            "dataset_schema",
                            {},
                        )
                    ),
                    ensure_ascii=False,
                )
            ),
        )

        logger.info(
            "experiment_design size: %d",
            len(
                json.dumps(
                    self._to_serializable(
                        specification.get(
                            "experiment_design",
                            {},
                        )
                    ),
                    ensure_ascii=False,
                )
            ),
        )

        logger.info(
            "code_generation_requirements size: %d",
            len(
                json.dumps(
                    self._to_serializable(
                        specification.get(
                            "code_generation_requirements",
                            {},
                        )
                    ),
                    ensure_ascii=False,
                )
            ),
        )

        logger.info(
            "evaluation_metrics size: %d",
            len(
                json.dumps(
                    self._to_serializable(
                        specification.get(
                            "evaluation_metrics",
                            [],
                        )
                    ),
                    ensure_ascii=False,
                )
            ),
        )

        logger.info(
            "reference_experiment size: %d",
            len(
                json.dumps(
                    self._to_serializable(
                        compact_reference_experiment
                    ),
                    ensure_ascii=False,
                )
            ),
        )

        logger.info(
            "evidence_metric_guidance size: %d",
            len(
                json.dumps(
                    self._to_serializable(
                        reference_metric_guidance
                    ),
                    ensure_ascii=False,
                )
            ),
        )

        specification_json = json.dumps(
            self._to_serializable(
                compact_specification
            ),
            ensure_ascii=False,
            indent=2,
        )

        return f"""
Generate the complete executable Python source code for the experiment
described by the AI Co-Scientist specification below.

The selected hypothesis is the final Rank #1 hypothesis.

IMPORTANT PRIORITY:

1. Selected Rank #1 hypothesis
2. Research goal
3. Experiment specification
4. Available dataset and dataset schema
5. Reference experiment and evidence sources

The reference experiment provides scientific guidance.

It must NOT replace or redefine the selected hypothesis.

============================================================
EXPERIMENT SPECIFICATION
============================================================

{specification_json}

============================================================
EVIDENCE-DERIVED METRIC RULES
============================================================

The `evidence_metric_guidance` section contains metrics extracted from
the supporting papers.

Use these rules:

1. Paper metrics are scientific evaluation guidance.
2. Implement a paper metric when it is compatible with the selected
   hypothesis and reproducible in the available environment.
3. Preserve the metric's meaning and unit.
4. The paper's `reference_metrics` are reference values only.
5. NEVER copy a paper reference value into the generated experiment.
6. The generated experiment must calculate its own values.
7. If a metric cannot be reproduced, do not fabricate it.
8. Clearly record unavailable metrics or scientifically justified proxy
   measurements in assumptions, adaptations, or limitations.
9. Do not replace experiment-specific metrics with generic classification
   metrics simply because the dataset happens to contain labels.
10. Do not turn the selected hypothesis into a reproduction of the
    evidence paper unless explicitly required.

============================================================
IMPLEMENTATION REQUIREMENTS
============================================================

0. Treat `experiment_type`, `experiment_design`,
   `evaluation_metrics`, and `code_generation_requirements` in the
   experiment specification as authoritative implementation constraints.

   Do not infer a different experiment type from the dataset name alone.

   Do not enable training, train/validation/test splitting,
   checkpointing, or training history when the specification disables
   those requirements.

1. Implement the selected Rank #1 hypothesis faithfully.

2. Follow the experiment specification and research goal.

3. Use the specified dataset when the experiment requires a dataset.

4. Use DATASET_PATH when available.

5. Use the provided dataset schema to validate available columns,
   data types, candidate targets, and observed target values.

6. Do not assume fixed dataset column names or categorical values.

7. For classification experiments, explicitly validate the target
   distribution before model construction and verify that at least
   two classes are available.

8. If a target is derived from another column, validate the source
   column and observed values before applying the transformation.

9. Do not invent labels or silently change the target definition.

10. Prevent methodological data leakage.

11. Use train/validation/test splitting, training, checkpointing,
    and held-out evaluation ONLY when required by the selected
    experiment type.

12. Do not force a classification or supervised-learning workflow
    onto an optimization, control, orchestration, simulation,
    networking, security, cryptographic, performance, or other
    non-classification experiment.

13. Implement the model or algorithm required by the selected
    hypothesis.

14. Use the reference experiment only as supporting scientific
    evidence.

15. Do not silently replace the selected experiment with the
    methodology used by the evidence paper.

16. If the implementation is an adaptation or proxy experiment,
    explicitly record the assumptions, adaptations, and limitations.

17. Use metrics appropriate to the actual experiment type.

18. Use compatible evidence-derived metrics when reproducible.

19. Do not invent classification metrics merely to satisfy an output
    requirement for a non-classification experiment.

20. Do not invent numerical paper results.

21. Do not copy paper reference values into the generated experiment.

22. Save all required machine-readable artifacts inside
    EXPERIMENT_OUTPUT_DIR.

23. Produce:
       metrics.json
       experiment_summary.json

24. Produce training_history.json when the experiment involves
    model training and training history is applicable.

25. Produce best_model.pt when a trained model/checkpoint is
    scientifically applicable.

26. Save relevant visualizations inside EXPERIMENT_OUTPUT_DIR.

27. Record timing information appropriate to the experiment.

28. The experiment must run unattended without interactive input.

29. The experiment must remain executable on CPU when applicable.

30. Use a deterministic random seed where appropriate and record it.

31. The generated program must be complete executable Python code.
    Do not generate pseudocode, TODO placeholders, or incomplete
    functions.

============================================================
FINAL CHECK
============================================================

Before returning the code, verify that:

- the selected hypothesis is still the experiment being implemented;
- the reference paper has not replaced the selected experiment;
- the dataset and target are handled according to the specification;
- the experiment type is appropriate to the hypothesis;
- evidence-derived metrics are used when scientifically compatible;
- the generated experiment calculates its own metric values;
- no paper result has been copied into the generated results;
- unavailable metrics are not fabricated;
- the evaluation metrics match the actual experiment type;
- required artifacts are actually written;
- the program can execute without manual intervention; and
- assumptions and limitations are recorded when applicable.

============================================================
RESPONSE FORMAT
============================================================

Return exactly one valid JSON object with these fields:

{{
  "model_recommendation": {{}},
  "experiment_plan": {{}},
  "assumptions": [],
  "dependencies": [],
  "pytorch_code": "..."
}}

The "pytorch_code" field must contain the complete executable
Python source code.

============================================================
MODEL RECOMMENDATION OUTPUT
============================================================

The `model_recommendation` field describes the approach implemented
by the automated experiment.

It MUST identify, when applicable:

- model_name
- algorithm
- architecture
- approach_type
- reason_for_selection
- relationship_to_rank1_hypothesis
reason_for_selection MUST be a substantive explanation of 3 to 6 sentences, written for a
supervisor, covering: (a) which limitation of the existing or baseline approach this model
addresses; (b) the mechanism by which the model is expected to achieve the goal of the
Rank #1 hypothesis (for unseen or novel attack detection: how it treats a class it has not
seen during training); (c) why it was chosen over simpler alternatives; (d) what this
experiment cannot prove. Do not claim results that have not been measured.

The recommendation MUST describe the model/algorithm selected from
the Rank #1 hypothesis.

Do NOT use the reference paper's model as the `model_recommendation`
unless the Rank #1 hypothesis explicitly proposes the same model.

The reference paper's model should remain reference metadata and
should not overwrite the Rank #1 model recommendation.

Do not return Markdown.
Do not return Markdown code fences.
Do not return explanations or commentary.
""".strip()

    # ========================================================
    # JSON Extraction
    # ========================================================

    @staticmethod
    def extract_json(
        response: str,
    ) -> Dict[str, Any]:
        """
        Extract a JSON object from an LLM response.

        Handles both normal JSON responses and responses that
        accidentally contain Markdown code fences.
        """
        if not response:
            raise ValueError(
                "LLM returned an empty response."
            )

        cleaned = response.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        try:
            payload = json.loads(
                cleaned
            )
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise ValueError(
                    "Could not locate a JSON object in the LLM response."
                )

            try:
                payload = json.loads(
                    cleaned[start:end + 1]
                )
            except json.JSONDecodeError as exc:
                try:
                    payload = ast.literal_eval(
                        cleaned[start:end + 1]
                    )
                except (
                    SyntaxError,
                    ValueError,
                ) as literal_error:
                    raise ValueError(
                        "Invalid structured response returned by "
                        "CodeGenerationAgent: "
                        f"{literal_error}"
                    ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "CodeGenerationAgent response must be a JSON object."
            )

        return payload

    @staticmethod
    def extract_fenced_python(
        response: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a minimal result when the model returns fenced Python only.
        """
        matches = re.findall(
            r"```(?:python|py)\s*\n(.*?)```",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if matches:
            code = max(
                matches,
                key=len,
            ).strip()
        else:
            opening = re.search(
                r"```(?:python|py)?[ \t]*\r?\n",
                response,
                flags=re.IGNORECASE,
            )

            if opening is None:
                return None

            code = CodeGenerationAgent.strip_code_fences(
                response[
                    opening.end():
                ]
            )

        if not code:
            return None

        return {
            "model_recommendation": {},
            "experiment_plan": {},
            "assumptions": [
                "The model returned executable Python in a fenced code block."
            ],
            "dependencies": [],
            "pytorch_code": code,
        }

    @staticmethod
    def extract_python_source(
        response: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract executable Python when the model ignores the JSON wrapper.
        """
        fenced = CodeGenerationAgent.extract_fenced_python(
            response
        )

        if fenced is not None:
            return fenced

        import_statement = re.search(
            r"(?m)^[ \t]*(?:from\s+[\w.]+\s+import\b|import\s+[\w.*]+)",
            response,
        )

        if import_statement is None:
            return None

        code = response[
            import_statement.start():
        ].strip()

        if not code:
            return None

        return {
            "model_recommendation": {},
            "experiment_plan": {},
            "assumptions": [
                "The model returned executable Python without a JSON wrapper."
            ],
            "dependencies": [],
            "pytorch_code": code,
        }

    # ========================================================
    # Generated Response Validation
    # ========================================================

    def _normalise_escaped_python_source(
        self,
        source: str,
    ) -> str:
        """
        Normalize Python source when the model returns a JSON-escaped string
        instead of plain Python source.

        This function is intentionally conservative: valid Python must be left
        alone. Converting escape sequences like ``\\n`` into real newlines inside
        an existing Python source string literal is what creates invalid
        ``f"...\n..."`` code, so we only decode a truly quoted-escaped payload.
        """
        if not isinstance(
            source,
            str,
        ):
            return source

        source = source.strip()

        if not source:
            return source

        try:
            ast.parse(source)
            return source
        except SyntaxError:
            pass

        # Only decode a whole quoted payload that looks like JSON-escaped
        # Python source, not an already valid Python file.
        if len(source) >= 2 and source[0] == source[-1] and source[0] in {'"', "'"}:
            try:
                decoded = ast.literal_eval(source)
            except (SyntaxError, ValueError):
                decoded = None

            if isinstance(decoded, str):
                try:
                    ast.parse(decoded)
                    return decoded
                except SyntaxError:
                    pass

        return source

    def validate_generated_response(
        self,
        response: Dict[str, Any],
    ) -> None:
        """
        Validate the structure of the generated experiment.
        """
        required_fields = [
            "model_recommendation",
            "experiment_plan",
            "assumptions",
            "dependencies",
            "pytorch_code",
        ]

        missing = [
            field
            for field in required_fields
            if field not in response
        ]

        if missing:
            raise ValueError(
                "Generated response is missing required fields: "
                + ", ".join(missing)
            )

        if not isinstance(
            response["model_recommendation"],
            dict,
        ):
            raise ValueError(
                "'model_recommendation' must be an object."
            )

        if not isinstance(
            response["experiment_plan"],
            dict,
        ):
            raise ValueError(
                "'experiment_plan' must be an object."
            )

        if not isinstance(
            response["assumptions"],
            list,
        ):
            raise ValueError(
                "'assumptions' must be a list."
            )

        if not isinstance(
            response["dependencies"],
            list,
        ):
            raise ValueError(
                "'dependencies' must be a list."
            )

        pytorch_code = response.get(
            "pytorch_code"
        )

        if not isinstance(
            pytorch_code,
            str,
        ):
            raise ValueError(
                "'pytorch_code' must be a string."
            )

        if not pytorch_code.strip():
            raise ValueError(
                "Generated PyTorch code is empty."
            )

        # ------------------------------------------------
        # Normalize LLM-generated escaped Python source.
        # ------------------------------------------------

        pytorch_code = self._normalise_escaped_python_source(
            pytorch_code
        )

        # Store the normalized source back into the response so
        # downstream code uses the corrected Python source.
        response["pytorch_code"] = pytorch_code

        try:
            ast.parse(
                pytorch_code
            )
        except SyntaxError as exc:
            raise ValueError(
                "Generated PyTorch code is not valid Python: "
                f"{exc.msg} at line {exc.lineno}."
            ) from exc

        forbidden_placeholders = [
            "TODO",
            "IMPLEMENT HERE",
            "YOUR CODE HERE",
            "PASS # IMPLEMENT",
            "NOTIMPLEMENTEDERROR",
        ]

        upper_code = pytorch_code.upper()

        for placeholder in forbidden_placeholders:
            if placeholder in upper_code:
                raise ValueError(
                    "Generated PyTorch code contains an incomplete "
                    f"placeholder: {placeholder}"
                )

    # ========================================================
    # Experiment Design Compliance Validation
    # ========================================================

    @staticmethod
    def validate_experiment_design_compliance(
        specification: Dict[str, Any],
        code: str,
    ) -> None:
        """
        Deterministically validate generated code against the
        experiment-design constraints.

        This is a safety check against obvious scientific-design
        violations. It does not attempt to prove that the experiment
        is scientifically correct.
        """
        if not isinstance(specification, dict):
            raise TypeError(
                "Experiment specification must be a dictionary."
            )

        if not isinstance(code, str) or not code.strip():
            raise ValueError(
                "Generated experiment code is empty."
            )

        experiment_design = specification.get(
            "experiment_design",
            {},
        )

        if not isinstance(experiment_design, dict):
            experiment_design = {}

        experiment_type = str(
            experiment_design.get(
                "experiment_type",
                "",
            )
        ).strip().lower()

        training_required = bool(
            experiment_design.get(
                "training_required",
                False,
            )
        )

        code_generation_requirements = specification.get(
            "code_generation_requirements",
            {},
        )

        if not isinstance(code_generation_requirements, dict):
            code_generation_requirements = {}

        checkpoint_setting = experiment_design.get(
            "checkpoint_required",
            code_generation_requirements.get("include_checkpoint"),
        )
        checkpoint_required = (
            None
            if checkpoint_setting is None
            else bool(checkpoint_setting)
        )

        training_history_required = bool(
            experiment_design.get(
                "training_history_required",
                False,
            )
        )

        code_lower = code.lower()

        # ----------------------------------------------------
        # Measurement benchmark constraints
        # ----------------------------------------------------

        if (
            experiment_type == "measurement_benchmark"
            and not training_required
        ):
            forbidden_training_patterns = (
                ".backward(",
                "optimizer.step(",
                "torch.optim",
                "training_history.json",
                "best_model.pt",
                "train_loader",
                "dataloader(",
                "num_epochs",
                "epochs =",
                "batch_size",
                "model.train(",
            )

            violations = [
                pattern
                for pattern in forbidden_training_patterns
                if pattern in code_lower
            ]

            if violations:
                raise ValueError(
                    "Generated measurement_benchmark experiment "
                    "contains training-related functionality even "
                    "though training_required=false: "
                    + ", ".join(violations)
                )

        # ----------------------------------------------------
        # Checkpoint constraint
        # ----------------------------------------------------

        if checkpoint_required is False:
            checkpoint_patterns = (
                "best_model.pt",
                "torch.save(",
                "checkpoint",
            )

            violations = [
                pattern
                for pattern in checkpoint_patterns
                if pattern in code_lower
            ]

            if violations:
                raise ValueError(
                    "Generated experiment contains checkpoint-related "
                    "functionality even though checkpoint_required=false: "
                    + ", ".join(violations)
                )

        # ----------------------------------------------------
        # Training-history constraint
        # ----------------------------------------------------

        if not training_history_required:
            if "training_history.json" in code_lower:
                raise ValueError(
                    "Generated experiment writes training_history.json "
                    "even though training_history_required=false."
                )

        # ----------------------------------------------------
        # Prevent obvious paper-result reuse
        # ----------------------------------------------------

        forbidden_reference_patterns = (
            "reference_metrics[",
            "reference_metrics.get(",
            "paper_results[",
            "paper_results.get(",
            "reference_values[",
            "reference_values.get(",
            "reference_result",
            "paper_result",
        )

        reference_violations = [
            pattern
            for pattern in forbidden_reference_patterns
            if pattern in code_lower
        ]

        if reference_violations:
            raise ValueError(
                "Generated experiment appears to use paper/reference "
                "results as experiment inputs or outputs: "
                + ", ".join(reference_violations)
            )

        # ----------------------------------------------------
        # Measurement benchmarks must not fabricate measurements
        # using random sampling.
        # ----------------------------------------------------

        if experiment_type == "measurement_benchmark":
            suspicious_random_patterns = (
                "np.random.uniform(",
                "random.uniform(",
                "np.random.normal(",
                "random.normalvariate(",
            )

            random_violations = [
                pattern
                for pattern in suspicious_random_patterns
                if pattern in code_lower
            ]

            if random_violations:
                raise ValueError(
                    "Generated measurement_benchmark experiment uses "
                    "random sampling to produce a measurement: "
                    + ", ".join(random_violations)
                    + ". Measurements must come from the actual "
                    "experiment or be recorded as unavailable."
                )

    # ========================================================
    # Incomplete Generation Recovery
    # ========================================================

    @staticmethod
    def strip_code_fences(
        response: str,
    ) -> str:
        """
        Remove Markdown fences, including an unterminated opening fence.

        Leading spaces are preserved because a continuation starts at the
        indentation of the statement it resumes.
        """
        text = response.lstrip(
            "\n"
        ).rstrip()

        text = re.sub(
            r"^[ \t]*```[A-Za-z0-9_+-]*[ \t]*\r?\n",
            "",
            text,
        )

        text = re.sub(
            r"(?:\r?\n)?[ \t]*```[ \t]*$",
            "",
            text,
        )

        return text.rstrip()

    @classmethod
    def syntax_error_is_truncation(
        cls,
        code: str,
        error: SyntaxError,
    ) -> bool:
        """
        Report whether a syntax error is the signature of a cut-off
        response.
        """
        message = str(
            getattr(
                error,
                "msg",
                "",
            )
            or ""
        ).lower()

        # A syntax error on the final line is not enough evidence of
        # truncation. Errors such as unexpected indent, invalid syntax,
        # and line-continuation errors must go through repair instead.
        return any(
            marker in message
            for marker in cls.TRUNCATION_SYNTAX_MARKERS
        )

    @classmethod
    def _is_truncation_validation_error(
        cls,
        code: str,
        validation_error: Exception,
    ) -> bool:
        """
        Decide whether validation failure should use continuation.

        Continuation is reserved for genuinely truncated Python. All other
        syntax/design errors use the normal repair path.
        """
        if not isinstance(code, str) or not code.strip():
            return False

        try:
            ast.parse(code)
        except SyntaxError as syntax_error:
            return cls.syntax_error_is_truncation(
                code,
                syntax_error,
            )

        return False

    @classmethod
    def longest_parsable_prefix(
        cls,
        code: str,
    ) -> Optional[str]:
        """
        Return the longest leading part of the source that still parses.

        The cut-off tail is dropped so the model can continue from a
        complete statement.
        """
        lines = code.splitlines()

        limit = min(
            len(lines),
            cls.MAX_CONTINUATION_REWIND_LINES,
        )

        for dropped in range(
            1,
            limit + 1,
        ):
            candidate = "\n".join(
                lines[
                    : len(lines) - dropped
                ]
            )

            if not candidate.strip():
                return None

            try:
                ast.parse(
                    candidate
                )
            except SyntaxError:
                continue

            return candidate

        return None

    def build_continuation_prompt(
        self,
        prefix: str,
    ) -> str:
        """
        Ask the model to finish an experiment that was cut off.
        """
        bounded_prefix = prefix[
            -self.MAX_CONTINUATION_TAIL_CHARS:
        ]

        return f"""
The following Python file is incomplete because the previous response
reached its output-token limit. Continue the file from exactly where it
stops.

IMPORTANT:

1. Do NOT repeat any line that is already written.
2. Do NOT restate imports, classes, or functions that already exist.
3. Continue with the indentation required where the file stops.
4. Complete the remaining implementation required by the selected
   experiment, including evaluation, metric calculation, artifact
   saving, and the executable entry point when applicable.
5. Do NOT redesign, simplify, or replace the experiment.
6. Preserve the selected hypothesis, experiment type, methodology,
   dataset, target, model/algorithm, and evaluation approach.
7. Preserve evidence-derived metrics when they are scientifically
   applicable.
8. Do not invent paper results or metric values.
9. If a metric cannot be calculated, record it as unavailable rather
   than fabricating a value.
10. Save all required artifacts inside EXPERIMENT_OUTPUT_DIR.
11. Return ONLY raw Python source code.
12. Do NOT return Markdown fences, JSON, explanations, or commentary.

============================================================
FILE WRITTEN SO FAR
============================================================

{bounded_prefix}

============================================================
CONTINUE THE FILE FROM THE NEXT CHARACTER
============================================================
""".strip()

    def continue_truncated_code(
        self,
        generated_code: str,
    ) -> str:
        """
        Complete an experiment that stopped at the output-token limit.

        Returns the source unchanged when it already parses.
        Raises ValueError when the source is invalid for another reason,
        or when the continuations do not complete it.
        """
        generated_code = self._normalise_escaped_python_source(
            generated_code
        )
        if not isinstance(
            generated_code,
            str,
        ) or not generated_code.strip():
            raise ValueError(
                "Generated PyTorch code is required for continuation."
            )

        code = self._normalise_escaped_python_source(
            generated_code
        )

        for attempt in range(
            self.MAX_CONTINUATION_ATTEMPTS + 1
        ):
            try:
                ast.parse(
                    code
                )
                return code
            except SyntaxError as error:
                if not self.syntax_error_is_truncation(
                    code,
                    error,
                ):
                    raise ValueError(
                        "Generated PyTorch code is not valid Python: "
                        f"{error.msg} at line {error.lineno}."
                    ) from error

                if attempt == self.MAX_CONTINUATION_ATTEMPTS:
                    raise ValueError(
                        "Generated PyTorch code is still incomplete after "
                        f"{self.MAX_CONTINUATION_ATTEMPTS} continuation "
                        "attempt(s). Increase llm_max_tokens.code_generation."
                    ) from error

            prefix = self.longest_parsable_prefix(
                code
            )

            if prefix is None:
                raise ValueError(
                    "Generated PyTorch code was cut off and no complete "
                    "prefix could be recovered."
                )

            logger.warning(
                "Generated experiment stopped at the output-token limit; "
                "requesting continuation %d/%d from line %d.",
                attempt + 1,
                self.MAX_CONTINUATION_ATTEMPTS,
                len(prefix.splitlines()),
            )

            response = _call_llm(
                self.build_continuation_prompt(
                    prefix
                ),
                temperature=0.0,
                model=self.model,
                system_prompt=self.CONTINUATION_SYSTEM_PROMPT,
                max_tokens=max(
                    self.CONTINUATION_MAX_TOKENS,
                    _output_token_limit(
                        "code_generation",
                        self.DEFAULT_MAX_TOKENS,
                    ),
                ),
                reasoning="off",
            )

            if not isinstance(
                response,
                str,
            ):
                response = str(
                    response
                )

            if response.startswith(
                "Error:"
            ):
                raise RuntimeError(
                    response
                )

            continuation = self.strip_code_fences(
                response
            )

            if not continuation:
                raise ValueError(
                    "The model returned no continuation for the "
                    "incomplete experiment."
                )

            code = (
                prefix.rstrip("\n")
                + "\n"
                + continuation.lstrip("\n")
            )

        return code

    def recover_incomplete_generation(
        self,
        specification: Dict[str, Any],
        generated: Optional[Dict[str, Any]],
        code_error: ValueError,
    ) -> Dict[str, Any]:
        """
        Recover a generated experiment that failed validation.

        A response that stopped at the output-token limit is continued
        from its last complete statement.

        Anything else is regenerated from the specification.
        """
        code = (
            generated.get("pytorch_code")
            if isinstance(
                generated,
                dict,
            )
            else None
        )

        if isinstance(
            code,
            str,
        ) and code.strip():
            try:
                completed = self.continue_truncated_code(
                    code
                )
            except (
                ValueError,
                RuntimeError,
            ) as continuation_error:
                logger.warning(
                    "Continuing the incomplete experiment failed: %s",
                    continuation_error,
                )
            else:
                recovered = dict(
                    generated
                )

                recovered[
                    "pytorch_code"
                ] = completed

                assumptions = list(
                    recovered.get(
                        "assumptions"
                    )
                    or []
                )

                if completed != code:
                    assumptions.append(
                        "The experiment source was completed with a "
                        "continuation request after the model reached its "
                        "output-token limit."
                    )

                recovered[
                    "assumptions"
                ] = assumptions

                self.validate_generated_response(
                    recovered
                )

                self.validate_experiment_design_compliance(
                    specification,
                    recovered["pytorch_code"],
                )

                return recovered

        reference_metric_guidance = (
            self._extract_reference_metric_requirements(
                specification.get(
                    "reference_experiment",
                    {},
                )
            )
        )

        code_only_prompt = f"""
Return only complete, executable Python source code for the selected
experiment.

Do not return JSON, Markdown, explanations, analysis, or commentary.
Start with a Python import and end with the executable experiment code.

The selected hypothesis and experiment specification are authoritative.

Do NOT redesign or replace the selected experiment.

Preserve:

- selected Rank #1 hypothesis
- research objective
- experiment type
- required methodology
- dataset and target definition when applicable
- specified model or algorithm
- evaluation methodology
- required metrics
- scientifically compatible evidence-derived metrics

Evidence-derived metric guidance:

{json.dumps(
    self._to_serializable(
        reference_metric_guidance
    ),
    ensure_ascii=False,
    indent=2,
)}

Important:

- Calculate experiment results from the actual experiment.
- Do not copy paper reference values into experiment results.
- Do not fabricate unavailable metrics.
- Record adaptations and limitations when a direct reproduction is
  impossible.

The generated program must:

1. Be complete and executable Python.
2. Use PyTorch when required by the selected experiment.
3. Use DATASET_PATH when a dataset is required.
4. Use EXPERIMENT_OUTPUT_DIR for all experiment artifacts.
5. Produce:
   - metrics.json
   - experiment_summary.json
6. Produce training_history.json when training history is applicable.
7. Produce best_model.pt when a trained model/checkpoint is applicable.
8. Save relevant visualizations inside EXPERIMENT_OUTPUT_DIR.
9. Record assumptions, adaptations, and proxy-experiment limitations
   when applicable.
10. Never rely only on console output for experiment results.
11. Remain compatible with ExperimentRunner.
12. Do not use interactive input.
13. Do not invent a different experiment merely to make the code easier
    to generate.

Selected hypothesis:
{specification["selected_hypothesis"].get("text", "")}

Research goal:
{json.dumps(
    self._to_serializable(
        specification.get(
            "research_goal",
            {},
        )
    ),
    ensure_ascii=False,
    indent=2,
)}

Experiment specification:
{json.dumps(
    self._to_serializable(
        specification
    ),
    ensure_ascii=False,
    indent=2,
)}

Dataset:
{specification["dataset"].get("name", "dataset")}
""".strip()

        code_only_response = _call_llm(
            code_only_prompt,
            temperature=0.0,
            model=self.model,
            system_prompt=self.CONTINUATION_SYSTEM_PROMPT,
            max_tokens=_output_token_limit(
                "code_generation",
                self.DEFAULT_MAX_TOKENS,
            ),
            reasoning="off",
        )

        if not isinstance(
            code_only_response,
            str,
        ):
            code_only_response = str(
                code_only_response
            )

        regenerated = self.extract_python_source(
            code_only_response
        )

        if regenerated is None:
            raise code_error

        regenerated[
            "pytorch_code"
        ] = self.continue_truncated_code(
            regenerated[
                "pytorch_code"
            ]
        )

        self.validate_generated_response(
            regenerated
        )

        self.validate_experiment_design_compliance(
            specification,
            regenerated["pytorch_code"],
        )

        return regenerated

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        specification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate an experiment from an experiment specification.

        Returns
        -------
        dict
            Structured CodeGenerationAgent result.
        """
        started_at = time.perf_counter()

        result: Dict[str, Any] = {
            "success": False,
            "model": self.model,
            "model_recommendation": None,
            "experiment_plan": None,
            "assumptions": [],
            "dependencies": [],
            "pytorch_code": None,
            "generation_seconds": None,
            "errors": [],
        }

        try:
            self.validate_specification(
                specification
            )

            # ------------------------------------------------
            # Inspect the dataset schema only when the
            # experiment actually requires the dataset.
            # ------------------------------------------------

            try:
                experiment_design = specification.get(
                    "experiment_design",
                    {},
                )

                if not isinstance(
                    experiment_design,
                    dict,
                ):
                    experiment_design = {}

                dataset = specification.get(
                    "dataset",
                    {},
                )

                if not isinstance(
                    dataset,
                    dict,
                ):
                    dataset = {}

                training_required = bool(
                    experiment_design.get(
                        "training_required",
                        False,
                    )
                )

                dataset_role = str(
                    dataset.get(
                        "role",
                        "",
                    )
                ).strip().lower()

                requires_dataset = (
                    training_required
                    or dataset_role == "required_experiment_dataset"
                )

                specification = dict(
                    specification
                )

                if requires_dataset:
                    dataset_name = dataset.get(
                        "name",
                        "dataset",
                    )

                    dataset_path = dataset.get(
                        "path"
                    )

                    dataset_manager = DatasetManager(
                        dataset_name=dataset_name,
                        dataset_path=dataset_path,
                    )

                    resolved_dataset_path = (
                        dataset_manager.get_latest_dataset()
                    )

                    dataset_schema = (
                        dataset_manager.inspect_schema(
                            resolved_dataset_path
                        )
                    )

                    specification[
                        "dataset_schema"
                    ] = dataset_schema

                    logger.info(
                        "Dataset schema inspected because the experiment "
                        "requires the dataset."
                    )

                else:
                    specification[
                        "dataset_schema"
                    ] = {}

                    logger.info(
                        "Dataset schema inspection skipped because the "
                        "experiment does not require the dataset."
                    )

            except Exception as error:
                logger.warning(
                    "Dataset schema inspection failed: %s",
                    error,
                )

                raise RuntimeError(
                    "Unable to prepare the dataset information before "
                    f"code generation: {error}"
                ) from error

            system_prompt = (
                self.build_system_prompt()
            )

            user_prompt = (
                self.build_user_prompt(
                    specification
                )
            )

            logger.info(
                "CodeGeneration user prompt size: %d characters",
                len(user_prompt),
            )

            logger.info(
                "CodeGeneration system prompt size: %d characters",
                len(system_prompt),
            )

            logger.info(
                "CodeGeneration total prompt size: %d characters",
                len(system_prompt) + len(user_prompt),
            )

            response = _call_llm(
                user_prompt,
                temperature=self.temperature,
                model=self.model,
                system_prompt=system_prompt,
                max_tokens=_output_token_limit(
                    "code_generation",
                    self.DEFAULT_MAX_TOKENS,
                ),
                reasoning="off",
            )

            if not isinstance(
                response,
                str,
            ):
                response = str(
                    response
                )

            if response.startswith(
                "Error:"
            ):
                raise RuntimeError(
                    response
                )

            # ------------------------------------------------
            # Extract structured response.
            # ------------------------------------------------

            try:
                generated = self.extract_json(
                    response
                )

            except ValueError as parse_error:
                generated = self.extract_python_source(
                    response
                )

                if generated is None:
                    repair_prompt = f"""
The previous response was not valid structured JSON.

Return exactly one valid JSON object with:

- model_recommendation
- experiment_plan
- assumptions
- dependencies
- pytorch_code

Preserve the complete experiment source code.

The selected Rank #1 hypothesis, experiment specification, and
scientifically compatible evidence-derived metrics are authoritative.

Do not add Markdown, explanations, or extra text.

Previous response:
{response}

Parser error:
{parse_error}
""".strip()

                    repaired_response = _call_llm(
                        repair_prompt,
                        temperature=0.0,
                        model=self.model,
                        system_prompt=system_prompt,
                        max_tokens=_output_token_limit(
                            "code_generation",
                            self.DEFAULT_MAX_TOKENS,
                        ),
                        reasoning="off",
                    )

                    if not isinstance(
                        repaired_response,
                        str,
                    ):
                        repaired_response = str(
                            repaired_response
                        )

                    if repaired_response.startswith(
                        "Error:"
                    ):
                        raise RuntimeError(
                            repaired_response
                        ) from parse_error

                    try:
                        generated = self.extract_json(
                            repaired_response
                        )

                    except ValueError as repaired_parse_error:
                        generated = self.extract_python_source(
                            repaired_response
                        )

                        if generated is None:
                            generated = self.extract_python_source(
                                response
                            )

                        if generated is None:
                            reference_metric_guidance = (
                                self._extract_reference_metric_requirements(
                                    specification.get(
                                        "reference_experiment",
                                        {},
                                    )
                                )
                            )

                            code_only_prompt = f"""
Generate only the complete executable Python source code for the
selected experiment.

Do not return JSON.
Do not return explanations.
Do not use Markdown fences.
Return raw Python source code only.

IMPORTANT:

The selected hypothesis and experiment specification are authoritative.

Do NOT redesign or replace the selected experiment.

Preserve:

- selected hypothesis
- research objective
- experiment type
- required methodology
- dataset and target definition when applicable
- specified model or algorithm
- evaluation methodology
- required metrics
- scientifically compatible evidence-derived metrics

Evidence-derived metric guidance:

{json.dumps(
    self._to_serializable(
        reference_metric_guidance
    ),
    ensure_ascii=False,
    indent=2,
)}

The generated experiment must calculate its own metric values.

Do not copy paper reference values into experiment results.

Do not fabricate unavailable metrics.

If a metric cannot be reproduced, record the limitation or use a clearly
identified scientific proxy where appropriate.

The generated program must:

1. Be complete and executable Python.
2. Use PyTorch when required by the experiment.
3. Use DATASET_PATH when a dataset is required.
4. Use EXPERIMENT_OUTPUT_DIR for all artifacts.
5. Produce metrics.json.
6. Produce experiment_summary.json.
7. Produce training_history.json when training history is applicable.
8. Produce best_model.pt when a trained model/checkpoint is applicable.
9. Save relevant visualizations when applicable.
10. Record assumptions and limitations.
11. Remain compatible with ExperimentRunner.
12. Do not use interactive input.

Selected hypothesis:
{specification["selected_hypothesis"].get("text", "")}

Research goal:
{json.dumps(
    self._to_serializable(
        specification.get(
            "research_goal",
            {},
        )
    ),
    ensure_ascii=False,
    indent=2,
)}

Experiment specification:
{json.dumps(
    self._to_serializable(
        specification
    ),
    ensure_ascii=False,
    indent=2,
)}

Dataset:
{specification["dataset"].get("name", "dataset")}
""".strip()

                            code_only_response = _call_llm(
                                code_only_prompt,
                                temperature=0.0,
                                model=self.model,
                                system_prompt=system_prompt,
                                max_tokens=_output_token_limit(
                                    "code_generation",
                                    self.DEFAULT_MAX_TOKENS,
                                ),
                                reasoning="off",
                            )

                            if not isinstance(
                                code_only_response,
                                str,
                            ):
                                code_only_response = str(
                                    code_only_response
                                )

                            generated = (
                                self.extract_python_source(
                                    code_only_response
                                )
                            )

                        if generated is None:
                            raise repaired_parse_error

            # ------------------------------------------------
            # Validate generated code.
            # ------------------------------------------------

            try:
                self.validate_generated_response(generated)
                self.validate_experiment_design_compliance(
                    specification,
                    generated["pytorch_code"],
                )
            except ValueError as code_error:
                if self._is_truncation_validation_error(
                    generated.get("pytorch_code", ""),
                    code_error,
                ):
                    generated = self.recover_incomplete_generation(
                        specification,
                        generated,
                        code_error,
                    )
                else:
                    generated = self.repair_generated_code(
                        specification,
                        generated["pytorch_code"],
                        {
                            "success": False,
                            "errors": [str(code_error)],
                        },
                    )

            # ------------------------------------------------
            # Preserve evidence-derived evaluation information
            # in the returned generation result.
            # ------------------------------------------------

            reference_metric_guidance = (
                self._extract_reference_metric_requirements(
                    specification.get(
                        "reference_experiment",
                        {},
                    )
                )
            )

            result.update(
                {
                    "success": True,
                    "model_recommendation": generated[
                        "model_recommendation"
                    ],
                    "experiment_plan": generated[
                        "experiment_plan"
                    ],
                    "assumptions": generated[
                        "assumptions"
                    ],
                    "dependencies": generated[
                        "dependencies"
                    ],
                    "pytorch_code": generated[
                        "pytorch_code"
                    ],

                    # These fields make the scientific provenance
                    # available to ExperimentRunner and downstream
                    # comparison/debugging code.
                    "evaluation_metrics": specification.get(
                        "evaluation_metrics",
                        [],
                    ),
                    "evidence_metric_guidance": (
                        reference_metric_guidance
                    ),
                    "experiment_design": specification.get(
                        "experiment_design",
                        {},
                    ),
                    "code_generation_requirements": specification.get(
                        "code_generation_requirements",
                        {},
                    ),
                    "reference_experiment": (
                        self._compact_reference_experiment(
                            specification.get(
                                "reference_experiment",
                                {},
                            )
                        )
                    ),
                    "selected_hypothesis": (
                        specification.get(
                            "selected_hypothesis",
                            {},
                        )
                    ),
                }
            )

        except Exception as error:
            logger.exception(
                "CodeGenerationAgent failed."
            )

            result[
                "errors"
            ].append(
                str(error)
            )

        finally:
            result[
                "generation_seconds"
            ] = (
                time.perf_counter()
                - started_at
            )

        return result

    # ========================================================
    # Automatic Experiment Code Repair
    # ========================================================

    def repair_generated_code(
        self,
        specification: Dict[str, Any],
        generated_code: str,
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Automatically repair a failed generated experiment using the LLM.

        The LLM receives:

            1. Experiment specification
            2. Selected Rank #1 hypothesis
            3. Evidence-derived metric guidance
            4. Current generated Python source
            5. Execution error / traceback
            6. Previous stdout
            7. Previous stderr

        The LLM must return a complete corrected Python experiment.

        This method intentionally does not contain hard-coded fixes for
        individual Python or machine-learning errors.
        """
        if not isinstance(
            generated_code,
            str,
        ) or not generated_code.strip():
            raise ValueError(
                "Generated PyTorch code is required for repair."
            )

        if not isinstance(
            execution_result,
            dict,
        ):
            raise TypeError(
                "execution_result must be a dictionary."
            )

        if not isinstance(
            specification,
            dict,
        ):
            raise TypeError(
                "specification must be a dictionary."
            )

        # ----------------------------------------------------
        # Keep repair prompt bounded.
        # ----------------------------------------------------

        bounded_source = generated_code[
            -self.MAX_REPAIR_SOURCE_CHARS:
        ]

        stdout = str(
            execution_result.get(
                "stdout",
                "",
            )
        )

        stderr = str(
            execution_result.get(
                "stderr",
                "",
            )
        )

        error_message = str(
            execution_result.get(
                "error",
                "",
            )
        )

        bounded_stdout = stdout[
            -self.MAX_REPAIR_LOG_CHARS:
        ]

        bounded_stderr = stderr[
            -self.MAX_REPAIR_LOG_CHARS:
        ]

        # ----------------------------------------------------
        # Build compact experiment specification.
        # ----------------------------------------------------

        dataset = specification.get(
            "dataset",
            {},
        )

        if not isinstance(
            dataset,
            dict,
        ):
            dataset = {}

        selected_hypothesis = specification.get(
            "selected_hypothesis",
            {},
        )

        if not isinstance(
            selected_hypothesis,
            dict,
        ):
            selected_hypothesis = {}

        reference_metric_guidance = (
            self._extract_reference_metric_requirements(
                specification.get(
                    "reference_experiment",
                    {},
                )
            )
        )

        compact_specification = {
            "dataset": dataset,
            "dataset_schema": specification.get(
                "dataset_schema",
                {},
            ),
            "research_goal": specification.get(
                "research_goal",
                {},
            ),
            "selected_hypothesis": {
                "hypothesis_id": selected_hypothesis.get(
                    "hypothesis_id"
                ),
                "title": selected_hypothesis.get(
                    "title"
                ),
                "text": selected_hypothesis.get(
                    "text"
                ),
                "evidence_source_ids": selected_hypothesis.get(
                    "evidence_source_ids",
                    [],
                ),
            },
            "code_generation_requirements": specification.get(
                "code_generation_requirements",
                {},
            ),
            "repair_requirements": specification.get(
                "repair_requirements",
                [],
            ),
            "evaluation_metrics": specification.get(
                "evaluation_metrics",
                [],
            ),
            "evidence_metric_guidance": (
                reference_metric_guidance
            ),
            "reference_experiment": (
                self._compact_reference_experiment(
                    specification.get(
                        "reference_experiment",
                        {},
                    )
                )
            ),
            "experiment_design": specification.get(
                "experiment_design",
                {},
            ),
        }

        # ----------------------------------------------------
        # Build execution-error context.
        # ----------------------------------------------------

        execution_context = {
            "status": execution_result.get(
                "status"
            ),
            "return_code": execution_result.get(
                "return_code"
            ),
            "error": error_message,
            "stderr": bounded_stderr,
            "stdout": bounded_stdout,
        }

        # ----------------------------------------------------
        # Repair prompt.
        # ----------------------------------------------------

        repair_prompt = f"""
You are repairing a failed automatically generated experiment inside
an AI Co-Scientist system.

The experiment was generated by another LLM and then executed
automatically by ExperimentRunner.

The experiment failed during execution.

Your task is to determine the ROOT CAUSE of the failure from the
traceback, stdout, stderr, experiment specification, and current
source code.

Then return a COMPLETE corrected Python source file.

IMPORTANT:

1. Return ONLY Python source code.
2. Do NOT return Markdown fences.
3. Do NOT return JSON.
4. Do NOT return explanations or commentary.
5. Do NOT return a patch or partial code.
6. Return the COMPLETE replacement for the current source file.
7. Preserve the original Rank #1 research hypothesis.
8. Preserve the intended model architecture whenever possible.
9. Preserve the intended experiment objective.
10. Preserve the evaluation metrics required by the selected experiment.
11. Preserve scientifically compatible evidence-derived metrics.
12. Do not replace latency, overhead, size, throughput, security,
    reward, cost, or other experiment-specific metrics with generic
    classification metrics merely to make the experiment run.
13. Do not invent paper results.
14. Do not copy paper reference values into experiment results.
15. Do not fabricate unavailable metrics.
16. If a required metric cannot be computed, clearly record it as
    unavailable or use an explicitly identified scientific proxy when
    appropriate.
17. Preserve the reference experiment only where it is applicable to
    the selected experiment.
18. Do not replace the selected experiment with the evidence paper's
    experiment.
19. Preserve train/validation/test evaluation design ONLY when it is
    applicable to the selected experiment type.
20. Follow experiment_design.checkpoint_required exactly. When it is false,
    remove checkpoint saving and references, including torch.save, best_model.pt,
    and checkpoint artifacts. When true, preserve required checkpoint behavior.
21. Preserve training-history generation when training history is
    scientifically applicable.
22. Preserve required visualization generation when scientifically
    applicable.
23. Save generated artifacts inside EXPERIMENT_OUTPUT_DIR.
24. Use DATASET_PATH when it is available.
25. Do not invent a different dataset.
26. Do not remove required experiment functionality merely to make the
    program run.
27. Fix the actual root cause instead of hiding the error.
28. Make the smallest scientifically reasonable correction.
29. If the dataset structure is different from what the original code
    assumed, adapt the preprocessing to the actual dataset information
    available in the specification and error.
30. Handle class-distribution and data-splitting problems robustly when
    necessary.
31. Handle missing, categorical, and numerical data appropriately.
32. Do not introduce data leakage.
33. The repaired source must be valid executable Python.
34. Do not use TODO, pass, placeholder code, or incomplete
    implementations.

The ExperimentRunner will execute the returned source again.

Therefore, your response must be the complete executable experiment,
not an explanation of what should be changed.

============================================================
EXPERIMENT SPECIFICATION
============================================================

{json.dumps(
    self._to_serializable(
        compact_specification
    ),
    ensure_ascii=False,
    indent=2,
)}

============================================================
EXECUTION RESULT
============================================================

{json.dumps(
    execution_context,
    ensure_ascii=False,
    indent=2,
)}

============================================================
CURRENT GENERATED SOURCE CODE
============================================================

{bounded_source}

============================================================
REPAIR INSTRUCTIONS
============================================================

Analyze the failure carefully.

Identify the actual root cause from the execution result.

Then rewrite the complete experiment so that the root cause is corrected
while preserving the original scientific experiment.

The repaired experiment must calculate its own results.

Return ONLY the complete corrected Python source code.
""".strip()

        repair_prompt += """

TRACEBACK-DRIVEN IMPORT CHECK
Compare every traceback-reported missing name or module against the complete
source. Preserve imports and definitions that are already present. Add an
import only when it resolves the reported failure and is an appropriate
dependency for the experiment. Do not remove code that depends on a missing
name or install/substitute an unrelated package.
"""

        repaired: Dict[str, Any] = {}

        def request_repair_candidate(prompt: str) -> str:
            nonlocal repaired

            response = _call_llm(
                prompt,
                temperature=0.0,
                model=self.model,
                system_prompt=self.REPAIR_SYSTEM_PROMPT,
                max_tokens=_output_token_limit(
                    "code_generation",
                    self.REPAIR_MAX_TOKENS,
                ),
                reasoning="medium",
            )

            if not isinstance(response, str):
                response = str(response)

            if response.startswith("Error:"):
                raise RuntimeError(response)

            repaired = self.extract_python_source(response)
            if repaired is None:
                try:
                    repaired = self.extract_json(response)
                except ValueError as error:
                    raise ValueError(
                        "LLM repair response did not contain "
                        "valid Python source code."
                    ) from error

            repaired_source = repaired.get("pytorch_code")
            if isinstance(repaired_source, str) and repaired_source.strip():
                repaired["pytorch_code"] = (
                    self._normalise_escaped_python_source(repaired_source)
                )

            self.validate_generated_response(repaired)
            self.validate_experiment_design_compliance(
                specification,
                repaired["pytorch_code"],
            )

            repaired_code = repaired.get("pytorch_code")
            if not isinstance(repaired_code, str) or not repaired_code.strip():
                raise ValueError("LLM repair returned empty Python code.")

            return repaired_code

        repaired_code = ""
        for repair_attempt in range(self.MAX_NOOP_REPAIR_RETRIES + 1):
            prompt = repair_prompt
            if repair_attempt:
                prompt += """

NO-OP RETRY FEEDBACK
The previous repair response was identical to the current source and was
rejected. Re-examine the execution result and traceback above, identify the
specific cause, and make the smallest concrete source change that addresses
it. Do not make arbitrary changes. Return the complete corrected Python file.
"""

            repaired_code = request_repair_candidate(prompt)
            if repaired_code.strip() != generated_code.strip():
                break
        else:
            raise ValueError(
                "LLM returned unchanged code after the bounded no-op repair retry."
            )

        # ----------------------------------------------------
        # Return repaired experiment.
        # ----------------------------------------------------

        return {
            "success": True,
            "model": self.model,
            "model_recommendation": repaired.get(
                "model_recommendation",
                {},
            ),
            "experiment_plan": repaired.get(
                "experiment_plan",
                {},
            ),
            "assumptions": repaired.get(
                "assumptions",
                [],
            ),
            "dependencies": repaired.get(
                "dependencies",
                [],
            ),
            "pytorch_code": repaired_code,
            "generation_seconds": 0.0,
            "errors": [],

            # Preserve scientific metric provenance.
            "evaluation_metrics": specification.get(
                "evaluation_metrics",
                [],
            ),
            "evidence_metric_guidance": (
                reference_metric_guidance
            ),
            "reference_experiment": (
                self._compact_reference_experiment(
                    specification.get(
                        "reference_experiment",
                        {},
                    )
                )
            ),
            "selected_hypothesis": specification.get(
                "selected_hypothesis",
                {},
            ),
        }

    # ========================================================
    # Generate From Components
    # ========================================================

    def generate_from_components(
        self,
        hypothesis: Any,
        research_goal: Optional[Any] = None,
        experiment_specification: Optional[
            Dict[str, Any]
        ] = None,
        dataset_name: str = "5G-NIDD",
        dataset_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate experiment code directly from a Hypothesis and
        ResearchGoal.

        If an experiment specification has already been produced by
        ExperimentOrchestrator, it is used directly.

        When constructing a fallback specification, this method does NOT
        impose universal classification metrics. The actual experiment
        specification should determine the required metrics.
        """
        if experiment_specification is None:
            hypothesis_data = (
                self.serialize_hypothesis(
                    hypothesis
                )
            )

            research_goal_data = (
                self.serialize_research_goal(
                    research_goal
                )
            )

            experiment_specification = {
                "dataset": {
                    "name": dataset_name,
                    "path": dataset_path,
                    "task": None,
                },
                "research_goal": (
                    research_goal_data
                ),
                "selected_hypothesis": (
                    hypothesis_data
                ),
                "scientific_evaluation": {},
                "experiment_design": {
                    "input_source": "selected_ai_co_scientist_hypothesis",
                    "experiment_type": "general_experiment",
                    "preprocessing_required": False,
                    "train_validation_test_split": False,
                    "reproducibility_required": True,
                    "checkpoint_required": False,
                    "training_history_required": False,
                    "training_required": False,
                },
                "code_generation_requirements": {
                    "framework": "PyTorch",
                    "language": "Python",
                    "dataset": dataset_name,
                    "dataset_path": dataset_path,
                    "experiment_type": "general_experiment",
                    "include_preprocessing": False,
                    "include_train_validation_test": False,
                    "include_checkpoint": False,
                    "include_training_history": False,
                    "include_reproducibility": True,
                    "include_evaluation": True,
                },

                # IMPORTANT:
                #
                # Do not universally force accuracy/precision/recall/F1.
                # If this fallback path is specifically being used for a
                # classification experiment, the orchestrator should
                # provide the classification metrics explicitly.
                "evaluation_metrics": [],

                "expected_artifacts": {
                    "metrics": "metrics.json",
                    "summary": "experiment_summary.json",
                },

                # No evidence is available in this fallback path unless
                # the caller supplies a complete experiment specification.
                "reference_experiment": {},
            }

        return self.generate(
            experiment_specification
        )

    # ========================================================
    # Save Generated Code
    # ========================================================

    def save_generated_code(
        self,
        generated_result: Dict[str, Any],
        output_path: Optional[str | Path] = None,
    ) -> Optional[Path]:
        """
        Save generated Python code to a .py file.

        Returns None when generation failed.
        """
        if not generated_result.get(
            "success",
            False,
        ):
            return None

        code = generated_result.get(
            "pytorch_code"
        )

        if not isinstance(
            code,
            str,
        ) or not code.strip():
            return None

        if output_path is not None:
            path = Path(
                output_path
            )
        elif self.output_directory is not None:
            path = (
                self.output_directory
                / "generated_experiment.py"
            )
        else:
            raise ValueError(
                "No output path was supplied and no output directory "
                "was configured."
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            code,
            encoding="utf-8",
        )

        generated_result[
            "generated_code_path"
        ] = str(path)

        return path

    # ========================================================
    # Save Generation Result
    # ========================================================

    def save_generation_result(
        self,
        generated_result: Dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """
        Save the complete CodeGenerationAgent result as JSON.
        """
        path = Path(
            output_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serializable_result = (
            self._to_serializable(
                generated_result
            )
        )

        path.write_text(
            json.dumps(
                serializable_result,
                indent=4,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        return path

    # ========================================================
    # Convenience Run Method
    # ========================================================

    def run(
        self,
        specification: Optional[
            Dict[str, Any]
        ] = None,
        hypothesis: Optional[Any] = None,
        research_goal: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Main public entry point.

        Preferred usage:

            agent.run(
                specification=experiment_specification
            )

        Or:

            agent.run(
                hypothesis=best_hypothesis,
                research_goal=research_goal,
            )
        """
        if specification is not None:
            return self.generate(
                specification
            )

        if hypothesis is None:
            return {
                "success": False,
                "model": self.model,
                "model_recommendation": None,
                "experiment_plan": None,
                "assumptions": [],
                "dependencies": [],
                "pytorch_code": None,
                "generation_seconds": 0.0,
                "errors": [
                    "Either specification or hypothesis "
                    "must be provided."
                ],
            }

        return self.generate_from_components(
            hypothesis=hypothesis,
            research_goal=research_goal,
        )