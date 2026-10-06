"""
Automated Experiment Orchestrator.

This module connects the completed AI Co-Scientist workflow to the
automated experiment pipeline.

Architecture:

    Research Goal
        |
        v
    SupervisorAgent
        |
        +--> Generation
        +--> Reflection
        +--> Ranking
        +--> Evolution
        +--> Reflection (Evolved)
        +--> Ranking
        +--> Proximity
        +--> Meta Review
        |
        v
    Final Active Hypotheses
        |
        v
    ExperimentOrchestrator
        |
        +--> Select final accepted hypothesis
        +--> Extract evidence sources
        +--> Read reference experiments from papers
        +--> Extract evidence-derived evaluation metrics
        +--> Build experiment specification
        +--> Generate PyTorch experiment code
        +--> Execute experiment
        +--> Compare experiment results against paper evidence
        +--> Save artifacts
        |
        v
    Experiment Results

Scientific design:

    Rank #1 Hypothesis
        |
        +--> defines WHAT is being tested
        |
        v
    Evidence Sources / Reference Experiments
        |
        +--> guide methodology
        +--> guide model/design where applicable
        +--> guide evaluation metrics
        +--> provide reference values when explicitly reported
        |
        v
    Dataset + Environment
        |
        +--> determines what can actually be reproduced
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

Important:
- The orchestrator does not create a new hypothesis.
- Paper reference values are never copied into experiment results.
- Evaluation metrics should be derived from the experiment specification,
  selected hypothesis, and supporting evidence rather than a universal
  hard-coded classification metric list.
- If a paper metric cannot be reproduced, the experiment should record
  that limitation rather than inventing a value.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.paper_library import ChromaPaperLibrary

from ..agents_modules.code_generation_agent import (
    CodeGenerationAgent,
    _call_llm,
)

from .experiment_runner import ExperimentRunner
from .experiment_comparator import ExperimentComparator
from .paper_reader import PaperReader

from ..data.dataset_manager import DatasetManager
from ..utils import logger

# import inspect

# print(
#     "ExperimentRunner source:",
#     inspect.getfile(ExperimentRunner),
# )

# print(
#     "ExperimentRunner signature:",
#     inspect.signature(ExperimentRunner.__init__),
# )

# print(
#     "ExperimentRunner.run_generated_result signature:",
#     inspect.signature(
#         ExperimentRunner.run_generated_result
#     ),
# )


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_EXPERIMENT_DIR = PROJECT_ROOT / "app/experiments"
RESULTS_DIR = BASE_EXPERIMENT_DIR / "results"

GENERATED_CODE_DIR = RESULTS_DIR / "generated_code"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
RUNS_DIR = RESULTS_DIR / "runs"


# ============================================================
# Experiment Orchestrator
# ============================================================

class ExperimentOrchestrator:
    """
    Bridge between the AI Co-Scientist and the automated
    experiment pipeline.

    The SupervisorAgent is responsible for:

        Generation
        Reflection
        Ranking
        Evolution
        Reflection of evolved hypotheses
        Ranking
        Proximity
        Meta-review

    This class begins AFTER that workflow has completed.

    Its responsibility is to:

        1. Inspect final ContextMemory.
        2. Identify final accepted hypotheses.
        3. Use the RankingAgent's Elo result to select the best candidate.
        4. Extract evidence sources associated with the hypothesis.
        5. Read reference experiments from supporting papers.
        6. Extract evidence-derived evaluation metrics.
        7. Build a structured experiment specification.
        8. Persist the specification and scientific provenance.
        9. Generate experiment code.
        10. Optionally execute the generated experiment.
        11. Compare experiment results against paper evidence.

    Scientific precedence:

        Rank #1 hypothesis
            ->
        Evidence/reference experiment
            ->
        Dataset/environment constraints
            ->
        Experiment implementation

    The selected hypothesis defines the research question.
    Supporting papers guide implementation and evaluation but
    must not silently replace the hypothesis.
    """

    def __init__(
        self,
        dataset_name: str = "5G-NIDD",
        dataset_path: Optional[str] = None,
        device: str = "cuda",
        python_executable: Optional[str] = None,
        dataset_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the experiment orchestrator.

        Parameters
        ----------
        dataset_name:
            Dataset used by the automated experiment.

        dataset_path:
            Local/offline path to the dataset.

        device:
            PyTorch device, e.g. "cpu" or "cuda".

        python_executable:
            Python executable used to execute generated
            experiment code. Defaults to the current interpreter.
        """

        self.dataset_name = dataset_name

        self.dataset_manager = DatasetManager(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_url=dataset_url,
            cache_dir=cache_dir,
        )

        self.dataset_path = Path(
            self.dataset_manager.get_latest_dataset()
        )

        self.device = device

        self.python_executable = (
            python_executable
            or sys.executable
        )

        self._create_directories()

        self.code_generation_agent = CodeGenerationAgent(
            output_directory=GENERATED_CODE_DIR
        )

        self.experiment_runner = ExperimentRunner(
            output_directory=RUNS_DIR,
            python_executable=self.python_executable,
        )

        self.paper_library = ChromaPaperLibrary()

        self.paper_reader = PaperReader(
            paper_library=self.paper_library,
            llm_callable=_call_llm,
        )

        self.experiment_comparator = ExperimentComparator(
            paper_library=self.paper_library
        )

    # ========================================================
    # Directory Management
    # ========================================================

    def _create_directories(self) -> None:
        """
        Create all experiment directories.
        """

        directories = [
            BASE_EXPERIMENT_DIR,
            RESULTS_DIR,
            GENERATED_CODE_DIR,
            CHECKPOINT_DIR,
            METRICS_DIR,
            VISUALIZATION_DIR,
            RUNS_DIR,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ========================================================
    # Utility Helpers
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: Optional[float] = None,
    ) -> Optional[float]:
        """
        Convert a value to float safely.
        """

        try:
            result = float(value)

            if not math.isfinite(result):
                return default

            return result

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_string(
        value: Any,
        default: str = "",
    ) -> str:
        """
        Convert a value to a safe string.
        """

        if value is None:
            return default

        return str(value)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """
        Convert common Python/model objects into JSON-safe data.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, dict):
            return {
                str(key): ExperimentOrchestrator._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                ExperimentOrchestrator._json_safe(item)
                for item in value
            ]

        if hasattr(value, "model_dump"):
            try:
                return ExperimentOrchestrator._json_safe(
                    value.model_dump()
                )
            except Exception:
                pass

        if hasattr(value, "to_dict"):
            try:
                return ExperimentOrchestrator._json_safe(
                    value.to_dict()
                )
            except Exception:
                pass

        return str(value)

    # ========================================================
    # Metric Helpers
    # ========================================================

    @staticmethod
    def _normalise_metric_name(
        metric_name: Any,
    ) -> str:
        """
        Normalize a metric name for deduplication while preserving
        the human-readable metric name elsewhere.
        """

        return re.sub(
            r"[^a-z0-9]+",
            "_",
            str(metric_name).strip().lower(),
        ).strip("_")

    @staticmethod
    def _flatten_metric_definition(value: Any) -> str:
        """
        Collapse nested metric metadata into a searchable string.

        The goal is to let metric semantics come from the actual metric
        definition, not from a fixed built-in list of metric names.
        """

        if value is None:
            return ""

        if isinstance(value, (str, int, float, bool)):
            return str(value).lower()

        if isinstance(value, (list, tuple, set)):
            return " ".join(
                ExperimentOrchestrator._flatten_metric_definition(item)
                for item in value
            )

        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                parts.append(str(key).lower())
                parts.append(
                    ExperimentOrchestrator._flatten_metric_definition(item)
                )
            return " ".join(parts)

        return str(value).lower()

    @classmethod
    def _metric_semantic_signal(
        cls,
        metric_name: str,
        definition: Any,
    ) -> str:
        """
        Build a semantic signal from metric name + metadata.

        This is intentionally broader than a hard-coded metric-name list and
        allows custom metrics with scientifically meaningful definitions to be
        classified by their actual semantics.
        """

        text = " ".join(
            part for part in (
                str(metric_name).lower(),
                cls._flatten_metric_definition(definition),
            )
            if part
        )

        return re.sub(r"[^a-z0-9_\s]+", " ", text)

    @classmethod
    def _looks_like_model_metric(
        cls,
        metric_name: str,
        definition: Any,
    ) -> bool:
        """
        Return True when a metric is likely a model-quality or prediction
        metric rather than a system benchmark metric.
        """

        signal = cls._metric_semantic_signal(metric_name, definition)

        if not signal:
            return False

        performance_tokens = (
            "score",
            "error",
            "loss",
            "quality",
            "prediction",
            "classification",
            "detection",
            "regression",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "mcc",
            "sensitivity",
            "specificity",
            "balanced",
            "reward",
            "risk",
            "confidence",
            "calibration",
            "hit",
            "match",
            "rate",
        )

        return any(token in signal for token in performance_tokens)

    @classmethod
    def _looks_like_system_metric(
        cls,
        metric_name: str,
        definition: Any,
    ) -> bool:
        """
        Return True when a metric describes runtime/system behavior rather than
        the model's predictive quality.
        """

        signal = cls._metric_semantic_signal(metric_name, definition)

        if not signal:
            return False

        system_tokens = (
            "latency",
            "delay",
            "throughput",
            "bandwidth",
            "runtime",
            "execution",
            "response",
            "overhead",
            "resource",
            "cpu",
            "gpu",
            "memory",
            "energy",
            "time",
            "cost",
            "traffic",
            "packet",
            "message",
            "processing",
            "network",
            "storage",
        )

        return any(token in signal for token in system_tokens)

    @classmethod
    def _extract_reference_metric_requirements(
        cls,
        reference_experiment: Any,
    ) -> Dict[str, Any]:
        """
        Extract evaluation guidance from PaperReader output.

        Returns
        -------
        dict
            {
                "metrics": [...],
                "metric_definitions": {...},
                "reference_metrics": {...}
            }

        Important:
        - metrics are metric names reported or identified by the paper.
        - metric_definitions describe meaning/unit/direction when available.
        - reference_metrics contain numerical values explicitly reported
          by the paper.
        - No reference value is treated as an experiment result.
        """

        result: Dict[str, Any] = {
            "metrics": [],
            "metric_definitions": {},
            "reference_metrics": {},
        }

        if not isinstance(reference_experiment, dict):
            return result

        sources = reference_experiment.get(
            "sources",
            [],
        )

        if not isinstance(sources, list):
            return result

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

            # ------------------------------------------------
            # Metric names
            # ------------------------------------------------

            metrics = details.get(
                "metrics",
                [],
            )

            if isinstance(metrics, list):
                for metric in metrics:
                    if not isinstance(metric, str):
                        continue

                    metric = metric.strip()

                    if not metric:
                        continue

                    key = cls._normalise_metric_name(
                        metric
                    )

                    if key and key not in seen_metrics:
                        result["metrics"].append(metric)
                        seen_metrics.add(key)

            # ------------------------------------------------
            # Metric definitions
            # ------------------------------------------------

            definitions = details.get(
                "metric_definitions",
                {},
            )

            if isinstance(definitions, dict):
                for name, definition in definitions.items():

                    name = str(name).strip()

                    if not name:
                        continue

                    key = cls._normalise_metric_name(
                        name
                    )

                    existing_key = next(
                        (
                            cls._normalise_metric_name(existing)
                            for existing in result[
                                "metric_definitions"
                            ]
                        ),
                        None,
                    )

                    if key and key not in {
                        cls._normalise_metric_name(existing)
                        for existing in result[
                            "metric_definitions"
                        ]
                    }:
                        result[
                            "metric_definitions"
                        ][name] = cls._json_safe(
                            definition
                        )

            # ------------------------------------------------
            # Explicit numerical reference values
            # ------------------------------------------------

            reference_metrics = details.get(
                "reference_metrics",
                {},
            )

            if isinstance(reference_metrics, dict):
                for name, value in reference_metrics.items():

                    name = str(name).strip()

                    if not name:
                        continue

                    key = cls._normalise_metric_name(
                        name
                    )

                    existing_keys = {
                        cls._normalise_metric_name(existing)
                        for existing in result[
                            "reference_metrics"
                        ]
                    }

                    if key and key not in existing_keys:
                        result[
                            "reference_metrics"
                        ][name] = cls._json_safe(
                            value
                        )

        return result

    @classmethod
    def _build_evaluation_guidance(
        cls,
        specification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build evaluation guidance for CodeGenerationAgent.

        Evidence metrics are classified according to whether they
        are directly reproducible, conditionally comparable, or
        reference-only for the current experiment.

        Paper reference values remain separate from experiment
        evaluation metrics.
        """

        explicit_metrics = specification.get(
            "evaluation_metrics",
            [],
        )

        if not isinstance(explicit_metrics, list):
            explicit_metrics = []

        reference_guidance = (
            cls._extract_reference_metric_requirements(
                specification.get(
                    "reference_experiment",
                    {},
                )
            )
        )

        evidence_metrics = reference_guidance.get(
            "metrics",
            [],
        )

        definitions = {}

        specification_definitions = specification.get(
            "evaluation_metric_definitions",
            {},
        )

        if isinstance(specification_definitions, dict):
            definitions.update(
                cls._json_safe(
                    specification_definitions
                )
            )

        definitions.update(
            reference_guidance.get(
                "metric_definitions",
                {},
            )
        )

        reference_metrics = {}

        specification_reference_metrics = specification.get(
            "reference_metrics",
            {},
        )

        if isinstance(specification_reference_metrics, dict):
            reference_metrics.update(
                cls._json_safe(
                    specification_reference_metrics
                )
            )

        reference_metrics.update(
            reference_guidance.get(
                "reference_metrics",
                {},
            )
        )

        experiment_type = specification.get(
            "experiment_design",
            {},
        ).get(
            "experiment_type",
            specification.get(
                "experiment",
                {},
            ).get(
                "experiment_type",
                "general_experiment",
            ),
        )

        metric_classification = (
            cls._classify_evidence_metrics(
                experiment_type=experiment_type,
                evidence_metrics=evidence_metrics,
                metric_definitions=definitions,
                reference_metrics=reference_metrics,
            )
        )

        # --------------------------------------------------------
        # Only directly reproducible evidence metrics become
        # automatic evaluation candidates.
        # --------------------------------------------------------

        reproducible_metrics = (
            metric_classification.get(
                "directly_reproducible",
                [],
            )
        )

        conditional_metrics = (
            metric_classification.get(
                "conditionally_comparable",
                [],
            )
        )

        preferred_comparison_metrics = (
            cls._merge_metric_names(
                explicit_metrics,
                reproducible_metrics,
            )
        )

        return {
            "required_by_specification": cls._json_safe(
                explicit_metrics
            ),

            "evidence_metrics": cls._json_safe(
                evidence_metrics
            ),

            "directly_reproducible_metrics": cls._json_safe(
                reproducible_metrics
            ),

            "conditionally_comparable_metrics": cls._json_safe(
                conditional_metrics
            ),

            "reference_only_metrics": cls._json_safe(
                metric_classification.get(
                    "reference_only",
                    [],
                )
            ),

            "preferred_comparison_metrics": cls._json_safe(
                preferred_comparison_metrics
            ),

            "metric_definitions": definitions,

            "reference_metrics": reference_metrics,

            "metric_roles": cls._json_safe(
                metric_classification.get(
                    "metric_roles",
                    {},
                )
            ),

            "policy": {
                "same_metrics_preferred": True,
                "never_copy_reference_values": True,
                "never_fabricate_metrics": True,
                "allow_proxy_metrics": True,
                "require_proxy_label": True,
                "reference_only_metrics_are_not_required": True,
                "conditional_metrics_require_protocol_compatibility": True,
            },
        }
    
    # ========================================================
    # Experiment ID
    # ========================================================

    def create_experiment_id(
        self,
        hypothesis_id: Optional[str] = None,
    ) -> str:
        """
        Create a unique experiment identifier.
        """

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        if hypothesis_id:
            safe_id = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                str(hypothesis_id),
            ).strip("_")

            if safe_id:
                return f"{safe_id}_{timestamp}"

        return f"experiment_{timestamp}"

    # ========================================================
    # Context / Hypothesis Access
    # ========================================================

    def get_active_hypotheses(
        self,
        context: Any,
    ) -> List[Any]:
        """
        Retrieve active hypotheses from ContextMemory.
        """

        if context is None:
            return []

        getter = getattr(
            context,
            "get_active_hypotheses",
            None,
        )

        if not callable(getter):
            return []

        try:
            hypotheses = getter()
        except Exception:
            return []

        if hypotheses is None:
            return []

        return list(hypotheses)

    @staticmethod
    def _get_hypothesis_value(
        hypothesis: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Read a value from either a hypothesis dictionary or object.
        """

        if hypothesis is None:
            return default

        if isinstance(hypothesis, dict):
            return hypothesis.get(
                key,
                default,
            )

        return getattr(
            hypothesis,
            key,
            default,
        )

    # ========================================================
    # Reference Experiment
    # ========================================================

    def extract_reference_experiment(
        self,
        hypothesis: Any,
    ) -> Dict[str, Any]:
        """
        Read the evidence papers associated with a hypothesis and
        extract experiment information for code generation.

        The extracted information is reference material only.

        It can guide:
            - methodology
            - model/design
            - baselines
            - assumptions
            - evaluation metrics
            - reported reference values

        It does not replace the selected hypothesis.
        """

        if hypothesis is None:
            return {}

        evidence_sources = self._get_hypothesis_value(
            hypothesis,
            "evidence_sources",
            default=[],
        )

        if not evidence_sources:
            logger.info(
                "No evidence sources found for the selected hypothesis."
            )
            return {}

        if isinstance(evidence_sources, dict):
            evidence_sources = [evidence_sources]

        reference_sources: List[Dict[str, Any]] = []

        for source in evidence_sources:

            paper_url = self._extract_evidence_url(
                source
            )

            if not paper_url:
                logger.warning(
                    "Evidence source does not contain a usable URL."
                )
                continue

            try:
                logger.info(
                    "Extracting experiment reference from: %s",
                    paper_url,
                )

                reference = (
                    self.paper_reader.read_experiment_reference(
                        paper_url,
                        source=source,
                    )
                )

                if not reference:
                    logger.warning(
                        "No experiment information extracted from: %s",
                        paper_url,
                    )
                    continue

                reference_sources.append(
                    reference
                )

            except Exception as error:
                logger.warning(
                    "Failed to read evidence paper %s: %s",
                    paper_url,
                    error,
                )

        if not reference_sources:
            return {}

        reference_experiment = {
            "available": True,
            "source_count": len(reference_sources),
            "sources": reference_sources,
        }

        # NEW:
        # Extract a compact summary of the metrics immediately so
        # downstream components do not need to rediscover them.
        metric_guidance = (
            self._extract_reference_metric_requirements(
                reference_experiment
            )
        )

        reference_experiment[
            "evaluation_guidance"
        ] = metric_guidance

        return self._json_safe(
            reference_experiment
        )

    # ========================================================
    # Reflection Routing
    # ========================================================

    def get_accepted_hypotheses(
        self,
        context: Any,
    ) -> List[Any]:
        """
        Retrieve hypotheses that Reflection explicitly accepted.
        """

        accepted: List[Any] = []

        for hypothesis in self.get_active_hypotheses(
            context
        ):

            if not self._is_valid_hypothesis(
                hypothesis
            ):
                continue

            report = self._get_hypothesis_value(
                hypothesis,
                "reflection_report",
                None,
            )

            if report is None:
                continue

            recommendation = (
                self._safe_string(
                    self._get_hypothesis_value(
                        report,
                        "recommendation",
                        "",
                    )
                )
                .strip()
                .upper()
            )

            if recommendation == "ACCEPT":
                accepted.append(
                    hypothesis
                )

        return accepted

    # ========================================================
    # Final Experiment Candidates
    # ========================================================

    def get_experiment_candidates(
        self,
        context: Any,
    ) -> List[Any]:
        """
        Return final active hypotheses explicitly accepted
        by the ReflectionAgent.
        """

        return self.get_accepted_hypotheses(
            context
        )

    def _is_valid_hypothesis(
        self,
        hypothesis: Any,
    ) -> bool:
        """
        Check whether a hypothesis contains the minimum
        information required by the experiment layer.
        """

        if hypothesis is None:
            return False

        is_active = self._get_hypothesis_value(
            hypothesis,
            "is_active",
            False,
        )

        if not is_active:
            return False

        text = self._safe_string(
            self._get_hypothesis_value(
                hypothesis,
                "text",
                "",
            )
        ).strip()

        return bool(text)

    # ========================================================
    # Ranking
    # ========================================================

    def get_ranked_hypotheses(
        self,
        context: Any,
    ) -> List[Any]:
        """
        Order final experiment candidates by the Elo score
        produced by the RankingAgent.

        The orchestrator does not run another tournament.
        """

        candidates = self.get_experiment_candidates(
            context
        )

        def elo_key(
            hypothesis: Any,
        ) -> float:

            score = self._safe_float(
                self._get_hypothesis_value(
                    hypothesis,
                    "elo_score",
                    1200.0,
                ),
                default=1200.0,
            )

            return (
                score
                if score is not None
                else 1200.0
            )

        return sorted(
            candidates,
            key=elo_key,
            reverse=True,
        )

    # ========================================================
    # Select Best Hypothesis
    # ========================================================

    def select_best_hypothesis(
        self,
        context: Any,
    ) -> Optional[Any]:
        """
        Select the highest-Elo hypothesis from the final
        Reflection-accepted candidates.
        """

        ranked = self.get_ranked_hypotheses(
            context
        )

        if not ranked:
            return None

        return ranked[0]

    # ========================================================
    # Reflection Serialization
    # ========================================================

    def serialize_reflection_report(
        self,
        report: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Serialize the actual ReflectionReport model.
        """

        if report is None:
            return None

        if hasattr(report, "model_dump"):
            try:
                return self._json_safe(
                    report.model_dump()
                )
            except Exception:
                pass

        return {
            "alignment_score": getattr(
                report,
                "alignment_score",
                None,
            ),
            "novelty_score": getattr(
                report,
                "novelty_score",
                None,
            ),
            "feasibility_score": getattr(
                report,
                "feasibility_score",
                None,
            ),
            "plausibility_score": getattr(
                report,
                "plausibility_score",
                None,
            ),
            "testability_score": getattr(
                report,
                "testability_score",
                None,
            ),
            "evidence_quality_score": getattr(
                report,
                "evidence_quality_score",
                None,
            ),
            "expected_research_value_score": getattr(
                report,
                "expected_research_value_score",
                None,
            ),
            "strengths": getattr(
                report,
                "strengths",
                [],
            ),
            "weaknesses": getattr(
                report,
                "weaknesses",
                [],
            ),
            "recommendation": getattr(
                report,
                "recommendation",
                None,
            ),
            "claims": self._json_safe(
                getattr(
                    report,
                    "claims",
                    [],
                )
            ),
            "overall_confidence": getattr(
                report,
                "overall_confidence",
                None,
            ),
        }

    # ========================================================
    # Hypothesis Serialization
    # ========================================================

    def serialize_hypothesis(
        self,
        hypothesis: Any,
    ) -> Dict[str, Any]:
        """
        Serialize the actual Hypothesis structure used by
        app/models.py.
        """

        if hypothesis is None:
            return {}

        return {
            "hypothesis_id": self._get_hypothesis_value(
                hypothesis,
                "hypothesis_id",
                None,
            ),
            "title": self._get_hypothesis_value(
                hypothesis,
                "title",
                None,
            ),
            "text": self._get_hypothesis_value(
                hypothesis,
                "text",
                None,
            ),
            "elo_score": self._get_hypothesis_value(
                hypothesis,
                "elo_score",
                None,
            ),
            "novelty_review": self._get_hypothesis_value(
                hypothesis,
                "novelty_review",
                None,
            ),
            "feasibility_review": self._get_hypothesis_value(
                hypothesis,
                "feasibility_review",
                None,
            ),
            "review_comments": self._json_safe(
                self._get_hypothesis_value(
                    hypothesis,
                    "review_comments",
                    [],
                )
            ),
            "references": self._json_safe(
                self._get_hypothesis_value(
                    hypothesis,
                    "references",
                    [],
                )
            ),
            "review_reference_ids": self._json_safe(
                self._get_hypothesis_value(
                    hypothesis,
                    "review_reference_ids",
                    [],
                )
            ),
            "is_active": self._get_hypothesis_value(
                hypothesis,
                "is_active",
                None,
            ),
            "deactivation_reason": self._get_hypothesis_value(
                hypothesis,
                "deactivation_reason",
                None,
            ),
            "parent_ids": self._json_safe(
                self._get_hypothesis_value(
                    hypothesis,
                    "parent_ids",
                    [],
                )
            ),
            "evolution_strategy": self._get_hypothesis_value(
                hypothesis,
                "evolution_strategy",
                None,
            ),
            "evidence_source_ids": self._json_safe(
                self._get_hypothesis_value(
                    hypothesis,
                    "evidence_source_ids",
                    [],
                )
            ),
            "evidence_sources": self._json_safe(
                self._get_hypothesis_value(
                    hypothesis,
                    "evidence_sources",
                    [],
                )
            ),
            "audit_score": self._get_hypothesis_value(
                hypothesis,
                "audit_score",
                None,
            ),
            "audit_verdict": self._get_hypothesis_value(
                hypothesis,
                "audit_verdict",
                None,
            ),
            "audit_report": self._json_safe(
                self._get_hypothesis_value(
                    hypothesis,
                    "audit_report",
                    {},
                )
            ),
            "reflection_report": self.serialize_reflection_report(
                self._get_hypothesis_value(
                    hypothesis,
                    "reflection_report",
                    None,
                )
            ),
        }

    # ========================================================
    # Research Goal Serialization
    # ========================================================

    def serialize_research_goal(
        self,
        research_goal: Any,
    ) -> Dict[str, Any]:

        if research_goal is None:
            return {}

        return {
            "description": self._get_hypothesis_value(
                research_goal,
                "description",
                "",
            ),
            "preferences": self._get_hypothesis_value(
                research_goal,
                "preferences",
                None,
            ),
            "idea_attributes": self._get_hypothesis_value(
                research_goal,
                "idea_attributes",
                None,
            ),
            "constraints": self._json_safe(
                self._get_hypothesis_value(
                    research_goal,
                    "constraints",
                    {},
                )
            ),
            "llm_model": self._get_hypothesis_value(
                research_goal,
                "llm_model",
                None,
            ),
            "query_rewrite_model": self._get_hypothesis_value(
                research_goal,
                "query_rewrite_model",
                None,
            ),
            "num_hypotheses": self._get_hypothesis_value(
                research_goal,
                "num_hypotheses",
                None,
            ),
            "generation_temperature": self._get_hypothesis_value(
                research_goal,
                "generation_temperature",
                None,
            ),
            "reflection_temperature": self._get_hypothesis_value(
                research_goal,
                "reflection_temperature",
                None,
            ),
            "elo_k_factor": self._get_hypothesis_value(
                research_goal,
                "elo_k_factor",
                None,
            ),
            "top_k_hypotheses": self._get_hypothesis_value(
                research_goal,
                "top_k_hypotheses",
                None,
            ),
        }

    # ========================================================
    # Evidence URL
    # ========================================================

    @classmethod
    def _extract_evidence_url(
        cls,
        evidence_source: Any,
    ) -> Optional[str]:
        """
        Extract a paper URL from an evidence-source object
        or dictionary.
        """

        if evidence_source is None:
            return None

        if isinstance(evidence_source, str):
            return (
                evidence_source.strip()
                or None
            )

        if isinstance(evidence_source, dict):

            possible_keys = (
                "url",
                "paper_url",
                "source_url",
                "link",
                "uri",
            )

            for key in possible_keys:

                value = evidence_source.get(
                    key
                )

                if value:
                    return str(value).strip()

            return None

        possible_attributes = (
            "url",
            "paper_url",
            "source_url",
            "link",
            "uri",
        )

        for attribute in possible_attributes:

            value = getattr(
                evidence_source,
                attribute,
                None,
            )

            if value:
                return str(value).strip()

        return None

    # ========================================================
    # Tournament Serialization
    # ========================================================

    def serialize_tournament_results(
        self,
        context: Any,
    ) -> List[Dict[str, Any]]:
        """
        Preserve the RankingAgent's tournament decisions.
        """

        if context is None:
            return []

        results = getattr(
            context,
            "tournament_results",
            [],
        )

        if not results:
            return []

        return self._json_safe(
            list(results)
        )

    # ========================================================
    # Proximity Serialization
    # ========================================================

    def serialize_proximity_analysis(
        self,
        context: Any,
    ) -> Dict[str, Any]:
        """
        Preserve the latest ProximityAgent output.
        """

        if context is None:
            return {}

        proximity = getattr(
            context,
            "proximity_analysis",
            {},
        )

        if proximity is None:
            return {}

        return self._json_safe(
            proximity
        )

    # ========================================================
    # Meta Review Serialization
    # ========================================================

    def serialize_meta_review(
        self,
        context: Any,
    ) -> List[Dict[str, Any]]:
        """
        Preserve MetaReviewAgent feedback.
        """

        if context is None:
            return []

        feedback = getattr(
            context,
            "meta_review_feedback",
            [],
        )

        if feedback is None:
            return []

        return self._json_safe(
            list(feedback)
        )

    @classmethod
    def _extract_hypothesis_metric_requirements(
        cls,
        hypothesis: Any,
    ) -> List[str]:
        """
        Extract explicitly structured evaluation metrics from
        the selected hypothesis.

        Metrics are only read from dedicated fields. They are
        not inferred from free-form hypothesis text.
        """

        if hypothesis is None:
            return []

        candidate_fields = (
            "evaluation_metrics",
            "metrics",
            "metric_requirements",
            "experiment_metrics",
        )

        metrics: List[str] = []
        seen = set()

        for field in candidate_fields:
            value = cls._get_hypothesis_value(
                hypothesis,
                field,
                None,
            )

            if isinstance(value, str):
                value = [value]

            if not isinstance(value, list):
                continue

            for metric in value:
                if not isinstance(metric, str):
                    continue

                name = metric.strip()

                if not name:
                    continue

                key = cls._normalise_metric_name(name)

                if key and key not in seen:
                    metrics.append(name)
                    seen.add(key)

        return metrics

    @classmethod
    def _merge_metric_names(
        cls,
        *metric_lists: Any,
    ) -> List[str]:
        """
        Merge metric-name lists while preserving order and
        removing duplicates.
        """

        merged: List[str] = []
        seen = set()

        for metric_list in metric_lists:

            if isinstance(metric_list, str):
                metric_list = [metric_list]

            if not isinstance(metric_list, list):
                continue

            for metric in metric_list:

                if not isinstance(metric, str):
                    continue

                name = metric.strip()

                if not name:
                    continue

                key = cls._normalise_metric_name(name)

                if key and key not in seen:
                    merged.append(name)
                    seen.add(key)

        return merged

    @classmethod
    def _classify_evidence_metrics(
        cls,
        experiment_type: str,
        evidence_metrics: Any,
        metric_definitions: Any = None,
        reference_metrics: Any = None,
    ) -> Dict[str, Any]:
        """
        Classify evidence-derived metrics according to whether they
        are appropriate for the current automated experiment.

        The paper's metrics remain scientific reference information.
        Only metrics that can reasonably be measured by the current
        experiment are promoted to evaluation metrics.

        Returns
        -------
        dict
            {
                "directly_reproducible": [...],
                "conditionally_comparable": [...],
                "reference_only": [...],
                "metric_roles": {...}
            }

        Important:
        - This function does not copy paper reference values.
        - A metric being reported by a paper does not automatically
          make it an evaluation requirement.
        - Conditional metrics require an explicit proxy/comparability
          explanation downstream.
        """

        if not isinstance(evidence_metrics, list):
            evidence_metrics = []

        if not isinstance(metric_definitions, dict):
            metric_definitions = {}

        if not isinstance(reference_metrics, dict):
            reference_metrics = {}

        directly_reproducible: List[str] = []
        conditionally_comparable: List[str] = []
        reference_only: List[str] = []

        metric_roles: Dict[str, Dict[str, Any]] = {}

        for metric in evidence_metrics:

            if not isinstance(metric, str):
                continue

            name = metric.strip()

            if not name:
                continue

            normalized = cls._normalise_metric_name(
                name
            )

            definition = {}

            for definition_name, definition_value in (
                metric_definitions.items()
            ):
                if (
                    cls._normalise_metric_name(
                        definition_name
                    )
                    == normalized
                ):
                    definition = definition_value
                    break

            metric_text = (
                cls._metric_semantic_signal(
                    metric_name=name,
                    definition=definition,
                )
            )

            has_ml_semantics = cls._looks_like_model_metric(
                metric_name=name,
                definition=definition,
            )

            has_system_semantics = cls._looks_like_system_metric(
                metric_name=name,
                definition=definition,
            )

            # ----------------------------------------------------
            # ML training experiment
            # ----------------------------------------------------

            if experiment_type == "ml_training":
                # Detection rate is potentially comparable, but its
                # definition must match the paper's detection setup.
                if (
                    normalized == "detection_rate"
                    or normalized.startswith(
                        "detection_rate_"
                    )
                ):
                    conditionally_comparable.append(name)

                    metric_roles[name] = {
                        "role": "conditionally_comparable",
                        "reason": (
                            "Detection rate can be computed from the "
                            "automated experiment, but direct comparison "
                            "requires compatible attack definitions, "
                            "labels, and evaluation protocol."
                        ),
                    }

                # [other-model-metric] metrics of ANOTHER model family in the paper (LLM prompting setups,
                # API deployments) cannot be reproduced by this experiment: keep them as reference only.
                elif any(
                    token in str(name).lower()
                    for token in ("llm", "zero_shot", "few_shot", "zeroshot", "fewshot", "api_based", "prompting")
                ):
                    reference_only.append(name)
                    metric_roles[name] = {
                        "role": "reference_only",
                        "reason": (
                            "The metric belongs to a different model family reported by the paper "
                            "(for example an LLM prompting setup) and cannot be reproduced by this experiment."
                        ),
                    }
                # Standard ML metrics can be measured directly.
                elif has_ml_semantics and not has_system_semantics:
                    directly_reproducible.append(name)

                    metric_roles[name] = {
                        "role": "directly_reproducible",
                        "reason": (
                            "The metric represents ML model "
                            "classification or detection performance "
                            "and can be computed from model predictions."
                        ),
                    }

                # Runtime/system measurements are not automatically
                # reproducible by a dataset-based PyTorch experiment.
                elif has_system_semantics:
                    reference_only.append(name)

                    metric_roles[name] = {
                        "role": "reference_only",
                        "reason": (
                            "The metric describes system/runtime "
                            "performance rather than the ML model's "
                            "predictive performance."
                        ),
                    }

                else:
                    reference_only.append(name)

                    metric_roles[name] = {
                        "role": "reference_only",
                        "reason": (
                            "The metric cannot be established as "
                            "directly reproducible from the current "
                            "experiment specification."
                        ),
                    }

            # ----------------------------------------------------
            # Measurement benchmark
            # ----------------------------------------------------

            elif experiment_type == "measurement_benchmark":
                if has_system_semantics:
                    directly_reproducible.append(name)

                    metric_roles[name] = {
                        "role": "directly_reproducible",
                        "reason": (
                            "The metric describes a system or runtime "
                            "measurement appropriate for a benchmark."
                        ),
                    }

                else:
                    reference_only.append(name)

                    metric_roles[name] = {
                        "role": "reference_only",
                        "reason": (
                            "The metric is not established as a "
                            "directly measurable benchmark metric."
                        ),
                    }

            # ----------------------------------------------------
            # Unknown/general experiment
            # ----------------------------------------------------

            else:

                reference_only.append(name)

                metric_roles[name] = {
                    "role": "reference_only",
                    "reason": (
                        "The current experiment type does not establish "
                        "a reliable reproduction method for this metric."
                    ),
                }

        # --------------------------------------------------------
        # Reference values are intentionally kept separate.
        # --------------------------------------------------------

        reference_only_set = {
            cls._normalise_metric_name(metric)
            for metric in reference_only
        }

        conditionally_comparable_set = {
            cls._normalise_metric_name(metric)
            for metric in conditionally_comparable
        }

        directly_reproducible_set = {
            cls._normalise_metric_name(metric)
            for metric in directly_reproducible
        }

        return {
            "directly_reproducible": directly_reproducible,
            "conditionally_comparable": conditionally_comparable,
            "reference_only": reference_only,
            "metric_roles": metric_roles,
            "reference_values": cls._json_safe(
                reference_metrics
            ),
            "counts": {
                "directly_reproducible": len(
                    directly_reproducible_set
                ),
                "conditionally_comparable": len(
                    conditionally_comparable_set
                ),
                "reference_only": len(
                    reference_only_set
                ),
            },
        }
    
    @classmethod
    def _infer_experiment_type(
        cls,
        hypothesis: Any,
        reference_experiment: Dict[str, Any],
    ) -> str:
        """
        Infer the primary experiment type.

        Scientific precedence:
            1. Selected hypothesis
            2. Supporting evidence

        Evidence must not override the primary experimental
        objective defined by the selected Rank #1 hypothesis.
        """

        hypothesis_text = cls._safe_string(
            cls._get_hypothesis_value(
                hypothesis,
                "text",
                "",
            )
        ).lower()

        hypothesis_title = cls._safe_string(
            cls._get_hypothesis_value(
                hypothesis,
                "title",
                "",
            )
        ).lower()

        hypothesis_text_combined = (
            f"{hypothesis_title} {hypothesis_text}"
        )

        # --------------------------------------------------------
        # 1. Determine the primary experiment type from the
        #    selected hypothesis FIRST.
        # --------------------------------------------------------

        ml_keywords = (
            "classification",
            "intrusion detection",
            "anomaly detection",
            "anomaly detector",
            "classifier",
            "machine learning",
            "deep learning",
            "neural network",
            "lstm",
            "bilstm",
            "gru",
            "transformer",
            "pytorch",
            "train",
            "training",
            "prediction",
            "predict",
            "detect",
            "detector",
        )

        if any(
            keyword in hypothesis_text_combined
            for keyword in ml_keywords
        ):
            return "ml_training"

        # --------------------------------------------------------
        # 2. Only if the hypothesis is not clearly an ML task,
        #    inspect the supporting evidence for a benchmark.
        # --------------------------------------------------------

        evidence_text_parts: List[str] = []

        for source in reference_experiment.get(
            "sources",
            [],
        ):
            if not isinstance(source, dict):
                continue

            details = source.get(
                "experiment_details",
                {},
            )

            if not isinstance(details, dict):
                continue

            for field in (
                "objective",
                "methodology",
                "setup",
                "experiment_type",
                "task",
            ):
                value = details.get(field)

                if isinstance(value, str):
                    evidence_text_parts.append(value)

                elif isinstance(value, list):
                    evidence_text_parts.extend(
                        str(item)
                        for item in value
                        if item is not None
                    )

        evidence_text = " ".join(
            evidence_text_parts
        ).lower()

        # --------------------------------------------------------
        # 3. Evidence-based measurement benchmark.
        # --------------------------------------------------------

        measurement_keywords = (
            "latency",
            "delay",
            "overhead",
            "throughput",
            "bandwidth",
            "tunnel setup",
            "setup time",
            "response time",
            "execution time",
            "stability",
            "benchmark",
            "performance measurement",
        )

        if any(
            keyword in evidence_text
            for keyword in measurement_keywords
        ):
            return "measurement_benchmark"

        return "general_experiment"

    @classmethod
    def _build_experiment_design(
        self,
        experiment_type: str,
    ) -> Dict[str, Any]:
        """
        Build experiment-design requirements according to the
        actual experiment type.

        Training-specific requirements must not be imposed on
        measurement or benchmarking experiments.
        """

        if experiment_type == "ml_training":
            return {
                "experiment_type": experiment_type,
                "preprocessing_required": True,
                "train_validation_test_split": True,
                "reproducibility_required": True,
                "checkpoint_required": True,
                "training_history_required": True,
                "training_required": True,
            }

        if experiment_type == "measurement_benchmark":
            return {
                "experiment_type": experiment_type,
                "preprocessing_required": False,
                "train_validation_test_split": False,
                "reproducibility_required": True,
                "checkpoint_required": False,
                "training_history_required": False,
                "training_required": False,
            }

        return {
            "experiment_type": experiment_type,
            "preprocessing_required": False,
            "train_validation_test_split": False,
            "reproducibility_required": True,
            "checkpoint_required": False,
            "training_history_required": False,
            "training_required": False,
        }
    
    @classmethod
    def _validate_experiment_specification_consistency(
        cls,
        specification: Dict[str, Any],
    ) -> List[str]:
        """
        Validate that the experiment specification is internally
        consistent with the inferred experiment type.

        This is a structural validation layer. It does not judge
        whether the scientific hypothesis itself is correct.

        Returns
        -------
        list[str]
            Validation errors. An empty list means the specification
            is internally consistent.
        """

        errors: List[str] = []

        if not isinstance(specification, dict):
            return [
                "Experiment specification must be a dictionary."
            ]

        experiment = specification.get(
            "experiment",
            {},
        )

        if not isinstance(experiment, dict):
            experiment = {}

        experiment_design = specification.get(
            "experiment_design",
            {},
        )

        if not isinstance(experiment_design, dict):
            errors.append(
                "experiment_design must be a dictionary."
            )
            experiment_design = {}

        code_generation_requirements = specification.get(
            "code_generation_requirements",
            {},
        )

        if not isinstance(
            code_generation_requirements,
            dict,
        ):
            errors.append(
                "code_generation_requirements must be a dictionary."
            )
            code_generation_requirements = {}

        dataset = specification.get(
            "dataset",
            {},
        )

        if not isinstance(dataset, dict):
            errors.append(
                "dataset must be a dictionary."
            )
            dataset = {}

        experiment_type = experiment_design.get(
            "experiment_type"
        )

        code_generation_type = (
            code_generation_requirements.get(
                "experiment_type"
            )
        )

        # ----------------------------------------------------
        # Experiment type consistency
        # ----------------------------------------------------

        if not experiment_type:
            errors.append(
                "experiment_design.experiment_type is missing."
            )

        if (
            code_generation_type
            and experiment_type
            and code_generation_type != experiment_type
        ):
            errors.append(
                "experiment_type is inconsistent between "
                "experiment_design and code_generation_requirements."
            )

        # ----------------------------------------------------
        # ML training consistency
        # ----------------------------------------------------

        if experiment_type == "ml_training":

            required_training_fields = {
                "preprocessing_required": True,
                "train_validation_test_split": True,
                "checkpoint_required": True,
                "training_history_required": True,
                "training_required": True,
            }

            for field, expected in (
                required_training_fields.items()
            ):

                actual = experiment_design.get(
                    field
                )

                if actual is not expected:
                    errors.append(
                        f"ml_training requires "
                        f"experiment_design.{field}={expected}."
                    )

            # Code-generation requirements must reflect the
            # same training configuration.

            code_generation_requirements_expected = {
                "include_preprocessing": True,
                "include_train_validation_test": True,
                "include_checkpoint": True,
                "include_training_history": True,
                "include_training": True,
            }

            for field, expected in (
                code_generation_requirements_expected.items()
            ):

                actual = code_generation_requirements.get(
                    field
                )

                if actual is not expected:
                    errors.append(
                        f"ml_training requires "
                        f"code_generation_requirements.{field}="
                        f"{expected}."
                    )

            # The dataset used by an ML experiment must be the
            # actual experiment dataset, not merely reference data.

            dataset_role = dataset.get(
                "role"
            )

            if dataset_role != "experiment_dataset":
                errors.append(
                    "ml_training requires dataset.role="
                    "'experiment_dataset'."
                )

        # ----------------------------------------------------
        # Measurement benchmark consistency
        # ----------------------------------------------------

        elif experiment_type == "measurement_benchmark":

            forbidden_training_fields = {
                "preprocessing_required": False,
                "train_validation_test_split": False,
                "checkpoint_required": False,
                "training_history_required": False,
                "training_required": False,
            }

            for field, expected in (
                forbidden_training_fields.items()
            ):

                actual = experiment_design.get(
                    field
                )

                if actual is not expected:
                    errors.append(
                        f"measurement_benchmark requires "
                        f"experiment_design.{field}={expected}."
                    )

            code_generation_requirements_expected = {
                "include_preprocessing": False,
                "include_train_validation_test": False,
                "include_checkpoint": False,
                "include_training_history": False,
                "include_training": False,
            }

            for field, expected in (
                code_generation_requirements_expected.items()
            ):

                actual = code_generation_requirements.get(
                    field
                )

                if actual is not expected:
                    errors.append(
                        f"measurement_benchmark requires "
                        f"code_generation_requirements.{field}="
                        f"{expected}."
                    )

        # ----------------------------------------------------
        # General experiment consistency
        # ----------------------------------------------------

        elif experiment_type == "general_experiment":

            if experiment_design.get(
                "training_required",
                False,
            ):
                errors.append(
                    "general_experiment must not require training "
                    "unless explicitly supported by the experiment type."
                )

        # ----------------------------------------------------
        # Evaluation metric consistency
        # ----------------------------------------------------

        evaluation_metrics = specification.get(
            "evaluation_metrics",
            [],
        )

        if not isinstance(
            evaluation_metrics,
            list,
        ):
            errors.append(
                "evaluation_metrics must be a list."
            )
            evaluation_metrics = []

        evaluation_guidance = specification.get(
            "evaluation_guidance",
            {},
        )

        if not isinstance(
            evaluation_guidance,
            dict,
        ):
            errors.append(
                "evaluation_guidance must be a dictionary."
            )
            evaluation_guidance = {}

        preferred_metrics = evaluation_guidance.get(
            "preferred_comparison_metrics",
            [],
        )

        if not isinstance(
            preferred_metrics,
            list,
        ):
            errors.append(
                "evaluation_guidance.preferred_comparison_metrics "
                "must be a list."
            )
            preferred_metrics = []

        # The top-level evaluation metrics and the metrics that
        # CodeGenerationAgent is told to implement must agree.

        if (
            cls._merge_metric_names(
                evaluation_metrics
            )
            != cls._merge_metric_names(
                preferred_metrics
            )
        ):
            errors.append(
                "evaluation_metrics must match "
                "evaluation_guidance.preferred_comparison_metrics."
            )

        # ----------------------------------------------------
        # Reference-only metrics must not become experiment
        # evaluation requirements.
        # ----------------------------------------------------

        reference_only_metrics = evaluation_guidance.get(
            "reference_only_metrics",
            [],
        )

        if not isinstance(
            reference_only_metrics,
            list,
        ):
            reference_only_metrics = []

        evaluation_metric_keys = {
            cls._normalise_metric_name(metric)
            for metric in evaluation_metrics
        }

        reference_only_keys = {
            cls._normalise_metric_name(metric)
            for metric in reference_only_metrics
        }

        overlap = (
            evaluation_metric_keys
            & reference_only_keys
        )

        if overlap:
            errors.append(
                "Reference-only metrics must not appear in "
                "evaluation_metrics."
            )

        # ----------------------------------------------------
        # Reference values must remain separate.
        # ----------------------------------------------------

        if not evaluation_guidance.get(
            "reference_values_are_not_experiment_results",
            False,
        ):
            errors.append(
                "Reference metric values must explicitly be marked "
                "as separate from experiment results."
            )

        if not evaluation_guidance.get(
            "policy",
            {},
        ).get(
            "never_copy_reference_values",
            False,
        ):
            errors.append(
                "Metric policy must prevent copying paper reference "
                "values into experiment results."
            )

        return errors

    # ========================================================
    # Build Experiment Specification
    # ========================================================

    def build_experiment_specification(
        self,
        hypothesis: Any,
        research_goal: Optional[Any] = None,
        context: Optional[Any] = None,
        reference_experiment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the structured contract between the
        AI Co-Scientist and the experiment pipeline.

        The selected hypothesis remains the scientific source
        of the experiment idea.

        Supporting evidence provides:
            - methodology guidance
            - reference experiment details
            - evaluation metric guidance
            - numerical reference values when explicitly reported

        Reference values are NOT experiment results.
        """

        if hypothesis is None:
            raise ValueError(
                "Cannot build an experiment specification "
                "without a selected hypothesis."
            )

        if reference_experiment is None:
            reference_experiment = {}

        if not isinstance(
            reference_experiment,
            dict,
        ):
            raise TypeError(
                "reference_experiment must be a dictionary."
            )

        hypothesis_data = self.serialize_hypothesis(
            hypothesis
        )

        research_goal_data = (
            self.serialize_research_goal(
                research_goal
            )
        )

        reflection_report = (
            hypothesis_data.get(
                "reflection_report"
            )
            or {}
        )

        # ----------------------------------------------------
        # NEW:
        # Extract evidence-derived metric guidance.
        # ----------------------------------------------------

        reference_metric_guidance = (
            self._extract_reference_metric_requirements(
                reference_experiment
            )
        )

        evidence_metrics = (
            reference_metric_guidance.get(
                "metrics",
                [],
            )
        )

        metric_definitions = (
            reference_metric_guidance.get(
                "metric_definitions",
                {},
            )
        )

        reference_metrics = (
            reference_metric_guidance.get(
                "reference_metrics",
                {},
            )
        )

        hypothesis_metrics = (
            self._extract_hypothesis_metric_requirements(
                hypothesis
            )
        )

        # ----------------------------------------------------
        # Determine experiment type FIRST.
        #
        # The selected hypothesis defines WHAT is being tested.
        # ----------------------------------------------------

        experiment_type = self._infer_experiment_type(
            hypothesis,
            reference_experiment,
        )

        experiment_design = self._build_experiment_design(
            experiment_type,
        )

        # ----------------------------------------------------
        # Classify evidence-derived metrics according to the
        # actual experiment type.
        # ----------------------------------------------------

        metric_classification = (
            self._classify_evidence_metrics(
                experiment_type=experiment_type,
                evidence_metrics=evidence_metrics,
                metric_definitions=metric_definitions,
                reference_metrics=reference_metrics,
            )
        )

        directly_reproducible_metrics = (
            metric_classification.get(
                "directly_reproducible",
                [],
            )
        )

        conditionally_comparable_metrics = (
            metric_classification.get(
                "conditionally_comparable",
                [],
            )
        )

        reference_only_metrics = (
            metric_classification.get(
                "reference_only",
                [],
            )
        )

        # ----------------------------------------------------
        # Only hypothesis metrics + directly reproducible
        # evidence metrics become experiment evaluation metrics.
        #
        # Reference-only paper metrics are NOT passed as
        # requirements for the generated experiment.
        # ----------------------------------------------------

        evaluation_metrics = (
            self._merge_metric_names(
                hypothesis_metrics,
                directly_reproducible_metrics,
            )
        )

        # ----------------------------------------------------
        # Experiment specification
        # ----------------------------------------------------

        specification = {

            "dataset": {
                "name": self.dataset_name,
                "path": (
                    str(self.dataset_path)
                    if self.dataset_path
                    else None
                ),
                "task": None,
                "role": (
                    "experiment_dataset"
                    if experiment_type == "ml_training"
                    else "supporting_or_reference_dataset"
                ),
            },

            "experiment": {
                "device": self.device,
                "framework": "PyTorch",
                "language": "Python",
                "experiment_type": experiment_type,
                "training_required": experiment_design[
                    "training_required"
                ],
            },

            "research_goal": research_goal_data,

            "selected_hypothesis": hypothesis_data,

            "reference_experiment": self._json_safe(
                reference_experiment
            ),

            # NEW:
            # Explicitly expose the metric information instead
            # of making CodeGenerationAgent rediscover it.
            "evaluation_guidance": {
                "hypothesis_metrics": self._json_safe(
                    hypothesis_metrics
                ),

                # All metrics reported/extracted from the paper.
                "evidence_metrics": self._json_safe(
                    evidence_metrics
                ),

                # Metrics that the current experiment can actually
                # calculate directly.
                "directly_reproducible_metrics": self._json_safe(
                    directly_reproducible_metrics
                ),

                # Metrics that could potentially be compared, but
                # only when the experimental protocol is compatible.
                "conditionally_comparable_metrics": self._json_safe(
                    conditionally_comparable_metrics
                ),

                # Paper metrics retained for scientific reference
                # but not required from the generated experiment.
                "reference_only_metrics": self._json_safe(
                    reference_only_metrics
                ),

                # Actual metrics that CodeGenerationAgent should
                # implement/evaluate.
                "experiment_evaluation_metrics": self._json_safe(
                    evaluation_metrics
                ),

                "preferred_comparison_metrics": self._json_safe(
                    evaluation_metrics
                ),

                "metric_definitions": self._json_safe(
                    metric_definitions
                ),

                # Paper values remain reference information only.
                "reference_metrics": self._json_safe(
                    reference_metrics
                ),

                "metric_roles": self._json_safe(
                    metric_classification.get(
                        "metric_roles",
                        {},
                    )
                ),

                "metric_counts": self._json_safe(
                    metric_classification.get(
                        "counts",
                        {},
                    )
                ),

                "reference_values_are_not_experiment_results": True,

                "policy": {
                    "same_metrics_preferred": True,
                    "never_copy_reference_values": True,
                    "never_fabricate_metrics": True,
                    "allow_proxy_metrics": True,
                    "require_proxy_label": True,
                    "reference_only_metrics_are_not_required": True,
                    "conditional_metrics_require_protocol_compatibility": True,
                },
            },

            "scientific_evaluation": {
                "alignment_score": reflection_report.get(
                    "alignment_score"
                ),
                "novelty_score": reflection_report.get(
                    "novelty_score"
                ),
                "feasibility_score": reflection_report.get(
                    "feasibility_score"
                ),
                "plausibility_score": reflection_report.get(
                    "plausibility_score"
                ),
                "testability_score": reflection_report.get(
                    "testability_score"
                ),
                "evidence_quality_score": reflection_report.get(
                    "evidence_quality_score"
                ),
                "expected_research_value_score": reflection_report.get(
                    "expected_research_value_score"
                ),
                "overall_confidence": reflection_report.get(
                    "overall_confidence"
                ),
                "recommendation": reflection_report.get(
                    "recommendation"
                ),
                "strengths": reflection_report.get(
                    "strengths",
                    [],
                ),
                "weaknesses": reflection_report.get(
                    "weaknesses",
                    [],
                ),
                "claims": reflection_report.get(
                    "claims",
                    [],
                ),
            },

            # ------------------------------------------------
            # CHANGED:
            # These describe the current default 5G-NIDD
            # training pipeline. They are implementation
            # defaults rather than universal scientific
            # requirements for every evidence source.
            # ------------------------------------------------

            "experiment_design": {
                "input_source": (
                    "selected_ai_co_scientist_hypothesis"
                ),

                # These requirements are determined by the
                # actual experiment type rather than being
                # universally forced.
                **experiment_design,

                "metric_selection_source": [
                    "selected_hypothesis",
                    "supporting_evidence",
                    "experiment_type",
                    "metric_reproducibility_classification",
                ],

                "directly_reproducible_metrics": self._json_safe(
                    directly_reproducible_metrics
                ),

                "conditionally_comparable_metrics": self._json_safe(
                    conditionally_comparable_metrics
                ),

                "reference_only_metrics": self._json_safe(
                    reference_only_metrics
                ),

                "reference_metrics_must_not_be_copied": True,

                "unreproducible_metrics_policy": (
                    "Mark unavailable/not_directly_comparable "
                    "or use a clearly identified proxy; "
                    "never fabricate a value."
                ),
            },

            "code_generation_requirements": {

            "framework": "PyTorch",
            "language": "Python",

            "dataset": self.dataset_name,

            "dataset_path": (
                str(self.dataset_path)
                if self.dataset_path
                else None
            ),

            "dataset_role": (
                "experiment_dataset"
                if experiment_type == "ml_training"
                else "supporting_or_reference_dataset"
            ),

            "device": self.device,

            "experiment_type": experiment_type,

            "include_preprocessing": experiment_design[
                "preprocessing_required"
            ],

            "include_train_validation_test": experiment_design[
                "train_validation_test_split"
            ],

            "include_training": experiment_design[
                "training_required"
            ],

            "include_checkpoint": experiment_design[
                "checkpoint_required"
            ],

            "include_training_history": experiment_design[
                "training_history_required"
            ],

            "include_reproducibility": experiment_design[
                "reproducibility_required"
            ],

            "include_evaluation": True,

            "use_evidence_derived_metrics": True,

            "do_not_copy_reference_metric_values": True,

            "record_unreproducible_metrics": True,

            "directly_reproducible_metrics": self._json_safe(
                directly_reproducible_metrics
            ),

            "conditionally_comparable_metrics": self._json_safe(
                conditionally_comparable_metrics
            ),

            "reference_only_metrics": self._json_safe(
                reference_only_metrics
            ),

            "do_not_implement_reference_only_metrics": True,

            "conditional_metrics_require_protocol_check": True,
        },

            # ------------------------------------------------
            # CHANGED:
            # No universal hard-coded accuracy/F1 list.
            #
            # Keep this empty so CodeGenerationAgent can derive
            # the evaluation requirements from the hypothesis,
            # evidence, and experiment type.
            # ------------------------------------------------

            "evaluation_metrics": self._json_safe(
                evaluation_metrics
            ),

            # NEW:
            "evaluation_metric_definitions": (
                self._json_safe(
                    metric_definitions
                )
            ),

            # NEW:
            "reference_metrics": (
                self._json_safe(
                    reference_metrics
                )
            ),

            # ------------------------------------------------
            # CHANGED:
            # Artifact requirements are conditional.
            # ------------------------------------------------

            "expected_artifacts": [
                "experiment_config.json",
                "generated_pytorch_code.py",
                "metrics.json",
                "experiment_summary.json",
            ],

            "optional_artifacts": [
                "checkpoint",
                "training_history.json",
                "visualizations",
                "latency_visualization",
                "performance_visualization",
                "stability_visualization",
            ],
        }

        # ----------------------------------------------------
        # AI Co-Scientist provenance
        # ----------------------------------------------------

        if context is not None:

            specification[
                "ai_co_scientist_provenance"
            ] = {

                "iteration_number": getattr(
                    context,
                    "iteration_number",
                    None,
                ),

                "tournament_results": (
                    self.serialize_tournament_results(
                        context
                    )
                ),

                "proximity_analysis": (
                    self.serialize_proximity_analysis(
                        context
                    )
                ),

                "meta_review_feedback": (
                    self.serialize_meta_review(
                        context
                    )
                ),
            }

        # ----------------------------------------------------
        # Validate specification consistency before returning.
        # ----------------------------------------------------

        consistency_errors = (
            self._validate_experiment_specification_consistency(
                specification
            )
        )

        if consistency_errors:
            raise ValueError(
                "Experiment specification is internally inconsistent: "
                + "; ".join(consistency_errors)
            )

        return self._json_safe(
            specification
        )

    # ========================================================
    # JSON Persistence
    # ========================================================

    def save_json(
        self,
        data: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """
        Save a dictionary as formatted JSON.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self._json_safe(data),
                file,
                indent=4,
                ensure_ascii=False,
            )

        return output_path

    # ========================================================
    # Run Directory
    # ========================================================

    def create_run_directory(
        self,
        experiment_id: str,
    ) -> Path:
        """
        Create a dedicated directory for an experiment.
        """

        run_directory = (
            RUNS_DIR
            / experiment_id
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return run_directory

    # ========================================================
    # Save Experiment Configuration
    # ========================================================

    def save_experiment_configuration(
        self,
        experiment_id: str,
        specification: Dict[str, Any],
    ) -> Path:
        """
        Save the complete experiment specification.
        """

        run_directory = (
            self.create_run_directory(
                experiment_id
            )
        )

        output_path = (
            run_directory
            / "experiment_config.json"
        )

        configuration = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
            "device": self.device,
            "dataset": self.dataset_name,
            "specification": specification,
        }

        return self.save_json(
            configuration,
            output_path,
        )

    # ========================================================
    # Save Final Rankings
    # ========================================================

    def save_final_rankings(
        self,
        experiment_id: str,
        context: Any,
    ) -> Path:
        """
        Save the final experiment candidate ranking.
        """

        ranked_hypotheses = (
            self.get_ranked_hypotheses(
                context
            )
        )

        ranking_data = []

        for rank, hypothesis in enumerate(
            ranked_hypotheses,
            start=1,
        ):

            item = (
                self.serialize_hypothesis(
                    hypothesis
                )
            )

            item["rank"] = rank

            ranking_data.append(
                item
            )

        output_path = (
            self.create_run_directory(
                experiment_id
            )
            / "final_rankings.json"
        )

        return self.save_json(
            {
                "experiment_id": experiment_id,
                "ranking_method": (
                    "RankingAgent Elo score"
                ),
                "rankings": ranking_data,
            },
            output_path,
        )

    # ========================================================
    # Save Co-Scientist Provenance
    # ========================================================

    def save_co_scientist_provenance(
        self,
        experiment_id: str,
        context: Any,
    ) -> Path:
        """
        Save the scientific provenance associated with
        experiment selection.
        """

        output_path = (
            self.create_run_directory(
                experiment_id
            )
            / "co_scientist_provenance.json"
        )

        provenance = {
            "iteration_number": getattr(
                context,
                "iteration_number",
                None,
            ),
            "tournament_results": (
                self.serialize_tournament_results(
                    context
                )
            ),
            "proximity_analysis": (
                self.serialize_proximity_analysis(
                    context
                )
            ),
            "meta_review_feedback": (
                self.serialize_meta_review(
                    context
                )
            ),
        }

        return self.save_json(
            provenance,
            output_path,
        )

    # ========================================================
    # Prepare Experiment
    # ========================================================

    def prepare_experiment(
        self,
        context: Any,
        research_goal: Optional[Any] = None,
        hypothesis: Optional[Any] = None,
        reference_experiment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare an automated experiment after the
        SupervisorAgent has completed its cycle.

        Steps:

            1. Retrieve final experiment candidates.
            2. Rank them using existing Elo values.
            3. Select the highest-ranked candidate.
            4. Extract reference experiments from evidence.
            5. Extract evidence-derived evaluation metrics.
            6. Build experiment specification.
            7. Save configuration.
            8. Save final rankings.
            9. Save Co-Scientist provenance.
        """

        started_at = time.perf_counter()

        result: Dict[str, Any] = {
            "success": False,
            "experiment_id": None,
            "selected_hypothesis": None,
            "ranked_hypotheses": [],
            "experiment_specification": None,
            "experiment_config_path": None,
            "final_rankings_path": None,
            "provenance_path": None,
            "errors": [],
        }

        try:

            if context is None:
                raise ValueError(
                    "ContextMemory is required."
                )

            # ------------------------------------------------
            # Select hypothesis
            # ------------------------------------------------

            if hypothesis is None:

                hypothesis = (
                    self.select_best_hypothesis(
                        context
                    )
                )

            if hypothesis is None:
                raise ValueError(
                    "No final accepted active hypothesis "
                    "is available for experiment generation."
                )

            if not self._is_valid_hypothesis(
                hypothesis
            ):
                raise ValueError(
                    "Selected hypothesis does not contain "
                    "valid experiment text."
                )

            # ------------------------------------------------
            # Experiment ID
            # ------------------------------------------------

            experiment_id = self.create_experiment_id(
                self._get_hypothesis_value(
                    hypothesis,
                    "hypothesis_id",
                    None,
                )
            )

            result[
                "experiment_id"
            ] = experiment_id

            # ------------------------------------------------
            # Final rankings
            # ------------------------------------------------

            ranked_hypotheses = (
                self.get_ranked_hypotheses(
                    context
                )
            )

            result[
                "ranked_hypotheses"
            ] = [
                self.serialize_hypothesis(
                    item
                )
                for item in ranked_hypotheses
            ]

            # ------------------------------------------------
            # Selected hypothesis
            # ------------------------------------------------

            result[
                "selected_hypothesis"
            ] = self.serialize_hypothesis(
                hypothesis
            )

            # ------------------------------------------------
            # Extract reference experiment from evidence
            # ------------------------------------------------

            if reference_experiment is None:

                reference_experiment = (
                    self.extract_reference_experiment(
                        hypothesis
                    )
                )

            # Ensure a dictionary is always passed downstream.

            if reference_experiment is None:
                reference_experiment = {}

            if not isinstance(
                reference_experiment,
                dict,
            ):
                raise TypeError(
                    "reference_experiment must be a dictionary."
                )

            result["reference_experiment"] = (
                self._json_safe(
                    reference_experiment
                )
            )

            # ------------------------------------------------
            # Build specification
            # ------------------------------------------------

            specification = (
                self.build_experiment_specification(
                    hypothesis=hypothesis,
                    research_goal=research_goal,
                    context=context,
                    reference_experiment=reference_experiment,
                )
            )

            result[
                "experiment_specification"
            ] = specification

            # ------------------------------------------------
            # Save configuration
            # ------------------------------------------------

            config_path = (
                self.save_experiment_configuration(
                    experiment_id,
                    specification,
                )
            )

            result[
                "experiment_config_path"
            ] = str(config_path)

            # ------------------------------------------------
            # Save rankings
            # ------------------------------------------------

            rankings_path = (
                self.save_final_rankings(
                    experiment_id,
                    context,
                )
            )

            result[
                "final_rankings_path"
            ] = str(rankings_path)

            # ------------------------------------------------
            # Save provenance
            # ------------------------------------------------

            provenance_path = (
                self.save_co_scientist_provenance(
                    experiment_id,
                    context,
                )
            )

            result[
                "provenance_path"
            ] = str(provenance_path)

            # ------------------------------------------------
            # NEW:
            # Expose evaluation guidance at preparation level.
            # ------------------------------------------------

            result[
                "evaluation_guidance"
            ] = specification.get(
                "evaluation_guidance",
                {},
            )

            result[
                "success"
            ] = True

        except Exception as error:

            result[
                "errors"
            ].append(
                str(error)
            )

        finally:

            result[
                "preparation_seconds"
            ] = (
                time.perf_counter()
                - started_at
            )

        return result

    # ========================================================
    # Code Generation Interface
    # ========================================================

    def generate_pytorch_code(
        self,
        specification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate PyTorch experiment code from the structured
        experiment specification.

        CodeGenerationAgent receives:

            - selected Rank #1 hypothesis
            - research goal
            - dataset/schema
            - reference experiment
            - evidence-derived metric guidance
            - reproducibility requirements

        The hypothesis remains authoritative for WHAT is tested.
        Supporting evidence guides HOW it is evaluated.
        """

        if not isinstance(
            specification,
            dict,
        ):
            raise TypeError(
                "specification must be a dictionary."
            )

        logger.info(
            "CodeGeneration specification size: %d characters",
            len(
                json.dumps(
                    specification,
                    ensure_ascii=False,
                )
            ),
        )

        logger.info(
            "Reference experiment size: %d characters",
            len(
                json.dumps(
                    specification.get(
                        "reference_experiment",
                        {},
                    ),
                    ensure_ascii=False,
                )
            ),
        )

        # NEW:
        # Log the actual evidence-derived metrics.

        evaluation_guidance = (
            specification.get(
                "evaluation_guidance",
                {},
            )
        )

        logger.info(
            "Evidence-derived evaluation metrics: %s",
            evaluation_guidance.get(
                "evidence_metrics",
                [],
            ),
        )

        logger.info(
            "Paper reference metric values: %s",
            evaluation_guidance.get(
                "reference_metrics",
                {},
            ),
        )

        return self.code_generation_agent.generate(
            specification
        )

    # ========================================================
    # Save Generated Code
    # ========================================================

    def save_generated_code(
        self,
        experiment_id: str,
        code: str,
    ) -> Path:
        """
        Save generated PyTorch source code inside the
        experiment run directory.
        """

        if not isinstance(
            code,
            str,
        ):
            raise TypeError(
                "Generated code must be a string."
            )

        output_path = (
            self.create_run_directory(
                experiment_id
            )
            / "generated_pytorch_code.py"
        )

        output_path.write_text(
            code,
            encoding="utf-8",
        )

        return output_path

    # ========================================================
    # Run Generated Experiment
    # ========================================================

    def run_generated_experiment(
        self,
        experiment_id: str,
        generated_result: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute generated experiment code through ExperimentRunner.

        ExperimentRunner is responsible for:

            - Creating the execution directory
            - Saving generated code
            - Saving experiment metadata
            - Executing the experiment
            - Capturing stdout/stderr
            - Collecting metrics
            - Collecting training history
            - Finding checkpoints
            - Finding visualizations
            - Validating outputs
        """

        if not isinstance(
            generated_result,
            dict,
        ):
            raise TypeError(
                "generated_result must be a dictionary."
            )

        original_timeout = (
            self.experiment_runner.timeout_seconds
        )

        if timeout_seconds is not None:
            self.experiment_runner.timeout_seconds = max(
                1,
                int(timeout_seconds),
            )

        try:

            logger.info(
                "Experiment execution started."
            )

            runner_result = (
                self.experiment_runner.run_generated_result(
                    generated_result=generated_result,
                    dataset_path=self.dataset_path,
                    experiment_id=experiment_id,
                )
            )

            runner_result["experiment_id"] = (
                experiment_id
            )

            return runner_result

        finally:

            self.experiment_runner.timeout_seconds = (
                original_timeout
            )

    # ========================================================
    # Compare Experiment Results
    # ========================================================

    def compare_experiment_results(
        self,
        hypothesis: Dict[str, Any],
        execution_result: Dict[str, Any],
        reference_experiment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compare successful automated experiment results against
        evidence sources associated with the selected hypothesis.

        The comparator is responsible for determining:

            - which paper metrics are available
            - which experiment metrics are available
            - unit compatibility
            - numerical differences
            - relative differences
            - comparability
            - limitations

        This method does not assume that accuracy/F1/etc. are
        universally applicable.
        """

        if not isinstance(
            hypothesis,
            dict,
        ):
            raise TypeError(
                "hypothesis must be a dictionary."
            )

        if not isinstance(
            execution_result,
            dict,
        ):
            raise TypeError(
                "execution_result must be a dictionary."
            )

        if not execution_result.get(
            "success",
            False,
        ):
            return {
                "success": False,
                "status": "experiment_failed",
                "errors": [
                    "Comparison skipped because the automated "
                    "experiment did not complete successfully."
                ],
            }

        evidence_sources = (
            hypothesis.get(
                "evidence_sources"
            )
            or []
        )

        if not evidence_sources:
            return {
                "success": False,
                "status": "no_evidence_sources",
                "errors": [
                    "Comparison skipped because the selected "
                    "hypothesis has no evidence sources."
                ],
            }

        return self.experiment_comparator.compare(
            hypothesis=hypothesis,
            experiment_result=execution_result,
            reference_experiment=reference_experiment,
        )

    # ========================================================
    # Full Experiment Pipeline
    # ========================================================

    def run_experiment(
        self,
        context: Any,
        research_goal: Optional[Any] = None,
        hypothesis: Optional[Any] = None,
        reference_experiment: Optional[Dict[str, Any]] = None,
        execute_generated_code: bool = False,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Pipeline:

            Completed Supervisor cycle
                    |
                    v
            Final accepted hypotheses
                    |
                    v
            Existing Elo ranking
                    |
                    v
            Rank #1 hypothesis
                    |
                    v
            Evidence Sources
                    |
                    v
            PaperReader
                    |
                    v
            Reference Experiment
                    |
                    v
            Evidence-derived Metrics
                    |
                    v
            Experiment Specification
                    |
                    v
            CodeGenerationAgent
                    |
                    v
            Generated PyTorch code
                    |
                    v
            ExperimentRunner
                    |
                    v
            Experiment Results
                    |
                    v
            ExperimentComparator
                    |
                    v
            Paper vs Experiment

        By default, generated code execution is disabled.
        """

        started_at = time.perf_counter()

        logger.info(
            "Experiment started"
        )

        preparation = (
            self.prepare_experiment(
                context=context,
                research_goal=research_goal,
                hypothesis=hypothesis,
                reference_experiment=reference_experiment,
            )
        )

        result: Dict[str, Any] = {
            "success": preparation.get(
                "success",
                False,
            ),
            "experiment_preparation": preparation,
            "code_generation": None,
            "execution": None,
            "comparison": None,
            "errors": list(
                preparation.get(
                    "errors",
                    [],
                )
            ),
            "reference_experiment": preparation.get(
                "reference_experiment",
                {},
            ),
        }

        if not preparation.get(
            "success",
            False,
        ):

            result[
                "total_seconds"
            ] = (
                time.perf_counter()
                - started_at
            )

            return result

        # ----------------------------------------------------
        # Code generation
        # ----------------------------------------------------

        specification = preparation.get(
            "experiment_specification"
        )

        try:

            code_generation = (
                self.generate_pytorch_code(
                    specification
                )
            )

            result[
                "code_generation"
            ] = code_generation

            if not isinstance(
                code_generation,
                dict,
            ):

                result[
                    "errors"
                ].append(
                    "Code generation did not return a dictionary."
                )

                result[
                    "success"
                ] = False

            elif not code_generation.get(
                "success",
                False,
            ):

                generation_errors = (
                    code_generation.get(
                        "errors",
                        [],
                    )
                )

                if isinstance(
                    generation_errors,
                    list,
                ):

                    result[
                        "errors"
                    ].extend(
                        str(error)
                        for error in generation_errors
                        if str(error).strip()
                    )

                elif generation_errors:

                    result[
                        "errors"
                    ].append(
                        str(generation_errors)
                    )

                else:

                    result[
                        "errors"
                    ].append(
                        code_generation.get(
                            "error",
                            "Code generation failed.",
                        )
                    )

                result[
                    "success"
                ] = False

        except Exception as error:

            result[
                "errors"
            ].append(
                f"Code generation failed: {error}"
            )

            result[
                "success"
            ] = False

        # ----------------------------------------------------
        # Optional execution
        # ----------------------------------------------------

        if (
            execute_generated_code
            and result.get(
                "success",
                False,
            )
        ):

            code_generation = (
                result[
                    "code_generation"
                ]
            )

            experiment_id = (
                preparation[
                    "experiment_id"
                ]
            )

            try:

                execution = (
                    self.run_generated_experiment(
                        experiment_id=experiment_id,
                        generated_result=code_generation,
                        timeout_seconds=timeout_seconds,
                    )
                )

            except Exception as error:

                execution = {
                    "success": False,
                    "status": "runner_error",
                    "errors": [
                        str(error)
                    ],
                    "error": str(error),
                }

            result[
                "execution"
            ] = execution

            output_validation = (
                execution.get(
                    "output_validation",
                    {},
                )
            )

            execution_valid = (
                execution.get(
                    "success",
                    False,
                )
                and output_validation.get(
                    "valid",
                    True,
                )
            )

            if execution_valid:

                result[
                    "success"
                ] = True

            else:

                result[
                    "success"
                ] = False

                execution_errors = (
                    execution.get(
                        "errors",
                        [],
                    )
                )

                if isinstance(
                    execution_errors,
                    list,
                ):

                    result[
                        "errors"
                    ].extend(
                        execution_errors
                    )

            # ------------------------------------------------
            # Compare successful experiment with evidence
            # ------------------------------------------------

            if execution_valid:

                try:

                    comparison_reference_experiment = (
                        preparation[
                            "experiment_specification"
                        ].get(
                            "reference_experiment",
                            {},
                        )
                    )

                    comparison = (
                        self.compare_experiment_results(
                            hypothesis=preparation[
                                "selected_hypothesis"
                            ],
                            execution_result=execution,
                            reference_experiment=(
                                comparison_reference_experiment
                            ),
                        )
                    )

                    result[
                        "comparison"
                    ] = comparison

                except Exception as error:

                    result[
                        "comparison"
                    ] = {
                        "success": False,
                        "status": "comparison_error",
                        "errors": [
                            str(error)
                        ],
                    }

                    result[
                        "errors"
                    ].append(
                        f"Experiment comparison failed: {error}"
                    )

        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        if execute_generated_code:

            execution = result.get(
                "execution"
            )

            if execution is not None:

                output_validation = (
                    execution.get(
                        "output_validation",
                        {},
                    )
                )

                result[
                    "success"
                ] = bool(
                    execution.get(
                        "success",
                        False,
                    )
                    and output_validation.get(
                        "valid",
                        True,
                    )
                )

            else:

                result[
                    "success"
                ] = False

        elif result.get(
            "code_generation"
        ):

            result[
                "success"
            ] = bool(
                result[
                    "code_generation"
                ].get(
                    "success",
                    False,
                )
            )

        result[
            "total_seconds"
        ] = (
            time.perf_counter()
            - started_at
        )

        return result