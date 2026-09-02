"""
Automated Experiment Orchestrator.

This module connects the completed AI Co-Scientist workflow to the
automated deep-learning experiment pipeline.

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
        +--> Build experiment specification
        +--> Generate PyTorch experiment code
        +--> Execute experiment
        +--> Evaluate results
        +--> Save artifacts
        |
        v
    Experiment Results

The orchestrator intentionally does NOT reproduce the SupervisorAgent's
scientific workflow. It consumes the final ContextMemory produced by the
Supervisor.

Current experiment domain:

    Dataset: 5G-NIDD
    Task: 5G network intrusion detection classification
    Framework: PyTorch
"""

from __future__ import annotations

import json
import math
import re
# import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents_modules.code_generation_agent import CodeGenerationAgent
from .experiment_runner import ExperimentRunner


# ============================================================
# Configuration
# ============================================================

BASE_EXPERIMENT_DIR = Path("app/experiments")

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
    deep-learning experiment pipeline.

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
        4. Build a structured experiment specification.
        5. Persist the specification and scientific provenance.
        6. Provide an integration point for code generation.
        7. Optionally execute generated PyTorch code.
        8. Collect experiment results.
    """

    def __init__(
        self,
        dataset_name: str = "5G-NIDD",
        dataset_path: Optional[str] = None,
        device: str = "cpu",
        python_executable: Optional[str] = None,
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

        self.dataset_path = (
            Path(dataset_path)
            if dataset_path
            else None
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

        This avoids forcing every object in the Co-Scientist
        context to implement its own serializer.
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
                return (
                    f"{safe_id}_{timestamp}"
                )

        return (
            f"experiment_{timestamp}"
        )

    # ========================================================
    # Context / Hypothesis Access
    # ========================================================

    def get_active_hypotheses(
        self,
        context: Any,
    ) -> List[Any]:
        """
        Retrieve active hypotheses from ContextMemory.

        The actual ContextMemory implementation provides:

            context.get_active_hypotheses()
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

    # ========================================================
    # Reflection Routing
    # ========================================================

    def get_accepted_hypotheses(
        self,
        context: Any,
    ) -> List[Any]:
        """
        Retrieve hypotheses that Reflection explicitly accepted.

        This mirrors the Supervisor's reflection routing:

            ACCEPT  -> rankable
            REJECT  -> inactive
            REVISE  -> revision path
            other   -> unreviewed

        The orchestrator does not change hypothesis state.
        It only reads the final state.
        """

        accepted: List[Any] = []

        for hypothesis in self.get_active_hypotheses(context):
            if not self._is_valid_hypothesis(
                hypothesis
            ):
                continue

            report = getattr(
                hypothesis,
                "reflection_report",
                None,
            )

            if report is None:
                continue

            recommendation = (
                self._safe_string(
                    getattr(
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

        Only hypotheses with an ACCEPT recommendation are
        eligible for automated experiment execution.
        """
        return self.get_accepted_hypotheses(context)

        # if hypothesis is None:
        #     raise ValueError(
        #         "No final Reflection-accepted active hypothesis "
        #         "is available for experiment generation."
        #     )

        # accepted = self.get_accepted_hypotheses(
        #     context
        # )

        # if accepted:
        #     return accepted

        # return [
        #     hypothesis
        #     for hypothesis in self.get_active_hypotheses(context)
        #     if self._is_valid_hypothesis(hypothesis)
        #     and getattr(
        #         hypothesis,
        #         "reflection_report",
        #         None,
        #     ) is None
        # ]

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

        if not getattr(
            hypothesis,
            "is_active",
            False,
        ):
            return False

        text = self._safe_string(
            getattr(
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

        RankingAgent has already updated:

            hypothesis.elo_score

        during the Supervisor cycle.
        """

        candidates = self.get_experiment_candidates(
            context
        )

        def elo_key(hypothesis: Any) -> float:
            score = self._safe_float(
                getattr(
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

        Selection order:

            1. Active
            2. Reflection ACCEPT
            3. Highest RankingAgent Elo score
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
            "novelty_review": getattr(
                hypothesis,
                "novelty_review",
                None,
            ),
            "feasibility_review": getattr(
                hypothesis,
                "feasibility_review",
                None,
            ),
            "review_comments": self._json_safe(
                getattr(
                    hypothesis,
                    "review_comments",
                    [],
                )
            ),
            "references": self._json_safe(
                getattr(
                    hypothesis,
                    "references",
                    [],
                )
            ),
            "review_reference_ids": self._json_safe(
                getattr(
                    hypothesis,
                    "review_reference_ids",
                    [],
                )
            ),
            "is_active": getattr(
                hypothesis,
                "is_active",
                None,
            ),
            "deactivation_reason": getattr(
                hypothesis,
                "deactivation_reason",
                None,
            ),
            "parent_ids": self._json_safe(
                getattr(
                    hypothesis,
                    "parent_ids",
                    [],
                )
            ),
            "evolution_strategy": getattr(
                hypothesis,
                "evolution_strategy",
                None,
            ),
            "evidence_source_ids": self._json_safe(
                getattr(
                    hypothesis,
                    "evidence_source_ids",
                    [],
                )
            ),
            "evidence_sources": self._json_safe(
                getattr(
                    hypothesis,
                    "evidence_sources",
                    [],
                )
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
            "audit_report": self._json_safe(
                getattr(
                    hypothesis,
                    "audit_report",
                    {},
                )
            ),
            "reflection_report": self.serialize_reflection_report(
                getattr(
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
        """
        Serialize the actual ResearchGoal runtime fields.
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
            "constraints": self._json_safe(
                getattr(
                    research_goal,
                    "constraints",
                    {},
                )
            ),
            "llm_model": getattr(
                research_goal,
                "llm_model",
                None,
            ),
            "query_rewrite_model": getattr(
                research_goal,
                "query_rewrite_model",
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
    # Tournament Serialization
    # ========================================================

    def serialize_tournament_results(
        self,
        context: Any,
    ) -> List[Dict[str, Any]]:
        """
        Preserve the RankingAgent's tournament decisions.

        This is scientific provenance rather than another ranking
        operation.
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
        Preserve the latest ProximityAgent output when available.
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
        Preserve MetaReviewAgent feedback from ContextMemory.
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

    # ========================================================
    # Build Experiment Specification
    # ========================================================

    def build_experiment_specification(
        self,
        hypothesis: Any,
        research_goal: Optional[Any] = None,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Build the structured contract between the
        AI Co-Scientist and the experiment pipeline.

        The selected hypothesis remains the scientific source
        of the experiment idea. The experiment layer does not
        invent a new hypothesis.
        """

        if hypothesis is None:
            raise ValueError(
                "Cannot build an experiment specification "
                "without a selected hypothesis."
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

        specification = {
            "dataset": {
                "name": self.dataset_name,
                "path": (
                    str(self.dataset_path)
                    if self.dataset_path
                    else None
                ),
                "task": (
                    "5G network intrusion "
                    "detection classification"
                ),
            },

            "experiment": {
                "device": self.device,
                "framework": "PyTorch",
                "language": "Python",
            },

            "research_goal": research_goal_data,

            "selected_hypothesis": hypothesis_data,

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

            "experiment_design": {
                "input_source": (
                    "selected_ai_co_scientist_hypothesis"
                ),
                "preprocessing_required": True,
                "train_validation_test_split": True,
                "reproducibility_required": True,
                "checkpoint_required": True,
                "training_history_required": True,
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
                "device": self.device,
                "include_preprocessing": True,
                "include_train_validation_test": True,
                "include_checkpoint": True,
                "include_training_history": True,
                "include_reproducibility": True,
                "include_evaluation": True,
            },

            "evaluation_metrics": [
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "confusion_matrix",
                "training_seconds",
                "evaluation_seconds",
                "total_execution_seconds",
            ],

            "expected_artifacts": [
                "experiment_config.json",
                "generated_pytorch_code.py",
                "checkpoint",
                "metrics.json",
                "training_history.json",
                "loss_visualization",
                "accuracy_visualization",
                "confusion_matrix_visualization",
                "performance_metrics_visualization",
            ],
        }

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

        This records the final state produced by the
        Reflection + Ranking workflow.
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
    ) -> Dict[str, Any]:
        """
        Prepare an automated experiment after the
        SupervisorAgent has completed its cycle.

        Steps:

            1. Retrieve final experiment candidates.
            2. Rank them using existing Elo values.
            3. Select the highest-ranked candidate.
            4. Build experiment specification.
            5. Save configuration.
            6. Save final rankings.
            7. Save Co-Scientist provenance.

        No Generation, Reflection, Ranking, Evolution,
        Proximity, or Meta-review agents are executed here.
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

            experiment_id = (
                self.create_experiment_id(
                    getattr(
                        hypothesis,
                        "hypothesis_id",
                        None,
                    )
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
            # Build specification
            # ------------------------------------------------

            specification = (
                self.build_experiment_specification(
                    hypothesis=hypothesis,
                    research_goal=research_goal,
                    context=context,
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

        The CodeGenerationAgent receives the experiment
        specification produced from the selected AI Co-Scientist
        hypothesis and returns generated experiment code.
        """

        # raise NotImplementedError(
        #     "CodeGenerationAgent has not yet "
        #     "been connected to ExperimentOrchestrator."
        # )
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

        if not isinstance(code, str):
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
        Execute generated PyTorch code through ExperimentRunner.

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
            runner_result = (
                self.experiment_runner.run_generated_result(
                    generated_result=generated_result,
                    dataset_path=self.dataset_path,
                    run_name=experiment_id,
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
    # Full Experiment Pipeline
    # ========================================================

    def run_experiment(
        self,
        context: Any,
        research_goal: Optional[Any] = None,
        hypothesis: Optional[Any] = None,
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
            Best hypothesis
                    |
                    v
            Experiment specification
                    |
                    v
            CodeGenerationAgent
                    |
                    v
            Generated PyTorch code
                    |
                    v
            Optional execution
                    |
                    v
            Metrics / artifacts

        By default, code execution is disabled until a real
        CodeGenerationAgent is connected.
        """

        started_at = time.perf_counter()

        preparation = (
            self.prepare_experiment(
                context=context,
                research_goal=research_goal,
                hypothesis=hypothesis,
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
            "errors": list(
                preparation.get(
                    "errors",
                    [],
                )
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
            code_generation = self.generate_pytorch_code(
                specification
            )

            result["code_generation"] = code_generation

            if not isinstance(code_generation, dict):
                result["errors"].append(
                    "Code generation did not return a dictionary."
                )
                result["success"] = False

            elif not code_generation.get("success", False):
                generation_errors = code_generation.get(
                    "errors",
                    [],
                )
                if isinstance(generation_errors, list):
                    result["errors"].extend(
                        str(error)
                        for error in generation_errors
                        if str(error).strip()
                    )
                elif generation_errors:
                    result["errors"].append(
                        str(generation_errors)
                    )
                else:
                    result["errors"].append(
                        code_generation.get(
                            "error",
                            "Code generation failed."
                        )
                    )
                result["success"] = False

        except Exception as error:
            result["errors"].append(
                f"Code generation failed: {error}"
            )
            result["success"] = False

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
                result.get(
                    "code_generation"
                )
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

                result[
                    "execution"
                ] = execution

                if not execution.get(
                    "success",
                    False,
                ):
                    result[
                        "errors"
                    ].extend(
                        execution.get(
                            "errors",
                            [],
                        )
                    )

                    if (
                        not execution.get(
                            "errors"
                        )
                    ):
                        result[
                            "errors"
                        ].append(
                            execution.get(
                                "status",
                                "Experiment execution failed."
                            )
                        )

                    result[
                        "success"
                    ] = False

            except Exception as error:
                result[
                    "errors"
                ].append(
                    f"Experiment execution failed: {error}"
                )

                result[
                    "success"
                ] = False


        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        if not result["errors"]:

            if execute_generated_code:
                execution = result.get(
                    "execution"
                )

                result[
                    "success"
                ] = bool(
                    execution
                    and execution.get(
                        "success",
                        False,
                    )
                )

            else:
                result[
                    "success"
                ] = bool(
                    result.get(
                        "code_generation",
                        {},
                    ).get(
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