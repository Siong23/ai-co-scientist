"""
Experiment Runner.

Executes generated experiment code, collects experiment outputs, and validates
the resulting artifacts for the automated AI Co-Scientist experiment pipeline.

Workflow:

    Rank #1 Hypothesis
            |
            +--> Evidence Sources
                      |
                      v
                 PaperReader
                      |
                      v
             Paper Experimental Details
             - metrics
             - metric definitions
             - reference values
             - methodology
                      |
                      v
             CodeGenerationAgent
                      |
                      v
              Generated Code
                      |
                      v
               ExperimentRunner
                      |
                      +--> Execute
                      +--> Collect metrics
                      +--> Collect metadata
                      +--> Collect optional artifacts
                      |
                      v
             ExperimentComparator
                      |
                      v
                Paper vs Experiment

Important design principle:

    ExperimentRunner is metric-agnostic.

The runner must NOT require a fixed set of metrics such as accuracy,
precision, recall, or F1. The metrics to be evaluated are determined by the
selected hypothesis, supporting evidence, and generated experiment.

For example, a generated experiment may produce:

    accuracy
    f1
    auc

or:

    handshake_latency_ms
    certificate_size_bytes
    throughput_mbps

or:

    attack_success_rate
    detection_delay_ms
    memory_usage_mb

The runner executes the generated experiment and preserves whatever scientific
metrics it produces. ExperimentComparator is responsible for determining which
paper and experiment metrics are actually comparable.
"""

import ast
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..utils import (
        execution_cancelled,
        execution_remaining_seconds,
    )
except ImportError:
    try:
        from ..utils import (
            execution_cancelled,
            execution_remaining_seconds,
        )
    except ImportError:
        def execution_cancelled() -> bool:
            return False

        def execution_remaining_seconds(default: Optional[float] = None) -> Optional[float]:
            return default


logger = logging.getLogger(__name__)

# import inspect

# ============================================================================
# Constants
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 7200

DEFAULT_OUTPUT_FILES = {
    "metrics": "metrics.json",
    "training_history": "training_history.json",
    "summary": "experiment_summary.json",
    "checkpoint": "best_model.pt",
}

DEFAULT_VISUALIZATION_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
}

# ---------------------------------------------------------------------------
# Backward-compatible constants
# ---------------------------------------------------------------------------
#
# These are retained so existing code/tests importing them do not break.
# They are NO LONGER used as mandatory output requirements.
#

REQUIRED_SCALAR_METRICS = {
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "training_seconds",
    "evaluation_seconds",
    "total_execution_seconds",
}

REQUIRED_STRUCTURED_METRICS = {
    "confusion_matrix",
}

TIMING_METRICS = {
    "training_seconds",
    "evaluation_seconds",
    "total_execution_seconds",
}

METRIC_ALIASES = {
    "accuracy": (
        "accuracy",
        "test_accuracy",
    ),
    "precision_weighted": (
        "precision_weighted",
        "test_precision_weighted",
    ),
    "recall_weighted": (
        "recall_weighted",
        "test_recall_weighted",
    ),
    "f1_weighted": (
        "f1_weighted",
        "test_f1_weighted",
    ),
}

REQUIRED_VISUALIZATION_STEMS = {
    "loss_visualization",
    "accuracy_visualization",
    "confusion_matrix_visualization",
    "performance_metrics_visualization",
}


# ============================================================================
# Experiment Runner
# ============================================================================


class ExperimentRunner:
    """
    Execute generated experiment code and collect its outputs.

    The runner deliberately does not decide which scientific metrics are
    required. That decision belongs to the experiment generation stage, which
    receives the selected hypothesis and evidence-derived experimental
    requirements.
    """

    def __init__(
        self,
        output_directory: str = "experiment_runs",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        python_executable: Optional[str] = None,
    ):
        self.output_directory = Path(output_directory)
        self.timeout_seconds = int(timeout_seconds)
        self.python_executable = python_executable or sys.executable

    # ========================================================================
    # General helpers
    # ========================================================================

    @staticmethod
    def _timestamp() -> str:
        """Return a UTC timestamp suitable for metadata."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict[str, Any]]:
        """Read a JSON object from disk."""
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                return data

            logger.warning("JSON file is not an object: %s", path)
            return None

        except Exception as exc:
            logger.warning("Failed to read JSON file %s: %s", path, exc)
            return None

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write JSON data to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        """
        Return True if value is a finite integer/float.

        Booleans are deliberately excluded because bool is a subclass of int.
        """
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    @classmethod
    def _has_nonfinite_number(cls, value: Any) -> bool:
        """
        Recursively detect NaN/Infinity values in nested structures.
        """
        if isinstance(value, float):
            return not math.isfinite(value)

        if isinstance(value, dict):
            return any(
                cls._has_nonfinite_number(item)
                for item in value.values()
            )

        if isinstance(value, (list, tuple)):
            return any(
                cls._has_nonfinite_number(item)
                for item in value
            )

        return False

    @staticmethod
    def _safe_filename(value: str) -> str:
        """Convert a string into a filesystem-safe filename."""
        value = str(value or "").strip()

        if not value:
            return "experiment"

        value = re.sub(r"[^\w\-\.]+", "_", value)
        value = re.sub(r"_+", "_", value)

        return value[:150]

    # ========================================================================
    # Run directory
    # ========================================================================

    def create_run_directory(
        self,
        experiment_id: Optional[str] = None,
    ) -> Path:
        """
        Create and return a unique experiment run directory.
        """
        self.output_directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

        if experiment_id:
            safe_id = self._safe_filename(experiment_id)
            run_name = f"{safe_id}_{timestamp}"
        else:
            run_name = f"experiment_{timestamp}"

        run_dir = self.output_directory / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        return run_dir

    # ========================================================================
    # Generated code preparation
    # ========================================================================

    def prepare_generated_code(
        self,
        generated_code: str,
        run_dir: Path,
        filename: str = "generated_experiment.py",
    ) -> Path:
        """
        Validate and save generated Python code.

        AST parsing is performed before execution to catch syntax errors early.
        """
        if not generated_code or not generated_code.strip():
            raise ValueError("Generated experiment code is empty.")

        try:
            ast.parse(generated_code)
        except SyntaxError as exc:
            raise ValueError(
                f"Generated experiment code contains a syntax error: {exc}"
            ) from exc

        code_path = run_dir / filename
        code_path.write_text(
            generated_code,
            encoding="utf-8",
        )

        return code_path

    def copy_generated_code(
        self,
        source_path: str,
        run_dir: Path,
        filename: str = "generated_experiment.py",
    ) -> Path:
        """
        Copy an existing generated Python file into the run directory.
        """
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(
                f"Generated code file does not exist: {source}"
            )

        generated_code = source.read_text(encoding="utf-8")

        return self.prepare_generated_code(
            generated_code=generated_code,
            run_dir=run_dir,
            filename=filename,
        )

    # ========================================================================
    # Metadata
    # ========================================================================

    def save_experiment_metadata(
        self,
        run_dir: Path,
        generated_result: Optional[Dict[str, Any]] = None,
        dataset_path: Optional[str] = None,
        model: Optional[str] = None,
        model_recommendation: Optional[str] = None,
        experiment_plan: Optional[Dict[str, Any]] = None,
        assumptions: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        hypothesis: Optional[Any] = None,
        evidence_sources: Optional[List[Any]] = None,
        evaluation_metrics: Optional[List[Any]] = None,
        metric_definitions: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save reproducibility metadata for the experiment.

        The metadata intentionally stores the scientific context that led to
        the generated experiment, including the selected hypothesis,
        evidence sources, and evidence-derived evaluation metrics.
        """
        generated_result = generated_result or {}

        # Extract values from generated_result when explicit arguments were
        # not provided.
        if experiment_plan is None:
            experiment_plan = generated_result.get(
                "experiment_plan",
                {},
            )

        if not dataset_path:
            dataset_path = (
                generated_result.get("dataset_path")
                or experiment_plan.get("dataset_path")
                if isinstance(experiment_plan, dict)
                else generated_result.get("dataset_path")
            )

        if not model:
            model = (
                generated_result.get("model")
                or generated_result.get("model_name")
                or experiment_plan.get("model")
                if isinstance(experiment_plan, dict)
                else generated_result.get("model")
            )

        if not model_recommendation:
            model_recommendation = generated_result.get(
                "model_recommendation"
            )

        if assumptions is None:
            assumptions = generated_result.get("assumptions", [])

        if dependencies is None:
            dependencies = generated_result.get("dependencies", [])

        if hypothesis is None:
            hypothesis = (
                generated_result.get("selected_hypothesis")
                or generated_result.get("hypothesis")
            )

        if evidence_sources is None:
            evidence_sources = (
                generated_result.get("evidence_sources")
                or generated_result.get("evidence")
                or []
            )

        if evaluation_metrics is None:
            evaluation_metrics = (
                generated_result.get("evaluation_metrics")
                or (
                    experiment_plan.get("evaluation_metrics", [])
                    if isinstance(experiment_plan, dict)
                    else []
                )
                or []
            )

        if metric_definitions is None:
            metric_definitions = (
                generated_result.get("metric_definitions")
                or (
                    experiment_plan.get("metric_definitions", {})
                    if isinstance(experiment_plan, dict)
                    else {}
                )
                or {}
            )

        metadata = {
            "created_at": self._timestamp(),
            "dataset_path": dataset_path,
            "model": model,
            "model_recommendation": model_recommendation,
            "hypothesis": hypothesis,
            "selected_hypothesis": hypothesis,
            "evidence_sources": evidence_sources,
            "evaluation_metrics": evaluation_metrics,
            "metric_definitions": metric_definitions,
            "experiment_plan": experiment_plan or {},
            "assumptions": assumptions or [],
            "dependencies": dependencies or [],
        }

        metadata_path = run_dir / "experiment_metadata.json"

        self._write_json(
            metadata_path,
            metadata,
        )

        return metadata_path

    # ========================================================================
    # Environment
    # ========================================================================

    def build_environment(
        self,
        run_dir: Path,
        dataset_path: Optional[str] = None,
        extra_environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Build the subprocess environment for the generated experiment.
        """
        environment = os.environ.copy()

        environment["EXPERIMENT_OUTPUT_DIR"] = str(run_dir)

        if dataset_path:
            environment["DATASET_PATH"] = str(dataset_path)
            environment["EXPERIMENT_DATASET_PATH"] = str(dataset_path)

        if extra_environment:
            for key, value in extra_environment.items():
                if value is not None:
                    environment[str(key)] = str(value)

        return environment

    # ========================================================================
    # Dependency handling
    # ========================================================================

    @staticmethod
    def _extract_missing_module(error_text: str) -> Optional[str]:
        """
        Extract a missing Python module from ModuleNotFoundError output.
        """
        if not error_text:
            return None

        patterns = [
            r"No module named ['\"]([^'\"]+)['\"]",
            r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_text)

            if match:
                module = match.group(1).strip()

                # Only use the top-level module for pip installation.
                return module.split(".")[0]

        return None

    def _install_package(
        self,
        package_name: str,
        remaining_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Install a missing Python package.

        Returns installation metadata rather than raising immediately so the
        caller can decide whether another experiment attempt is appropriate.
        """
        if not package_name:
            return {
                "success": False,
                "package": package_name,
                "error": "Empty package name.",
            }

        install_timeout = 300

        if remaining_seconds is not None:
            install_timeout = max(
                1,
                min(
                    install_timeout,
                    int(max(1, remaining_seconds)),
                ),
            )

        package_mapping = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "bs4": "beautifulsoup4",
            "dotenv": "python-dotenv",
            "dateutil": "python-dateutil",
            "imblearn": "imbalanced-learn",
        }

        pypi_package_name = package_mapping.get(
            package_name,
            package_name,
        )

        command = [
            self.python_executable,
            "-m",
            "pip",
            "install",
            pypi_package_name,
        ]

        logger.info(
            "Installing missing dependency: %s",
            package_name,
        )

        started = time.monotonic()

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=install_timeout,
            )

            duration = time.monotonic() - started

            return {
                "success": process.returncode == 0,
                "package": package_name,
                "pypi_package": pypi_package_name,
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "duration_seconds": duration,
            }

        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started

            return {
                "success": False,
                "package": package_name,
                "return_code": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "duration_seconds": duration,
                "error": (
                    f"Package installation timed out after "
                    f"{install_timeout} seconds."
                ),
            }

        except Exception as exc:
            duration = time.monotonic() - started

            return {
                "success": False,
                "package": package_name,
                "return_code": None,
                "stdout": "",
                "stderr": "",
                "duration_seconds": duration,
                "error": str(exc),
            }

    # ========================================================================
    # LLM repair
    # ========================================================================

    def _repair_experiment_with_llm(
        self,
        generated_code_path: Path,
        generated_result: Dict[str, Any],
        run_dir: Path,
    ) -> Optional[str]:
        """
        Ask CodeGenerationAgent to repair failed generated experiment code.

        Crucially, the repair specification preserves the actual scientific
        requirements from the selected hypothesis and evidence instead of
        replacing them with hard-coded classification metrics.
        """
        try:
            from ..agents_modules.code_generation_agent import (
                CodeGenerationAgent,
            )
        except ImportError:
            try:
                from agents_modules.code_generation_agent import (
                    CodeGenerationAgent,
                )
            except ImportError as exc:
                logger.warning(
                    "Could not import CodeGenerationAgent for repair: %s",
                    exc,
                )
                return None

        try:
            current_code = generated_code_path.read_text(
                encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(
                "Could not read generated code for repair: %s",
                exc,
            )
            return None

        experiment_plan = generated_result.get(
            "experiment_plan",
            {},
        )

        if not isinstance(experiment_plan, dict):
            experiment_plan = {}

        selected_hypothesis = (
            generated_result.get("selected_hypothesis")
            or generated_result.get("hypothesis")
            or experiment_plan.get("selected_hypothesis")
            or experiment_plan.get("hypothesis")
            or ""
        )

        evidence_sources = (
            generated_result.get("evidence_sources")
            or generated_result.get("evidence")
            or experiment_plan.get("evidence_sources")
            or []
        )

        evaluation_metrics = (
            generated_result.get("evaluation_metrics")
            or experiment_plan.get("evaluation_metrics")
            or []
        )

        metric_definitions = (
            generated_result.get("metric_definitions")
            or experiment_plan.get("metric_definitions")
            or {}
        )

        research_goal = (
            generated_result.get("research_goal")
            or generated_result.get("research_question")
            or experiment_plan.get("research_goal")
            or experiment_plan.get("research_question")
            or ""
        )

        dataset_path = (
            generated_result.get("dataset_path")
            or experiment_plan.get("dataset_path")
            or ""
        )

        model = (
            generated_result.get("model")
            or generated_result.get("model_name")
            or experiment_plan.get("model")
            or ""
        )

        # Preserve all evidence-derived scientific requirements.
        specification = {
            "research_goal": research_goal,
            "selected_hypothesis": selected_hypothesis,
            "evidence_sources": evidence_sources,
            "dataset_path": dataset_path,
            "model": model,
            "evaluation_metrics": evaluation_metrics,
            "metric_definitions": metric_definitions,
            "experiment_plan": experiment_plan,
            "repair_requirements": [
                (
                    "Fix the execution failure without silently changing "
                    "the selected hypothesis."
                ),
                (
                    "Preserve the intended dataset, target variable, "
                    "model architecture, and methodology whenever possible."
                ),
                (
                    "Preserve the evidence-derived evaluation metrics. "
                    "Do not replace them with generic accuracy, precision, "
                    "recall, or F1 unless those metrics are actually part "
                    "of the experiment requirements."
                ),
                (
                    "Do not fabricate scientific results or metric values."
                ),
                (
                    "If a requested metric cannot be computed from the "
                    "available experiment, report it as unavailable rather "
                    "than inventing a value."
                ),
                (
                    "Keep generated outputs compatible with "
                    "ExperimentRunner and ExperimentComparator."
                ),
            ],
        }

        try:
            agent = CodeGenerationAgent()

            repair_method = getattr(
                agent,
                "repair_generated_code",
                None,
            )

            if not callable(repair_method):
                logger.warning(
                    "CodeGenerationAgent does not provide "
                    "repair_generated_code()."
                )
                return None

            repair_result = repair_method(
                specification=specification,
                generated_code=current_code,
                execution_result=generated_result.get(
                    "execution",
                    {},
                ),
            )

            if isinstance(repair_result, dict):
                if not repair_result.get("success", False):
                    logger.warning(
                        "LLM experiment repair returned failure: %s",
                        repair_result.get("errors", "no error details"),
                    )
                    return None
                repaired_code = repair_result.get("pytorch_code")
            else:
                repaired_code = repair_result

        except Exception as exc:
            logger.warning(
                "LLM experiment repair failed: %s",
                exc,
            )
            return None

        if not repaired_code or not isinstance(repaired_code, str):
            logger.warning(
                "LLM repair returned no usable code."
            )
            return None

        try:
            ast.parse(repaired_code)
        except SyntaxError as exc:
            logger.warning(
                "LLM repair returned invalid Python: %s",
                exc,
            )
            return None

        if repaired_code.strip() == current_code.strip():
            logger.warning(
                "LLM repair returned unchanged code."
            )
            return None

        repaired_path = run_dir / "generated_experiment_repaired.py"

        try:
            repaired_path.write_text(
                repaired_code,
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                "Could not save repaired experiment code: %s",
                exc,
            )

        return repaired_code

    # ========================================================================
    # Execution
    # ========================================================================

    # print(
    #     "execution_remaining_seconds signature:",
    #     inspect.signature(execution_remaining_seconds),
    # )

    def execute(
        self,
        code_path: Path,
        run_dir: Path,
        dataset_path: Optional[str] = None,
        extra_environment: Optional[Dict[str, str]] = None,
        generated_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute generated experiment code.

        Up to 10 attempts may be made. Attempts can include:

        1. Normal execution.
        2. Missing dependency installation.
        3. LLM-assisted code repair.

        The total execution budget is bounded by timeout_seconds.
        """
        generated_result = generated_result or {}

        started_at = time.monotonic()

        attempts: List[Dict[str, Any]] = []
        installations: List[Dict[str, Any]] = []
        repairs: List[Dict[str, Any]] = []

        current_code_path = code_path

        final_return_code: Optional[int] = None
        final_stdout = ""
        final_stderr = ""
        process: Optional[subprocess.CompletedProcess] = None

        max_attempts = 10

        for attempt_number in range(1, max_attempts + 1):
            elapsed = time.monotonic() - started_at
            remaining = self.timeout_seconds - elapsed

            if remaining <= 0:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "timeout_before_attempt",
                    }
                )
                break

            if execution_cancelled():
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "cancelled_before_attempt",
                    }
                )
                break

            # Allow the global execution-control mechanism to further reduce
            # the available time.
            controlled_remaining = execution_remaining_seconds()

            if controlled_remaining is not None:
                remaining = min(
                    remaining,
                    max(0.0, float(controlled_remaining)),
                )

            if remaining <= 0:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "timeout_before_attempt",
                    }
                )
                break

            environment = self.build_environment(
                run_dir=run_dir,
                dataset_path=dataset_path,
                extra_environment=extra_environment,
            )

            command = [
                self.python_executable,
                str(current_code_path),
            ]

            attempt_started = time.monotonic()

            attempt_record: Dict[str, Any] = {
                "attempt": attempt_number,
                "code_path": str(current_code_path),
                "command": command,
            }

            logger.info(
                "Running generated experiment attempt %s/%s",
                attempt_number,
                max_attempts,
            )

            try:
                process = subprocess.run(
                    command,
                    cwd=str(run_dir),
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=max(1, int(remaining)),
                )

                attempt_duration = time.monotonic() - attempt_started

                final_return_code = process.returncode
                final_stdout = process.stdout or ""
                final_stderr = process.stderr or ""

                attempt_record.update(
                    {
                        "status": (
                            "success"
                            if process.returncode == 0
                            else "failed"
                        ),
                        "return_code": process.returncode,
                        "duration_seconds": attempt_duration,
                        "stdout": final_stdout,
                        "stderr": final_stderr,
                    }
                )

                attempts.append(attempt_record)

                # Successful execution ends the retry loop.
                if process.returncode == 0:
                    break

                # ----------------------------------------------------------
                # Missing dependency handling
                # ----------------------------------------------------------

                missing_module = self._extract_missing_module(
                    final_stderr
                )

                if missing_module:
                    elapsed = time.monotonic() - started_at
                    remaining_after_failure = (
                        self.timeout_seconds - elapsed
                    )

                    installation = self._install_package(
                        missing_module,
                        remaining_seconds=remaining_after_failure,
                    )

                    installations.append(installation)

                    if installation.get("success"):
                        continue

                # ----------------------------------------------------------
                # LLM repair
                # ----------------------------------------------------------

                elapsed = time.monotonic() - started_at
                remaining_after_failure = (
                    self.timeout_seconds - elapsed
                )

                if remaining_after_failure <= 0:
                    break

                repair_started = time.monotonic()

                repaired_code = self._repair_experiment_with_llm(
                    generated_code_path=current_code_path,
                    generated_result={
                        **generated_result,
                        "execution": {
                            "return_code": process.returncode,
                            "stdout": final_stdout,
                            "stderr": final_stderr,
                        },
                    },
                    run_dir=run_dir,
                )

                repair_duration = time.monotonic() - repair_started

                # Support both the current repair API:
                #
                #     repaired_code -> str | None
                #
                # and older/test-compatible APIs:
                #
                #     (success, repaired_code, error_message)
                #
                # This prevents a failure tuple from being interpreted as
                # successfully repaired code merely because the tuple is truthy.
                repair_error = None

                if isinstance(repaired_code, tuple):
                    repair_success = bool(
                        repaired_code[0]
                        if len(repaired_code) > 0
                        else False
                    )

                    repaired_code_text = (
                        repaired_code[1]
                        if len(repaired_code) > 1
                        else ""
                    )

                    repair_error = (
                        repaired_code[2]
                        if len(repaired_code) > 2
                        else None
                    )

                    if not repair_success:
                        repaired_code_text = ""

                    repaired_code = repaired_code_text

                if repaired_code and isinstance(repaired_code, str):
                    repaired_path = (
                        run_dir / "generated_experiment_repaired.py"
                    )

                    current_code_path = repaired_path

                    repairs.append(
                        {
                            "attempt": attempt_number,
                            "status": "repaired",
                            "code_path": str(repaired_path),
                            "duration_seconds": repair_duration,
                        }
                    )

                    continue

                repair_record = {
                    "attempt": attempt_number,
                    "status": "repair_unavailable",
                    "duration_seconds": repair_duration,
                }

                if repair_error:
                    repair_record["error"] = str(repair_error)

                repairs.append(repair_record)

            except subprocess.TimeoutExpired as exc:
                attempt_duration = time.monotonic() - attempt_started

                stdout = exc.stdout or ""
                stderr = exc.stderr or ""

                if isinstance(stdout, bytes):
                    stdout = stdout.decode(
                        "utf-8",
                        errors="replace",
                    )

                if isinstance(stderr, bytes):
                    stderr = stderr.decode(
                        "utf-8",
                        errors="replace",
                    )

                final_return_code = None
                final_stdout = stdout
                final_stderr = stderr

                attempt_record.update(
                    {
                        "status": "timeout",
                        "return_code": None,
                        "duration_seconds": attempt_duration,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                )

                attempts.append(attempt_record)

                stdout_path = run_dir / "stdout.txt"
                stderr_path = run_dir / "stderr.txt"

                stdout_path.write_text(
                    stdout,
                    encoding="utf-8",
                )

                stderr_path.write_text(
                    stderr,
                    encoding="utf-8",
                )

                total_execution_seconds = time.monotonic() - started_at

                result = {
                    "success": False,
                    "status": "timeout",
                    "return_code": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "timeout_seconds": self.timeout_seconds,
                    "error": (
                        f"Experiment execution timeout after "
                        f"{self.timeout_seconds} seconds."
                    ),
                    "execution_seconds": total_execution_seconds,
                    "total_execution_seconds": total_execution_seconds,
                    "attempts": attempts,
                    "installations": installations,
                    "repairs": repairs,
                    "code_path": str(current_code_path),
                    "cancelled": False,
                }

                self._write_json(
                    run_dir / "execution_result.json",
                    result,
                )

                return result

            except Exception as exc:
                attempt_duration = time.monotonic() - attempt_started

                final_return_code = None
                final_stdout = ""
                final_stderr = str(exc)

                attempt_record.update(
                    {
                        "status": "exception",
                        "return_code": None,
                        "duration_seconds": attempt_duration,
                        "stdout": "",
                        "stderr": str(exc),
                    }
                )

                attempts.append(attempt_record)

                # Try another attempt only if meaningful time remains.
                elapsed = time.monotonic() - started_at

                if elapsed >= self.timeout_seconds:
                    break

        total_execution_seconds = time.monotonic() - started_at

        cancelled = execution_cancelled()

        if cancelled:
            status = "cancelled"
        elif final_return_code == 0:
            status = "success"
        elif total_execution_seconds >= self.timeout_seconds:
            status = "timeout"
        else:
            status = "failed"

        result = {
            "success": final_return_code == 0 and not cancelled,
            "status": status,
            "return_code": final_return_code,
            "stdout": final_stdout,
            "stderr": final_stderr,
            "execution_seconds": total_execution_seconds,
            "total_execution_seconds": total_execution_seconds,
            "attempts": attempts,
            "installations": installations,
            "repairs": repairs,
            "code_path": str(current_code_path),
            "cancelled": cancelled,
        }

        # Save execution information independently so that failed experiments
        # are still diagnosable.
        self._write_json(
            run_dir / "execution_result.json",
            result,
        )

        return result

    # ========================================================================
    # Output discovery
    # ========================================================================

    def find_output_file(
        self,
        run_dir: Path,
        filename: str,
    ) -> Optional[Path]:
        """
        Find an output file in the run directory.
        """
        direct_path = run_dir / filename

        if direct_path.exists() and direct_path.is_file():
            return direct_path

        matches = list(run_dir.rglob(filename))

        if matches:
            return matches[0]

        return None

    def collect_metrics(
        self,
        run_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Collect metrics from metrics.json.

        No specific scientific metric names are required.
        """
        metrics_path = self.find_output_file(
            run_dir,
            DEFAULT_OUTPUT_FILES["metrics"],
        )

        if not metrics_path:
            return None

        return self._read_json(metrics_path)

    def collect_training_history(
        self,
        run_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Collect optional training history.
        """
        history_path = self.find_output_file(
            run_dir,
            DEFAULT_OUTPUT_FILES["training_history"],
        )

        if not history_path:
            return None

        return self._read_json(history_path)

    def find_checkpoint(
        self,
        run_dir: Path,
    ) -> Optional[Path]:
        """
        Find an optional model checkpoint.
        """
        checkpoint_name = DEFAULT_OUTPUT_FILES["checkpoint"]

        checkpoint = self.find_output_file(
            run_dir,
            checkpoint_name,
        )

        return checkpoint

    def find_visualizations(
        self,
        run_dir: Path,
    ) -> List[str]:
        """
        Find optional visualization files.
        """
        visualizations: List[str] = []

        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() in DEFAULT_VISUALIZATION_EXTENSIONS:
                visualizations.append(str(path))

        return sorted(visualizations)

    def collect_outputs(
        self,
        run_dir: Path,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Collect experiment outputs.

        Scientific metrics are preserved from metrics.json without requiring
        specific metric names. Optional experiment artifacts are collected
        separately.
        """
        execution_result = execution_result or {}

        metrics = self.collect_metrics(run_dir)
        training_history = self.collect_training_history(run_dir)
        checkpoint = self.find_checkpoint(run_dir)
        visualizations = self.find_visualizations(run_dir)

        summary_path = self.find_output_file(
            run_dir,
            DEFAULT_OUTPUT_FILES["summary"],
        )

        summary = (
            self._read_json(summary_path)
            if summary_path
            else None
        )

        # Preserve the absence of metrics.json.
        # Do not create a synthetic scientific metric from runner execution time.
        # Otherwise an experiment that produced no metrics.json could incorrectly
        # pass validation just because the runner itself measured execution time.
        if isinstance(metrics, dict):
            execution_seconds = execution_result.get(
                "execution_seconds"
            )

            if self._is_finite_number(execution_seconds):
                metrics.setdefault(
                    "total_execution_seconds",
                    execution_seconds,
                )

        outputs = {
            "metrics": metrics,
            "training_history": training_history,
            "summary": summary,
            "experiment_summary": summary,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "visualizations": visualizations,
        }

        return outputs

    # ========================================================================
    # Output validation
    # ========================================================================

    def validate_outputs(
        self,
        outputs: Dict[str, Any],
        execution_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate experiment outputs.

        Only the existence of usable scientific metrics is mandatory.

        Optional:
            - training history
            - checkpoint
            - visualizations
            - confusion matrix
            - timing metrics

        This prevents a latency, throughput, cryptographic-overhead, or other
        non-classification experiment from being rejected simply because it
        does not produce accuracy/F1/confusion-matrix artifacts.
        """
        warnings: List[str] = []

        execution_success = bool(
            execution_result.get("success")
        )

        if not execution_success:
            warnings.append(
                "Experiment execution was not successful."
            )

        metrics = outputs.get("metrics")

        if metrics is None:
            warnings.append(
                "metrics.json was not found."
            )
            metrics = {}

        if not isinstance(metrics, dict):
            warnings.append(
                "Experiment metrics must be stored as a JSON object."
            )
            metrics = {}

        if not metrics:
            warnings.append(
                "No experiment metrics were produced."
            )

        if self._has_nonfinite_number(metrics):
            warnings.append(
                "Experiment metrics contain NaN or Infinity values."
            )

        numeric_metrics: Dict[str, Any] = {}
        structured_metrics: Dict[str, Any] = {}

        for name, value in metrics.items():
            if self._is_finite_number(value):
                numeric_metrics[str(name)] = value
            elif isinstance(value, (dict, list, tuple)):
                structured_metrics[str(name)] = value

        if not numeric_metrics and not structured_metrics:
            warnings.append(
                "No usable metrics were found in metrics.json."
            )

        metric_definitions = outputs.get(
            "metric_definitions"
        )

        if metric_definitions is not None:
            if not isinstance(metric_definitions, dict):
                warnings.append(
                    "metric_definitions must be a JSON object when provided."
                )

        # Training history is optional.
        training_history = outputs.get(
            "training_history"
        )

        if training_history is not None:
            if not isinstance(training_history, dict):
                warnings.append(
                    "training_history.json is present but is not a JSON object."
                )
            elif self._has_nonfinite_number(training_history):
                warnings.append(
                    "Training history contains NaN or Infinity values."
                )

        # Checkpoint is optional.
        checkpoint = outputs.get("checkpoint")

        if checkpoint:
            checkpoint_path = Path(checkpoint)

            if not checkpoint_path.exists():
                warnings.append(
                    "Checkpoint path was reported but the file does not exist."
                )

        # Visualizations are optional and do not invalidate the experiment.
        # When visualization output is collected, report missing standard
        # visualization artifacts as warnings for diagnostics.
        visualizations = outputs.get(
            "visualizations",
            [],
        )

        if visualizations is None:
            visualizations = []

        if not isinstance(visualizations, list):
            warnings.append(
                "Visualization output must be a list."
            )
            visualizations = []

        else:
            visualization_stems = {
                Path(str(path)).stem
                for path in visualizations
            }

            missing_visualizations = (
                REQUIRED_VISUALIZATION_STEMS
                - visualization_stems
            )

            if missing_visualizations:
                warnings.append(
                    "Missing required visualizations: "
                    + ", ".join(sorted(missing_visualizations))
                )

        # Do not make warnings automatically invalidate a scientifically
        # usable experiment. Only execution failure or absence of usable
        # metrics makes the result invalid.
        metrics_valid = bool(
            metrics
            and isinstance(metrics, dict)
            and (
                numeric_metrics
                or structured_metrics
            )
            and not self._has_nonfinite_number(metrics)
        )

        valid = execution_success and metrics_valid

        return {
            "valid": valid,
            "warnings": warnings,
            "metrics_available": sorted(
                numeric_metrics.keys()
            ),
            "structured_metrics_available": sorted(
                structured_metrics.keys()
            ),
            "metric_count": len(
                numeric_metrics
            ) + len(
                structured_metrics
            ),
            "has_training_history": training_history is not None,
            "has_checkpoint": bool(checkpoint),
            "visualization_count": len(visualizations)
            if isinstance(visualizations, list)
            else 0,
        }

    # ========================================================================
    # Final result
    # ========================================================================

    def _load_metric_definitions(
        self,
        run_dir: Path,
        generated_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load metric definitions from generated result or metadata.
        """
        generated_result = generated_result or {}

        definitions = (
            generated_result.get("metric_definitions")
            or (
                generated_result.get("experiment_plan", {}).get(
                    "metric_definitions",
                    {}
                )
                if isinstance(
                    generated_result.get("experiment_plan"),
                    dict,
                )
                else {}
            )
            or {}
        )

        if isinstance(definitions, dict) and definitions:
            return definitions

        metadata_path = run_dir / "experiment_metadata.json"

        metadata = self._read_json(metadata_path)

        if metadata:
            metadata_definitions = metadata.get(
                "metric_definitions",
                {},
            )

            if isinstance(metadata_definitions, dict):
                return metadata_definitions

        return {}

    def _load_evaluation_metrics(
        self,
        run_dir: Path,
        generated_result: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Load the metrics that the generated experiment was intended to
        evaluate.
        """
        generated_result = generated_result or {}

        experiment_plan = generated_result.get(
            "experiment_plan",
            {},
        )

        if not isinstance(experiment_plan, dict):
            experiment_plan = {}

        evaluation_metrics = (
            generated_result.get("evaluation_metrics")
            or experiment_plan.get("evaluation_metrics")
            or []
        )

        if isinstance(evaluation_metrics, list) and evaluation_metrics:
            return evaluation_metrics

        metadata_path = run_dir / "experiment_metadata.json"

        metadata = self._read_json(metadata_path)

        if metadata:
            metadata_metrics = metadata.get(
                "evaluation_metrics",
                [],
            )

            if isinstance(metadata_metrics, list):
                return metadata_metrics

        return []

    # ========================================================================
    # Main runner
    # ========================================================================

    def run(
        self,
        generated_code: str,
        dataset_path: Optional[str] = None,
        experiment_id: Optional[str] = None,
        model: Optional[str] = None,
        model_recommendation: Optional[str] = None,
        experiment_plan: Optional[Dict[str, Any]] = None,
        assumptions: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        hypothesis: Optional[Any] = None,
        evidence_sources: Optional[List[Any]] = None,
        evaluation_metrics: Optional[List[Any]] = None,
        metric_definitions: Optional[Dict[str, Any]] = None,
        extra_environment: Optional[Dict[str, str]] = None,
        generated_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run generated experiment code from start to finish.
        """
        generated_result = generated_result or {}

        # --------------------------------------------------------------
        # Resolve scientific context
        # --------------------------------------------------------------

        if evaluation_metrics is None:
            evaluation_metrics = (
                generated_result.get("evaluation_metrics")
                or (
                    experiment_plan.get("evaluation_metrics", [])
                    if isinstance(experiment_plan, dict)
                    else []
                )
                or []
            )

        if metric_definitions is None:
            metric_definitions = (
                generated_result.get("metric_definitions")
                or (
                    experiment_plan.get("metric_definitions", {})
                    if isinstance(experiment_plan, dict)
                    else {}
                )
                or {}
            )

        if hypothesis is None:
            hypothesis = (
                generated_result.get("selected_hypothesis")
                or generated_result.get("hypothesis")
                or (
                    experiment_plan.get("selected_hypothesis")
                    if isinstance(experiment_plan, dict)
                    else None
                )
            )

        if evidence_sources is None:
            evidence_sources = (
                generated_result.get("evidence_sources")
                or generated_result.get("evidence")
                or []
            )

        # --------------------------------------------------------------
        # Validate generated code
        # --------------------------------------------------------------

        if not generated_code or not generated_code.strip():
            return {
                "success": False,
                "status": "invalid_generated_code",
                "error": "Generated experiment code is empty.",
            }

        try:
            ast.parse(generated_code)
        except SyntaxError as exc:
            return {
                "success": False,
                "status": "invalid_generated_code",
                "error": f"Generated experiment code is invalid: {exc}",
            }

        # --------------------------------------------------------------
        # Validate dataset if one was explicitly supplied
        # --------------------------------------------------------------

        if dataset_path:
            dataset = Path(dataset_path)

            if not dataset.exists():
                return {
                    "success": False,
                    "status": "dataset_not_found",
                    "error": (
                        f"Dataset path does not exist: "
                        f"{dataset_path}"
                    ),
                }

        # --------------------------------------------------------------
        # Create run directory
        # --------------------------------------------------------------

        try:
            run_dir = self.create_run_directory(
                experiment_id=experiment_id,
            )
        except Exception as exc:
            return {
                "success": False,
                "status": "run_directory_error",
                "error": str(exc),
            }

        # --------------------------------------------------------------
        # Save metadata
        # --------------------------------------------------------------

        try:
            metadata_path = self.save_experiment_metadata(
                run_dir=run_dir,
                generated_result=generated_result,
                dataset_path=dataset_path,
                model=model,
                model_recommendation=model_recommendation,
                experiment_plan=experiment_plan,
                assumptions=assumptions,
                dependencies=dependencies,
                hypothesis=hypothesis,
                evidence_sources=evidence_sources,
                evaluation_metrics=evaluation_metrics,
                metric_definitions=metric_definitions,
            )
        except Exception as exc:
            logger.warning(
                "Could not save experiment metadata: %s",
                exc,
            )
            metadata_path = None

        # --------------------------------------------------------------
        # Save generated code
        # --------------------------------------------------------------

        try:
            code_path = self.prepare_generated_code(
                generated_code=generated_code,
                run_dir=run_dir,
            )
        except Exception as exc:
            return {
                "success": False,
                "status": "code_preparation_error",
                "error": str(exc),
                "run_dir": str(run_dir),
            }

        # --------------------------------------------------------------
        # Execute
        # --------------------------------------------------------------

        execution_result = self.execute(
            code_path=code_path,
            run_dir=run_dir,
            dataset_path=dataset_path,
            extra_environment=extra_environment,
            generated_result={
                **generated_result,
                "selected_hypothesis": hypothesis,
                "evidence_sources": evidence_sources,
                "evaluation_metrics": evaluation_metrics,
                "metric_definitions": metric_definitions,
                "experiment_plan": experiment_plan or {},
            },
        )

        # --------------------------------------------------------------
        # Collect outputs
        # --------------------------------------------------------------

        outputs = self.collect_outputs(
            run_dir=run_dir,
            execution_result=execution_result,
        )

        # Attach scientific context to collected outputs.
        outputs["evaluation_metrics"] = self._load_evaluation_metrics(
            run_dir=run_dir,
            generated_result={
                **generated_result,
                "evaluation_metrics": evaluation_metrics,
                "experiment_plan": experiment_plan or {},
            },
        )

        outputs["metric_definitions"] = self._load_metric_definitions(
            run_dir=run_dir,
            generated_result={
                **generated_result,
                "metric_definitions": metric_definitions,
                "experiment_plan": experiment_plan or {},
            },
        )

        # --------------------------------------------------------------
        # Validate outputs
        # --------------------------------------------------------------

        validation = self.validate_outputs(
            outputs=outputs,
            execution_result=execution_result,
        )

        success = bool(
            execution_result.get("success")
            and validation.get("valid")
        )

        if success:
            status = "success"
        elif execution_result.get("status") == "cancelled":
            status = "cancelled"
        elif execution_result.get("status") == "timeout":
            status = "timeout"
        elif execution_result.get("success") and not validation.get("valid"):
            status = "invalid_outputs"
        else:
            status = execution_result.get("status", "failed")

        # --------------------------------------------------------------
        # Final result
        # --------------------------------------------------------------

        result = {
            "success": success,
            "status": status,

            # Standardized paths
            "run_directory": str(run_dir),
            "generated_code_path": str(code_path),
            "dataset_path": dataset_path,

            # Backward-compatible names
            "run_dir": str(run_dir),
            "code_path": str(code_path),

            "metadata_path": (
                str(metadata_path)
                if metadata_path
                else None
            ),

            "execution": execution_result,
            "outputs": outputs,

            # Standardized validation name
            "output_validation": validation,

            # Backward-compatible name
            "validation": validation,

            "total_execution_seconds": execution_result.get(
                "total_execution_seconds",
                execution_result.get("execution_seconds", 0.0),
            ),

            "errors": [],

            "selected_hypothesis": hypothesis,
            "hypothesis": hypothesis,
            "evidence_sources": evidence_sources,

            "evaluation_metrics": outputs.get(
                "evaluation_metrics",
                evaluation_metrics or [],
            ),

            "metric_definitions": outputs.get(
                "metric_definitions",
                metric_definitions or {},
            ),

            "metrics": outputs.get(
                "metrics",
                {},
            ),
        }

        errors = []

        if not execution_result.get("success"):
            error_message = execution_result.get("error")

            if error_message:
                errors.append(str(error_message))

            stderr = execution_result.get("stderr")
            if stderr:
                errors.append(str(stderr))

        if not validation.get("valid"):
            errors.extend(
                validation.get("warnings", [])
            )

        result["errors"] = errors

        # --------------------------------------------------------------
        # Save complete runner result
        # --------------------------------------------------------------

        result_path = run_dir / "runner_result.json"

        result["result_path"] = str(result_path)

        try:
            self._write_json(
                result_path,
                result,
            )
        except Exception as exc:
            logger.warning(
                "Could not save runner_result.json: %s",
                exc,
            )

        return result

    # ========================================================================
    # Generated-result adapter
    # ========================================================================

    def run_generated_result(
        self,
        generated_result: Dict[str, Any],
        dataset_path: Optional[str] = None,
        experiment_id: Optional[str] = None,
        extra_environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run an experiment returned by CodeGenerationAgent.

        This adapter supports common generated-result layouts while preserving
        the scientific context required by the downstream comparator.
        """
        if not isinstance(generated_result, dict):
            return {
                "success": False,
                "status": "invalid_generated_result",
                "error": "generated_result must be a dictionary.",
            }

        generated_code = (
            generated_result.get("pytorch_code")
            or generated_result.get("generated_code")
            or generated_result.get("code")
        )

        if not generated_code:
            experiment = generated_result.get(
                "experiment",
                {},
            )

            if isinstance(experiment, dict):
                generated_code = (
                    experiment.get("pytorch_code")
                    or experiment.get("generated_code")
                    or experiment.get("code")
                )

        if not generated_code:
            return {
                "success": False,
                "status": "missing_generated_code",
                "error": (
                    "No generated experiment code was found in "
                    "generated_result."
                ),
            }

        experiment_plan = generated_result.get(
            "experiment_plan",
            {},
        )

        if not isinstance(experiment_plan, dict):
            experiment_plan = {}

        if dataset_path is None:
            dataset_path = (
                generated_result.get("dataset_path")
                or experiment_plan.get("dataset_path")
            )

        experiment_id = (
            experiment_id
            or generated_result.get("experiment_id")
            or generated_result.get("id")
        )

        return self.run(
            generated_code=generated_code,
            dataset_path=dataset_path,
            experiment_id=experiment_id,
            model=(
                generated_result.get("model")
                or generated_result.get("model_name")
                or experiment_plan.get("model")
            ),
            model_recommendation=generated_result.get(
                "model_recommendation"
            ),
            experiment_plan=experiment_plan,
            assumptions=generated_result.get(
                "assumptions",
                [],
            ),
            dependencies=generated_result.get(
                "dependencies",
                [],
            ),
            hypothesis=(
                generated_result.get("selected_hypothesis")
                or generated_result.get("hypothesis")
                or experiment_plan.get("selected_hypothesis")
                or experiment_plan.get("hypothesis")
            ),
            evidence_sources=(
                generated_result.get("evidence_sources")
                or generated_result.get("evidence")
                or []
            ),
            evaluation_metrics=(
                generated_result.get("evaluation_metrics")
                or experiment_plan.get("evaluation_metrics")
                or []
            ),
            metric_definitions=(
                generated_result.get("metric_definitions")
                or experiment_plan.get("metric_definitions")
                or {}
            ),
            extra_environment=extra_environment,
            generated_result=generated_result,
        )


# ============================================================================
# Convenience function
# ============================================================================


def run_experiment(
    generated_code: str,
    dataset_path: Optional[str] = None,
    output_directory: str = "experiment_runs",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    experiment_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience wrapper around ExperimentRunner.
    """
    runner = ExperimentRunner(
        output_directory=output_directory,
        timeout_seconds=timeout_seconds,
    )

    return runner.run(
        generated_code=generated_code,
        dataset_path=dataset_path,
        experiment_id=experiment_id,
        **kwargs,
    )