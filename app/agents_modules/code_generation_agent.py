"""
Code Generation Agent.

This module connects the AI Co-Scientist hypothesis workflow with the
automated deep-learning experiment pipeline.

Workflow:

    Selected Hypothesis
            +
    Research Goal
            +
    Reflection Report
            +
    Experiment Specification
            ↓
    CodeGenerationAgent
            ↓
    Model / Approach Recommendation
            +
    Experiment Plan
            +
    Executable PyTorch Code

The generated code is intended to be executed later by the experiment
runner. This agent is responsible for CODE GENERATION and does not
execute training itself.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import config
from ..utils import logger

# ============================================================
# LLM Boundary
# ============================================================

def _call_llm(*args, **kwargs):
    """
    Use the existing application LLM façade.

    This follows the same pattern used by generation_helpers.py
    so that the project's existing LLM configuration and mocks
    remain effective.
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

    Falls back to the supplied default when the configuration
    does not contain a valid value.
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
    Generates an executable PyTorch experiment from a selected
    AI Co-Scientist hypothesis.

    The agent does not execute generated code.

    Responsibilities:
        1. Validate the experiment specification.
        2. Build a detailed code-generation prompt.
        3. Ask the configured LLM for a structured experiment plan.
        4. Extract the generated PyTorch code.
        5. Validate the generated response.
        6. Save generated code when requested.
        7. Return structured results to ExperimentOrchestrator.
    """

    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 12000

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
            Optional directory where generated Python files
            are saved.
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
        Serialize the selected Hypothesis using the actual
        fields used by app/models.py.
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
                reflection_data = (
                    self._to_serializable(
                        reflection_report
                    )
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
        Serialize ResearchGoal using the actual fields in
        app/models.py.
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

    # ========================================================
    # Prompt Construction
    # ========================================================

    def build_system_prompt(self) -> str:
        """
        Build the system prompt used for experiment code generation.
        """
        return """
You are the Code Generation Agent in an AI Co-Scientist system.

Your task is to convert a scientifically evaluated machine-learning
hypothesis into a reproducible PyTorch experiment.

The selected hypothesis has already passed the AI Co-Scientist stages
including generation, reflection, ranking, evolution, proximity
analysis, and meta-review.

You must therefore implement the selected research idea faithfully.

IMPORTANT RULES:

1. Use PyTorch.
2. Generate executable Python code.
3. Do not execute the generated code.
4. Do not invent unavailable datasets.
5. Use the dataset path supplied in the experiment specification.
6. The current dataset is 5G-NIDD.
7. Treat the dataset as an offline/local dataset.
8. Include deterministic/reproducible random seeds.
9. Include preprocessing appropriate for tabular/network intrusion data.
10. Handle categorical and numerical features appropriately.
11. Avoid data leakage.
12. Create separate training, validation, and test partitions.
13. Automatically determine the number of classes from the training data
    where practical.
14. Use a PyTorch Dataset/DataLoader design.
15. Implement the architecture described or implied by the hypothesis.
16. Do not silently replace the proposed architecture with an unrelated
    model.
17. Include a complete training loop.
18. Include validation after each epoch.
19. Save the best model checkpoint based on validation performance.
20. Evaluate on the held-out test set.
21. Report accuracy, weighted precision, weighted recall, weighted F1,
    and confusion matrix.
22. Record training and evaluation execution time.
23. Save training history.
24. Generate useful visualizations.
25. Keep the code self-contained.
26. Include comments explaining important implementation choices.
27. Make reasonable assumptions explicit in the generated experiment
    metadata.
28. Do not use placeholder code such as "TODO", "implement here", or
    "pass" for required experiment functionality.
29. Return ONLY complete executable Python source code.
30. The ExperimentRunner provides the environment variable
    EXPERIMENT_OUTPUT_DIR. All generated artifacts MUST be saved
    inside this directory.
31. Use:

        output_dir = Path(
            os.environ.get(
                "EXPERIMENT_OUTPUT_DIR",
                "."
            )
        )
32. Save the following files using these exact names:

        metrics.json
        training_history.json
        best_model.pt
33. Save all visualization files inside EXPERIMENT_OUTPUT_DIR
    or one of its subdirectories.
34. The dataset path may be provided through the DATASET_PATH
    environment variable. Prefer DATASET_PATH when it is available.

The generated code must be suitable for later automated execution by
an Experiment Runner.
""".strip()

    def build_user_prompt(
        self,
        specification: Dict[str, Any],
    ) -> str:
        """
        Build the user prompt containing the complete experiment
        specification.
        """
        prompt_specification = dict(specification)
        selected_hypothesis = specification.get(
            "selected_hypothesis",
            {},
        )
        if isinstance(selected_hypothesis, dict):
            prompt_specification["selected_hypothesis"] = {
                key: selected_hypothesis.get(key)
                for key in ("hypothesis_id", "title", "text")
            }

        prompt_specification.pop(
            "ai_co_scientist_provenance",
            None,
        )
        prompt_specification["scientific_evaluation"] = {}

        specification_json = json.dumps(
            self._to_serializable(
                prompt_specification
            ),
            ensure_ascii=False,
            indent=2,
        )

        return f"""
    Generate only the complete executable Python source code for a PyTorch
    experiment from the following AI Co-Scientist experiment specification.

EXPERIMENT SPECIFICATION
========================

{specification_json}

OUTPUT REQUIREMENTS
===================

Return only Python source code. Do not return JSON, Markdown fences,
explanations, analysis, or commentary.

The generated code must:

- load the specified local dataset;
- preprocess the data;
- split the data into train/validation/test sets;
- prevent preprocessing leakage from validation/test data;
- construct the selected model;
- train the model;
- validate the model;
- save the best checkpoint;
- load the best checkpoint for final testing;
- calculate the requested evaluation metrics;
- save metrics to JSON;
- save training history;
- generate visualizations;
- print a concise final experiment summary.

If the hypothesis proposes a particular neural architecture,
implement that architecture rather than defaulting to a generic
MLP.

If the hypothesis does not provide enough implementation detail,
make the smallest scientifically reasonable assumptions in the code.
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

        # Remove Markdown fences if present.
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
            # Attempt to locate the first JSON object.
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
                except (SyntaxError, ValueError) as literal_error:
                    raise ValueError(
                        f"Invalid structured response returned by CodeGenerationAgent: {literal_error}"
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
        """Build a minimal result when the model returns fenced Python only."""
        matches = re.findall(
            r"```(?:python|py)\s*\n(.*?)```",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not matches:
            return None

        code = max(matches, key=len).strip()
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
        """Extract executable Python when the model ignores the JSON wrapper."""
        fenced = CodeGenerationAgent.extract_fenced_python(response)
        if fenced is not None:
            return fenced

        source_start = response.find("import torch")
        if source_start == -1:
            source_start = response.find("from torch")
        if source_start == -1:
            return None

        code = response[source_start:].strip()
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

    @staticmethod
    def validate_generated_response(
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

        try:
            ast.parse(pytorch_code)
        except SyntaxError as exc:
            raise ValueError(
                "Generated PyTorch code is not valid Python: "
                f"{exc.msg} at line {exc.lineno}."
            ) from exc

        # Basic protection against incomplete generation.
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
        started_at = __import__(
            "time"
        ).perf_counter()

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

            system_prompt = (
                self.build_system_prompt()
            )

            user_prompt = (
                self.build_user_prompt(
                    specification
                )
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
                response = str(response)

            if response.startswith(
                "Error:"
            ):
                raise RuntimeError(
                    response
                )

            try:
                generated = self.extract_json(
                    response
                )
            except ValueError as parse_error:
                generated = self.extract_python_source(response)
                if generated is not None:
                    self.validate_generated_response(generated)
                else:
                    repair_prompt = f"""
    The previous response was not valid structured JSON. Return exactly one
    valid JSON object with model_recommendation, experiment_plan, assumptions,
    dependencies, and pytorch_code. Preserve the complete PyTorch source code.
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
                    if not isinstance(repaired_response, str):
                        repaired_response = str(repaired_response)
                    if repaired_response.startswith("Error:"):
                        raise RuntimeError(repaired_response) from parse_error
                    try:
                        generated = self.extract_json(repaired_response)
                    except ValueError as repaired_parse_error:
                        generated = self.extract_python_source(repaired_response)
                        if generated is None:
                            generated = self.extract_python_source(response)
                        if generated is None:
                            code_only_prompt = f"""
Generate only the complete executable Python source code for this PyTorch
experiment. Do not return JSON. Do not return explanations. Do not use
Markdown fences. The source must import torch, load the local dataset from
DATASET_PATH, train and evaluate the model, and save all artifacts inside
EXPERIMENT_OUTPUT_DIR.

Selected hypothesis:
{specification["selected_hypothesis"].get("text", "")}

Dataset:
{specification["dataset"].get("name", "5G-NIDD")}
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
                            if not isinstance(code_only_response, str):
                                code_only_response = str(code_only_response)
                            generated = self.extract_python_source(code_only_response)
                        if generated is None:
                            raise repaired_parse_error

            try:
                self.validate_generated_response(
                    generated
                )
            except ValueError as code_error:
                code_only_prompt = f"""
Return only complete, executable Python source code for a PyTorch experiment.
Do not return JSON, Markdown, explanations, analysis, or commentary. Start
with a Python import and end with the executable experiment code.

Selected hypothesis:
{specification["selected_hypothesis"].get("text", "")}

Dataset:
{specification["dataset"].get("name", "5G-NIDD")}
""".strip()
                code_only_response = _call_llm(
                    code_only_prompt,
                    temperature=0.0,
                    model=self.model,
                    max_tokens=_output_token_limit(
                        "code_generation",
                        self.DEFAULT_MAX_TOKENS,
                    ),
                    reasoning="off",
                )
                if not isinstance(code_only_response, str):
                    code_only_response = str(code_only_response)
                repaired_code = self.extract_python_source(code_only_response)
                if repaired_code is None:
                    raise code_error
                self.validate_generated_response(repaired_code)
                generated = repaired_code

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
                }
            )

        except Exception as error:
            logger.exception(
                "CodeGenerationAgent failed."
            )

            result["errors"].append(
                str(error)
            )

        finally:
            import time

            result["generation_seconds"] = (
                time.perf_counter()
                - started_at
            )

        return result

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
    ) -> Dict[str, Any]:
        """
        Generate experiment code directly from a Hypothesis and
        ResearchGoal.

        If an experiment specification has already been produced
        by ExperimentOrchestrator, it is used directly.
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
                    "name": "5G-NIDD",
                    "path": None,
                    "task": (
                        "5G network intrusion "
                        "detection classification"
                    ),
                },
                "research_goal": (
                    research_goal_data
                ),
                "selected_hypothesis": (
                    hypothesis_data
                ),
                "scientific_evaluation": {},
                "code_generation_requirements": {
                    "framework": "PyTorch",
                    "language": "Python",
                    "dataset": "5G-NIDD",
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
        Save generated PyTorch code to a .py file.

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
