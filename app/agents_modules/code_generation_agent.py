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
    REPAIR_MAX_TOKENS = 8000
    MAX_REPAIR_SOURCE_CHARS = 28000
    MAX_REPAIR_LOG_CHARS = 8000

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
hypothesis into a complete, reproducible, executable PyTorch experiment.

The selected hypothesis has already passed the AI Co-Scientist workflow,
including generation, reflection, ranking, evolution, proximity analysis,
and meta-review.

You must therefore implement the selected research idea faithfully.

The generated Python code will NOT be executed by you. It will be saved
and later executed automatically by the ExperimentRunner on a remote
CPU/GPU server.

IMPORTANT RULES:

============================================================
1. SCIENTIFIC FIDELITY
============================================================

1. Implement the selected hypothesis faithfully.

2. Do not silently change the research objective, dataset, target variable,
   model architecture, or experimental methodology.

3. If the hypothesis proposes a specific machine-learning or deep-learning
   architecture, implement that architecture rather than replacing it with
   a generic model.

4. Do not simplify the proposed architecture merely to make the experiment
   faster.

5. Performance optimizations are allowed only when they do not invalidate
   the scientific objective.

6. If implementation details are missing from the hypothesis, make the
   smallest scientifically reasonable assumptions and record them in the
   experiment metadata.

============================================================
2. DATASET
============================================================

7. The current dataset is 5G-NIDD.

8. Treat 5G-NIDD as an offline/local dataset.

9. The dataset path may be provided through the DATASET_PATH environment
   variable.

10. Prefer DATASET_PATH when it is available.

11. Do not depend on a machine-specific absolute path as the only dataset
    location.

12. Validate that the dataset exists before loading it.

13. Raise a clear and informative error if the dataset cannot be found.

14. Automatically identify and validate the target column according to the
    experiment specification.

============================================================
3. DATA PREPROCESSING
============================================================

15. Include preprocessing appropriate for tabular/network intrusion data.

16. Handle numerical and categorical features appropriately.

17. Never use unconditional df.dropna() on the entire dataset.

18. Missing numerical values must be handled using training-set statistics,
    such as the training-set median.

19. Missing categorical values must be handled explicitly using an
    appropriate sentinel such as "Unknown".

20. Fit imputers, encoders, and scalers using training data only.

21. Apply fitted preprocessing to validation and test data without fitting
    on those partitions.

22. Verify that preprocessing produces valid data.

23. Verify that the processed dataset contains at least one usable sample.

24. Verify that the processed features do not contain NaN or infinite values.

25. Raise a clear error if preprocessing produces invalid or empty data.

============================================================
4. DATA SPLITTING AND DATA LEAKAGE
============================================================

26. Create separate training, validation, and test partitions.

27. Use stratified splitting for classification when appropriate.

28. Handle class-distribution problems gracefully.

29. Never use test data to fit preprocessing components.

30. Never use test data for model selection or hyperparameter tuning.

31. Identify the target variable explicitly before preprocessing.

32. Do not use the target variable itself as an input feature.

33. Inspect feature names and experiment metadata for variables that may
    directly encode, derive from, or reveal the target.

34. Features such as attack labels, attack categories, attack tools,
    outcome indicators, or post-event annotations may contain target
    information.

35. If a feature is clearly derived from the target or would not be
    available at prediction time in a real-world intrusion-detection
    setting, exclude it from the predictive feature set.

36. Do not remove potentially informative features merely because they are
    correlated with the target. Exclude a feature when there is a clear
    methodological, temporal, or target-leakage reason.

37. Record excluded target-related features and the reason for excluding
    them in the experiment summary.

38. Ensure that the final feature set represents information that would
    realistically be available to the intrusion-detection model at
    prediction time.

39. Check for duplicated or near-duplicated records when appropriate,
    particularly when unusually high validation or test performance occurs.

40. Extremely high or perfect validation/test performance must not
    automatically be interpreted as evidence of a successful model.
    Consider possible target leakage, duplicated records, or other
    methodological issues.

41. Preserve the scientific intent of the hypothesis while preventing
    methodological data leakage.

============================================================
5. REPRODUCIBILITY
============================================================

42. Set deterministic random seeds for Python, NumPy, and PyTorch where
    appropriate.

43. When CUDA is available, configure PyTorch reproducibility appropriately.

44. Do not unnecessarily sacrifice performance for reproducibility.

45. Record the random seed in the final experiment summary.

============================================================
6. PYTORCH MODEL
============================================================

46. Use PyTorch for the deep-learning experiment.

47. Implement the architecture specified by the selected hypothesis.

48. Use torch.nn.Module appropriately.

49. Use an appropriate loss function.

50. Use an appropriate optimizer.

51. Use model.train() during training.

52. Use model.eval() during validation and testing.

53. Use torch.no_grad() during validation and testing when gradients are
    not required.

54. Save the best-performing model checkpoint according to validation
    performance when appropriate.

55. Reload the best checkpoint before final test evaluation when appropriate.

============================================================
7. GPU AND DEVICE HANDLING
============================================================

56. Automatically detect whether CUDA is available.

57. Use:

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

58. Fall back to CPU automatically when CUDA is unavailable.

59. Never assume that a GPU is available.

60. Print the selected device.

61. When CUDA is available, print the GPU name.

62. Move the model to the selected device.

63. Move training, validation, and test tensors to the selected device
    efficiently.

64. Avoid unnecessary CPU-to-GPU and GPU-to-CPU transfers.

65. Do not repeatedly transfer the same tensors between CPU and GPU inside
    performance-critical loops.

66. When CUDA is available, consider using pin_memory=True in DataLoader.

67. When appropriate, use non_blocking=True for tensor transfers.

68. The experiment must remain executable on CPU when CUDA is unavailable.

============================================================
8. COMPUTATIONAL EFFICIENCY
============================================================

69. The generated experiment must be suitable for automated execution on
    a shared CPU/GPU server.

70. Choose a batch size appropriate for the dataset size, model complexity,
    and available hardware.

71. For large tabular datasets with relatively small neural networks,
    prefer a moderately large batch size when GPU memory permits.

72. Do not default to unnecessarily small batch sizes such as 32, 64, or
    128 for very large tabular datasets unless the experiment specifically
    requires them.

73. When appropriate, consider batch sizes such as 512, 1024, 2048, or
    another hardware-appropriate value.

74. The selected batch size must remain safe for available GPU memory.
    If necessary, use a smaller batch size to avoid out-of-memory errors.

75. Do not automatically use an extremely large batch size.

76. Do not create unnecessarily large models.

77. Do not use unnecessarily many layers, hidden units, or parameters unless
    required by the hypothesis.

78. Do not use an unnecessarily large number of training epochs.

79. Use a reasonable maximum epoch limit.

80. Use early stopping based on validation performance when scientifically
    appropriate and when it does not conflict with the experiment
    specification.

81. Early stopping should stop training when validation performance has
    converged and has not meaningfully improved for a reasonable patience
    period.

82. Do not continue training for many additional epochs after validation
    performance has clearly converged.

83. Use a reasonable early-stopping patience value rather than an
    excessively large patience value.

84. Avoid repeated dataset loading.

85. Avoid repeated preprocessing.

86. Avoid redundant model evaluation.

87. Avoid unnecessary computations inside the training loop.

88. When the dataset is already loaded into memory, configure the DataLoader
    efficiently.

89. When CUDA is available, consider pin_memory=True when it provides a
    benefit.

90. When appropriate, use non_blocking=True for CPU-to-GPU tensor transfers.

91. Avoid unnecessary CPU-to-GPU and GPU-to-CPU transfers.

92. Do not use unnecessarily expensive visualizations.

93. Do not optimize for speed by changing the scientific objective.

94. If the hypothesis does not specify an exact batch size or number of
    epochs, choose values that provide a reasonable balance between
    scientific validity and computational efficiency.

95. Do not artificially increase model size, batch size, or training
    duration merely to increase GPU utilization.

96. Computational-efficiency improvements must not change the research
    objective, target variable, dataset, or proposed model architecture.

============================================================
9. OPTIONAL MIXED PRECISION
============================================================

97. Mixed-precision training may be used when appropriate for the selected
    model and CUDA hardware.

98. If mixed precision is used, it must safely fall back to normal precision
    when CUDA is unavailable.

99. Do not use mixed precision solely for the purpose of using GPU features
    if it is unlikely to provide a meaningful benefit.

============================================================
10. TRAINING
============================================================

100. Implement a complete training loop.

101. Validate the model after each epoch when appropriate.

102. Record training loss and validation metrics.

103. Save the best model checkpoint.

104. Print concise training progress.

105. Do not produce unnecessarily large console output.

106. Training must have a bounded maximum number of epochs.

107. Never create an infinite training loop.

============================================================
11. EVALUATION
============================================================

108. Evaluate the final model on the held-out test set.

109. For classification experiments, calculate:

    - Accuracy
    - Weighted Precision
    - Weighted Recall
    - Weighted F1-score
    - Confusion Matrix

110. Use additional metrics when required by the experiment specification.

111. Do not use the test set during model selection.

112. Save evaluation metrics in metrics.json.

============================================================
12. TIMING
============================================================

113. Measure training execution time separately.

114. Measure evaluation execution time separately.

115. Measure complete experiment wall-clock execution time.

116. Use time.perf_counter() for timing.

117. Total experiment execution time must include all required work from
    experiment start until all required artifacts have been generated and
    saved.

118. This includes:

    - dataset loading
    - preprocessing
    - training
    - validation
    - test evaluation
    - visualization generation
    - metric saving
    - training-history saving
    - checkpoint saving

119. Record:

    - training_seconds
    - evaluation_seconds
    - total_execution_seconds

120. Record the selected device and GPU name when available.

============================================================
13. VISUALIZATION
============================================================

121. Generate useful visualizations relevant to the experiment.

122. For training experiments, generate training-history visualizations
    where appropriate.

123. For classification experiments, generate a confusion matrix visualization.

124. Generate a performance metrics visualization when appropriate.

125. Do not display plots interactively.

126. Use a non-interactive matplotlib backend suitable for remote/server
     execution.

127. Save all visualization files inside EXPERIMENT_OUTPUT_DIR.

128. Use these exact filenames when applicable:

    loss_visualization.png
    accuracy_visualization.png
    confusion_matrix_visualization.png
    performance_metrics_visualization.png

============================================================
14. OUTPUT ARTIFACTS
============================================================

129. The ExperimentRunner provides the environment variable
     EXPERIMENT_OUTPUT_DIR.

130. All generated experiment artifacts MUST be saved inside
     EXPERIMENT_OUTPUT_DIR.

131. Do not save experiment artifacts to arbitrary system directories.

132. Create EXPERIMENT_OUTPUT_DIR if it does not exist.

133. Save these files using the exact filenames:

    metrics.json
    training_history.json
    best_model.pt

134. Save all required visualization files inside the same output directory.

============================================================
15. EXPERIMENT SUMMARY
============================================================

135. Generate a final experiment summary containing, when applicable:

    - Experiment name
    - Research hypothesis
    - Dataset
    - Target variable
    - Number of samples
    - Number of features
    - Number of classes
    - Model architecture
    - Batch size
    - Number of epochs completed
    - Best validation performance
    - Test performance
    - Training time
    - Evaluation time
    - Total execution time
    - Device
    - GPU name
    - Random seed

136. Save the experiment summary in a structured JSON file.

============================================================
16. AUTOMATED SERVER EXECUTION
============================================================

137. The generated experiment will run unattended.

138. Do not require interactive user input.

139. Do not use input().

140. Do not require a graphical desktop environment.

141. Do not open interactive matplotlib windows.

142. Do not require manual confirmation.

143. Do not assume files have been manually created by the user.

144. Use environment variables and portable paths where appropriate.

============================================================
17. DEPENDENCIES
============================================================

145. Use only libraries that are required by the experiment.

146. Avoid unnecessary dependencies.

147. Do not include unused imports.

148. Use commonly available scientific Python libraries where appropriate.

149. Do not introduce unnecessary external packages merely for convenience.

============================================================
18. CODE QUALITY
============================================================

150. Generate clean, readable, modular Python code.

151. Use meaningful variable and function names.

152. Use functions for logically separate operations.

153. Include concise comments explaining important implementation choices.

154. Avoid duplicated code.

155. Keep configuration values clearly defined.

156. Do not generate pseudocode.

157. Do not generate incomplete code.

158. Do not use TODO placeholders.

159. Do not use "pass" as a replacement for required functionality.

160. Do not leave required functions unimplemented.

============================================================
19. HARDWARE INDEPENDENCE
============================================================

161. Do not assume that a larger GPU is required.

162. Generate code that efficiently uses the available hardware.

163. Do not artificially increase model size or batch size merely to increase
     GPU utilization.

164. If the model or dataset is small, recognize that GPU acceleration may
     provide limited benefit.

165. Preserve CPU compatibility.

============================================================
20. PYTORCH COMPATIBILITY
============================================================

166. Target the installed PyTorch API.

167. Do not use unsupported arguments for the installed PyTorch version.

168. In particular, do not pass verbose to
     torch.optim.lr_scheduler.ReduceLROnPlateau because the project's
     installed PyTorch version may not support that argument.

============================================================
21. FINAL REQUIREMENTS
============================================================

169. The final generated file must be complete and executable Python.

170. The experiment must automatically:

    1. Load the specified local dataset.
    2. Validate the dataset.
    3. Preprocess the data.
    4. Split the data into training, validation, and test sets.
    5. Prevent data leakage.
    6. Build the model specified by the hypothesis.
    7. Select CUDA when available.
    8. Fall back to CPU when necessary.
    9. Train the model.
   10. Validate the model.
   11. Save the best checkpoint.
   12. Evaluate on the held-out test set.
   13. Calculate the required metrics.
   14. Generate required visualizations.
   15. Save all required artifacts.
   16. Record training, evaluation, and total execution time.
   17. Generate a final experiment summary.

171. Most importantly, preserve the scientific intent of the selected
     Rank #1 hypothesis while making the generated implementation robust,
     reproducible, efficient, and suitable for automated execution.

172. Return ONLY complete executable Python source code when generating the
     experiment.
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
Generate the complete executable Python source code for the PyTorch
experiment described by the following AI Co-Scientist experiment
specification.

EXPERIMENT SPECIFICATION
========================

{specification_json}

IMPLEMENTATION REQUIREMENTS
===========================

The generated experiment MUST:

1. Implement the selected Rank #1 hypothesis faithfully.
2. Use the specified dataset and task.
3. Use DATASET_PATH when available.
4. Use PyTorch.
5. Automatically use CUDA when available and CPU otherwise.
6. Handle numerical and categorical features correctly.
7. Handle missing values without dropping valid network records unnecessarily.
8. Fit preprocessing components using training data only.
9. Prevent data leakage.
10. Create train, validation, and test partitions.
11. Implement the model architecture required by the hypothesis.
12. Train and validate the model.
13. Save the best model checkpoint.
14. Evaluate on the held-out test set.
15. Calculate the requested evaluation metrics.
16. Generate the required visualizations.
17. Save all artifacts inside EXPERIMENT_OUTPUT_DIR.
18. Record training, evaluation, and complete wall-clock execution time.
19. Print the selected device and GPU name when available.
20. Remain executable without manual intervention.

IMPORTANT:

Do not replace the selected model architecture with a generic MLP unless
the hypothesis itself specifies an MLP.

Do not change the scientific objective merely to improve execution speed.

Use reasonable computational-efficiency techniques such as:

- selecting a batch size appropriate for the dataset size, model complexity,
  and available GPU memory;
- preferring moderately large batch sizes for very large tabular datasets
  when appropriate;
- using efficient DataLoader configuration;
- minimizing unnecessary CPU-to-GPU transfers;
- using bounded training epochs;
- using early stopping when validation performance has converged;
- avoiding unnecessary repeated preprocessing or evaluation; and
- avoiding unnecessarily large models or excessive training duration.

Do not use a fixed batch size, epoch count, or early-stopping patience
unless the experiment specification explicitly requires it.

Do not change the scientific objective, dataset, target variable, or
proposed architecture merely to reduce execution time.

Before finalizing the experiment, consider whether any feature may leak
information about the target variable, particularly features representing
attack type, attack tool, outcome, post-event information, or other
target-derived information. Exclude such features only when there is a
clear methodological or prediction-time leakage reason, and record the
decision in the experiment summary.

Return ONLY complete executable Python source code.

Do not return JSON.
Do not return Markdown.
Do not use Markdown code fences.
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
            1. The experiment specification
            2. The current generated Python source code
            3. The execution error / traceback
            4. The previous stdout output
            5. The previous stderr output

        The LLM must return a complete corrected Python experiment.

        This method intentionally does not contain hard-coded fixes for
        individual Python or machine-learning errors. The LLM determines
        the cause of the failure and modifies the generated experiment
        accordingly.
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
        # Keep the repair prompt bounded.
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

        # Keep only the most recent part of the logs because
        # complete training logs can become extremely large.
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

        compact_specification = {
            "dataset": dataset,
            "selected_hypothesis": {
                key: selected_hypothesis.get(key)
                for key in (
                    "hypothesis_id",
                    "title",
                    "text",
                )
            },
            "code_generation_requirements": specification.get(
                "code_generation_requirements",
                {},
            ),
            "evaluation_metrics": specification.get(
                "evaluation_metrics",
                [],
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
You are repairing a failed automatically generated PyTorch
experiment inside an AI Co-Scientist system.

The experiment was generated by another LLM and then executed
automatically by an ExperimentRunner.

The experiment failed during execution.

Your task is to determine the ROOT CAUSE of the failure from
the traceback, stdout, stderr, experiment specification, and
current source code.

Then return a COMPLETE corrected Python source file.

IMPORTANT:

1. Return ONLY Python source code.
2. Do NOT return Markdown fences.
3. Do NOT return JSON.
4. Do NOT return explanations or commentary.
5. Do NOT return a patch or partial code.
6. Return the COMPLETE replacement for the current source file.
7. Preserve the original research hypothesis.
8. Preserve the intended model architecture whenever possible.
9. Preserve the intended experiment objective.
10. Preserve the required evaluation metrics.
11. Preserve the train/validation/test evaluation design.
12. Preserve checkpoint generation.
13. Preserve training-history generation.
14. Preserve required visualization generation.
15. Save generated artifacts inside EXPERIMENT_OUTPUT_DIR.
16. Use DATASET_PATH when it is available.
17. Do not invent a different dataset.
18. Do not remove required experiment functionality merely to
    make the program run.
19. Fix the actual root cause instead of hiding the error.
20. Make the smallest scientifically reasonable correction.
21. If the dataset structure is different from what the original
    code assumed, adapt the preprocessing to the actual dataset
    information available in the specification and error.
22. Handle class-distribution and data-splitting problems
    robustly when necessary.
23. Handle missing, categorical, and numerical data appropriately.
24. Do not introduce data leakage.
25. The repaired source must be valid executable Python.
26. Do not use TODO, pass, placeholder code, or incomplete
    implementations.

The ExperimentRunner will execute the returned source again.
Therefore, your response must be the complete executable
experiment, not an explanation of what should be changed.

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

Then rewrite the complete experiment so that the root
cause is corrected while preserving the original
scientific experiment.

Return ONLY the complete corrected Python source code.
""".strip()

        # ----------------------------------------------------
        # Call LLM.
        # ----------------------------------------------------

        response = _call_llm(
            repair_prompt,
            temperature=0.0,
            model=self.model,
            system_prompt=self.build_system_prompt(),
            max_tokens=_output_token_limit(
                "code_generation",
                self.REPAIR_MAX_TOKENS,
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

        # ----------------------------------------------------
        # Extract Python source.
        # ----------------------------------------------------

        repaired = self.extract_python_source(
            response
        )

        # The preferred repair response is Python source.
        #
        # The JSON fallback is retained because the model may
        # occasionally return the normal CodeGenerationAgent
        # structured format.
        if repaired is None:
            try:
                repaired = self.extract_json(
                    response
                )
            except ValueError as error:
                raise ValueError(
                    "LLM repair response did not contain "
                    "valid Python source code."
                ) from error

        # ----------------------------------------------------
        # Validate repaired experiment.
        # ----------------------------------------------------

        self.validate_generated_response(
            repaired
        )

        repaired_code = repaired.get(
            "pytorch_code"
        )

        if not isinstance(
            repaired_code,
            str,
        ) or not repaired_code.strip():
            raise ValueError(
                "LLM repair returned empty Python code."
            )

        # ----------------------------------------------------
        # Prevent useless repair loops.
        # ----------------------------------------------------

        if repaired_code.strip() == generated_code.strip():
            raise ValueError(
                "LLM returned the same code without making "
                "a repair."
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
