"""
Regression tests for the experiment pipeline (orchestrator, code generation, runner).

Offline tests (default):

    pytest tests/test_experiment_pipeline.py -v -s -m "not integration"

Integration tests (live LM Studio):

    pytest tests/test_experiment_pipeline.py -k integration -v -s

    pytest tests/test_experiment_pipeline.py -v -s -m "integration"

    pytest tests/test_experiment_pipeline.py::test_code_generation_agent_live_lmstudio_call -v -s -m integration

    pytest tests/test_experiment_pipeline.py::test_display_generated_pytorch_code -v -s -m integration
"""

import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from app.agents_modules.code_generation_agent import CodeGenerationAgent
from app.config import load_config
from app.experiments.experiment_orchestrator import ExperimentOrchestrator
from app.experiments.experiment_runner import ExperimentRunner
from app.data.dataset_manager import DatasetManager
from app.utils import call_llm

VALID_SPECIFICATION = {
    "dataset": {
        "name": "5G-NIDD",
        "path": None,
        "task": "classification",
    },
    "selected_hypothesis": {
        "hypothesis_id": "H-1",
        "title": "Adaptive PQC selection",
        "text": ("Adaptive PQC selection reduces 5G handshake latency under bursty load."),
    },
    "research_goal": {"description": "Test a new model for 5G security orchestration."},
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
    ],
}


# ============================================================
# CodeGenerationAgent - Offline Tests
# ============================================================


def test_code_generation_agent_generates_valid_experiment(monkeypatch):
    calls = []

    fake_response = {
        "model_recommendation": {
            "name": "lstm",
            "reason": "temporal structure in the dataset",
        },
        "experiment_plan": {
            "architecture": "LSTM",
            "epochs": 3,
            "batch_size": 32,
        },
        "assumptions": ["The dataset is tabular and time-ordered."],
        "dependencies": [
            "torch",
            "pandas",
            "numpy",
        ],
        "pytorch_code": (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class TinyLSTM(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(\n"
            "            nn.Linear(8, 16),\n"
            "            nn.ReLU(),\n"
            "            nn.Linear(16, 2),\n"
            "        )\n\n"
            "    def forward(self, x):\n"
            "        return self.net(x)\n"
        ),
    }

    def fake_call_llm(*args, **kwargs):
        calls.append(kwargs)
        return json.dumps(fake_response)

    monkeypatch.setattr(
        "app.agents_modules.code_generation_agent._call_llm",
        fake_call_llm,
    )

    agent = CodeGenerationAgent(model="qwen/qwen3.8-27b")

    result = agent.generate(VALID_SPECIFICATION)

    assert result["success"] is True
    assert result["model_recommendation"]["name"] == "lstm"
    assert "class TinyLSTM" in result["pytorch_code"]
    assert calls
    assert calls[0]["reasoning"] == "off"


def test_code_generation_agent_extracts_fenced_python_response():
    result = CodeGenerationAgent.extract_fenced_python(
        "Here is the experiment:\n```python\nimport torch\nprint('ok')\n```"
    )

    assert result is not None
    assert result["pytorch_code"] == ("import torch\nprint('ok')")


def test_code_generation_agent_recovers_unterminated_fenced_python():
    result = CodeGenerationAgent.extract_fenced_python("```python\nimport os\nimport torch\nprint('partial'")

    assert result is not None
    # The leading imports must survive. Recovering the source from the first
    # torch import instead would drop them and leave the experiment unrunnable.
    assert result["pytorch_code"].startswith("import os")


def test_code_generation_agent_continues_truncated_experiment(monkeypatch):
    truncated = (
        "```python\n"
        "import os\n"
        "import torch\n"
        "\n"
        "\n"
        "def main():\n"
        "    device = torch.device('cpu')\n"
        "    print(os.getcwd(), device)\n"
        "    summary = {\n"
        '        "ffnn\n'
    )

    continuation = (
        "    summary = {\n"
        '        "ffnn": 0.91,\n'
        "    }\n"
        "    print(summary)\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    responses = [truncated, continuation]
    calls = []

    def fake_call_llm(*args, **kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr(
        "app.agents_modules.code_generation_agent._call_llm",
        fake_call_llm,
    )

    agent = CodeGenerationAgent(model="qwen/qwen3.8-27b")

    result = agent.generate(VALID_SPECIFICATION)

    assert result["success"] is True
    assert len(calls) == 2
    assert calls[1]["system_prompt"] == (CodeGenerationAgent.CONTINUATION_SYSTEM_PROMPT)

    code = result["pytorch_code"]

    assert code.startswith("import os")
    assert code.rstrip().endswith("main()")
    ast.parse(code)


def test_code_generation_agent_keeps_complete_code_unchanged(monkeypatch):
    def fail_call_llm(*args, **kwargs):
        raise AssertionError("A complete experiment must not be continued.")

    monkeypatch.setattr(
        "app.agents_modules.code_generation_agent._call_llm",
        fail_call_llm,
    )

    agent = CodeGenerationAgent(model="test-model")
    source = "import torch\nprint(torch.__version__)\n"

    assert agent.continue_truncated_code(source) == source


def test_code_generation_agent_does_not_continue_invalid_python(monkeypatch):
    def fail_call_llm(*args, **kwargs):
        raise AssertionError("Invalid Python must not be continued.")

    monkeypatch.setattr(
        "app.agents_modules.code_generation_agent._call_llm",
        fail_call_llm,
    )

    agent = CodeGenerationAgent(model="test-model")
    source = "import torch\nthis is not valid Python\n" + "print(1)\n" * 60

    with pytest.raises(ValueError, match="not valid Python"):
        agent.continue_truncated_code(source)


def test_code_generation_agent_uses_dedicated_model_by_default(monkeypatch):
    from app.agents_modules.code_generation_agent import config

    monkeypatch.setitem(config, "llm_model", "research-model")
    monkeypatch.setitem(config, "code_generation_model", "code-model")
    agent = CodeGenerationAgent()

    assert agent.model == "code-model"


def test_code_repair_prompt_is_bounded(monkeypatch):
    captured = {}

    def fake_call_llm(*args, **kwargs):
        captured["prompt"] = args[0] if args else ""
        captured["kwargs"] = kwargs
        return "import torch\nprint('fixed')"

    monkeypatch.setattr(
        "app.agents_modules.code_generation_agent._call_llm",
        fake_call_llm,
    )

    agent = CodeGenerationAgent(model="test-model")

    result = agent.repair_generated_code(
        specification={
            "dataset": {"name": "5G-NIDD"},
            "selected_hypothesis": {"text": "test hypothesis"},
            "large_provenance": "x" * 200000,
        },
        generated_code="x" * 200000,
        execution_result={
            "status": "invalid_outputs",
            "stderr": "error" * 10000,
            "stdout": "output" * 10000,
        },
    )

    assert result["success"] is True
    # The bound holds a full-length experiment plus both bounded logs, so the
    # repair model sees the whole file instead of its tail.
    assert len(captured["prompt"]) < 90000
    assert captured["kwargs"]["max_tokens"] == agent.REPAIR_MAX_TOKENS


def test_code_generation_agent_rejects_invalid_python():
    response = {
        "model_recommendation": {},
        "experiment_plan": {},
        "assumptions": [],
        "dependencies": [],
        "pytorch_code": ("import torch\nthis is not valid Python"),
    }

    with pytest.raises(ValueError, match="not valid Python"):
        CodeGenerationAgent.validate_generated_response(response)


# ============================================================
# ExperimentRunner - Output Tests
# ============================================================


def test_experiment_runner_collects_standard_output_files(tmp_path):
    runner = ExperimentRunner(output_directory=tmp_path / "runs")

    run_directory = runner.create_run_directory("demo_run")

    (run_directory / "metrics.json").write_text(
        '{"accuracy": 0.91}',
        encoding="utf-8",
    )

    (run_directory / "training_history.json").write_text(
        '{"loss": [1.0, 0.5]}',
        encoding="utf-8",
    )

    (run_directory / "experiment_summary.json").write_text(
        '{"status": "ok"}',
        encoding="utf-8",
    )

    (run_directory / "best_model.pt").write_bytes(b"checkpoint")

    outputs = runner.collect_outputs(run_directory)

    assert outputs["metrics"]["accuracy"] == 0.91
    assert outputs["training_history"]["loss"] == [1.0, 0.5]
    assert outputs["experiment_summary"]["status"] == "ok"
    assert outputs["checkpoint_path"].endswith("best_model.pt")


def test_experiment_runner_rejects_nonfinite_metrics_and_missing_visualizations():
    execution = {"success": True}

    outputs = {
        "metrics": {
            "accuracy": float("nan"),
            "precision_weighted": 0.5,
            "recall_weighted": 0.5,
            "f1_weighted": 0.5,
            "confusion_matrix": [[1]],
            "training_seconds": 1.0,
            "evaluation_seconds": 1.0,
            "total_execution_seconds": 2.0,
        },
        "training_history": {"train_loss": [0.5]},
        "checkpoint_path": "best_model.pt",
        "visualizations": ["loss_visualization.png"],
    }

    validation = ExperimentRunner.validate_outputs(
        execution,
        outputs,
    )

    assert validation["valid"] is False

    assert "NaN or infinite" in " ".join(validation["warnings"])

    assert "Missing required visualizations" in " ".join(validation["warnings"])


def test_experiment_runner_rejects_metrics_missing_total_execution_seconds():
    """
    Regression test for incomplete metrics.json output.

    The generated experiment must include total_execution_seconds
    in metrics.json. This test ensures ExperimentRunner rejects
    metrics.json when that required field is missing.
    """
    execution = {
        "success": True
    }

    outputs = {
        "metrics": {
            "accuracy": 0.999175,
            "precision_weighted": 0.9991756449662996,
            "recall_weighted": 0.999175,
            "f1_weighted": 0.9991750865944501,
            "confusion_matrix": [
                [15761, 7],
                [26, 24206],
            ],
            "training_seconds": 39.23,
            "evaluation_seconds": 0.54,
            # Deliberately missing:
            # "total_execution_seconds": ...
        },
        "training_history": {
            "train_loss": [0.5]
        },
        "checkpoint_path": "best_model.pt",
        "visualizations": [
            "loss_visualization.png",
            "accuracy_visualization.png",
            "confusion_matrix_visualization.png",
            "performance_metrics_visualization.png",
        ],
    }

    validation = ExperimentRunner.validate_outputs(
        execution,
        outputs,
    )

    assert validation["valid"] is False

    warnings = " ".join(validation["warnings"])

    assert "total_execution_seconds" in warnings


def _classification_metrics():
    return {
        "accuracy": 0.99,
        "precision_weighted": 0.99,
        "recall_weighted": 0.99,
        "f1_weighted": 0.99,
        "confusion_matrix": [[11825, 2], [33, 18140]],
    }


def _complete_outputs(**overrides):
    outputs = {
        "metrics": _classification_metrics(),
        "training_history": {"train_loss": [0.5]},
        "checkpoint_path": "best_model.pt",
        "visualizations": [
            "loss_visualization.png",
            "accuracy_visualization.png",
            "confusion_matrix_visualization.png",
            "performance_metrics_visualization.png",
        ],
    }
    outputs.update(overrides)
    return outputs


def test_run_timings_may_come_from_the_experiment_summary():
    """A completed run must not be rejected over which file holds its timings."""

    outputs = _complete_outputs(
        experiment_summary={
            "training_seconds": 21.16,
            "evaluation_seconds": 0.32,
            "total_execution_seconds": 28.0,
        }
    )

    validation = ExperimentRunner.validate_outputs({"success": True}, outputs)

    assert validation["valid"] is True
    assert validation["warnings"] == []


def test_timings_are_still_required_somewhere():
    validation = ExperimentRunner.validate_outputs(
        {"success": True},
        _complete_outputs(experiment_summary={"status": "ok"}),
    )

    assert validation["valid"] is False
    assert "missing required metrics" in " ".join(validation["warnings"])


def test_a_nonfinite_summary_timing_does_not_satisfy_the_contract():
    validation = ExperimentRunner.validate_outputs(
        {"success": True},
        _complete_outputs(
            experiment_summary={
                "training_seconds": float("nan"),
                "evaluation_seconds": 0.32,
                "total_execution_seconds": 28.0,
            }
        ),
    )

    assert validation["valid"] is False
    assert "training_seconds" in " ".join(validation["warnings"])


# ============================================================
# ExperimentRunner - Dependency Handling
# ============================================================


def test_experiment_runner_stops_retrying_once_the_cycle_budget_is_gone(
    tmp_path,
    monkeypatch,
):
    import threading
    import time

    from app.experiments import experiment_runner as runner_module
    from app.utils import execution_budget

    run_directory = tmp_path / "run"
    run_directory.mkdir()

    code_path = run_directory / "generated_experiment.py"
    code_path.write_text("raise SystemExit(1)", encoding="utf-8")

    runner = ExperimentRunner(
        output_directory=tmp_path / "runs",
        timeout_seconds=10,
    )

    attempts = []

    def fake_subprocess_run(command, **kwargs):
        attempts.append(command)
        return subprocess.CompletedProcess(command, 1, "", "RuntimeError: boom")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        runner,
        "_repair_experiment_with_llm",
        lambda **kwargs: (True, "print('repaired')", "repaired"),
    )

    cancel_event = threading.Event()
    cancel_event.set()

    with execution_budget(time.monotonic() + 600, cancel_event):
        result = runner.execute(
            code_path=code_path,
            run_directory=run_directory,
        )

    assert len(attempts) == 1
    assert result["success"] is False
    assert result["status"] == "cancelled_no_cycle_budget"


def test_experiment_runner_extracts_missing_python_library(
    tmp_path,
):
    runner = ExperimentRunner(
        output_directory=tmp_path / "runs",
        timeout_seconds=10,
    )

    stderr = (
        "Traceback (most recent call last):\n"
        "  File 'generated_experiment.py', line 1, in <module>\n"
        "    import fake_missing_library\n"
        "ModuleNotFoundError: No module named "
        "'fake_missing_library'\n"
    )

    module_name = runner._extract_missing_module(stderr)

    assert module_name == "fake_missing_library"


def test_experiment_runner_installs_missing_python_library(
    tmp_path,
    monkeypatch,
):
    runner = ExperimentRunner(
        output_directory=tmp_path / "runs",
        timeout_seconds=10,
    )

    installed = []

    def fake_install_package(module_name):
        installed.append(module_name)
        return True, "installed"

    monkeypatch.setattr(
        runner,
        "_install_package",
        fake_install_package,
    )

    success, message = runner._install_package("fake_missing_library")

    assert success is True
    assert message == "installed"

    assert installed == ["fake_missing_library"]


def test_experiment_runner_rejects_invalid_package_name():
    """
    Verify that unsafe/invalid package names are rejected before
    pip is executed.
    """
    runner = ExperimentRunner()

    success, message = runner._install_package(
        "invalid package name!"
    )

    assert success is False
    assert "Rejected invalid package name" in message


def test_experiment_runner_maps_python_module_to_pypi_package(
    monkeypatch,
):
    """
    Verify that Python import names are mapped to their corresponding
    PyPI package names.
    """
    runner = ExperimentRunner()

    captured_command = []

    def fake_run(command, **kwargs):
        captured_command.append(command)

        return SimpleNamespace(
            returncode=0,
            stdout="Successfully installed scikit-learn",
            stderr="",
        )

    monkeypatch.setattr(
        "subprocess.run",
        fake_run,
    )

    success, output = runner._install_package("sklearn")

    assert success is True
    assert "scikit-learn" in captured_command[0]
    assert "sklearn" not in captured_command[0][-1]


def test_experiment_runner_handles_dependency_install_timeout(
    monkeypatch,
):
    """
    Verify that a timed-out pip installation is converted into a
    structured failure rather than raising an exception.
    """
    runner = ExperimentRunner()

    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
        )

    monkeypatch.setattr(
        "subprocess.run",
        fake_subprocess_run,
    )

    success, output = runner._install_package(
        "some_package"
    )

    assert success is False
    assert "Timed out while installing some_package" in output


# ============================================================
# ExperimentRunner - Generic LLM Repair
# ============================================================


def test_experiment_runner_automatically_repairs_failed_experiment_with_llm(
    tmp_path,
    monkeypatch,
):
    """
    Verify the generic automatic repair workflow.

    The runner must:

        1. Execute the generated experiment.
        2. Detect the failed execution.
        3. Send the error to the LLM repair mechanism.
        4. Replace generated_experiment.py with repaired code.
        5. Retry the experiment.
        6. Return success after the repaired experiment succeeds.

    The test deliberately uses a generic ValueError rather than
    a specific ML error. This verifies that the repair mechanism
    is generic rather than hard-coded for one particular exception.
    """

    runner = ExperimentRunner(
        output_directory=tmp_path / "runs",
        timeout_seconds=10,
    )

    run_directory = runner.create_run_directory("generic_llm_repair")

    original_code = "raise ValueError('some generated experiment error')\n"

    repaired_code = "print('experiment repaired successfully')\n"

    code_path = runner.prepare_generated_code(
        original_code,
        run_directory,
    )

    repair_calls = []

    def fake_repair_experiment_with_llm(
        code_path,
        stderr,
        stdout,
        dataset_path=None,
        generated_result=None,
    ):
        repair_calls.append(
            {
                "stderr": stderr,
                "stdout": stdout,
            }
        )

        assert "some generated experiment error" in stderr

        # Simulate the LLM returning corrected code.
        return (
            True,
            repaired_code,
            "Experiment repaired successfully.",
        )

    monkeypatch.setattr(
        runner,
        "_repair_experiment_with_llm",
        fake_repair_experiment_with_llm,
    )

    process_results = [
        SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=("Traceback (most recent call last):\nValueError: some generated experiment error\n"),
        ),
        SimpleNamespace(
            returncode=0,
            stdout=("experiment repaired successfully\n"),
            stderr="",
        ),
    ]

    def fake_subprocess_run(*args, **kwargs):
        assert process_results, "Unexpected extra experiment execution"

        # Before the second execution, simulate what the
        # ExperimentRunner should do after receiving repaired code.
        if len(process_results) == 1:
            code_path.write_text(
                repaired_code,
                encoding="utf-8",
            )

        return process_results.pop(0)

    monkeypatch.setattr(
        "app.experiments.experiment_runner.subprocess.run",
        fake_subprocess_run,
    )

    result = runner.execute(
        code_path,
        run_directory,
        generated_result={
            "model_recommendation": {},
            "experiment_plan": {},
            "assumptions": [],
            "dependencies": [],
        },
    )

    print("\n=== EXECUTION RESULT ===")
    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )
    print("========================\n")

    assert result["success"] is True
    assert result["return_code"] == 0

    assert result["experiment_attempts"] == 2
    assert result["repair_attempts"] == 1

    assert len(repair_calls) == 1

    repaired_file = code_path.read_text(encoding="utf-8")

    assert repaired_file == repaired_code

    assert "some generated experiment error" not in repaired_file


def test_experiment_runner_does_not_use_hard_coded_error_fix(
    tmp_path,
    monkeypatch,
):
    """
    Verify that an arbitrary Python error is sent to the
    generic LLM repair mechanism instead of being handled by
    a hard-coded error-specific fixer.
    """

    runner = ExperimentRunner(
        output_directory=tmp_path / "runs",
        timeout_seconds=10,
    )

    run_directory = runner.create_run_directory("generic_error")

    code_path = runner.prepare_generated_code(
        "raise RuntimeError('completely unrelated failure')\n",
        run_directory,
    )

    repair_calls = []

    def fake_repair(
        code_path,
        stderr,
        stdout,
        dataset_path=None,
        generated_result=None,
    ):
        repair_calls.append(
            {
                "stderr": stderr,
                "stdout": stdout,
            }
        )

        return (
            False,
            "",
            "LLM repair intentionally failed for test",
        )

    monkeypatch.setattr(
        runner,
        "_repair_experiment_with_llm",
        fake_repair,
    )

    process = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr=("Traceback (most recent call last):\nRuntimeError: completely unrelated failure\n"),
    )

    monkeypatch.setattr(
        "app.experiments.experiment_runner.subprocess.run",
        lambda *args, **kwargs: process,
    )

    result = runner.execute(
        code_path,
        run_directory,
    )

    assert result["success"] is False

    assert repair_calls
    assert "completely unrelated failure" in repair_calls[0]["stderr"]


def test_experiment_runner_stops_after_max_experiment_attempts(
    monkeypatch,
    tmp_path,
):
    """
    Verify that ExperimentRunner does not retry indefinitely.

    The current implementation limits execution to 10 attempts.
    """
    runner = ExperimentRunner(
        output_directory=tmp_path,
        timeout_seconds=10,
    )

    run_directory = runner.create_run_directory("max_attempt_test")

    code_path = runner.prepare_generated_code(
        "print('test')",
        run_directory,
    )

    execution_calls = []

    def fake_subprocess_run(*args, **kwargs):
        execution_calls.append(args)

        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="RuntimeError: experiment failed",
        )

    monkeypatch.setattr(
        "subprocess.run",
        fake_subprocess_run,
    )

    monkeypatch.setattr(
        runner,
        "_repair_experiment_with_llm",
        lambda **kwargs: (
            False,
            "",
            "No repair available",
        ),
    )

    result = runner.execute(
        code_path=code_path,
        run_directory=run_directory,
    )

    assert result["success"] is False
    assert result["status"] == "repair_failed"

    assert result["experiment_attempts"] == 1
    assert len(execution_calls) == 1


def test_experiment_runner_stops_at_max_experiment_attempts(
    monkeypatch,
    tmp_path,
):
    """
    Verify that the experiment execution loop is bounded at
    MAX_EXPERIMENT_ATTEMPTS = 10.
    """
    runner = ExperimentRunner(
        output_directory=tmp_path,
        timeout_seconds=10,
    )

    run_directory = runner.create_run_directory(
        "max_attempt_test"
    )

    code_path = runner.prepare_generated_code(
        "print('test')",
        run_directory,
    )

    execution_calls = []

    def fake_subprocess_run(command, **kwargs):
        execution_calls.append(command)

        attempt_number = len(execution_calls)

        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "ModuleNotFoundError: No module named "
                f"'fake_dependency_{attempt_number}'"
            ),
        )

    monkeypatch.setattr(
        "subprocess.run",
        fake_subprocess_run,
    )

    installed_modules = []

    def fake_install_package(module_name):
        installed_modules.append(module_name)
        return True, f"Installed {module_name}"

    monkeypatch.setattr(
        runner,
        "_install_package",
        fake_install_package,
    )

    result = runner.execute(
        code_path=code_path,
        run_directory=run_directory,
    )

    # The runner must stop after the maximum of 10 attempts.
    assert len(execution_calls) == 10

    # The experiment must not be reported as successful.
    assert result["success"] is False

    # The runner should finish with a failure status.
    assert result["status"] == "failed"

    # Verify that dependency recovery was attempted 10 times.
    assert len(installed_modules) == 10


# ============================================================
# ExperimentRunner - General Execution
# ============================================================


def test_experiment_runner_includes_stderr_in_nonzero_exit_error(
    tmp_path,
    monkeypatch,
):
    """
    Verify that stderr is preserved when the generated experiment
    exits with a non-zero return code.

    The LLM repair is disabled for this specific regression test
    because this test only checks error reporting.
    """

    runner = ExperimentRunner(
        output_directory=tmp_path / "runs",
        timeout_seconds=10,
    )

    run_directory = runner.create_run_directory("stderr_details")

    code_path = runner.prepare_generated_code(
        "raise RuntimeError('external server failure')",
        run_directory,
    )

    def no_repair(
        code_path,
        stderr,
        stdout,
        dataset_path=None,
        generated_result=None,
    ):
        return (
            False,
            "",
            "repair disabled for error-reporting test",
        )

    monkeypatch.setattr(
        runner,
        "_repair_experiment_with_llm",
        no_repair,
    )

    result = runner.execute(
        code_path,
        run_directory,
    )

    assert result["success"] is False
    assert result["return_code"] == 1
    assert "external server failure" in result["error"]


def test_experiment_runner_executes_relative_code_path_from_run_directory(
    tmp_path,
):
    runner = ExperimentRunner(
        output_directory=tmp_path / "runs",
        timeout_seconds=10,
    )

    run_directory = runner.create_run_directory("relative_path")

    code_path = runner.prepare_generated_code(
        "print('experiment ran')",
        run_directory,
    )

    result = runner.execute(
        code_path,
        run_directory,
    )

    assert result["success"] is True
    assert result["return_code"] == 0
    assert result["stdout"].strip() == "experiment ran"


def test_experiment_runner_handles_timeout(
    monkeypatch,
    tmp_path,
):
    """
    Verify that a subprocess timeout is handled gracefully and
    returned as a structured failed execution result.
    """
    runner = ExperimentRunner(
        output_directory=tmp_path,
        timeout_seconds=5,
    )

    run_directory = runner.create_run_directory(
        "timeout_test"
    )

    code_path = runner.prepare_generated_code(
        "print('test')",
        run_directory,
    )

    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(
        "subprocess.run",
        fake_subprocess_run,
    )

    result = runner.execute(
        code_path=code_path,
        run_directory=run_directory,
    )

    assert result["success"] is False
    assert result["status"] == "timeout"

    assert result["timeout_seconds"] == 5

    assert "timeout" in result["error"].lower()

    assert Path(result["stdout_path"]).exists()
    assert Path(result["stderr_path"]).exists()


def test_experiment_runner_rejects_successful_execution_with_missing_outputs(
    monkeypatch,
    tmp_path,
):
    """
    A Python experiment can exit with return code 0 while failing
    to produce the required ML experiment artifacts.

    The Runner must therefore return invalid_outputs rather than
    treating the experiment as a successful ML experiment.
    """
    runner = ExperimentRunner(
        output_directory=tmp_path,
        timeout_seconds=10,
    )

    def fake_subprocess_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="Training finished successfully.",
            stderr="",
        )

    monkeypatch.setattr(
        "subprocess.run",
        fake_subprocess_run,
    )

    monkeypatch.setattr(
        runner,
        "_repair_experiment_with_llm",
        lambda **kwargs: (
            False,
            "",
            "No repair available",
        ),
    )

    result = runner.run(
        generated_code="print('Training finished successfully.')"
    )

    assert result["success"] is False
    assert result["status"] == "invalid_outputs"

    assert result["execution"]["success"] is True

    assert result["output_validation"]["valid"] is False

    assert result["errors"]

    assert any(
        "metrics.json" in error
        for error in result["errors"]
    )


def test_experiment_runner_writes_valid_runner_result_json(
    monkeypatch,
    tmp_path,
):
    """
    Verify that run() writes runner_result.json and that the file
    contains the expected top-level structure.
    """
    runner = ExperimentRunner(
        output_directory=tmp_path,
        timeout_seconds=10,
    )

    def fake_subprocess_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="Training completed.",
            stderr="",
        )

    monkeypatch.setattr(
        "subprocess.run",
        fake_subprocess_run,
    )

    result = runner.run(
        generated_code="print('Training completed.')"
    )

    assert result["run_directory"] is not None

    run_directory = Path(
        result["run_directory"]
    )

    result_path = run_directory / "runner_result.json"

    assert result_path.exists()

    saved_result = json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )

    assert isinstance(saved_result, dict)

    assert "success" in saved_result
    assert "status" in saved_result
    assert "run_directory" in saved_result
    assert "generated_code_path" in saved_result
    assert "dataset_path" in saved_result
    assert "execution" in saved_result
    assert "outputs" in saved_result
    assert "output_validation" in saved_result
    assert "total_execution_seconds" in saved_result
    assert "errors" in saved_result

    assert saved_result["run_directory"] == result["run_directory"]

    assert isinstance(
        saved_result["execution"],
        dict,
    )

    assert isinstance(
        saved_result["outputs"],
        dict,
    )

    assert isinstance(
        saved_result["output_validation"],
        dict,
    )

    assert isinstance(
        saved_result["errors"],
        list,
    )


# ============================================================
# ExperimentOrchestrator - Dataset / Hypothesis Tests
# ============================================================


def test_experiment_orchestrator_uses_repository_dataset_by_default(
    monkeypatch,
):
    monkeypatch.setattr(
        DatasetManager,
        "get_latest_dataset",
        lambda self: "data/5g_nidd/5g_nidd.csv",
    )

    orchestrator = ExperimentOrchestrator()


def test_config_loads_from_repository_when_cwd_is_elsewhere(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config["logging_level"] is not None


def test_experiment_orchestrator_selects_best_accepted_hypothesis(
    monkeypatch,
):
    monkeypatch.setattr(
        DatasetManager,
        "get_latest_dataset",
        lambda self: "data/5g_nidd/5g_nidd.csv",
    )

    class FakeReport:
        def __init__(self, recommendation):
            self.recommendation = recommendation

    class FakeHypothesis:
        def __init__(
            self,
            hypothesis_id,
            text,
            elo_score,
            recommendation,
        ):
            self.hypothesis_id = hypothesis_id
            self.text = text
            self.elo_score = elo_score
            self.is_active = True
            self.reflection_report = FakeReport(recommendation)

    better = FakeHypothesis(
        "H-2",
        "Better hypothesis",
        1600.0,
        "ACCEPT",
    )

    weaker = FakeHypothesis(
        "H-1",
        "Weaker hypothesis",
        1400.0,
        "ACCEPT",
    )

    rejected = FakeHypothesis(
        "H-3",
        "Rejected hypothesis",
        1800.0,
        "REJECT",
    )

    context = SimpleNamespace(
        get_active_hypotheses=lambda: [
            better,
            weaker,
            rejected,
        ]
    )

    orchestrator = ExperimentOrchestrator()

    candidates = orchestrator.get_experiment_candidates(context)

    assert [h.hypothesis_id for h in candidates] == ["H-2", "H-1"]

    assert orchestrator.select_best_hypothesis(context).hypothesis_id == "H-2"


def test_experiment_orchestrator_delegates_execution_to_runner(
    monkeypatch,
):
    monkeypatch.setattr(
        DatasetManager,
        "get_latest_dataset",
        lambda self: "data/5g_nidd/5g_nidd.csv",
    )

    orchestrator = ExperimentOrchestrator()

    specification = dict(VALID_SPECIFICATION)

    preparation = {
        "success": True,
        "experiment_id": "H-1_test",
        "experiment_specification": specification,
    }

    generation = {
        "success": True,
        "pytorch_code": "print('experiment')",
    }

    runner_result = {
        "success": True,
        "status": "completed",
        "output_validation": {
            "valid": True,
            "warnings": [],
        },
    }

    monkeypatch.setattr(
        orchestrator,
        "prepare_experiment",
        lambda **kwargs: preparation,
    )

    monkeypatch.setattr(
        orchestrator,
        "generate_pytorch_code",
        lambda specification: generation,
    )

    runner_calls = []

    def fake_run_generated_experiment(**kwargs):
        runner_calls.append(kwargs)
        return runner_result

    monkeypatch.setattr(
        orchestrator,
        "run_generated_experiment",
        fake_run_generated_experiment,
    )

    result = orchestrator.run_experiment(
        context=object(),
        execute_generated_code=True,
    )

    assert result["success"] is True
    assert result["execution"] == runner_result

    assert len(runner_calls) == 1
    assert runner_calls[0]["experiment_id"] == "H-1_test"
    assert runner_calls[0]["generated_result"] == generation


# ============================================================
# Integration Tests - Live LM Studio
# ============================================================


@pytest.mark.integration
def test_code_generation_agent_live_lmstudio_call():
    """
    Test that CodeGenerationAgent can call a live LM Studio
    server and produce a valid experiment specification.

    This test requires a running LM Studio server configured
    in config.yaml.
    """

    agent = CodeGenerationAgent(model="qwen/qwen3.8-27b")

    result = agent.generate(VALID_SPECIFICATION)

    assert isinstance(result, dict)
    assert "success" in result
    assert "model" in result
    assert "generation_seconds" in result

    if result["success"]:
        assert result["model_recommendation"] is not None
        assert result["experiment_plan"] is not None
        assert isinstance(
            result["pytorch_code"],
            str,
        )
        assert len(result["pytorch_code"]) > 0

        assert "import torch" in result["pytorch_code"] or "torch" in result["pytorch_code"].lower()
    else:
        assert len(result["errors"]) > 0

        print(f"Code generation failed: {result['errors']}")


@pytest.mark.integration
def test_lmstudio_native_chat_endpoint_is_reachable():
    """
    Test that the LM Studio native chat API endpoint is
    reachable and can accept a request payload.
    """

    import requests

    from app.utils import (
        get_lmstudio_native_chat_url,
    )

    native_url = get_lmstudio_native_chat_url()

    payload = {
        "model": "qwen/qwen3.8-27b",
        "input": ("Hello, please respond with a single word."),
        "temperature": 0.5,
        "max_output_tokens": 10,
        "reasoning": "off",
        "store": False,
        "stream": False,
    }

    try:
        response = requests.post(
            native_url,
            json=payload,
            timeout=10,
        )

        assert response.status_code in [
            200,
            500,
        ], f"Unexpected status {response.status_code} from LM Studio at {native_url}"

    except requests.ConnectionError as exc:
        pytest.skip(f"LM Studio not reachable at {native_url}: {exc}")

    except requests.Timeout:
        pytest.skip(f"LM Studio timeout at {native_url}")


@pytest.mark.integration
def test_call_llm_resolves_configured_model():
    """
    Test that call_llm() uses the configured model
    and can reach LM Studio.
    """

    from app.utils import get_lmstudio_model

    configured_model = get_lmstudio_model()

    assert configured_model, "No LLM model configured"

    result = call_llm(
        "Respond with the word 'acknowledged'.",
        temperature=0.2,
        model=configured_model,
        max_tokens=20,
    )

    assert isinstance(result, str)
    assert len(result) > 0

    if not result.startswith("Error:"):
        print(f"LM Studio response: {result[:100]}")


@pytest.mark.integration
def test_display_generated_pytorch_code(capsys):
    """
    Display the full generated PyTorch code and experiment
    metadata from LM Studio.

    Run with:

        pytest tests/test_experiment_pipeline.py::test_display_generated_pytorch_code -v -s -m integration
    """

    print("\n" + "=" * 80)
    print("Calling LM Studio CodeGenerationAgent...")
    print("=" * 80)

    agent = CodeGenerationAgent(model="qwen/qwen3.8-27b")

    result = agent.generate(VALID_SPECIFICATION)

    print(f"\nGeneration Success: {result['success']}")

    print(f"Model Used: {result['model']}")

    print(f"Generation Time: {result['generation_seconds']:.2f}s")

    if result["success"]:
        print("\n" + "=" * 80)
        print("MODEL RECOMMENDATION")
        print("=" * 80)

        print(
            json.dumps(
                result["model_recommendation"],
                indent=2,
            )
        )

        print("\n" + "=" * 80)
        print("EXPERIMENT PLAN")
        print("=" * 80)

        print(
            json.dumps(
                result["experiment_plan"],
                indent=2,
            )
        )

        print("\n" + "=" * 80)
        print("ASSUMPTIONS")
        print("=" * 80)

        for i, assumption in enumerate(
            result["assumptions"],
            1,
        ):
            print(f"  {i}. {assumption}")

        print("\n" + "=" * 80)
        print("DEPENDENCIES")
        print("=" * 80)

        for i, dependency in enumerate(
            result["dependencies"],
            1,
        ):
            print(f"  {i}. {dependency}")

        print("\n" + "=" * 80)
        print("GENERATED PYTORCH CODE")
        print("=" * 80)

        print(result["pytorch_code"])

        assert result["model_recommendation"] is not None
        assert result["experiment_plan"] is not None

        assert isinstance(
            result["pytorch_code"],
            str,
        )

        assert len(result["pytorch_code"]) > 0

        assert "import torch" in result["pytorch_code"] or "torch" in result["pytorch_code"].lower()

    else:
        print("\nGeneration Failed:")

        for error in result["errors"]:
            print(f"  - {error}")

        assert False, f"Code generation failed: {result['errors']}"
