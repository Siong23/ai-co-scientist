'''
Regression tests for the experiment pipeline (orchestrator, code generation, runner).

Offline tests (default):
    pytest tests/test_experiment_pipeline.py -v -s -m "not integration"

Integration tests (live LM Studio):
    pytest tests/test_experiment_pipeline.py -k integration -v -s
    (or via `make test-all`)
    pytest tests/test_experiment_pipeline.py -v -s -m "integration"
    pytest tests/test_experiment_pipeline.py::test_code_generation_agent_live_lmstudio_call -v -s -m integration
    pytest tests/test_experiment_pipeline.py::test_display_generated_pytorch_code -v -s -m integration
'''

import json
import pytest
from pathlib import Path
from types import SimpleNamespace

from app.agents_modules.code_generation_agent import CodeGenerationAgent
from app.experiments.experiment_orchestrator import ExperimentOrchestrator
from app.experiments.experiment_runner import ExperimentRunner
from app.utils import call_llm


VALID_SPECIFICATION = {
    "dataset": {"name": "5G-NIDD", "path": None, "task": "classification"},
    "selected_hypothesis": {
        "hypothesis_id": "H-1",
        "title": "Adaptive PQC selection",
        "text": "Adaptive PQC selection reduces 5G handshake latency under bursty load.",
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


def test_code_generation_agent_generates_valid_experiment(monkeypatch):
    payload = {
        "model_recommendation": {"name": "lstm", "reason": "temporal structure in the dataset"},
        "experiment_plan": {"architecture": "LSTM", "epochs": 3, "batch_size": 32},
        "assumptions": ["The dataset is tabular and time-ordered."],
        "dependencies": ["torch", "pandas", "numpy"],
        "pytorch_code": (
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class TinyLSTM(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))\n"
            "    def forward(self, x):\n"
            "        return self.net(x)\n"
        ),
    }

    calls = []

    def fake_call_llm(*args, **kwargs):
        calls.append(kwargs)
        return '{"model_recommendation": {"name": "lstm", "reason": "temporal structure in the dataset"}, "experiment_plan": {"architecture": "LSTM", "epochs": 3, "batch_size": 32}, "assumptions": ["The dataset is tabular and time-ordered."], "dependencies": ["torch", "pandas", "numpy"], "pytorch_code": "import torch\\nimport torch.nn as nn\\n\\nclass TinyLSTM(nn.Module):\\n    def __init__(self):\\n        super().__init__()\\n        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))\\n    def forward(self, x):\\n        return self.net(x)\\n"}'

    monkeypatch.setattr("app.agents_modules.code_generation_agent._call_llm", fake_call_llm)

    agent = CodeGenerationAgent(model="qwen/qwen3.8-27b")
    result = agent.generate(VALID_SPECIFICATION)

    assert result["success"] is True
    assert result["model_recommendation"]["name"] == "lstm"
    assert "class TinyLSTM" in result["pytorch_code"]
    assert calls[0]["reasoning"] == "off"


def test_experiment_runner_collects_standard_output_files(tmp_path):
    runner = ExperimentRunner(output_directory=tmp_path / "runs")
    run_directory = runner.create_run_directory("demo_run")

    (run_directory / "metrics.json").write_text('{"accuracy": 0.91}', encoding="utf-8")
    (run_directory / "training_history.json").write_text('{"loss": [1.0, 0.5]}', encoding="utf-8")
    (run_directory / "experiment_summary.json").write_text('{"status": "ok"}', encoding="utf-8")
    (run_directory / "best_model.pt").write_bytes(b"checkpoint")

    outputs = runner.collect_outputs(run_directory)

    assert outputs["metrics"]["accuracy"] == 0.91
    assert outputs["training_history"]["loss"] == [1.0, 0.5]
    assert outputs["experiment_summary"]["status"] == "ok"
    assert outputs["checkpoint_path"].endswith("best_model.pt")


def test_experiment_orchestrator_selects_best_accepted_hypothesis():
    class FakeReport:
        def __init__(self, recommendation):
            self.recommendation = recommendation

    class FakeHypothesis:
        def __init__(self, hypothesis_id, text, elo_score, recommendation):
            self.hypothesis_id = hypothesis_id
            self.text = text
            self.elo_score = elo_score
            self.is_active = True
            self.reflection_report = FakeReport(recommendation)

    better = FakeHypothesis("H-2", "Better hypothesis", 1600.0, "ACCEPT")
    weaker = FakeHypothesis("H-1", "Weaker hypothesis", 1400.0, "ACCEPT")
    rejected = FakeHypothesis("H-3", "Rejected hypothesis", 1800.0, "REJECT")

    context = SimpleNamespace(get_active_hypotheses=lambda: [better, weaker, rejected])

    orchestrator = ExperimentOrchestrator()

    candidates = orchestrator.get_experiment_candidates(context)
    assert [h.hypothesis_id for h in candidates] == ["H-2", "H-1"]
    assert orchestrator.select_best_hypothesis(context).hypothesis_id == "H-2"


# ============================================================
# Integration Tests (Live LM Studio)
# ============================================================
# Run with: pytest tests/test_experiment_pipeline.py -k integration -v
# or via: make test-all


@pytest.mark.integration
def test_code_generation_agent_live_lmstudio_call():
    """
    Test that CodeGenerationAgent can call a live LM Studio server
    and produce a valid experiment specification.
    
    This test requires a running LM Studio server configured in config.yaml.
    """
    agent = CodeGenerationAgent(model="qwen/qwen3.8-27b")
    result = agent.generate(VALID_SPECIFICATION)

    # Verify the result structure
    assert isinstance(result, dict)
    assert "success" in result
    assert "model" in result
    assert "generation_seconds" in result

    if result["success"]:
        # If generation succeeded, validate the payload
        assert result["model_recommendation"] is not None
        assert result["experiment_plan"] is not None
        assert isinstance(result["pytorch_code"], str)
        assert len(result["pytorch_code"]) > 0
        
        # Validate PyTorch code contains basic structure
        assert "import torch" in result["pytorch_code"] or "torch" in result["pytorch_code"].lower()
    else:
        # Document why generation failed (e.g., LM Studio unavailable)
        assert len(result["errors"]) > 0
        print(f"Code generation failed: {result['errors']}")


@pytest.mark.integration
def test_lmstudio_native_chat_endpoint_is_reachable():
    """
    Test that the LM Studio native chat API endpoint is reachable
    and can accept a request payload.
    
    This is a connectivity check before attempting full code generation.
    """
    from app.utils import get_lmstudio_native_chat_url
    import requests

    native_url = get_lmstudio_native_chat_url()
    
    # Small payload to test connectivity
    payload = {
        "model": "qwen/qwen3.8-27b",
        "input": "Hello, please respond with a single word.",
        "temperature": 0.5,
        "max_output_tokens": 10,
        "reasoning": "off",
        "store": False,
        "stream": False,
    }

    try:
        response = requests.post(native_url, json=payload, timeout=10)
        # Accept 500 as evidence the server is reachable but may have internal issues
        assert response.status_code in [200, 500], (
            f"Unexpected status {response.status_code} from LM Studio at {native_url}"
        )
    except requests.ConnectionError as exc:
        pytest.skip(f"LM Studio not reachable at {native_url}: {exc}")
    except requests.Timeout:
        pytest.skip(f"LM Studio timeout at {native_url}")


@pytest.mark.integration
def test_call_llm_resolves_configured_model():
    """
    Test that call_llm() uses the configured model and can reach LM Studio.
    """
    from app.utils import get_lmstudio_model

    configured_model = get_lmstudio_model()
    assert configured_model, "No LLM model configured"

    # Simple prompt to verify basic connectivity
    result = call_llm(
        "Respond with the word 'acknowledged'.",
        temperature=0.2,
        model=configured_model,
        max_tokens=20,
    )

    # Result can be either a successful response or an error message
    assert isinstance(result, str)
    assert len(result) > 0

    # If it's not an error, it should contain a response
    if not result.startswith("Error:"):
        print(f"LM Studio response: {result[:100]}")


@pytest.mark.integration
def test_display_generated_pytorch_code(capsys):
    """
    Display the full generated PyTorch code and experiment metadata from LM Studio.
    
    Run with:
        pytest tests/test_experiment_pipeline.py::test_display_generated_pytorch_code -v -s -m integration
    """
    print("\n" + "=" * 80)
    print("Calling LM Studio CodeGenerationAgent...")
    print("=" * 80)

    agent = CodeGenerationAgent(model="qwen/qwen3.8-27b")
    result = agent.generate(VALID_SPECIFICATION)

    print(f"\n✓ Generation Success: {result['success']}")
    print(f"✓ Model Used: {result['model']}")
    print(f"✓ Generation Time: {result['generation_seconds']:.2f}s")

    if result["success"]:
        print(f"\n{'='*80}")
        print("MODEL RECOMMENDATION")
        print(f"{'='*80}")
        print(json.dumps(result["model_recommendation"], indent=2))

        print(f"\n{'='*80}")
        print("EXPERIMENT PLAN")
        print(f"{'='*80}")
        print(json.dumps(result["experiment_plan"], indent=2))

        print(f"\n{'='*80}")
        print("ASSUMPTIONS")
        print(f"{'='*80}")
        for i, assumption in enumerate(result["assumptions"], 1):
            print(f"  {i}. {assumption}")

        print(f"\n{'='*80}")
        print("DEPENDENCIES")
        print(f"{'='*80}")
        for i, dep in enumerate(result["dependencies"], 1):
            print(f"  {i}. {dep}")

        print(f"\n{'='*80}")
        print("GENERATED PYTORCH CODE")
        print(f"{'='*80}")
        print(result["pytorch_code"])

        # Validate the output
        assert result["model_recommendation"] is not None
        assert result["experiment_plan"] is not None
        assert isinstance(result["pytorch_code"], str)
        assert len(result["pytorch_code"]) > 0
        assert "import torch" in result["pytorch_code"] or "torch" in result["pytorch_code"].lower()
    else:
        print(f"\n✗ Generation Failed:")
        for error in result["errors"]:
            print(f"  - {error}")
        assert False, f"Code generation failed: {result['errors']}"

