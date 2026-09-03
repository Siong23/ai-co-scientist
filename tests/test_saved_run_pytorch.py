"""
Opt-in test for generating and running PyTorch code from a saved run.

Run with:
    $env:RUN_SAVED_PYTORCH_TEST = "1"
    pytest tests/test_saved_run_pytorch.py -v -s -m "integration"

"""

import json
import os
from pathlib import Path

import pytest

from app.agents_modules.code_generation_agent import CodeGenerationAgent
from app.experiments.experiment_orchestrator import ExperimentOrchestrator

RUN_ID = "G5159_20260903_101435_953158"
RUN_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "experiments"
    / "results"
    / "runs"
    / RUN_ID
)
GENERATED_CODE_DIRECTORY = RUN_DIRECTORY.parents[1] / "generated_code"


@pytest.mark.integration
def test_generate_and_optionally_run_saved_pytorch_experiment():
    """Generate code from the saved top hypothesis and optionally execute it."""

    if os.getenv("RUN_SAVED_PYTORCH_TEST") != "1":
        pytest.skip("Set RUN_SAVED_PYTORCH_TEST=1 to call LM Studio.")

    configuration_path = RUN_DIRECTORY / "experiment_config.json"
    if not configuration_path.exists():
        pytest.fail(f"Saved experiment configuration not found: {configuration_path}")

    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    saved_specification = configuration["specification"]
    rankings_path = RUN_DIRECTORY / "final_rankings.json"
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    top_hypotheses = rankings.get("rankings", [])
    if not top_hypotheses:
        pytest.fail(f"No ranked hypotheses found: {rankings_path}")

    saved_hypothesis = top_hypotheses[0]
    specification = {
        "dataset": saved_specification["dataset"],
        "experiment": saved_specification["experiment"],
        "research_goal": saved_specification.get("research_goal", {}),
        "selected_hypothesis": {
            key: saved_hypothesis.get(key)
            for key in ("hypothesis_id", "title", "text")
        },
        "scientific_evaluation": {},
        "experiment_design": saved_specification["experiment_design"],
        "code_generation_requirements": saved_specification[
            "code_generation_requirements"
        ],
        "evaluation_metrics": saved_specification["evaluation_metrics"],
    }
    agent = CodeGenerationAgent(output_directory=GENERATED_CODE_DIRECTORY)
    generated_result = agent.generate(specification)

    assert generated_result["success"], generated_result.get("errors")
    code_path = agent.save_generated_code(
        generated_result,
        GENERATED_CODE_DIRECTORY / f"{RUN_ID}_generated_pytorch_code.py",
    )
    assert code_path is not None
    print(f"Generated PyTorch code: {code_path}")

    if os.getenv("EXECUTE_SAVED_PYTORCH_TEST") != "1":
        return

    dataset_path = specification.get("dataset", {}).get("path")
    orchestrator = ExperimentOrchestrator(
        dataset_name=specification["dataset"]["name"],
        dataset_path=dataset_path,
        device=specification["experiment"].get("device", "cpu"),
    )
    execution = orchestrator.experiment_runner.execute(
        code_path=code_path,
        run_directory=RUN_DIRECTORY,
        dataset_path=dataset_path,
    )

    assert execution["success"], execution