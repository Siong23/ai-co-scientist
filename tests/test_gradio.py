"""Offline tests for imports and Gradio UI construction.

LM Studio model discovery is mocked so these tests are deterministic and make
no network calls.
"""

import importlib.util
import os
import time
from unittest.mock import patch

import pytest


def test_core_imports():
    import gradio  # noqa: F401

    from app.agents import SupervisorAgent  # noqa: F401
    from app.models import ContextMemory, ResearchGoal  # noqa: F401
    from app.tools.arxiv_search import ArxivSearchTool  # noqa: F401
    from app.utils import fetch_lmstudio_models, get_lmstudio_base_url, logger  # noqa: F401


@pytest.fixture(scope="module")
def gradio_app_module():
    """Load the root app.py as a module (the app/ package shadows it on import)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("gradio_app", os.path.join(repo_root, "app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gradio_interface_constructs_without_network(gradio_app_module):
    with patch.object(gradio_app_module, "fetch_lmstudio_models", return_value=[]):
        demo = gradio_app_module.create_gradio_interface()
    assert demo is not None
    # The fetch failed, so the module must have fallen back to a non-empty default model list.
    assert gradio_app_module.available_models


def test_run_history_loads_existing_runs_and_delete_controls(gradio_app_module, monkeypatch, tmp_path):
    from app.models import ResearchGoal
    from app.run_store import RUNS_DIR_ENV, save_run

    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
    save_run(
        research_goal=ResearchGoal(description="Existing saved run"),
        cycle_details={"iteration": 1, "steps": {}},
        status="done",
        references_html="",
        results_html="",
        run_id="run-existing",
    )

    with patch.object(gradio_app_module, "fetch_lmstudio_models", return_value=[]):
        demo = gradio_app_module.create_gradio_interface()

    saved_runs = [
        component
        for component in demo.config["components"]
        if component["type"] == "html" and component["props"].get("label") == "Saved Runs"
    ]
    delete_dropdowns = [
        component
        for component in demo.config["components"]
        if component["type"] == "dropdown" and component["props"].get("label") == "Saved Run to Delete"
    ]
    delete_buttons = [
        component
        for component in demo.config["components"]
        if component["type"] == "button" and component["props"].get("value") == "Delete Selected Run"
    ]

    assert "Existing saved run" in saved_runs[0]["props"]["value"]
    assert any("run-existing" in str(choice) for choice in delete_dropdowns[0]["props"]["choices"])
    assert delete_buttons


def test_default_model_is_selected_and_first_choice(gradio_app_module):
    gradio_app_module.available_models = [
        "another/model",
        gradio_app_module.CONFIGURED_LLM_MODEL,
    ]

    choices = gradio_app_module.get_model_dropdown_choices()

    assert choices[0] == gradio_app_module.CONFIGURED_LLM_MODEL
    assert choices.count(gradio_app_module.CONFIGURED_LLM_MODEL) == 1


def test_first_available_model_is_default_when_configured_model_is_unavailable(gradio_app_module, monkeypatch):
    monkeypatch.setattr(gradio_app_module, "CONFIGURED_LLM_MODEL", "unavailable-model")

    choices = gradio_app_module.get_model_dropdown_choices(["local/model-a", "local/model-b"])

    assert choices[0] == "local/model-a"
    assert "unavailable-model" not in choices


def test_local_model_list_comes_from_lmstudio(gradio_app_module):
    with patch.object(
        gradio_app_module,
        "fetch_lmstudio_models",
        return_value=["local/model-a", "local/model-b"],
    ) as mock_fetch:
        models = gradio_app_module.fetch_available_models()

    assert models == ["local/model-a", "local/model-b"]
    mock_fetch.assert_called_once()


def test_references_render_only_sources_used_for_generation(
    gradio_app_module,
):
    cycle_details = {
        "steps": {
            "generation": {
                "sources": [
                    {
                        "title": "Selected evidence",
                        "authors": ["Researcher"],
                        "arxiv_id": "1234.5678v1",
                        "published": "2024-01-01",
                        "abstract": "Directly relevant evidence.",
                        "arxiv_url": ("https://arxiv.org/abs/1234.5678"),
                        "pdf_url": ("https://arxiv.org/pdf/1234.5678"),
                    }
                ]
            }
        }
    }

    html = gradio_app_module.get_references_html(cycle_details)

    assert "Retrieved Evidence Used for Generation" in html
    assert "Selected evidence" in html
    assert "Space VLBI" not in html


def test_references_do_not_search_again_when_no_source_was_used(
    gradio_app_module,
):
    html = gradio_app_module.get_references_html({"steps": {"generation": {"sources": []}}})

    assert html == ("<p>No retrieved evidence was used for generation.</p>")


def test_hypothesis_evidence_sources_are_clickable_and_validated(
    gradio_app_module,
):
    cycle_details = {
        "iteration": 1,
        "steps": {
            "generation": {
                "sources": [
                    {
                        "source_id": "arXiv:1111.1111v2",
                        "arxiv_id": "1111.1111v2",
                    },
                    {
                        "source_id": "arXiv:hep-th/9901001",
                        "arxiv_id": "hep-th/9901001",
                    },
                ],
                "hypotheses": [
                    {
                        "id": "G1",
                        "title": "Grounded hypothesis",
                        "text": "Testable claim.",
                        "evidence_source_ids": [
                            "arXiv:1111.1111v2",
                            "arXiv:hep-th/9901001",
                            "arXiv:9999.9999",
                        ],
                    }
                ],
            }
        },
    }

    html = gradio_app_module.format_cycle_results(cycle_details)

    assert "Evidence Sources:" in html
    assert 'href="https://arxiv.org/abs/1111.1111v2"' in html
    assert 'href="https://arxiv.org/abs/hep-th/9901001"' in html
    assert "arXiv:9999.9999" not in html
    assert 'target="_blank" rel="noopener noreferrer"' in html


def test_advanced_settings_exposes_available_model_choices(gradio_app_module):
    models = [gradio_app_module.CONFIGURED_LLM_MODEL, "local/alternative-model"]

    with patch.object(gradio_app_module, "fetch_available_models", return_value=models):
        gradio_app_module.available_models = models
        demo = gradio_app_module.create_gradio_interface()

    model_dropdowns = [
        component
        for component in demo.config["components"]
        if component["type"] == "dropdown" and str(component["props"].get("label", "")).startswith("LLM Model")
    ]

    assert len(model_dropdowns) == 1
    assert "local/alternative-model" in str(model_dropdowns[0]["props"]["choices"])
    assert model_dropdowns[0]["props"]["interactive"] is True


def test_run_cycle_with_progress_streams_active_status(gradio_app_module, monkeypatch, tmp_path):
    from app.models import ContextMemory, ResearchGoal
    from app.run_store import RUNS_DIR_ENV

    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
    gradio_app_module.current_research_goal = ResearchGoal(description="status test")
    gradio_app_module.global_context = ContextMemory()

    def slow_cycle(research_goal, context, cycle_supervisor):
        time.sleep(0.02)
        context.iteration_number += 1
        return {
            "status": "done",
            "results_html": "<p>done</p>",
            "references_html": "<p>refs</p>",
            "cycle_details": {"iteration": context.iteration_number, "steps": {}},
            "log_file": "",
        }

    monkeypatch.setattr(gradio_app_module, "execute_cycle", slow_cycle)
    monkeypatch.setattr(gradio_app_module, "write_report", lambda run: "report.html")
    monkeypatch.setattr(gradio_app_module, "report_file_url", lambda path: "/report.html")

    updates = list(gradio_app_module.run_cycle_with_progress(timeout_seconds=1, poll_seconds=0.001))

    assert any(
        "Active work: generating, reviewing, ranking, and evolving hypotheses." in update[0] for update in updates
    )
    assert all("Streamed hypothesis" not in update[1] for update in updates[:-1])
    assert any("Elapsed:" in update[0] for update in updates)
    assert updates[-1][0].startswith("done")
    assert updates[-1][1:] == ("<p>done</p>", "<p>refs</p>")
    assert gradio_app_module.global_context.iteration_number == 1


def test_run_cycle_with_progress_times_out(gradio_app_module, monkeypatch, tmp_path):
    from app.models import ContextMemory, ResearchGoal
    from app.run_store import RUNS_DIR_ENV

    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
    gradio_app_module.current_research_goal = ResearchGoal(description="timeout test")
    gradio_app_module.global_context = ContextMemory()

    def stuck_cycle(research_goal, context, cycle_supervisor):
        time.sleep(0.05)
        context.iteration_number = 99
        return {
            "status": "late success",
            "results_html": "<p>late</p>",
            "references_html": "",
            "cycle_details": {"iteration": 99, "steps": {}},
            "log_file": "",
        }

    monkeypatch.setattr(gradio_app_module, "execute_cycle", stuck_cycle)

    updates = list(gradio_app_module.run_cycle_with_progress(timeout_seconds=0.01, poll_seconds=0.001))

    assert "timed out" in updates[-1][0]
    assert "time limit" in updates[-1][1]
    run_files = list((tmp_path / "runs").glob("*.json"))
    assert len(run_files) == 1
    assert gradio_app_module.global_context.iteration_number == 0
    time.sleep(0.06)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1
