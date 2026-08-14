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
    research_process = [
        component
        for component in demo.config["components"]
        if component["type"] == "html" and component["props"].get("label") == "Research Process"
    ]
    assert len(research_process) == 1


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
    sidebars = [
        component
        for component in demo.config["components"]
        if component["type"] == "sidebar" and component["props"].get("elem_id") == "research-history-sidebar"
    ]
    sidebar_history = [
        component
        for component in demo.config["components"]
        if component["type"] == "radio" and component["props"].get("label") == "Recent research goals"
    ]
    assert "Existing saved run" in saved_runs[0]["props"]["value"]
    assert any("run-existing" in str(choice) for choice in delete_dropdowns[0]["props"]["choices"])
    assert saved_runs[0]["id"] < delete_dropdowns[0]["id"]
    assert delete_buttons
    assert sidebars and sidebars[0]["props"]["open"] is False
    assert any("Existing saved run" in str(choice) for choice in sidebar_history[0]["props"]["choices"])
    assert sidebar_history[0]["props"]["buttons"][0]["value"] == "Delete"
    assert sidebar_history[0]["props"]["buttons"][0]["variant"] == "stop"
    assert "#research-history-sidebar {\n            background: #ffffff !important;" in demo.css
    assert "#sidebar-run-list {\n            background: #ffffff !important;" in demo.css
    assert "var(--body-text-color) 6%, transparent" in demo.css
    assert "var(--body-text-color) 10%, transparent" in demo.css
    assert "#sidebar-run-list label:has(input:checked) span" in demo.css
    assert "color: var(--body-text-color) !important;" in demo.css
    assert not any(
        component["type"] == "group" and component["props"].get("elem_id") == "run-history-content"
        for component in demo.config["components"]
    )


def test_sidebar_delete_refreshes_history_table_and_choices(gradio_app_module, monkeypatch, tmp_path):
    from app.models import ResearchGoal
    from app.run_store import RUNS_DIR_ENV, save_run

    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
    save_run(
        research_goal=ResearchGoal(description="Delete from sidebar"),
        cycle_details={"iteration": 1, "steps": {}},
        status="done",
        references_html="",
        results_html="",
        run_id="run-sidebar-delete",
    )

    status, history, delete_dropdown, sidebar_history = gradio_app_module.delete_history_run(
        "run-sidebar-delete"
    )

    assert status == "Deleted saved run run-sidebar-delete."
    assert "Delete from sidebar" not in history
    assert delete_dropdown["choices"] == []
    assert sidebar_history["choices"] == []
    assert not (tmp_path / "runs" / "run-sidebar-delete.json").exists()


def test_sidebar_selection_restores_saved_goal_results_and_settings(gradio_app_module, monkeypatch, tmp_path):
    from app.models import ResearchGoal
    from app.run_store import RUNS_DIR_ENV, save_run

    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
    save_run(
        research_goal=ResearchGoal(
            description="Restore this research goal",
            llm_model="local/saved-model",
            num_hypotheses=6,
            generation_temperature=0.8,
            reflection_temperature=0.4,
            elo_k_factor=48,
            top_k_hypotheses=3,
        ),
        cycle_details={"iteration": 1, "steps": {}},
        status="Saved run completed.",
        references_html="<p>Saved references</p>",
        results_html="<p>Saved results</p>",
        run_id="run-to-restore",
    )
    gradio_app_module.available_models = ["local/current-model"]

    restored = gradio_app_module.load_history_run("run-to-restore")

    assert restored[0] == "Restore this research goal"
    assert "Loaded saved run run-to-restore" in restored[1]
    assert "agent workflow will appear here" in restored[2]
    assert restored[3] == "<p>Saved results</p>"
    assert restored[4] == "<p>Saved references</p>"
    assert restored[5]["value"] == "local/saved-model"
    assert "local/saved-model" in restored[5]["choices"]
    assert restored[6:11] == (6, 0.8, 0.4, 48, 3)
    assert restored[11]["selected"] == "current-run"


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
                        "full_text_indexed": True,
                        "full_text_chunks_used": 3,
                    }
                ]
            }
        }
    }

    html = gradio_app_module.get_references_html(cycle_details)

    assert "Retrieved Evidence Used for Generation" in html
    assert "Selected evidence" in html
    assert "Indexed in local ChromaDB; 3 relevant full-text chunk(s) used" in html
    assert "Space VLBI" not in html


def test_references_do_not_search_again_when_no_source_was_used(
    gradio_app_module,
):
    html = gradio_app_module.get_references_html({"steps": {"generation": {"sources": []}}})

    assert html == ("<p>No retrieved evidence was used for generation.</p>")


def test_references_hide_pdf_link_when_source_has_no_pdf(gradio_app_module):
    cycle_details = {
        "steps": {
            "generation": {
                "sources": [
                    {
                        "title": "Web-only evidence",
                        "source_id": "web:web-only",
                        "source_type": "web",
                        "provider": "tavily",
                        "url": "https://example.org/article",
                        "summary": "Current web guidance.",
                        "pdf_url": None,
                    }
                ]
            }
        }
    }

    html = gradio_app_module.get_references_html(cycle_details)

    assert "View source" in html
    assert "Download PDF" not in html
    assert "Web content" in html
    assert "Retrieved web content used directly" in html


def test_generation_results_show_every_search_provider_call(gradio_app_module):
    cycle_details = {
        "iteration": 1,
        "steps": {
            "generation": {
                "hypotheses": [],
                "sources": [],
                "search_stats": [
                    {
                        "round": 1,
                        "source": "arXiv",
                        "queries_completed": 5,
                        "queries_requested": 5,
                        "results": 20,
                        "elapsed_ms": 420,
                        "status": "ok",
                    },
                    {
                        "round": 1,
                        "source": "Tavily",
                        "queries_completed": 5,
                        "queries_requested": 5,
                        "results": 10,
                        "elapsed_ms": 510,
                        "status": "ok",
                    },
                ],
                "query_plan": {
                    "provisional_hypotheses": [
                        {
                            "role": "primary",
                            "statement": "A <provisional> mechanism may help.",
                        }
                    ],
                    "queries": [
                        {
                            "query": "targeted <query>",
                            "search_intent": "counterevidence",
                            "source_type": "academic",
                        }
                    ],
                },
                "query_fidelity": [
                    {
                        "kind": "query",
                        "query": "targeted <query>",
                        "accepted": True,
                    }
                ],
            }
        },
    }

    html = gradio_app_module.format_cycle_results(cycle_details)

    search_details_start = html.index('<details style="margin: 5px 0 10px;">')
    search_details_end = html.index("</details>", search_details_start)
    search_details = html[search_details_start:search_details_end]

    assert "<summary" in search_details
    assert "Search details</summary>" in search_details
    assert " open" not in search_details.split(">", 1)[0]
    assert "Search providers called" in html
    assert "arXiv" in html
    assert "5/5 queries, 20 results" in html
    assert "Tavily" in html
    assert "Provisional retrieval hypotheses (not evidence)" in html
    assert "A &lt;provisional&gt; mechanism may help." in html
    assert "counterevidence · academic: targeted &lt;query&gt;" in html
    assert "Query fidelity:</strong> 1/1 accepted" in html


def test_generation_results_explain_quality_gate_outcomes(gradio_app_module):
    cycle_details = {
        "iteration": 1,
        "steps": {
            "generation": {
                "hypotheses": [],
                "sources": [],
                "audits": [
                    {
                        "verdict": "REJECT",
                        "weighted_score": 66.5,
                        "hard_failures": [
                            "The final hypothesis contains unsupported claims."
                        ],
                        "warnings": [
                            "Weighted audit score is below 70/100."
                        ],
                    }
                ],
            }
        },
    }

    html = gradio_app_module.format_cycle_results(cycle_details)

    assert "Quality audit details" in html
    assert "REJECT · 66.5/100" in html
    assert "The final hypothesis contains unsupported claims." in html
    assert "Weighted audit score is below 70/100." in html


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
                        "title": "First arXiv source",
                    },
                    {
                        "source_id": "arXiv:hep-th/9901001",
                        "arxiv_id": "hep-th/9901001",
                        "title": "Second arXiv source",
                    },
                    {
                        "source_id": "s2:1e34a577580c659127152539f550bc02c8ca1644",
                        "arxiv_id": "s2:1e34a577580c659127152539f550bc02c8ca1644",
                        "title": "Semantic Scholar evidence",
                        "arxiv_url": ("https://www.semanticscholar.org/paper/1e34a577580c659127152539f550bc02c8ca1644"),
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
                            "s2:1e34a577580c659127152539f550bc02c8ca1644",
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
    assert 'href="https://www.semanticscholar.org/paper/1e34a577580c659127152539f550bc02c8ca1644"' in html
    assert "Semantic Scholar evidence" in html
    assert "s2:1e34a577580c659127152539f550bc02c8ca1644" not in html
    assert "arXiv:9999.9999" not in html
    assert 'target="_blank" rel="noopener noreferrer"' in html


def test_hypothesis_evidence_sources_omit_sources_without_a_useful_url(
    gradio_app_module,
):
    html = gradio_app_module.format_evidence_sources_html(
        {"evidence_source_ids": ["s2:missing-link"]},
        [
            {
                "source_id": "s2:missing-link",
                "title": "Untraceable source",
            }
        ],
    )

    assert html == "<p><strong>Evidence Sources:</strong> None recorded</p>"


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

    def slow_cycle(research_goal, context, cycle_supervisor, progress_callback=None):
        running_event = {
            "step": "generation",
            "status": "running",
            "title": "Discovering evidence",
            "summary": "Searching selected literature sources.",
            "details": [],
        }
        if progress_callback:
            progress_callback(running_event)
        time.sleep(0.02)
        context.iteration_number += 1
        completed_event = {
            **running_event,
            "status": "completed",
            "summary": "Generated two candidates.",
            "details": ["Evidence: <unsafe title>"],
            "elapsed_seconds": 0.02,
        }
        if progress_callback:
            progress_callback(completed_event)
        return {
            "status": "done",
            "results_html": "<p>done</p>",
            "references_html": "<p>refs</p>",
            "cycle_details": {
                "iteration": context.iteration_number,
                "steps": {},
                "research_trace": [completed_event],
            },
            "log_file": "",
        }

    monkeypatch.setattr(gradio_app_module, "execute_cycle", slow_cycle)
    monkeypatch.setattr(gradio_app_module, "write_report", lambda run: "report.html")
    monkeypatch.setattr(gradio_app_module, "report_file_url", lambda path: "/report.html")

    updates = list(gradio_app_module.run_cycle_with_progress(timeout_seconds=1, poll_seconds=0.001))

    assert any("Active work: Discovering evidence." in update[0] for update in updates)
    assert all("Streamed hypothesis" not in update[1] for update in updates[:-1])
    assert any("Elapsed:" in update[0] for update in updates)
    assert any("Searching selected literature sources." in update[3] for update in updates[:-1])
    assert updates[-1][0].startswith("done")
    assert updates[-1][1:3] == ("<p>done</p>", "<p>refs</p>")
    assert "Generated two candidates." in updates[-1][3]
    assert "&lt;unsafe title&gt;" in updates[-1][3]
    assert gradio_app_module.global_context.iteration_number == 1


def test_run_cycle_with_progress_times_out(gradio_app_module, monkeypatch, tmp_path):
    from app.models import ContextMemory, ResearchGoal
    from app.run_store import RUNS_DIR_ENV

    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
    gradio_app_module.current_research_goal = ResearchGoal(description="timeout test")
    gradio_app_module.global_context = ContextMemory()

    def stuck_cycle(research_goal, context, cycle_supervisor, progress_callback=None):
        if progress_callback:
            progress_callback(
                {
                    "step": "generation",
                    "status": "running",
                    "title": "Discovering evidence",
                    "summary": "Waiting for the model provider.",
                    "details": [],
                }
            )
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
    assert "Cycle time limit reached" in updates[-1][3]
    run_files = list((tmp_path / "runs").glob("*.json"))
    assert len(run_files) == 1
    assert gradio_app_module.global_context.iteration_number == 0
    time.sleep(0.06)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1
