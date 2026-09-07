import json

from app.models import ResearchGoal
from app.run_store import (
    _final_hypotheses,
    delete_run,
    history_html,
    list_runs,
    render_report,
    report_file_url,
    save_run,
    write_report,
)

FAKE_KEY = "LMSTUDIO-THIS-FAKE-KEY-MUST-NOT-PERSIST"


def _cycle_details():
    return {
        "iteration": 1,
        "research_trace": [
            {
                "step": "generation",
                "status": "completed",
                "title": "Generating hypotheses",
                "summary": "Generated one evidence-grounded candidate.",
                "details": ["Evidence: source-1"],
                "elapsed_seconds": 1.5,
            }
        ],
        "steps": {
            "ranking2": {
                "hypotheses": [
                    {
                        "id": "H1",
                        "title": "Safer catalyst",
                        "text": "Use a lower-temperature catalyst.",
                        "novelty_review": "HIGH",
                        "feasibility_review": "MEDIUM",
                        "elo_score": 1220.5,
                        "review_comments": ["Promising next step."],
                    }
                ]
            },
            "meta_review": {
                "meta_review_critique": ["Good grounding."],
                "research_overview": {"top_ranked_hypotheses": [], "suggested_next_steps": []},
            },
        },
    }


def test_final_hypotheses_uses_latest_dynamic_ranking_round():
    steps = {
        "ranking_3": {"hypotheses": [{"id": "H3", "elo_score": 1300}]},
        "ranking_4": {"hypotheses": [{"id": "H4", "elo_score": 1260}]},
    }

    assert _final_hypotheses(steps) == [{"id": "H4", "elo_score": 1260}]


def test_save_run_persists_json_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("LMSTUDIO_API_KEY", FAKE_KEY)
    goal = ResearchGoal(description=f"Study catalyst with api_key={FAKE_KEY}")

    details = _cycle_details()
    details["research_trace"][0]["summary"] = f"Completed with api_key={FAKE_KEY}"
    run = save_run(
        research_goal=goal,
        cycle_details=details,
        status=f"done Authorization: Bearer {FAKE_KEY}",
        references_html="<p>refs</p>",
        results_html="<p>results</p>",
        log_file="results/app_log_test.txt",
        run_id="run-test",
    )

    saved = json.loads((tmp_path / "runs" / "run-test.json").read_text(encoding="utf-8"))
    serialized = json.dumps(saved)
    assert run["run_id"] == "run-test"
    assert FAKE_KEY not in serialized
    assert "***REDACTED***" in serialized


def test_save_run_persists_experiment_result(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    experiment_result = {
        "code_generation": {"pytorch_code": "print('ok')"},
        "execution": {
            "outputs": {
                "metrics": {"accuracy": 0.9999},
                "training_history": {"loss": [1.0, 0.5]},
                "checkpoint_path": "runs/experiment/best_model.pt",
                "visualizations": ["runs/experiment/loss.png"],
            }
        },
    }

    save_run(
        research_goal=ResearchGoal(description="Run an experiment"),
        cycle_details={"experiment_result": experiment_result},
        status="done",
        references_html="",
        results_html="",
        experiment_result=experiment_result,
        run_id="run-experiment",
    )

    saved = json.loads((tmp_path / "runs" / "run-experiment.json").read_text(encoding="utf-8"))
    assert saved["experiment_result"] == experiment_result


def test_report_escapes_user_and_model_content(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    goal = ResearchGoal(description="<script>alert('goal')</script>")
    details = _cycle_details()
    details["steps"]["ranking2"]["hypotheses"][0]["title"] = "<img src=x onerror=alert(1)>"
    details["research_trace"][0]["summary"] = "<script>alert('trace')</script>"

    run = save_run(
        research_goal=goal,
        cycle_details=details,
        status="done",
        references_html="<script>alert('refs')</script>",
        results_html="<p>results</p>",
        run_id="run-report",
    )

    report = render_report(run)
    assert "<script>alert('goal')</script>" not in report
    assert "<img src=x onerror=alert(1)>" not in report
    assert "<script>alert('trace')</script>" not in report
    assert "&lt;script&gt;alert" in report
    assert "Research Process" in report


def test_report_persists_and_renders_evidence_funnel_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    details = _cycle_details()
    details["steps"]["generation"] = {
        "hypotheses": [],
        "evidence_funnel": {
            "raw_search_hits": 129,
            "unique_candidates": 37,
            "selected_sources": 10,
            "acquisition_attempts": 4,
            "committed_sources": 2,
            "retrieved_passages": 2,
            "coverage_approved_sources": 1,
            "generation_consumed_sources": 0,
        },
        "evidence_pipeline": [
            {
                "requirement_id": "spikes",
                "query": "traffic spike measurements",
                "candidate_source_id": "arXiv:1234.5678",
                "provider": "arxiv",
                "raw_result_count": 80,
                "candidate_rank": 2,
                "reserved_for_requirement": True,
                "pdf_eligible": True,
                "acquisition_attempted": True,
                "acquisition_result": "committed",
                "index_status": "COMMITTED",
                "full_text_chunk_count": 12,
                "selected_chunk_ids": ["chunk-1"],
                "strict_gate_rejection_reason": "retained_committed_full_text_passage",
                "coverage_contribution": True,
            }
        ],
    }
    save_run(
        research_goal=ResearchGoal("Diagnose evidence loss"),
        cycle_details=details,
        status="failed",
        references_html="",
        results_html="",
        run_id="run-evidence-funnel",
    )

    saved = json.loads((tmp_path / "runs" / "run-evidence-funnel.json").read_text(encoding="utf-8"))
    generation = saved["cycle_details"]["steps"]["generation"]
    assert generation["evidence_funnel"]["raw_search_hits"] == 129
    assert generation["evidence_pipeline"][0]["requirement_id"] == "spikes"
    report = render_report(saved)
    assert "Evidence funnel" in report
    assert "Evidence path diagnostics" in report
    assert "arXiv:1234.5678" in report
    assert "traffic spike measurements" in report
    assert "chunk-1" in report


def test_history_lists_runs_and_creates_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    goal = ResearchGoal(description="Build a better battery")
    run = save_run(
        research_goal=goal,
        cycle_details=_cycle_details(),
        status="done",
        references_html="<p>refs</p>",
        results_html="<p>results</p>",
        run_id="run-history",
    )

    report_path = write_report(run)
    rows = list_runs()
    html = history_html()

    assert report_path.exists()
    assert rows[0]["run_id"] == "run-history"
    assert "Build a better battery" in html
    assert "/gradio_api/file=" in html


def test_delete_run_removes_saved_run_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    run = save_run(
        research_goal=ResearchGoal(description="Delete this run"),
        cycle_details=_cycle_details(),
        status="done",
        references_html="<p>refs</p>",
        results_html="<p>results</p>",
        run_id="run-delete",
    )
    report_path = write_report(run)

    assert (tmp_path / "runs" / "run-delete.json").exists()
    assert report_path.exists()
    assert delete_run("run-delete") is True
    assert not (tmp_path / "runs" / "run-delete.json").exists()
    assert not report_path.exists()
    assert "Delete this run" not in history_html()
    assert delete_run("run-delete") is False


def test_report_handles_nullable_hypothesis_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    details = _cycle_details()
    hypothesis = details["steps"]["ranking2"]["hypotheses"][0]
    hypothesis["title"] = None
    hypothesis["text"] = None
    hypothesis["novelty_review"] = None
    hypothesis["feasibility_review"] = None

    run = save_run(
        research_goal=ResearchGoal(description="Handle sparse model output"),
        cycle_details=details,
        status="done",
        references_html="<p>refs</p>",
        results_html="<p>results</p>",
        run_id="run-nullable",
    )

    report = render_report(run)
    assert "Untitled" in report
    assert "Traceback" not in report


def test_report_file_url_uses_gradio_file_endpoint(tmp_path):
    report_path = tmp_path / "reports" / "run with spaces.html"
    assert report_file_url(report_path).startswith("/gradio_api/file=")
    assert "run%20with%20spaces.html" in report_file_url(report_path)
