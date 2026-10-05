"""Local persistence and static reports for app research runs."""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .research_trace import format_research_trace_html
from .utils import logger, redact_secrets

DEFAULT_RESULTS_DIR = Path("results")
RUNS_DIR_ENV = "CO_SCIENTIST_RUNS_DIR"

# Stamped into every rendered report so ensure_report() can tell a report built
# by the current template from one left over by an older version. Bump the
# version whenever render_report()'s output changes in a way that should
# invalidate reports already on disk.
REPORT_TEMPLATE_MARKER = "<!-- co-scientist-report-template: v9 -->"

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s<>'\"]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s<>'\"]+"),
]


# ExperimentOrchestrator stores experiment artifacts here:
# app/experiments/results/runs/
EXPERIMENT_RESULTS_DIR_ENV = "CO_SCIENTIST_EXPERIMENT_RESULTS_DIR"
DEFAULT_EXPERIMENT_RESULTS_DIR = Path("app/experiments/results")

_PERCENT_METRIC_NAMES = {
    "accuracy", "acc", "precision", "precision_weighted", "weighted_precision",
    "recall", "recall_weighted", "weighted_recall", "f1", "f1_score",
    "f1_weighted", "weighted_f1", "macro_f1", "micro_f1", "macro_precision",
    "macro_recall", "balanced_accuracy",
}
_PERCENT_UNITS = {"%", "percent", "percentage", "percentage_point", "percentage_points", "pp"}
_RUN_STATUS_LABELS = {
    "success": "Completed",
    "timeout": "Timed out",
    "cancelled": "Cancelled",
    "invalid_outputs": "Finished, but outputs failed validation",
    "failed": "Failed",
    "runner_error": "Runner error",
}
_MAX_INLINE_CHARS = 20000
 
 
def _selected_hypothesis_id(selected: Any) -> Any:
    """The orchestrator stores 'hypothesis_id'; cycle steps store 'id'."""
    if not isinstance(selected, dict):
        return None
    return _first_non_empty(selected.get("hypothesis_id"), selected.get("id"))
 
 
def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
 
 
def _count(value: Any) -> int:
    """The runner stores attempts/repairs/installations as LISTS of records."""
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0
 
 
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
 
 
def _format_value(name: str, value: Any, unit: Any = None) -> str:
    """Format a scalar metric. Proportion-type metrics are shown as percentages."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if not _is_number(value):
        return str(value)
    number = float(value)
    unit_text = str(unit).strip() if unit else ""
    key = str(name).strip().lower()
    if unit_text.lower() in _PERCENT_UNITS or (not unit_text and key in _PERCENT_METRIC_NAMES):
        if 0.0 <= number <= 1.0:
            # Show the proportion as a percentage and keep the raw value, e.g. "95.00% (0.95)".
            return f"{number * 100.0:.2f}% ({value})"
        return f"{number:.2f}%"
    text = f"{number:.4g}"
    return f"{text} {unit_text}" if unit_text else text
 
 
def _truncate(text: Any, limit: int = _MAX_INLINE_CHARS) -> str:
    text = str(text if text is not None else "")
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} earlier characters omitted ...]\n" + text[-limit:]
 
 
def _json_block(value: Any, summary: str) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        rendered = json.dumps(value, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return (
        f"<details><summary>{_escape(summary)}</summary>"
        f"<pre>{_escape(_truncate(rendered))}</pre></details>"
    )
 
 
def _find_in_run_dir(run_directory: Any, filename: str) -> Optional[Path]:
    if not run_directory:
        return None
    try:
        root = Path(str(run_directory)).expanduser()
        if not root.is_dir():
            return None
        direct = root / filename
        if direct.is_file():
            return direct.resolve()
        for match in root.rglob(filename):
            if match.is_file():
                return match.resolve()
    except OSError:
        return None
    return None


_TWO_MODEL_HEADLINE = ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "false_alarm_rate")
 
 
def _recommended_model_label(experiment_result: Optional[Dict[str, Any]]) -> str:
    generation = _as_dict(_as_dict(experiment_result).get("code_generation"))
    recommendation = generation.get("model_recommendation")
    if isinstance(recommendation, str):
        return recommendation.strip()
    recommendation = _as_dict(recommendation)
    name = str(recommendation.get("model_name") or "").strip()
    architecture = recommendation.get("architecture")
    architecture = " ".join(str(architecture).split()) if architecture and not isinstance(architecture, (dict, list)) else ""
    if architecture and architecture.lower() not in name.lower():
        return f"{name} - {architecture[:200]}" if name else architecture[:200]
    return name
 
 
def _two_model_report(metrics: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> str:
    """Model 1 (existing; never saw the unseen attack) vs Model 2 (proposed; trained on all attacks).
 
    Uses the metric names the "TWO-MODEL PROTOCOL" prompt section requires: headline metrics are the
    proposed model's; baseline_<metric> are Model 1's; per_class_recall_baseline / per_class_recall_proposed;
    unseen_attack. Returns "" when they are absent.
    """
    summary = _as_dict(summary)
    unseen = str(_first_non_empty(metrics.get("unseen_attack"), summary.get("unseen_attack")) or "").strip()
    headline = []
    for name in _TWO_MODEL_HEADLINE:
        base, new = metrics.get(f"baseline_{name}"), metrics.get(name)
        if _is_number(base) and _is_number(new):
            headline.append(
                f"<tr><td>{_escape(name.replace('_', ' ').title())}</td>"
                f"<td>{_escape(_format_value(name, base))}</td><td>{_escape(_format_value(name, new))}</td></tr>"
            )
    per_old, per_new = metrics.get("per_class_recall_baseline"), metrics.get("per_class_recall_proposed")
    per_rows = []
    if isinstance(per_old, dict) and isinstance(per_new, dict):
        for cls in sorted(set(per_old) | set(per_new), key=lambda c: (str(c) != unseen, str(c))):
            a, b = per_old.get(cls), per_new.get(cls)
            tag = ' <span class="muted">(unseen for Model 1)</span>' if str(cls) == unseen else ""
            per_rows.append(
                f"<tr><td>{_escape(cls)}{tag}</td>"
                f"<td>{_escape(_format_value('recall', a) if _is_number(a) else 'N/A')}</td>"
                f"<td>{_escape(_format_value('recall', b) if _is_number(b) else 'N/A')}</td></tr>"
            )
    if not headline and not per_rows:
        return ""
    head = (
        "<thead><tr><th>{}</th><th>Model 1: existing (not trained on the unseen attack)</th>"
        "<th>Model 2: proposed (all attacks)</th></tr></thead>"
    )
    out = "<h3>Model 1 vs Model 2</h3>"
    if unseen:
        out += f"<p><strong>Unseen attack for Model 1:</strong> {_escape(unseen)}</p>"
    if headline:
        out += "<table>" + head.format("Overall metric (same test split)") + f"<tbody>{''.join(headline)}</tbody></table>"
    if per_rows:
        out += "<table>" + head.format("Recall per attack type") + f"<tbody>{''.join(per_rows)}</tbody></table>"
    return out
 
 
def _model_rationale_report(experiment_result: Dict[str, Any]) -> str:
    """Proposed model + why it was chosen (CodeGenerationAgent's model_recommendation)."""
    generation = _as_dict(experiment_result.get("code_generation"))
    recommendation = generation.get("model_recommendation")
    if isinstance(recommendation, str) and recommendation.strip():
        recommendation = {"model_name": recommendation}
    recommendation = _as_dict(recommendation)
    if not recommendation:
        return ""
 
    def show(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "; ".join(str(item) for item in value if str(item).strip())
        if isinstance(value, dict):
            return "; ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)
 
    known = [
        ("model_name", "Proposed model"),
        ("algorithm", "Algorithm"),
        ("architecture", "Architecture"),
        ("approach_type", "Approach type"),
        ("reason_for_selection", "Why this model"),
        ("relationship_to_rank1_hypothesis", "Link to the Rank #1 hypothesis"),
    ]
    rows, used = [], set()
    for key, label in known:
        used.add(key)
        value = recommendation.get(key)
        if value not in (None, "", [], {}):
            rows.append((label, show(value)))
    for key, value in recommendation.items():
        if key not in used and value not in (None, "", [], {}):
            rows.append((str(key).replace("_", " ").capitalize(), show(value)))
    if not rows:
        return ""
    out = [
        "<h3>Proposed Model and Rationale</h3>",
        '<p class="muted">The model/algorithm the automated experiment implemented, as chosen from the '
        "Rank #1 hypothesis, and the reason it was selected.</p>",
        "<table><tbody>"
        + "".join(f"<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>" for label, value in rows)
        + "</tbody></table>",
    ]
    assumptions = [a for a in _as_list(generation.get("assumptions")) if str(a).strip()]
    if assumptions:
        out.append("<h3>Assumptions</h3><ul>" + "".join(f"<li>{_escape(show(a))}</li>" for a in assumptions) + "</ul>")
    plan = generation.get("experiment_plan")
    if plan:
        out.append(_json_block(plan, "Experiment plan (from the code generator)"))
    return "".join(out)


def get_results_dir() -> Path:
    """Return the directory used for AI Co-Scientist run persistence."""
    return Path(os.getenv(RUNS_DIR_ENV, DEFAULT_RESULTS_DIR)).expanduser()


def get_runs_dir() -> Path:
    """Return the directory containing saved AI Co-Scientist runs."""
    return get_results_dir() / "runs"


def get_reports_dir() -> Path:
    """Return the directory containing generated HTML reports."""
    return get_results_dir() / "reports"


def get_experiment_results_dir() -> Path:
    """
    Return the root directory containing automated experiment artifacts.

    This matches ExperimentOrchestrator.RESULTS_DIR and can be overridden
    for deployment/server environments using CO_SCIENTIST_EXPERIMENT_RESULTS_DIR.
    """
    return Path(
        os.getenv(
            EXPERIMENT_RESULTS_DIR_ENV,
            DEFAULT_EXPERIMENT_RESULTS_DIR,
        )
    ).expanduser()


def get_gradio_allowed_paths() -> List[str]:
    """
    Return filesystem directories that Gradio is allowed to serve.

    Reports and automated experiment artifacts must both be included because
    reports contain links to generated code, logs, checkpoints, metrics,
    and visualizations stored outside the normal report directory.
    """
    paths = [
        get_reports_dir(),
        get_experiment_results_dir(),
    ]

    return [
        str(path.resolve())
        for path in paths
    ]


def report_file_url(report_path: Path) -> str:
    """Return a Gradio file-serving URL for a generated report/file."""
    path = Path(report_path).expanduser().resolve()

    return (
        f"/gradio_api/file="
        f"{quote(path.as_posix())}"
    )


def generate_run_id(created_at: Optional[dt.datetime] = None) -> str:
    timestamp = (created_at or dt.datetime.now(dt.timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:8]}"


def redact_text(text: str) -> str:
    redacted = redact_secrets(text)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}***REDACTED***" if match.groups() else "***REDACTED***", redacted
        )
    return redacted


def sanitize(value: Any) -> Any:
    """Recursively convert run data into JSON-safe, redacted values."""
    if isinstance(value, Path):
        return redact_text(str(value))
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, set):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): sanitize(item)
            for key, item in value.items()
        }
    return value


def research_goal_to_dict(research_goal: Any) -> Dict[str, Any]:
    if research_goal is None:
        return {}
    return sanitize(
        {
            "description": getattr(research_goal, "description", ""),
            "preferences": getattr(research_goal, "preferences", ""),
            "idea_attributes": getattr(research_goal, "idea_attributes", ""),
            "constraints": getattr(research_goal, "constraints", {}),
            "research_id": getattr(research_goal, "research_id", None),
            "research_type": getattr(research_goal, "research_type", "auto"),
            "resolved_research_type": getattr(research_goal, "resolved_research_type", None),
            "llm_model": getattr(research_goal, "llm_model", None),
            "query_rewrite_model": getattr(research_goal, "query_rewrite_model", None),
            "num_hypotheses": getattr(research_goal, "num_hypotheses", None),
            "generation_temperature": getattr(research_goal, "generation_temperature", None),
            "reflection_temperature": getattr(research_goal, "reflection_temperature", None),
            "elo_k_factor": getattr(research_goal, "elo_k_factor", None),
            "top_k_hypotheses": getattr(research_goal, "top_k_hypotheses", None),
        }
    )


def save_run(
    *,
    research_goal: Any,
    cycle_details: Dict[str, Any],
    status: str,
    references_html: str,
    results_html: str,
    log_file: Optional[str] = None,
    experiment_result: Optional[Dict[str, Any]] = None,
    comparison_result: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    created_at: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    created = created_at or dt.datetime.now(dt.timezone.utc)
    run = sanitize(
        {
            "run_id": run_id or generate_run_id(created),
            "created_at": created.isoformat(),
            "research_goal": research_goal_to_dict(research_goal),
            "status": status,
            "log_file": log_file,
            "cycle_details": cycle_details,
            "references_html": references_html,
            "results_html": results_html,
            "experiment_result": experiment_result,
            "comparison_result": comparison_result,
        }
    )
    get_runs_dir().mkdir(parents=True, exist_ok=True)
    run_path = get_run_path(run["run_id"])
    # Run JSON is an immutable audit snapshot. Research sessions evolve in the
    # separate research-state store; reusing a run ID must never rewrite history.
    with run_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(run, indent=2, sort_keys=True))
    return run


def get_run_path(run_id: str) -> Path:
    safe_run_id = Path(run_id).name
    return get_runs_dir() / f"{safe_run_id}.json"


def load_run(run_id: str) -> Dict[str, Any]:
    return json.loads(get_run_path(run_id).read_text(encoding="utf-8"))


def delete_run(run_id: str) -> bool:
    """Delete a saved run and its generated HTML report.

    Returns True when the persisted run JSON existed and was removed. The report
    file is best-effort because reports can be regenerated and may not exist.
    """
    if not run_id:
        return False

    safe_run_id = Path(run_id).name
    run_path = get_run_path(safe_run_id)
    existed = run_path.exists()
    if not existed:
        return False

    run_path.unlink()
    report_path = get_reports_dir() / f"{safe_run_id}.html"
    report_path.unlink(missing_ok=True)
    return True


def list_runs(limit: Optional[int] = 20) -> List[Dict[str, Any]]:
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []

    summaries = []
    for path in runs_dir.glob("*.json"):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        goal = run.get("research_goal", {})
        cycle = run.get("cycle_details", {})
        summaries.append(
            {
                "run_id": run.get("run_id", path.stem),
                "created_at": run.get("created_at", ""),
                "goal": goal.get("description", ""),
                "model": goal.get("llm_model", ""),
                "iteration": cycle.get("iteration", ""),
                "status": run.get("status", ""),
            }
        )

    sorted_runs = sorted(summaries, key=lambda item: item.get("created_at", ""), reverse=True)
    if limit is None:
        return sorted_runs
    return sorted_runs[:limit]


def render_report(run: Dict[str, Any]) -> str:
    goal = run.get("research_goal", {})
    cycle = run.get("cycle_details", {})
    steps = cycle.get("steps", {})
    research_trace = cycle.get("research_trace", [])

    final_hypotheses = _final_hypotheses(steps)

    experiment_result = run.get("experiment_result")
    if not isinstance(experiment_result, dict):
        experiment_result = {}

    # Prefer the explicitly saved comparison result. Fall back to the
    # cycle_details copy for compatibility with older saved runs.
    comparison_result = run.get("comparison_result")
    if not isinstance(comparison_result, dict) or not comparison_result:
        comparison_result = cycle.get("comparison_result")
    if not isinstance(comparison_result, dict) or not comparison_result:
        comparison_result = (experiment_result or {}).get("comparison")
    if not isinstance(comparison_result, dict):
        comparison_result = {}

    status = _escape(run.get("status"), "Unknown")
    status_class = status.lower().replace(" ", "-")

    html_parts = [
        "<!doctype html>",
        REPORT_TEMPLATE_MARKER,
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(run.get('run_id'), 'Run report')}</title>",
        """
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                margin: 32px;
                color: #1f2933;
                background: #f8fafc;
            }

            main {
                max-width: 1100px;
                margin: 0 auto;
            }

            section {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 10px;
                padding: 22px;
                margin-top: 24px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            }

            h1 {
                margin-bottom: 4px;
            }

            h2 {
                margin-top: 0;
            }

            h3 {
                margin-top: 20px;
            }

            .meta {
                color: #52606d;
            }

            .status {
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: 600;
            }

            .status.completed,
            .status.success {
                color: #166534;
                background: #dcfce7;
            }

            .status.warning {
                color: #92400e;
                background: #fef3c7;
            }

            .status.error,
            .status.failed {
                color: #991b1b;
                background: #fee2e2;
            }

            .hypothesis {
                border-left: 4px solid #2f80ed;
                background: #f7fbff;
                padding: 16px 18px;
                margin: 14px 0;
                border-radius: 6px;
            }

            .selected-hypothesis {
                border-left: 5px solid #059669;
                background: #f0fdf4;
                padding: 18px;
                border-radius: 8px;
                margin-top: 12px;
            }

            .metric-good {
                font-weight: 600;
            }

            .metric-difference {
                font-weight: 600;
                white-space: nowrap;
            }

            pre {
                white-space: pre-wrap;
                background: #f5f7fa;
                padding: 12px;
                border-radius: 6px;
                overflow: auto;
            }

            table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
            }

            td,
            th {
                border: 1px solid #d9e2ec;
                padding: 9px;
                text-align: left;
                vertical-align: top;
            }

            th {
                background: #f5f7fa;
            }

            .file-link {
                display: inline-block;
                padding: 7px 11px;
                border: 1px solid #bcccdc;
                border-radius: 6px;
                text-decoration: none;
            }

            .file-link:hover {
                text-decoration: underline;
            }

            .file-missing {
                color: #9b1c1c;
            }

            .source-list {
                margin: 8px 0;
                padding-left: 22px;
            }

            .muted {
                color: #6b7280;
            }

            details {
                margin-top: 12px;
                border: 1px solid #d9e2ec;
                border-radius: 7px;
                padding: 10px 14px;
                background: #fafbfc;
            }

            summary {
                cursor: pointer;
                font-weight: 600;
            }
        </style>
        """,
        "</head>",
        "<body><main>",
        f"<h1>Research Run {_escape(run.get('run_id'))}</h1>",
        f'<p class="meta">Created: {_escape(run.get("created_at"))}</p>',
        f'<p><span class="status {status_class}">{status}</span></p>',
        "<section><h2>Research Goal</h2>",
        f"<p>{_escape(goal.get('description'))}</p>",
        _settings_table(goal),
        "</section>",
        "<section><h2>Final Hypotheses</h2>",
    ]

    if final_hypotheses:
        for index, hypothesis in enumerate(final_hypotheses, start=1):
            html_parts.append(_hypothesis_block(index, hypothesis))
    else:
        html_parts.append("<p>No final hypotheses were available for this run.</p>")

    html_parts.append("</section>")
    experiment_result = run.get("experiment_result")

    # ------------------------------------------------------------
    # Selected Rank #1 hypothesis
    # ------------------------------------------------------------

    selected_hypothesis = _get_selected_experiment_hypothesis(
        experiment_result,
        final_hypotheses,
    )

    if selected_hypothesis:
        html_parts.append(
            "<section>"
            "<h2>Selected Rank #1 Hypothesis (For Automated Experiment)</h2>"
            '<div class="selected-hypothesis">'
            f"<p><strong>Title:</strong> "
            f"{_escape(selected_hypothesis.get('title'), 'Untitled')}</p>"
            f"<p><strong>ID:</strong> "
            f"{_escape(_selected_hypothesis_id(selected_hypothesis))}</p>"
            f"<p>{_escape(selected_hypothesis.get('text'))}</p>"
            "</div>"
            "</section>"
        )

    # ------------------------------------------------------------
    # Evidence / reference experiment
    # ------------------------------------------------------------

    if experiment_result:
        evidence_section = _evidence_reference_section(experiment_result)

        if evidence_section:
            html_parts.append(evidence_section)

        specification_section = _experiment_specification_section(experiment_result)
        if specification_section:
            html_parts.append(specification_section)

    # ------------------------------------------------------------
    # Automated experiment
    # ------------------------------------------------------------

    if experiment_result:
        html_parts.append(_experiment_report_section(experiment_result))

    # ------------------------------------------------------------
    # Paper vs experiment
    # ------------------------------------------------------------

    if comparison_result:
        html_parts.append(
            _comparison_report_section(comparison_result)
        )

    # ------------------------------------------------------------
    # Research trace
    # ------------------------------------------------------------

    if research_trace:
        html_parts.extend(
            [
                "<section><h2>Research Process</h2>",
                format_research_trace_html(
                    research_trace,
                    elapsed_seconds=cycle.get("execution_time"),
                ),
                "</section>",
            ]
        )

    # ------------------------------------------------------------
    # Cycle steps
    # ------------------------------------------------------------

    html_parts.append("<section><h2>Cycle Steps</h2>")

    for step_name, step_data in steps.items():
        hypotheses = (
            step_data.get("hypotheses", [])
            if isinstance(step_data, dict)
            else []
        )

        html_parts.append(
            f"<h3>{_escape(step_name)}</h3>"
            f"<p>{len(hypotheses)} hypotheses</p>"
        )

        if step_name == "generation":
            funnel = step_data.get("evidence_funnel", {})

            if isinstance(funnel, dict) and funnel:
                labels = (
                    ("raw_search_hits", "Raw search hits"),
                    ("unique_candidates", "Unique candidates"),
                    ("selected_sources", "Selected sources"),
                    ("abstract_candidates", "Abstract candidates"),
                    ("abstract_screened", "Abstracts screened"),
                    ("abstract_accepted", "Abstracts accepted"),
                    ("abstract_maybe", "Abstracts marked MAYBE"),
                    ("abstract_rejected", "Abstracts rejected"),
                    ("full_text_requested", "Full text requested"),
                    ("full_text_cache_hits", "Full-text cache hits"),
                    ("full_text_downloads", "Full-text downloads"),
                    ("acquisition_attempts", "Acquisition attempts"),
                    ("committed_sources", "COMMITTED sources"),
                    ("retrieved_passages", "Retrieved passages"),
                    ("coverage_approved_sources", "Coverage-approved sources"),
                    ("generation_consumed_sources", "Generation-consumed sources"),
                )

                html_parts.append(
                    "<h4>Evidence funnel</h4>"
                    "<table><tbody>"
                )

                for key, label in labels:
                    html_parts.append(
                        f"<tr><th>{_escape(label)}</th>"
                        f"<td>{_escape(funnel.get(key, 0))}</td></tr>"
                    )

                html_parts.append("</tbody></table>")

            pipeline = step_data.get("evidence_pipeline", [])

            if isinstance(pipeline, list) and pipeline:
                html_parts.append(
                    '<details style="max-height: 400px; overflow-y: auto; overflow-x: auto; border: 1px solid #ddd; border-radius: 6px;">'
                    '<summary>Evidence path diagnostics</summary>'
                    '<table><thead><tr>'
                    '<th>Requirement</th>'
                    '<th>Query</th>'
                    '<th>Provider</th>'
                    "<th>Raw results</th>"
                    "<th>Source</th>"
                    "<th>Rank</th>"
                    "<th>Reserved</th>"
                    "<th>PDF eligible</th>"
                    "<th>Attempted</th>"
                    "<th>Acquisition</th>"
                    "<th>Index</th>"
                    "<th>Indexed chunks</th>"
                    "<th>Selected chunk IDs</th>"
                    "<th>Strict gate</th>"
                    "<th>Coverage</th>"
                    "</tr></thead><tbody>"
                )

                for item in pipeline:
                    if not isinstance(item, dict):
                        continue

                    selected_chunk_ids = ", ".join(
                        str(value)
                        for value in item.get("selected_chunk_ids") or []
                    )

                    html_parts.append(
                        "<tr>"
                        f"<td>{_escape(item.get('requirement_id') or 'unscoped')}</td>"
                        f"<td>{_escape(item.get('query') or '')}</td>"
                        f"<td>{_escape(item.get('provider') or 'unknown')}</td>"
                        f"<td>{_escape(item.get('raw_result_count') or 0)}</td>"
                        f"<td>{_escape(item.get('candidate_source_id') or 'unknown')}</td>"
                        f"<td>{_escape(item.get('candidate_rank') or '')}</td>"
                        f"<td>{_escape(bool(item.get('reserved_for_requirement')))}</td>"
                        f"<td>{_escape(bool(item.get('pdf_eligible')))}</td>"
                        f"<td>{_escape(bool(item.get('acquisition_attempted')))}</td>"
                        f"<td>{_escape(item.get('acquisition_result') or 'not_attempted')}</td>"
                        f"<td>{_escape(item.get('index_status') or 'MISSING')}</td>"
                        f"<td>{_escape(item.get('full_text_chunk_count') or 0)}</td>"
                        f"<td>{_escape(selected_chunk_ids)}</td>"
                        f"<td>{_escape(item.get('strict_gate_rejection_reason') or 'not_evaluated')}</td>"
                        f"<td>{_escape(bool(item.get('coverage_contribution')))}</td>"
                        "</tr>"
                    )

                html_parts.append(
                    "</tbody></table>"
                    "</details>"
                )

        if step_name.startswith("ranking"):
            tournament = step_data.get("tournament_results", [])

            if tournament:
                html_parts.append(
                    "<details>"
                    "<summary>Ranking Tournament Details</summary>"
                    "<table>"
                    "<thead><tr>"
                    "<th>Comparison</th>"
                    "<th>Outcome</th>"
                    "<th>Confidence</th>"
                    "<th>Criteria</th>"
                    "<th>Reasoning</th>"
                    "</tr></thead><tbody>"
                )

                title_lookup = {
                    h["id"]: h["title"]
                    for h in hypotheses
                    if isinstance(h, dict) and h.get("id")
                }

                for result in tournament:
                    confidence = (
                        f"{result.get('confidence', 1)}/10"
                    )
                    criteria = ", ".join(
                        result.get("criteria", [])
                    )

                    title_a = title_lookup.get(
                        result.get("hypothesis_a"),
                        result.get("hypothesis_a"),
                    )
                    title_b = title_lookup.get(
                        result.get("hypothesis_b"),
                        result.get("hypothesis_b"),
                    )

                    html_parts.append(
                        "<tr>"
                        "<td>"
                        f"{_escape(title_a)} "
                        f"(ID: <b>{_escape(result.get('hypothesis_a'))}</b>)"
                        " vs "
                        f"{_escape(title_b)} "
                        f"(ID: <b>{_escape(result.get('hypothesis_b'))}</b>)"
                        "</td>"
                        f"<td>{_escape(result.get('outcome'))}</td>"
                        f"<td>{_escape(confidence)}</td>"
                        f"<td>{_escape(criteria)}</td>"
                        f"<td>{_escape(result.get('reasoning', ''))}</td>"
                        "</tr>"
                    )

                html_parts.append(
                    "</tbody></table>"
                    "</details>"
                )

        if step_name == "meta_review":
            html_parts.append(
                "<details>"
                "<summary>Meta Review Data</summary>"
                f"<pre>{_escape(json.dumps(step_data, indent=2, sort_keys=True))}</pre>"
                "</details>"
            )

    html_parts.extend(
        [
            "</section>",
            "<section><h2>References</h2>",
            "<p>Reference results are stored from the app display for this run.</p>",
            f"<pre>{_escape(run.get('references_html'))}</pre>",
            "</section>",
            "</main></body></html>",
        ]
    )

    return "\n".join(html_parts)


def _format_report_value(value: Any) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)

    return str(value)


def _format_report_metric(value: Any, metric_name: str = "") -> str:
    """
    Format a metric for the evidence/reference section.

    Classification metrics represented as proportions are displayed as
    percentages when the value is between 0 and 1.
    """
    if value is None:
        return "N/A"

    if isinstance(value, (int, float)):
        numeric = float(value)

        percentage_metrics = {
            "accuracy",
            "acc",
            "precision",
            "precision_weighted",
            "weighted_precision",
            "recall",
            "recall_weighted",
            "weighted_recall",
            "f1",
            "f1_score",
            "f1_weighted",
            "weighted_f1",
        }

        if metric_name.lower() in percentage_metrics and 0.0 <= numeric <= 1.0:
            return f"{numeric * 100:.2f}%"

        return f"{numeric:.4f}"

    return str(value)


def _evidence_reference_section(
    experiment_result: Dict[str, Any],
) -> str:
    """
    Render the evidence sources and reference experiment used to guide the
    automated experiment.
    """
    if not isinstance(experiment_result, dict):
        return ""

    preparation = experiment_result.get(
        "experiment_preparation",
        {},
    )

    if not isinstance(preparation, dict):
        preparation = {}

    evidence_sources = (
        preparation.get("evidence_sources")
        or experiment_result.get("evidence_sources")
        or []
    )

    if isinstance(evidence_sources, dict):
        evidence_sources = [evidence_sources]

    if not isinstance(evidence_sources, list):
        evidence_sources = []

    reference_experiment = (
        preparation.get("reference_experiment")
        or experiment_result.get("reference_experiment")
        or {}
    )

    if not isinstance(reference_experiment, dict):
        reference_experiment = {}

    evaluation_guidance = (
        preparation.get("evaluation_guidance")
        or experiment_result.get("evaluation_guidance")
        or {}
    )

    if not isinstance(evaluation_guidance, dict):
        evaluation_guidance = {}

    if not evidence_sources and not reference_experiment:
        return ""

    parts = [
        "<section>",
        "<h2>Evidence & Reference</h2>",
        "<p class=\"muted\">"
        "Research evidence used to guide the automated experiment."
        "</p>",
    ]

    # ------------------------------------------------------------
    # Evidence sources
    # ------------------------------------------------------------

    if evidence_sources:
        parts.append("<h3>Evidence Sources</h3><ul class=\"source-list\">")

        for source in evidence_sources:
            if isinstance(source, dict):
                title = (
                    source.get("title")
                    or source.get("name")
                    or source.get("source_id")
                    or "Untitled source"
                )

                url = source.get("url") or source.get("source_url")
                source_id = source.get("source_id")

                label = _escape(title)

                if url:
                    safe_url = _escape(url)
                    label = (
                        f'<a href="{safe_url}" target="_blank">'
                        f"{label}</a>"
                    )

                if source_id:
                    label += (
                        f" <span class=\"muted\">"
                        f"(ID: {_escape(source_id)})</span>"
                    )

                parts.append(f"<li>{label}</li>")

            else:
                parts.append(
                    f"<li>{_escape(source)}</li>"
                )

        parts.append("</ul>")

    # ------------------------------------------------------------
    # Reference experiment
    # ------------------------------------------------------------

    if reference_experiment:
        parts.append("<h3>Reference Experiment</h3>")

        available = reference_experiment.get("available")

        if available is not None:
            parts.append(
                f"<p><strong>Available:</strong> "
                f"{_escape(available)}</p>"
            )

        models = (
            reference_experiment.get("models")
            or reference_experiment.get("model")
        )

        datasets = (
            reference_experiment.get("datasets")
            or reference_experiment.get("dataset")
        )

        sources = reference_experiment.get("sources")

        if models:
            parts.append(
                f"<p><strong>Model:</strong> "
                f"{_escape(_format_report_value(models))}</p>"
            )

        if datasets:
            parts.append(
                f"<p><strong>Dataset:</strong> "
                f"{_escape(_format_report_value(datasets))}</p>"
            )

        if sources:
            parts.append(
                f"<p><strong>Sources:</strong> "
                f"{_escape(_format_report_value(sources))}</p>"
            )

        reference_metrics = (
            reference_experiment.get("reference_metrics")
            or reference_experiment.get("metrics")
            or {}
        )

        if isinstance(reference_metrics, dict) and reference_metrics:
            parts.append(
                "<h4>Reported Reference Metrics</h4>"
                "<table>"
                "<thead><tr>"
                "<th>Metric</th>"
                "<th>Reported Value</th>"
                "</tr></thead>"
                "<tbody>"
            )

            for name, value in reference_metrics.items():
                parts.append(
                    "<tr>"
                    f"<td>{_escape(name)}</td>"
                    f"<td>{_escape(_format_report_metric(value, name))}</td>"
                    "</tr>"
                )

            parts.append("</tbody></table>")

    # ------------------------------------------------------------
    # Evaluation guidance
    # ------------------------------------------------------------

    evidence_metrics = _first_non_empty(
        evaluation_guidance.get("evidence_derived_metrics"),
        evaluation_guidance.get("evidence_metrics"),
        evaluation_guidance.get("preferred_comparison_metrics"),
        evaluation_guidance.get("reference_metrics"),
    )

    if evidence_metrics:
        parts.append(
            "<h3>Evidence-Derived Evaluation Metrics</h3>"
            "<p class=\"muted\">"
            "These metrics are reported from the evidence/reference material "
            "and are shown as the metric requirements produced by the experiment "
            "preparation stage. run_store.py does not select or replace them."
            "</p>"
        )
        parts.append(_metric_name_list_table(evidence_metrics))

    reference_only_metrics = _first_non_empty(
        evaluation_guidance.get("reference_only_metrics"),
        evaluation_guidance.get("unreproducible_metrics"),
        evaluation_guidance.get("not_reproducible_metrics"),
    )

    if reference_only_metrics:
        parts.append(
            "<h4>Reference-Only / Not Currently Reproducible Metrics</h4>"
            "<p class=\"muted\">"
            "These metrics remain part of the evidence context but are not "
            "represented as automated experiment measurements unless the "
            "experiment explicitly produced them."
            "</p>"
            + _metric_name_list_table(reference_only_metrics)
        )

    parts.append(
        "<p class=\"muted\"><strong>Reference-value policy:</strong> "
        "values reported by the evidence are reference context only and are "
        "never copied into the automated experiment results.</p>"
    )

    parts.append("</section>")

    return "\n".join(parts)


def _first_non_empty(*values: Any) -> Any:
    """Return the first value that is meaningfully populated."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, set, dict)) and not value:
            continue
        return value
    return None


def _normalise_metric_names(value: Any) -> List[str]:
    """Convert an orchestrator-provided metric collection to display names."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        # Structured metric definitions may be keyed by metric name.
        return [str(key) for key in value.keys()]

    if isinstance(value, (list, tuple, set)):
        names: List[str] = []
        for item in value:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = _first_non_empty(
                    item.get("name"),
                    item.get("metric"),
                    item.get("metric_name"),
                    item.get("id"),
                )
                if name:
                    names.append(str(name))
        return names

    return [str(value)]


def _metric_name_list_table(metrics: Any) -> str:
    """Render metric names without inventing metric values."""
    names = _normalise_metric_names(metrics)
    if not names:
        return ""

    rows = "".join(
        f"<tr><td>{_escape(name)}</td></tr>"
        for name in names
    )
    return (
        "<table><thead><tr><th>Metric</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _get_experiment_preparation(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
    preparation = experiment_result.get("experiment_preparation", {})
    return preparation if isinstance(preparation, dict) else {}


def _get_experiment_specification(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the orchestrator-produced specification, if present."""
    preparation = _get_experiment_preparation(experiment_result)
    specification = _first_non_empty(
        preparation.get("experiment_specification"),
        experiment_result.get("experiment_specification"),
        preparation.get("specification"),
        experiment_result.get("specification"),
    )
    return specification if isinstance(specification, dict) else {}


def _get_evaluation_guidance(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
    preparation = _get_experiment_preparation(experiment_result)
    specification = _get_experiment_specification(experiment_result)
    guidance = _first_non_empty(
        preparation.get("evaluation_guidance"),
        specification.get("evaluation_guidance"),
        experiment_result.get("evaluation_guidance"),
    )
    return guidance if isinstance(guidance, dict) else {}
 
 
def _get_evidence_metric_names(experiment_result: Dict[str, Any]) -> List[str]:
    """Names of metrics that came from the paper evidence (for the 'Source' column)."""
    guidance = _get_evaluation_guidance(experiment_result)
    candidates = [
        guidance.get("directly_reproducible_metrics"),
        guidance.get("conditionally_comparable_metrics"),
        guidance.get("evidence_metrics"),
        guidance.get("evidence_derived_metrics"),  # legacy key
    ]
    names: List[str] = []
    seen = set()
    for candidate in candidates:
        for name in _normalise_metric_names(candidate):
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                names.append(name)
    return names
 
 
def _find_gpu_name(*sources: Any) -> Optional[str]:
    """Find the GPU model name wherever the generated experiment (or the runner) put it.
 
    The code-generation prompt only says "GPU name"; the LLM-written experiment may save it as
    gpu_name, "GPU name", gpu_device, cuda_device_name, device_name, gpu_model, or nest it under
    e.g. {"environment": {...}} / {"hardware": {...}}. Keys are matched case-insensitively.
    """
    exact = ("gpu_name", "gpu name", "gpu", "gpu_device", "gpu_device_name", "gpu_model",
             "cuda_device_name", "cuda_device", "device_name")
    skip_words = ("count", "memory", "mem", "util", "available", "used", "free", "total", "id", "index")
 
    def clean(value: Any) -> Optional[str]:
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, str):
            text = value.strip()
            if text and text.lower() not in {"n/a", "na", "none", "null", "cpu", "cuda", "unknown", "false", "true"}:
                return text
        return None
 
    def search(node: Any, depth: int = 0) -> Optional[str]:
        if not isinstance(node, dict) or depth > 3:
            return None
        lowered = {str(k).strip().lower().replace("-", "_"): v for k, v in node.items()}
        for key in exact:
            found = clean(lowered.get(key))
            if found:
                return found
        for key, value in lowered.items():  # e.g. "gpu_product_name", "primary_gpu"
            if ("gpu" in key or "cuda_device" in key) and not any(w in key for w in skip_words):
                found = clean(value)
                if found:
                    return found
        for value in node.values():
            found = search(value, depth + 1)
            if found:
                return found
        return None
 
    for source in sources:
        found = search(source)
        if found:
            return found
    return None

 
def _experiment_specification_section(experiment_result: Dict[str, Any]) -> str:
    if not isinstance(experiment_result, dict):
        return ""
    preparation = _get_experiment_preparation(experiment_result)
    specification = _get_experiment_specification(experiment_result)
    guidance = _get_evaluation_guidance(experiment_result)
 
    selected = _as_dict(
        _first_non_empty(
            preparation.get("selected_hypothesis"),
            specification.get("selected_hypothesis"),
            experiment_result.get("selected_hypothesis"),
        )
    )
    experiment_block = _as_dict(specification.get("experiment"))
    dataset_block = specification.get("dataset")
    if isinstance(dataset_block, dict):
        dataset_name = dataset_block.get("name")
        dataset_path = dataset_block.get("path")
        dataset_role = dataset_block.get("role")
    else:
        dataset_name, dataset_path, dataset_role = dataset_block, None, None
 
    experiment_type = _first_non_empty(
        experiment_block.get("experiment_type"), specification.get("experiment_type")
    )
    device = experiment_block.get("device")
    framework = experiment_block.get("framework")
    training_required = experiment_block.get("training_required")
 
    evaluation_metrics = _first_non_empty(
        guidance.get("experiment_evaluation_metrics"),
        guidance.get("preferred_comparison_metrics"),
        specification.get("evaluation_metrics"),
    )
    conditional = guidance.get("conditionally_comparable_metrics")
    reference_only = _first_non_empty(
        guidance.get("reference_only_metrics"), specification.get("reference_only_metrics")
    )
 
    if not any(
        value not in (None, "", [], {}, ())
        for value in (selected, dataset_name, experiment_type, evaluation_metrics, reference_only)
    ):
        return ""
 
    rows = []
    if experiment_type is not None:
        rows.append(("Experiment Type", _escape(experiment_type)))
    if dataset_name is not None:
        text = _escape(dataset_name)
        if dataset_role:
            text += f' <span class="muted">({_escape(dataset_role)})</span>'
        if dataset_path:
            text += f"<br><code>{_escape(dataset_path)}</code>"
        rows.append(("Dataset", text))
    if framework or device:
        rows.append(("Framework / Requested Device", _escape(f"{framework or 'N/A'} / {device or 'N/A'}")))
    if training_required is not None:
        rows.append(("Training Required", _escape(training_required)))
    if selected:
        hypothesis_id = _selected_hypothesis_id(selected)
        text = f"<strong>{_escape(selected.get('title') or 'Rank #1 hypothesis')}</strong>"
        if hypothesis_id:
            text += f' <span class="muted">(ID: {_escape(hypothesis_id)})</span>'
        if selected.get("text"):
            text += f"<br>{_escape(selected.get('text'))}"
        rows.append(("Selected Rank #1 Approach", text))
 
    parts = [
        "<section>",
        "<h2>Automated Experiment Specification</h2>",
        '<p class="muted">Specification produced by ExperimentOrchestrator. The Rank #1 '
        "hypothesis supplies the approach; evidence-derived metrics supply the evaluation "
        "requirements. Metrics that cannot be measured with the available dataset are not "
        "fabricated.</p>",
        "<table><tbody>"
        + "".join(f"<tr><th>{_escape(label)}</th><td>{value}</td></tr>" for label, value in rows)
        + "</tbody></table>",
    ]
    if evaluation_metrics:
        parts.append(
            "<h3>Metrics the Experiment Must Evaluate</h3>"
            '<p class="muted">Hypothesis metrics plus evidence metrics the dataset can '
            "reproduce directly.</p>" + _metric_name_list_table(evaluation_metrics)
        )
    if conditional:
        parts.append(
            "<h3>Conditionally Comparable Metrics</h3>"
            '<p class="muted">Comparable with the paper only if the experimental protocol '
            "is compatible.</p>" + _metric_name_list_table(conditional)
        )
    if reference_only:
        parts.append(
            "<h3>Reference-Only / Not Reproducible Here</h3>"
            '<p class="muted">Kept as evidence context; not required from the generated '
            "experiment.</p>" + _metric_name_list_table(reference_only)
        )
    parts.append("</section>")
    return "\n".join(parts)


def _experiment_report_section(experiment_result: Dict[str, Any]) -> str:
    if not isinstance(experiment_result, dict):
        experiment_result = {}
 
    status_code = str(experiment_result.get("status") or "")
    if status_code.startswith("skipped"):
        return (
            "<section><h2>Automated Experiment</h2>"
            f"<p><strong>Status:</strong> Skipped ({_escape(status_code.replace('_', ' '))})</p>"
            f"<p>{_escape(experiment_result.get('reason') or 'The automated experiment did not run.')}</p>"
            "</section>"
        )
 
    runner = _as_dict(experiment_result.get("execution"))
    raw = _as_dict(runner.get("execution"))
    outputs = _as_dict(runner.get("outputs"))
    summary = _as_dict(_first_non_empty(outputs.get("summary"), outputs.get("experiment_summary")))
    validation = _as_dict(_first_non_empty(runner.get("output_validation"), runner.get("validation")))
    code_generation = _as_dict(experiment_result.get("code_generation"))
    specification = _get_experiment_specification(experiment_result)
 
    metrics = _as_dict(outputs.get("metrics"))
    metric_definitions = _as_dict(
        _first_non_empty(outputs.get("metric_definitions"), runner.get("metric_definitions"))
    )
 
    # ---- status -------------------------------------------------
    runner_status = str(_first_non_empty(runner.get("status"), raw.get("status"), "") or "")
    if experiment_result.get("success"):
        status_text = "Completed"
    elif not runner and code_generation and not code_generation.get("success", True):
        status_text = "Failed during code generation"
    elif not runner:
        status_text = "Failed before execution"
    else:
        status_text = _RUN_STATUS_LABELS.get(runner_status, runner_status.replace("_", " ").title() or "Failed")
 
    attempts = _count(raw.get("attempts"))
    repairs = _count(raw.get("repairs"))
    installs = _count(raw.get("installations"))
    execution_seconds = _first_non_empty(
        runner.get("total_execution_seconds"),
        raw.get("total_execution_seconds"),
        raw.get("execution_seconds"),
        metrics.get("total_execution_seconds"),
    )
 
    run_directory = _first_non_empty(runner.get("run_directory"), runner.get("run_dir"))
    generated_code_path = _first_non_empty(
        runner.get("generated_code_path"),
        runner.get("code_path"),
        raw.get("code_path"),
        _as_dict(experiment_result.get("code_generation")).get("code_path"),
    )
    stdout_path = _first_non_empty(raw.get("stdout_path"), runner.get("stdout_path"))
    stderr_path = _first_non_empty(raw.get("stderr_path"), runner.get("stderr_path"))
    if not stdout_path:
        found = _find_in_run_dir(run_directory, "stdout.txt")
        stdout_path = str(found) if found else None
    if not stderr_path:
        found = _find_in_run_dir(run_directory, "stderr.txt")
        stderr_path = str(found) if found else None
 
    def resolve_file_path(file_path: Any) -> Optional[Path]:
        if not file_path:
            return None
        path = Path(str(file_path)).expanduser()
        if not path.is_absolute() and run_directory:
            path = Path(str(run_directory)) / path
        try:
            path = path.resolve()
        except OSError:
            return None
        return path if path.is_file() else None
 
    def file_link(file_path: Any, label: str) -> str:
        path = resolve_file_path(file_path)
        if path is None:
            return f'<span class="file-missing">{_escape(label)} not found</span>'
        return (
            f'<a class="file-link" href="{_escape(report_file_url(path))}" '
            f'target="_blank">{_escape(label)}</a>'
        )
 
    parts = [
        "<section><h2>Automated Experiment</h2>",
        f"<p><strong>Status:</strong> {_escape(status_text)}</p>",
    ]
    rationale = _model_rationale_report(experiment_result)
    if rationale:
        parts.append(rationale)
 
    # ---- execution summary -------------------------------------------
    rows = [
        ("Experiment ID", _first_non_empty(
            _get_experiment_preparation(experiment_result).get("experiment_id"),
            runner.get("experiment_id"),
        )),
        ("Execution Attempts", attempts or None),
        ("LLM Repair Attempts", repairs),
        ("Dependency Installation Attempts", installs),
    ]
    if _is_number(execution_seconds):
        rows.append(("Total Execution Time", f"{float(execution_seconds):.1f} s"))
    if raw.get("return_code") is not None:
        rows.append(("Process Return Code", raw.get("return_code")))
    parts.append(
        "<h3>Execution Summary</h3><table><tbody>"
        + "".join(
            f"<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>"
            for label, value in rows
            if value is not None
        )
        + "</tbody></table>"
    )
 
    # ---- environment ---------------------------------------------------
    requested_device = _as_dict(specification.get("experiment")).get("device")
    reported_device = _first_non_empty(summary.get("device"), outputs.get("device"))
    gpu_name = _find_gpu_name(summary, outputs, raw, runner)
    if any(value is not None for value in (requested_device, reported_device, gpu_name)):
        parts.append(
            "<h3>Execution Environment</h3><table><tbody>"
            f"<tr><th>Requested Device</th><td>{_escape(requested_device, 'N/A')}</td></tr>"
            f"<tr><th>Device Used By Experiment</th><td>{_escape(reported_device, 'Not reported')}</td></tr>"
            f"<tr><th>GPU</th><td>{_escape(gpu_name, 'N/A')}</td></tr>"
            "</tbody></table>"
        )
 
    # ---- files -----------------------------------------------------------
    file_rows = [("Generated Experiment", file_link(generated_code_path, "View generated_experiment.py"))]
    for label, filename, link_text in (
        ("Metrics", "metrics.json", "View metrics.json"),
        ("Experiment Summary", "experiment_summary.json", "View experiment_summary.json"),
        ("Training History", "training_history.json", "View training_history.json"),
        ("Runner Result", "runner_result.json", "View runner_result.json"),
    ):
        found = _find_in_run_dir(run_directory, filename)
        if found:
            file_rows.append((label, file_link(found, link_text)))
    if stdout_path:
        file_rows.append(("Standard Output", file_link(stdout_path, "View stdout")))
    if stderr_path:
        file_rows.append(("Error Output", file_link(stderr_path, "View stderr")))
    checkpoint = _first_non_empty(outputs.get("checkpoint_path"), outputs.get("checkpoint"))
    if checkpoint:
        file_rows.append(("Checkpoint", file_link(checkpoint, "Download checkpoint")))
    if run_directory:
        file_rows.append(("Run Directory", f"<code>{_escape(run_directory)}</code>"))
    parts.append(
        "<h3>Experiment Files</h3><table><tbody>"
        + "".join(f"<tr><th>{_escape(label)}</th><td>{value}</td></tr>" for label, value in file_rows)
        + "</tbody></table>"
    )
 
    # Logs are only written to disk on timeout; show the captured text otherwise.
    if not stdout_path and raw.get("stdout"):
        parts.append(
            f"<details><summary>Standard output</summary><pre>{_escape(_truncate(raw.get('stdout')))}</pre></details>"
        )
    if not stderr_path and raw.get("stderr"):
        parts.append(
            f"<details><summary>Error output</summary><pre>{_escape(_truncate(raw.get('stderr')))}</pre></details>"
        )
 
    # ---- metrics ------------------------------------------------------------
    scalar_rows = []
    structured = {}
    evidence_names = {name.strip().lower() for name in _get_evidence_metric_names(experiment_result)}
    for name, value in metrics.items():
        if name == "total_execution_seconds":
            continue  # runner timing, shown in the Execution Summary
        if isinstance(value, (dict, list, tuple)):
            structured[name] = value
            continue
        definition = _as_dict(metric_definitions.get(name))
        unit = definition.get("unit")
        source = "Evidence-derived" if str(name).strip().lower() in evidence_names else "Experiment output"
        description = definition.get("description") or definition.get("definition")
        scalar_rows.append(
            "<tr>"
            f"<td>{_escape(name)}"
            + (f'<br><span class="muted">{_escape(description)}</span>' if description else "")
            + "</td>"
            f"<td>{_escape(source)}</td>"
            f"<td>{_escape(_format_value(name, value, unit))}</td>"
            "</tr>"
        )
    parts.append("<h3>Evaluation Metrics</h3>")
    if scalar_rows:
        parts.append(
            "<table><thead><tr><th>Metric</th><th>Source</th><th>Automated Experiment Value</th></tr></thead>"
            "<tbody>" + "".join(scalar_rows) + "</tbody></table>"
        )
    elif not structured:
        parts.append("<p>No evaluation metrics were produced.</p>")
    for name, value in structured.items():
        parts.append(_json_block(value, f"{name} (structured metric)"))
 
    two_models = _two_model_report(metrics, summary)
    if two_models:
        parts.append(two_models)
 
    # ---- validation / history / summary -------------------------------------
    if validation and validation.get("valid") is False:
        warnings = validation.get("warnings") or []
        parts.append(
            "<h3>Output Validation</h3><p><strong>Outputs failed validation.</strong></p><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in warnings)
            + "</ul>"
        )
    elif validation.get("warnings"):
        parts.append(
            "<h3>Output Validation Warnings</h3><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in validation["warnings"])
            + "</ul>"
        )
    history = outputs.get("training_history")
    if history:
        epochs = len(history) if isinstance(history, list) else None
        label = f"Training history ({epochs} records)" if epochs is not None else "Training history"
        parts.append(_json_block(history[-5:] if isinstance(history, list) else history, label + " - last records"))
    parts.append(_json_block(summary, "Experiment summary (experiment_summary.json)"))
 
    # ---- errors ---------------------------------------------------------------
    errors = runner.get("errors") or experiment_result.get("errors") or []
    if not isinstance(errors, list):
        errors = [errors]
    seen_errors = set()
    error_blocks = []
    for error in errors:
        text = str(error).strip()
        if not text or text in seen_errors:
            continue
        seen_errors.add(text)
        first_line = text.splitlines()[0][:200]
        if "\n" in text:
            error_blocks.append(
                f"<li><details><summary>{_escape(first_line)}</summary>"
                f"<pre>{_escape(_truncate(text))}</pre></details></li>"
            )
        else:
            error_blocks.append(f"<li>{_escape(first_line)}</li>")
    if error_blocks:
        parts.append("<h3>Errors</h3><ul>" + "".join(error_blocks) + "</ul>")
 
    # ---- visualizations --------------------------------------------------------
    visualizations = outputs.get("visualizations") or []
    if not isinstance(visualizations, list):
        visualizations = [visualizations]
    rendered_visuals = []
    for visualization in visualizations:
        path = resolve_file_path(visualization)
        if path is None:
            continue
        url = _escape(report_file_url(path))
        label = _escape(path.name)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            rendered_visuals.append(
                f'<figure><a href="{url}" target="_blank"><img src="{url}" alt="{label}" '
                f'style="max-width:100%;height:auto"></a><figcaption>{label}</figcaption></figure>'
            )
        else:
            rendered_visuals.append(f'<p><a href="{url}" target="_blank">{label}</a></p>')
    parts.append(
        "<h3>Visualizations</h3><div>" + "".join(rendered_visuals) + "</div>"
        if rendered_visuals
        else "<p>No visualizations were produced.</p>"
    )
    parts.append("</section>")
    return "\n".join(parts)

def _comparison_metric_cell(value: Any, unit: Any, percentage: bool) -> str:
    if value is None:
        return "Not available"
    if not _is_number(value):
        return _escape(value)
    number = float(value)
    if percentage:
        if abs(number) <= 1:
            # Percentage plus the raw proportion, e.g. "91.00% (0.91)".
            return _escape(f"{number * 100.0:.2f}% ({value})")
        return _escape(f"{number:.2f}%")
    text = f"{number:.4g}"
    return _escape(f"{text} {unit}" if unit else text)
 
 
def _as_list(value: Any) -> List[Any]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]
 
 
def _text_list(*values: Any) -> List[str]:
    out: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        for item in (value if isinstance(value, (list, tuple, set)) else [value]):
            if isinstance(item, dict):
                item = item.get("name") or item.get("model") or item.get("title") or ""
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
    return out
 
 
def _model_comparison_report(comparison_result: Dict[str, Any], full_experiment_result: Optional[Dict[str, Any]]) -> str:
    """Published (paper evidence) model vs our Rank #1 hypothesis / implemented model.
 
    PaperReader stores `models_or_systems` / `datasets_or_testbeds` but ExperimentComparator reads
    `models` / `datasets`, so comparability.paper_model is normally empty; read the paper keys directly.
    """
    def short(text: Any, limit: int = 800) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: limit - 3] + "..."
 
    comparability = _as_dict(comparison_result.get("comparability"))
    paper_result = _as_dict(comparison_result.get("paper_result"))
    reference = _as_dict(comparison_result.get("reference_experiment"))
    models, baselines, datasets, objectives, urls = [], [], [], [], []
    for source in _as_list(reference.get("sources")):
        source = _as_dict(source)
        details = _as_dict(source.get("experiment_details"))
        models += _text_list(details.get("models_or_systems"), details.get("models"), source.get("models"))
        baselines += _text_list(details.get("baselines"))
        datasets += _text_list(details.get("datasets_or_testbeds"), details.get("datasets"), source.get("datasets"))
        objectives += _text_list(details.get("experiment_objective"), details.get("objective"))
        urls += _text_list(source.get("source_url"))
    models = _text_list(models, paper_result.get("models"), comparability.get("paper_model"))
    datasets = _text_list(datasets, paper_result.get("datasets"), comparability.get("paper_dataset"))
 
    ours = _as_dict(full_experiment_result)
    preparation = _as_dict(ours.get("experiment_preparation"))
    hypothesis = _as_dict(preparation.get("selected_hypothesis"))
    summary = _as_dict(_as_dict(_as_dict(ours.get("execution")).get("outputs")).get("summary"))
    dataset_spec = _as_dict(_get_experiment_specification(ours)).get("dataset")
    spec_dataset = dataset_spec.get("name") if isinstance(dataset_spec, dict) else dataset_spec
    recommendation = _as_dict(_as_dict(ours.get("code_generation")).get("model_recommendation"))
    our_models = _text_list(
        _recommended_model_label(ours), summary.get("model"), summary.get("model_name"), summary.get("model_algorithm"),
        summary.get("model/algorithm"), summary.get("algorithm"), comparability.get("experiment_model"),
    )
    our_datasets = _text_list(summary.get("dataset"), spec_dataset, comparability.get("experiment_dataset"))
    title = hypothesis.get("title") or comparison_result.get("hypothesis_title")
    hyp_id = _selected_hypothesis_id(hypothesis) or comparison_result.get("hypothesis_id")
    text = hypothesis.get("text")
 
    if not any([models, baselines, objectives, our_models, title, text]):
        return ""
 
    def cell(items: List[str], empty: str = "Not reported") -> str:
        return _escape(", ".join(items)) if items else f'<span class="muted">{_escape(empty)}</span>'
 
    approach = ""
    if title:
        approach = f"<strong>{_escape(title)}</strong>"
        if hyp_id:
            approach += f' <span class="muted">(ID: {_escape(hyp_id)})</span>'
    if text:
        approach += ("<br>" if approach else "") + _escape(short(text))
    links = "<br>".join(f'<a href="{_escape(u)}" target="_blank">{_escape(short(u, 90))}</a>' for u in urls)
    rows = [
        ("Proposed model / system", cell(models), cell(our_models, "Not reported by the experiment")),
        ("Approach / objective", cell([short(o) for o in objectives[:3]]), approach or '<span class="muted">Not available</span>'),
        ("Baselines", cell(baselines), '<span class="muted">Not applicable</span>'),
        ("Dataset / testbed", cell(datasets), cell(our_datasets)),
        ("Source", links or '<span class="muted">Not available</span>', "Rank #1 hypothesis"),
    ]
    return (
        "<h3>Proposed Models: Paper vs Rank #1 Hypothesis</h3><table><thead><tr><th></th>"
        "<th>Published (paper evidence)</th><th>Automated experiment (Rank #1 hypothesis)</th></tr></thead><tbody>"
        + "".join(f"<tr><th>{_escape(label)}</th><td>{a}</td><td>{b}</td></tr>" for label, a, b in rows)
        + "</tbody></table>"
    )

 
def _comparison_report_section(comparison_result: Dict[str, Any], full_experiment_result: Optional[Dict[str, Any]] = None) -> str:
    if not isinstance(comparison_result, dict) or not comparison_result:
        return ""
 
    status = str(comparison_result.get("status") or "unknown")
    paper_result = _as_dict(comparison_result.get("paper_result"))
    experiment_side = _as_dict(comparison_result.get("experiment_result"))
    comparability = _as_dict(comparison_result.get("comparability"))
    explanation = _as_dict(comparison_result.get("explanation"))
    errors = [str(item) for item in _as_list(comparison_result.get("errors") or comparison_result.get("error"))]
 
    metric_comparison = _as_dict(comparison_result.get("metric_comparison"))
    if isinstance(metric_comparison.get("metrics"), dict):
        comparison_metrics = metric_comparison["metrics"]
    else:  # older saved runs stored {metric: {...}} directly
        comparison_metrics = {k: v for k, v in metric_comparison.items() if isinstance(v, dict)}
 
    paper_metrics = _as_dict(_first_non_empty(paper_result.get("metrics"), comparison_result.get("paper_metrics")))
    experiment_metrics = _as_dict(
        _first_non_empty(experiment_side.get("metrics"), comparison_result.get("experiment_metrics"))
    )
 
    conclusion = _first_non_empty(
        comparison_result.get("conclusion"),
        explanation.get("overall_assessment"),
        comparability.get("reason"),
        comparison_result.get("reason"),
        "; ".join(errors),
        "No comparison conclusion was generated.",
    )
 
    primary_metrics = set()
    reference_experiment = _as_dict(comparison_result.get("reference_experiment"))
    for source in _as_list(reference_experiment.get("sources")):
        details = _as_dict(_as_dict(source).get("experiment_details"))
        for metric in _as_list(details.get("primary_metrics")):
            if isinstance(metric, str) and metric.strip():
                primary_metrics.add(re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_"))
 
    parts = [
        "<section><h2>Paper vs Automated Experiment</h2>",
        f"<p><strong>Status:</strong> {_escape(status.replace('_', ' ').title())}</p>",
        f"<p><strong>Conclusion:</strong> {_escape(conclusion)}</p>",
    ]
    if explanation.get("reproduction_level"):
        parts.append(f"<p><strong>Reproduction level:</strong> {_escape(explanation['reproduction_level'])}</p>")
    if primary_metrics:
        parts.append(f"<p><strong>Paper-declared primary metrics:</strong> {_escape(', '.join(sorted(primary_metrics)))}</p>")
 
    model_table = _model_comparison_report(comparison_result, full_experiment_result)
    if model_table:
        parts.append(model_table)
 
    # ---- comparability -----------------------------------------------------
    if comparability:
        context_rows = [
            ("Comparable", comparability.get("comparable")),
            ("Comparison level", comparability.get("comparison_level")),
            ("Reason", comparability.get("reason")),
            ("Published model/system", comparability.get("paper_model")),
            ("Automated experiment model", comparability.get("experiment_model")),
            ("Published dataset/testbed", comparability.get("paper_dataset")),
            ("Experiment dataset", comparability.get("experiment_dataset")),
            ("Dataset warning", comparability.get("dataset_warning")),
        ]
        rows = "".join(
            f"<tr><th>{_escape(label)}</th><td>{_escape(_format_report_value(value))}</td></tr>"
            for label, value in context_rows
            if value not in (None, "", [], {})
        )
        if rows:
            parts.append(f"<h3>Comparability</h3><table><tbody>{rows}</tbody></table>")
        warnings = _as_list(comparability.get("warnings"))
        if warnings:
            parts.append("<ul>" + "".join(f"<li>{_escape(w)}</li>" for w in warnings) + "</ul>")
 
    # ---- metric table ---------------------------------------------------------
    metric_names = sorted(comparison_metrics or set(paper_metrics) | set(experiment_metrics))
    if metric_names:
        body = []
        for name in metric_names:
            record = _as_dict(comparison_metrics.get(name))
            unit = _first_non_empty(record.get("reference_unit"), record.get("unit"))
            diff_pp = record.get("difference_percentage_points")
            percentage = (
                diff_pp is not None
                or str(unit or "").lower() in _PERCENT_UNITS
                or (not unit and str(name).lower() in _PERCENT_METRIC_NAMES)
            )
            paper_value = record.get("paper", paper_metrics.get(name))
            experiment_value = record.get("experiment", experiment_metrics.get(name))
            paper_text = _comparison_metric_cell(paper_value, unit, percentage)
 
            value_type = str(record.get("reference_value_type") or "measured_value").lower()
            relation = str(record.get("reference_relation") or "exact").lower()
            if value_type == "upper_bound":
                paper_text = {"less_than": "&lt; "}.get(relation, "&lt;= ") + paper_text
            elif value_type == "lower_bound":
                paper_text = {"greater_than": "&gt; "}.get(relation, "&gt;= ") + paper_text
 
            difference = record.get("difference")
            if diff_pp is not None:
                diff_text = f"{float(diff_pp):+.2f} pp"
            elif value_type != "measured_value":
                diff_text = _first_non_empty(
                    record.get("comparison_interpretation"), "Compared with reported constraint"
                )
                if record.get("constraint_satisfied") is not None:
                    diff_text = f"{diff_text} (satisfied: {record['constraint_satisfied']})"
            elif _is_number(difference):
                diff_text = f"{float(difference) * 100:+.2f} pp" if percentage else f"{float(difference):+.4g}"
                if unit and not percentage:
                    diff_text += f" {unit}"
            else:
                diff_text = "Not comparable"
 
            label = str(name).replace("_", " ").title()
            if re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_") in primary_metrics:
                label += " (paper primary)"
            body.append(
                "<tr>"
                f"<td>{_escape(label)}</td>"
                f"<td>{paper_text}</td>"
                f"<td>{_comparison_metric_cell(experiment_value, unit, percentage)}</td>"
                f'<td class="metric-difference">{_escape(diff_text)}</td>'
                "</tr>"
            )
        parts.append(
            "<h3>Metric Comparison</h3><table><thead><tr><th>Metric</th><th>Published evidence</th>"
            "<th>Automated experiment</th><th>Difference / interpretation</th></tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table>"
        )
        for key, title in (
            ("satisfied_metrics", "Constraints satisfied"),
            ("not_satisfied_metrics", "Constraints not satisfied"),
            ("inconclusive_metrics", "Inconclusive"),
        ):
            names = _as_list(metric_comparison.get(key))
            if names:
                parts.append(f"<p><strong>{title}:</strong> {_escape(', '.join(map(str, names)))}</p>")
    else:
        parts.append("<p>No shared, comparable numerical metrics were available.</p>")
 
    # ---- explanation --------------------------------------------------------------
    for key, title in (
        ("confirmed_observations", "Confirmed observations"),
        ("possible_explanations", "Possible explanations"),
        ("limitations", "Limitations"),
        ("recommendation", "Recommendation"),
    ):
        items = _as_list(explanation.get(key))
        if items:
            parts.append(
                f"<h3>{title}</h3><ul>"
                + "".join(
                    f"<li>{_escape(_format_report_value(item))}</li>" for item in items
                )
                + "</ul>"
            )
 
    shown_errors = [e for e in errors if e and e != str(conclusion)]
    if shown_errors:
        parts.append("<h3>Errors</h3><ul>" + "".join(f"<li>{_escape(e)}</li>" for e in shown_errors) + "</ul>")
    parts.append("</section>")
    return "\n".join(parts)


def write_report(run: Dict[str, Any]) -> Path:
    get_reports_dir().mkdir(parents=True, exist_ok=True)
    report_path = get_reports_dir() / f"{Path(run['run_id']).name}.html"
    report_path.write_text(render_report(run), encoding="utf-8")
    return report_path


def _report_is_reusable(report_path: Path, run_path: Path) -> bool:
    """True when an existing report file can stand in for a fresh render."""
    try:
        report_stat = report_path.stat()
        run_stat = run_path.stat()
    except OSError:
        return False

    # A run JSON is an immutable audit snapshot, so a report written after it is
    # still an accurate rendering of that run.
    if not report_stat.st_size or report_stat.st_mtime < run_stat.st_mtime:
        return False

    # Reject reports left over from an older report template.
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            head = handle.read(len(REPORT_TEMPLATE_MARKER) + 128)
    except OSError:
        return False
    return REPORT_TEMPLATE_MARKER in head


def ensure_report(run_id: str) -> Path:
    """Return the run's HTML report, rendering it only when necessary.

    Run History refreshes call this once per listed run. Re-rendering every past
    report on each refresh is wasted work, so an existing report that is newer
    than its (immutable) run JSON and carries the current template marker is
    reused as-is.
    """
    report_path = get_reports_dir() / f"{Path(run_id).name}.html"
    if _report_is_reusable(report_path, get_run_path(run_id)):
        return report_path
    return write_report(load_run(run_id))


def history_html(limit: int = 20) -> str:
    runs = list_runs(limit=limit)
    if not runs:
        return "<p>No saved runs yet.</p>"

    rows = []
    for run in runs:
        try:
            report_path = ensure_report(run["run_id"])
            report_link = report_file_url(report_path)
        except OSError:
            report_link = "#"
        rows.append(
            "<tr>"
            f"<td>{_escape(run.get('created_at'))}</td>"
            f"<td>{_escape(run.get('goal'))}</td>"
            f"<td>{_escape(run.get('iteration'))}</td>"
            f"<td><code>{_escape(run.get('run_id'))}</code></td>"
            f'<td><a href="{_escape(report_link)}" target="_blank">Open report</a></td>'
            "</tr>"
        )

    return (
        "<table><thead><tr><th>Created</th><th>Goal</th><th>Iteration</th><th>Run ID</th><th>Report</th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _settings_table(goal: Dict[str, Any]) -> str:
    fields = [
        "research_type",
        "llm_model",
        "num_hypotheses",
        "generation_temperature",
        "reflection_temperature",
        "elo_k_factor",
        "top_k_hypotheses",
    ]
    rows = "".join(f"<tr><th>{_escape(field)}</th><td>{_escape(goal.get(field))}</td></tr>" for field in fields)
    return f"<table><tbody>{rows}</tbody></table>"


def _get_selected_experiment_hypothesis(
    experiment_result: Dict[str, Any],
    final_hypotheses: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Return the hypothesis explicitly selected for the automated experiment.

    Prefer the hypothesis stored by ExperimentOrchestrator. Fall back to the
    highest-ranked final hypothesis for compatibility with older run data.
    """
    if isinstance(experiment_result, dict):
        preparation = experiment_result.get(
            "experiment_preparation",
            {},
        )

        if isinstance(preparation, dict):
            selected = preparation.get("selected_hypothesis")

            if isinstance(selected, dict) and selected:
                return selected

        selected = experiment_result.get("selected_hypothesis")

        if isinstance(selected, dict) and selected:
            return selected

    if final_hypotheses:
        return final_hypotheses[0]

    return None


def _final_hypotheses(steps: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranking_steps = []
    for index, step_name in enumerate(steps):
        match = re.fullmatch(r"ranking(?:_?(\d+)|_final)?", step_name)
        if not match:
            continue
        priority = float("inf") if step_name == "ranking_final" else int(match.group(1) or 0)
        ranking_steps.append((priority, index, step_name))
    for _, _, step_name in sorted(ranking_steps, reverse=True):
        hypotheses = steps.get(step_name, {}).get("hypotheses", [])
        if hypotheses:
            return sorted(hypotheses, key=lambda item: item.get("elo_score", 0), reverse=True)
    for step_data in steps.values():
        hypotheses = step_data.get("hypotheses", []) if isinstance(step_data, dict) else []
        if hypotheses:
            return hypotheses
    return []


# def _hypothesis_block(index: int, hypothesis: Dict[str, Any]) -> str:
#     comments = hypothesis.get("review_comments") or []
#     comments_html = "".join(f"<li>{_escape(comment)}</li>" for comment in comments)
#     return (
#         '<div class="hypothesis">'
#         f"<h3>{index}. {_escape(hypothesis.get('title'), 'Untitled')}</h3>"
#         f"<p><strong>ID:</strong> {_escape(hypothesis.get('id'))} | "
#         f"<strong>Elo:</strong> {_escape(hypothesis.get('elo_score'))}</p>"
#         f"<p>{_escape(hypothesis.get('text'))}</p>"
#         f"<p><strong>Novelty:</strong> {_escape(hypothesis.get('novelty_review'))} | "
#         f"<strong>Feasibility:</strong> {_escape(hypothesis.get('feasibility_review'))}</p>"
#         f"<ul>{comments_html}</ul>"
#         "</div>"
#     )
def _hypothesis_block(index: int, hypothesis: Dict[str, Any]) -> str:
    comments = hypothesis.get("review_comments") or []
    comments_html = "".join(f"<li>{_escape(comment)}</li>" for comment in comments)
    return (
        '<div class="hypothesis">'
        f"<h3>Rank #{index}</h3>"
        f"<p><strong>Title:</strong> {_escape(hypothesis.get('title'), 'Untitled')}</p>"
        f"<p><strong>ID:</strong> {_escape(hypothesis.get('id'))}</p>"
        f"<p><strong>Elo Score:</strong> {_escape(hypothesis.get('elo_score'))}</p>"
        f"<p><strong>Novelty:</strong> {_escape(hypothesis.get('novelty_review'))}</p>"
        f"<p><strong>Feasibility:</strong> {_escape(hypothesis.get('feasibility_review'))}</p>"
        f"<p>{_escape(hypothesis.get('text'))}</p>"
        "<strong>Reviewer Comments</strong>"
        f"<ul>{comments_html}</ul>"
        "</div>"
    )


def _escape(value: Any, default: str = "") -> str:
    if value is None:
        value = default
    return html.escape(str(value))
