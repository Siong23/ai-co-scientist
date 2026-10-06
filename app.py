import base64
import os
import re
import threading
import time
import html as html_lib
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import gradio as gr
from numpy.ma import count  # noqa: F401

from app.agents import SupervisorAgent
from app.config import config
from app.data.dataset_manager import DatasetManager
from app.experiments.experiment_comparator import (
    ExperimentComparator,
)
from app.experiments.experiment_orchestrator import (
    ExperimentOrchestrator,
)
from app.models import ContextMemory, ResearchGoal
from app.research_state import LocalJSONResearchStateStore, ResearchStateError
from app.research_trace import format_research_trace_html, merge_trace_event, normalize_trace_event
from app.run_store import (
    _escape,
    delete_run,
    get_gradio_allowed_paths,
    get_reports_dir,
    history_html,
    list_runs,
    load_run,
    report_file_url,
    save_run,
    write_report,
)
from app.runtime_logging import configure_runtime_logging
from app.tools.tavily_search import cycle_usage as web_search_usage
from app.tools.tavily_search import reset_cycle_usage as reset_web_search_usage
from app.utils import (
    classify_llm_error,
    execution_budget,
    execution_remaining_seconds,
    fetch_lmstudio_models,
    get_lmstudio_base_url,
    get_lmstudio_model,
    logger,
    redact_secrets,
)

# Global state for the Gradio app
global_context = ContextMemory()
supervisor = SupervisorAgent()
experiment_comparator = ExperimentComparator()
current_research_goal: Optional[ResearchGoal] = None
research_state_store = LocalJSONResearchStateStore(
    root_dir=(config.get("research_state", {}) or {}).get("directory") or None
)
available_models: List[str] = []
CONFIGURED_LLM_MODEL = get_lmstudio_model()
SAFE_FALLBACK_LLM_MODEL = CONFIGURED_LLM_MODEL or "-- Select Model --"
CYCLE_TIMEOUT_SECONDS = int(os.getenv("CO_SCIENTIST_CYCLE_TIMEOUT_SECONDS", "1800"))
EXPERIMENT_DATASET_NAME = os.getenv(
    "EXPERIMENT_DATASET_NAME",
    "5G-NIDD",
)
EXPERIMENT_DATASET_PATH = (
    os.getenv("EXPERIMENT_DATASET_PATH", "data/5g_nidd/5g_nidd.csv").strip()
    or None
)
EXPERIMENT_DEVICE = os.getenv(
    "EXPERIMENT_DEVICE",
    "cuda",
)
EXPERIMENT_TIMEOUT_SECONDS = int(
    os.getenv(
        "EXPERIMENT_TIMEOUT_SECONDS",
        "600",
    )
)
# Smallest slice of the remaining cycle budget worth handing to the experiment.
# Below this the app skips it and reports the finished hypotheses instead of
# spending the rest of the cycle on work the deadline would discard anyway.
EXPERIMENT_MIN_BUDGET_SECONDS = int(
    os.getenv(
        "EXPERIMENT_MIN_BUDGET_SECONDS",
        "300",
    )
)
CYCLE_PROGRESS_INTERVAL_SECONDS = 5
_cycle_run_lock = threading.Lock()

_PCT_METRICS = {
    "accuracy", "acc", "precision", "precision_weighted", "weighted_precision",
    "recall", "recall_weighted", "weighted_recall", "f1", "f1_score",
    "f1_weighted", "weighted_f1", "macro_f1", "micro_f1", "balanced_accuracy",
}
_PCT_UNITS = {"%", "percent", "percentage", "percentage_point", "percentage_points", "pp"}
 
 
def _fmt_experiment_metric(name: str, value: Any, unit: Any = None) -> str:
    """Two decimals for numbers (the Gradio panel convention); units appended."""
    if value is None:
        return "N/A"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    text = f"{value:.2f}"
    unit_text = str(unit).strip() if unit else ""
    return f"{text} {unit_text}" if unit_text else text
 
 
def _first_error_line(error: Any, limit: int = 300) -> str:
    text = str(error).strip()
    first = text.splitlines()[0] if text else ""
    return first[:limit]
 
 
def _model_rationale_html(experiment_result: Dict[str, Any]) -> str:
    """"Proposed Model and Rationale" from the code generator's `model_recommendation`.
 
    CodeGenerationAgent returns experiment_result["code_generation"]["model_recommendation"] with
    model_name, algorithm, architecture, approach_type, reason_for_selection and
    relationship_to_rank1_hypothesis, plus `assumptions` and `experiment_plan`.
    """
    generation = experiment_result.get("code_generation")
    generation = generation if isinstance(generation, dict) else {}
    recommendation = generation.get("model_recommendation")
    if isinstance(recommendation, str) and recommendation.strip():
        recommendation = {"model_name": recommendation}
    if not isinstance(recommendation, dict) or not recommendation:
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
        value = recommendation.get(key)
        used.add(key)
        if value not in (None, "", [], {}):
            rows.append((label, show(value)))
    for key, value in recommendation.items():  # any extra fields the model returned
        if key not in used and value not in (None, "", [], {}):
            rows.append((str(key).replace("_", " ").capitalize(), show(value)))
    if not rows:
        return ""
 
    body = "".join(
        f"<tr><th style='text-align:left;vertical-align:top'>{html_lib.escape(label)}</th>"
        f"<td>{html_lib.escape(value)}</td></tr>"
        for label, value in rows
    )
    assumptions = generation.get("assumptions")
    assumption_html = ""
    if isinstance(assumptions, list) and assumptions:
        assumption_html = (
            "<p><strong>Assumptions:</strong></p><ul>"
            + "".join(f"<li>{html_lib.escape(show(item))}</li>" for item in assumptions)
            + "</ul>"
        )
    return (
        "<h3>🧠 Proposed Model and Rationale</h3>"
        f"<table><tbody>{body}</tbody></table>{assumption_html}"
    )
 
 
_TWO_MODEL_HEADLINE = ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "false_alarm_rate")
 
 
def _recommended_model_label(experiment_result: Optional[Dict[str, Any]]) -> str:
    generation = (experiment_result or {}).get("code_generation") if isinstance(experiment_result, dict) else None
    recommendation = generation.get("model_recommendation") if isinstance(generation, dict) else None
    if isinstance(recommendation, str):
        return recommendation.strip()
    if not isinstance(recommendation, dict):
        return ""
    name = str(recommendation.get("model_name") or "").strip()
    architecture = recommendation.get("architecture")
    architecture = " ".join(str(architecture).split()) if architecture and not isinstance(architecture, (dict, list)) else ""
    if architecture and architecture.lower() not in name.lower():
        return f"{name} - {architecture[:200]}" if name else architecture[:200]
    return name
 
 
def _two_model_html(metrics: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> str:
    """Model 1 (existing; never saw the unseen attack) vs Model 2 (proposed; trained on all attacks).
 
    Reads the metric names required by the "TWO-MODEL PROTOCOL" prompt section: headline metrics
    (= proposed model), baseline_<metric> (= Model 1), per_class_recall_baseline /
    per_class_recall_proposed, and unseen_attack. Shows nothing when they are absent.
    """
    summary = summary if isinstance(summary, dict) else {}
 
    def num(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
 
    unseen = str(metrics.get("unseen_attack") or summary.get("unseen_attack") or "").strip()
    headline = []
    for name in _TWO_MODEL_HEADLINE:
        base, new = metrics.get(f"baseline_{name}"), metrics.get(name)
        if num(base) and num(new):
            headline.append(
                f"<tr><td>{html_lib.escape(name.replace('_', ' ').title())}</td>"
                f"<td>{base:.2f}</td><td>{new:.2f}</td></tr>"
            )
    per_old, per_new = metrics.get("per_class_recall_baseline"), metrics.get("per_class_recall_proposed")
    per_rows = []
    if isinstance(per_old, dict) and isinstance(per_new, dict):
        for cls in sorted(set(per_old) | set(per_new), key=lambda c: (str(c) != unseen, str(c))):
            a, b = per_old.get(cls), per_new.get(cls)
            tag = " <em>(unseen for Model 1)</em>" if str(cls) == unseen else ""
            per_rows.append(
                f"<tr><td>{html_lib.escape(str(cls))}{tag}</td>"
                f"<td>{f'{a:.2f}' if num(a) else 'N/A'}</td><td>{f'{b:.2f}' if num(b) else 'N/A'}</td></tr>"
            )
    if not headline and not per_rows:
        return ""
    head = (
        "<thead><tr><th>{}</th><th>Model 1: existing (not trained on the unseen attack)</th>"
        "<th>Model 2: proposed (all attacks)</th></tr></thead>"
    )
    out = "<h3>Model 1 vs Model 2</h3>"
    if unseen:
        out += f"<p><strong>Unseen attack for Model 1:</strong> {html_lib.escape(unseen)}</p>"
    if headline:
        out += "<table>" + head.format("Overall metric (same test split)") + f"<tbody>{''.join(headline)}</tbody></table>"
    if per_rows:
        out += "<table>" + head.format("Recall per attack type") + f"<tbody>{''.join(per_rows)}</tbody></table>"
    return out

def fetch_available_models():
    """Fetch selectable models from the local LM Studio server."""
    global available_models

    discovered_models = fetch_lmstudio_models()
    available_models = discovered_models or ([CONFIGURED_LLM_MODEL] if CONFIGURED_LLM_MODEL else [])
    logger.info("LM Studio exposed %d selectable models.", len(discovered_models))
    return available_models


def get_default_model_choice(models: Optional[List[str]] = None) -> str:
    """Prefer the configured local model when available."""
    model_choices = models or available_models
    if CONFIGURED_LLM_MODEL and (not model_choices or CONFIGURED_LLM_MODEL in model_choices):
        return CONFIGURED_LLM_MODEL
    if model_choices:
        return model_choices[0]
    return CONFIGURED_LLM_MODEL or SAFE_FALLBACK_LLM_MODEL


def get_model_dropdown_choices(models: Optional[List[str]] = None) -> List[str]:
    """Return local model choices with the default first and de-duplicated."""
    model_choices = models or available_models
    choices = [get_default_model_choice(model_choices)]
    for model in model_choices:
        if model and model not in choices:
            choices.append(model)
    return choices


def get_deployment_status():
    """Get local LM Studio connection status information."""
    status = f"💻 Local LM Studio | {len(available_models)} model(s) available"
    return status, "blue"


def history_run_choices() -> List[Tuple[str, str]]:
    """Return dropdown choices for deleting saved runs."""
    choices = []
    for run in list_runs(limit=None):
        goal = run.get("goal") or "Untitled goal"
        if len(goal) > 80:
            goal = f"{goal[:77]}..."
        label = f"{run.get('created_at') or 'Unknown date'} — {goal} ({run.get('run_id')})"
        choices.append((label, run.get("run_id")))
    return choices


def sidebar_run_choices(limit: int = 50) -> List[Tuple[str, str]]:
    """Return compact, newest-first choices for the research history sidebar."""
    choices = []
    for run in list_runs(limit=limit):
        goal = run.get("goal") or "Untitled research goal"
        if len(goal) > 58:
            goal = f"{goal[:55]}..."
        created_at = str(run.get("created_at") or "")[:16].replace("T", " ")
        label = f"{goal}  ·  {created_at}" if created_at else goal
        choices.append((label, run.get("run_id")))
    return choices


def refresh_history_view() -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
    """Refresh the saved-run table, delete dropdown, and sidebar list."""
    return (
        history_html(),
        gr.update(choices=history_run_choices(), value=None),
        gr.update(choices=sidebar_run_choices(), value=None),
        "",
    )


def delete_history_run(selected_run_id: Optional[str]) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Delete the selected saved run and refresh the history display."""
    if not selected_run_id:
        return (
            "Select a saved run to delete.",
            history_html(),
            gr.update(choices=history_run_choices(), value=None),
            gr.update(choices=sidebar_run_choices(), value=None),
        )

    deleted = delete_run(selected_run_id)
    message = f"Deleted saved run {selected_run_id}." if deleted else f"Saved run {selected_run_id} was not found."
    return (
        message,
        history_html(),
        gr.update(choices=history_run_choices(), value=None),
        gr.update(choices=sidebar_run_choices(), value=None),
    )


def load_history_run(selected_run_id: Optional[str]) -> Tuple[Any, ...]:
    """Load a saved run into the main view without making an LLM call."""
    global current_research_goal, global_context

    if not selected_run_id:
        return (gr.skip(),) * 12

    try:
        run = load_run(selected_run_id)
    except (OSError, ValueError):
        skipped = [gr.skip() for _ in range(12)]
        skipped[1] = f"Saved run {selected_run_id} could not be loaded. Refresh the history and try again."
        return tuple(skipped)

    goal_data = run.get("research_goal") or {}
    description = goal_data.get("description") or ""
    loaded_goal = ResearchGoal(
        description=description,
        preferences=goal_data.get("preferences")
        or "Novelty, feasibility, scientific validity, practical applicability, clarity, and potential impact.",
        idea_attributes=goal_data.get("idea_attributes")
        or "novelty, feasibility, correctness, utility, specificity, and originality",
        constraints=goal_data.get("constraints") or {},
        llm_model=goal_data.get("llm_model"),
        query_rewrite_model=goal_data.get("query_rewrite_model"),
        num_hypotheses=goal_data.get("num_hypotheses"),
        generation_temperature=goal_data.get("generation_temperature"),
        reflection_temperature=goal_data.get("reflection_temperature"),
        elo_k_factor=goal_data.get("elo_k_factor"),
        top_k_hypotheses=goal_data.get("top_k_hypotheses"),
        research_type=goal_data.get("research_type") or goal_data.get("resolved_research_type") or "auto",
        research_id=goal_data.get("research_id"),
    )
    resumed = False
    resume_warning = ""
    research_id = str(goal_data.get("research_id") or "").strip()
    state_enabled = bool((config.get("research_state", {}) or {}).get("enabled", True))
    if state_enabled and research_id:
        try:
            if research_state_store.exists(research_id):
                resumed_session = research_state_store.load(research_id)
                loaded_goal = resumed_session.research_goal
                current_research_goal = loaded_goal
                global_context = resumed_session.context
                resumed = True
        except (ResearchStateError, OSError, ValueError) as exc:
            resume_warning = (
                f"\n\nA research-state checkpoint exists but could not be resumed safely: {redact_secrets(str(exc))}"
            )
    model_choices = list(available_models)
    if loaded_goal.llm_model and loaded_goal.llm_model not in model_choices:
        model_choices.append(loaded_goal.llm_model)

    stored_status = run.get("status") or "No status was recorded for this run."
    load_action = "Resumed current research session associated with" if resumed else "Loaded saved run"
    status = f"{load_action} {run.get('run_id', selected_run_id)}.\n\n{stored_status}{resume_warning}"
    cycle_details = run.get("cycle_details") or {}
    trace = cycle_details.get("research_trace") or []
    return (
        loaded_goal.description,
        status,
        format_research_trace_html(trace, elapsed_seconds=cycle_details.get("execution_time")),
        run.get("results_html") or "<p>No results were recorded for this run.</p>",
        run.get("references_html") or "<p>No references were recorded for this run.</p>",
        gr.update(choices=get_model_dropdown_choices(model_choices), value=loaded_goal.llm_model),
        loaded_goal.num_hypotheses,
        loaded_goal.generation_temperature,
        loaded_goal.reflection_temperature,
        loaded_goal.elo_k_factor,
        loaded_goal.top_k_hypotheses,
        gr.update(selected="current-run"),
    )


# Create a small helper function to turn plain text into bold text
def to_bold(text):
    # Mapping for a-z and A-Z to Mathematical Bold Capital/Small letters
    return "".join(
        chr(ord(c) + 119743) if "A" <= c <= "Z" else chr(ord(c) + 119737) if "a" <= c <= "z" else c for c in text
    )


def set_research_goal(
    description: str,
    llm_model: str = None,
    num_hypotheses: int = 3,
    generation_temperature: float = 0.7,
    reflection_temperature: float = 0.5,
    elo_k_factor: int = 32,
    top_k_hypotheses: int = 2,
) -> Tuple[str, str]:
    """Set the research goal and initialize the system."""
    global current_research_goal, global_context

    if not description.strip():
        return "❌ Error: Please enter a research goal.", ""

    try:
        normalized_description = description.strip()
        requested_model = (
            llm_model
            if llm_model and llm_model != "-- Select Model --"
            else config.get("llm_model", "google/gemini-flash-1.5")
        )
        same_session = bool(
            current_research_goal
            and global_context.research_id
            and global_context.research_id == current_research_goal.research_id
            and current_research_goal.description == normalized_description
            and current_research_goal.llm_model == requested_model
            and current_research_goal.num_hypotheses == num_hypotheses
            and current_research_goal.generation_temperature == generation_temperature
            and current_research_goal.reflection_temperature == reflection_temperature
            and current_research_goal.elo_k_factor == elo_k_factor
            and current_research_goal.top_k_hypotheses == top_k_hypotheses
        )
        if same_session:
            logger.info(
                "Continuing research session %s at cycle %d.",
                current_research_goal.research_id,
                global_context.iteration_number + 1,
            )
            status_msg = (
                "✅ Continuing existing research session!\n\n"
                f"{to_bold('Goal:')} {normalized_description}\n"
                f"{to_bold('Research ID:')} {current_research_goal.research_id}\n"
                f"{to_bold('Next cycle:')} {global_context.iteration_number + 1}"
            )
            return status_msg, "Existing research state retained."

        # Create research goal with settings
        current_research_goal = ResearchGoal(
            description=normalized_description,
            constraints={},
            llm_model=requested_model,
            num_hypotheses=num_hypotheses,
            generation_temperature=generation_temperature,
            reflection_temperature=reflection_temperature,
            elo_k_factor=elo_k_factor,
            top_k_hypotheses=top_k_hypotheses,
        )

        # Reset context
        global_context = ContextMemory(
            research_id=current_research_goal.research_id,
            research_type=current_research_goal.resolved_research_type or "hypothesis_testing",
        )

        logger.info(f"Research goal set: {description}")
        logger.info(f"Settings: model={current_research_goal.llm_model}, num={current_research_goal.num_hypotheses}")

        # status_msg = f"✅ Research goal set successfully!\n\n**Goal:** {description}\n**Model:** {current_research_goal.llm_model or 'Default'}\n**Hypotheses per cycle:** {num_hypotheses}"
        status_msg = f"✅ Research goal set successfully!\n\n{to_bold('Goal:')} {description}\n{to_bold('Model:')} {current_research_goal.llm_model or 'Default'}\n{to_bold('Hypotheses per cycle:')} {num_hypotheses}"

        return status_msg, "Ready to run first cycle. Click 'Run Cycle' to begin."

    except Exception as e:
        error_msg = f"❌ Error setting research goal: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""


def format_execution_time(seconds: float) -> str:
    """Format execution time as X min Y sec."""

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    if minutes > 0:
        return f"{minutes} mins {seconds} sec"

    return f"{seconds} sec"


def format_experiment_results_html(
    experiment_result: Dict[str, Any],
    comparison_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Format the automated experiment for the Gradio UI.
 
    Expected structure (ExperimentOrchestrator.run_experiment):
      experiment_result["experiment_preparation"]["selected_hypothesis"]["hypothesis_id"|"title"]
      experiment_result["experiment_preparation"]["experiment_id"]
      experiment_result["execution"]                      -> ExperimentRunner result
      experiment_result["execution"]["status"|"total_execution_seconds"|"output_validation"]
      experiment_result["execution"]["outputs"]["metrics"|"metric_definitions"|"summary"]
      experiment_result["comparison"]                     -> ExperimentComparator result
    """
    if not experiment_result:
        return ""
 
    status_code = str(experiment_result.get("status") or "")
    if status_code.startswith("skipped"):
        research_type = html_lib.escape(str(experiment_result.get("research_type") or "this research mode"))
        reason = html_lib.escape(
            str(experiment_result.get("reason") or "No hypothesis candidate was available to test.")
        )
        return (
            '<div style="margin-top: 20px; padding: 15px; border: 2px solid #17a2b8; border-radius: 8px;">'
            "<h2>🧪 Automated Experiment Skipped</h2>"
            f"<p><strong>Research type:</strong> {research_type}</p><p>{reason}</p></div>"
        )
 
    preparation = experiment_result.get("experiment_preparation")
    preparation = preparation if isinstance(preparation, dict) else {}
    selected = preparation.get("selected_hypothesis")
    selected = selected if isinstance(selected, dict) else {}
    runner = experiment_result.get("execution")
    runner = runner if isinstance(runner, dict) else {}
    outputs = runner.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    validation = runner.get("output_validation")
    validation = validation if isinstance(validation, dict) else {}
 
    experiment_id = html_lib.escape(str(preparation.get("experiment_id") or "Unknown"))
    hypothesis_title = html_lib.escape(str(selected.get("title") or "Untitled Hypothesis"))
    hypothesis_id = html_lib.escape(str(selected.get("hypothesis_id") or selected.get("id") or "Unknown"))
    header = (
        f"<p><strong>Experiment ID:</strong> {experiment_id}</p>"
        f"<p><strong>Selected Hypothesis:</strong> {hypothesis_title}</p>"
        f"<p><strong>Hypothesis ID:</strong> {hypothesis_id}</p>"
    )
 
    rationale_html = _model_rationale_html(experiment_result)
 
    # ---------------- failed experiment ----------------
    if not experiment_result.get("success", False):
        runner_status = str(runner.get("status") or "").replace("_", " ")
        errors = runner.get("errors") or experiment_result.get("errors") or []
        if not isinstance(errors, list):
            errors = [errors]
        seen, items = set(), []
        for error in errors:
            line = _first_error_line(error)
            if line and line not in seen:  # full stderr is kept in the run report
                seen.add(line)
                items.append(f"<li>{html_lib.escape(line)}</li>")
        status_line = (
            f"<p><strong>Status:</strong> {html_lib.escape(runner_status.title())}</p>" if runner_status else ""
        )
        return (
            '<div style="margin-top: 20px; padding: 15px; border: 2px solid #e74c3c; border-radius: 8px;">'
            "<h2>❌ Automated Experiment Failed</h2>"
            f"{header}{status_line}{rationale_html}"
            f"<ul>{''.join(items) or '<li>No detailed error was returned.</li>'}</ul>"
            "<p>The full generated code, logs and error output are in the saved run report.</p></div>"
        )
 
    # ---------------- successful experiment ----------------
    metrics = outputs.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    definitions = outputs.get("metric_definitions") or runner.get("metric_definitions") or {}
    definitions = definitions if isinstance(definitions, dict) else {}
 
    rows, structured = [], []
    for name, value in metrics.items():
        if name == "total_execution_seconds":
            continue  # runner timing, shown separately below
        if isinstance(value, (dict, list, tuple)):
            structured.append(str(name))
            continue
        definition = definitions.get(name)
        unit = definition.get("unit") if isinstance(definition, dict) else None
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(name).replace('_', ' ').title())}</td>"
            f"<td>{html_lib.escape(_fmt_experiment_metric(name, value, unit))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="2">No scalar experiment metrics were produced.</td></tr>')
    structured_note = (
        f"<p><em>Structured outputs (see run report): {html_lib.escape(', '.join(structured))}</em></p>"
        if structured
        else ""
    )
 
    two_model_html = _two_model_html(metrics, outputs.get("summary"))
 
    seconds = runner.get("total_execution_seconds")
    time_line = (
        f"<p><strong>Execution time:</strong> {float(seconds):.1f} s</p>"
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
        else ""
    )
    warnings = validation.get("warnings") or []
    warning_html = (
        "<p><strong>Output validation warnings:</strong></p><ul>"
        + "".join(f"<li>{html_lib.escape(_first_error_line(w))}</li>" for w in warnings)
        + "</ul>"
        if warnings
        else ""
    )
 
    comparison_html = (
        format_comparison_html(comparison_result, experiment_result) if comparison_result else ""
    )
 
    return (
        '<div style="margin-top: 20px; padding: 20px; border: 2px solid #28a745; border-radius: 8px;">'
        "<h2>🧪 Automated Experiment Results</h2>"
        f"{header}{time_line}<hr>{rationale_html}<h3>📊 Evaluation Metrics</h3>"
        "<table><thead><tr><th>Metric</th><th>Experiment result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{two_model_html}{structured_note}{warning_html}"
        "<p>Generated code, logs, checkpoints, metrics, training history and visualizations "
        "are saved in the run history report.</p></div>"
        f"{comparison_html}"
    )


def _cmp_is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
 
 
def _text_list(*values: Any) -> List[str]:
    """Flatten strings / lists / dicts of names into a de-duplicated list of strings."""
    out: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            if isinstance(item, dict):
                item = item.get("name") or item.get("model") or item.get("title") or ""
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
    return out


def _model_comparison_html(comparison_result: Dict[str, Any], experiment_result: Optional[Dict[str, Any]]) -> str:
    """Side-by-side: model proposed in the paper evidence vs. our Rank #1 hypothesis.
 
    NOTE: PaperReader stores `models_or_systems` / `datasets_or_testbeds` while
    ExperimentComparator reads `models` / `datasets`, so comparability.paper_model is
    usually empty. This reads the paper's own keys directly (with fallbacks).
    """
    def d(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}
 
    def short(text: str, limit: int = 500) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: limit - 3] + "..."
 
    comparability = d(comparison_result.get("comparability"))
    paper_result = d(comparison_result.get("paper_result"))
    reference = d(comparison_result.get("reference_experiment"))
 
    paper_models, paper_baselines, paper_datasets, paper_objectives, paper_urls = [], [], [], [], []
    for source in reference.get("sources") or []:
        source = d(source)
        details = d(source.get("experiment_details"))
        paper_models += _text_list(details.get("models_or_systems"), details.get("models"), source.get("models"))
        paper_baselines += _text_list(details.get("baselines"))
        paper_datasets += _text_list(details.get("datasets_or_testbeds"), details.get("datasets"), source.get("datasets"))
        paper_objectives += _text_list(details.get("experiment_objective"), details.get("objective"))
        paper_urls += _text_list(source.get("source_url"))
    paper_models = _text_list(paper_models, paper_result.get("models"), comparability.get("paper_model"))
    paper_datasets = _text_list(paper_datasets, paper_result.get("datasets"), comparability.get("paper_dataset"))
 
    ours = d(experiment_result)
    preparation = d(ours.get("experiment_preparation"))
    hypothesis = d(preparation.get("selected_hypothesis"))
    runner = d(ours.get("execution"))
    outputs = d(runner.get("outputs"))
    summary = d(outputs.get("summary"))
    specification = d(preparation.get("experiment_specification"))
    dataset_spec = specification.get("dataset")
    spec_dataset = dataset_spec.get("name") if isinstance(dataset_spec, dict) else dataset_spec
    recommendation = d(d(ours.get("code_generation")).get("model_recommendation"))
    our_model = _text_list(
        _recommended_model_label(ours), summary.get("model"), summary.get("model_name"), summary.get("model_algorithm"),
        summary.get("model/algorithm"), summary.get("algorithm"), comparability.get("experiment_model"),
    )
    our_dataset = _text_list(summary.get("dataset"), spec_dataset, comparability.get("experiment_dataset"))
    our_title = hypothesis.get("title") or comparison_result.get("hypothesis_title")
    our_id = hypothesis.get("hypothesis_id") or hypothesis.get("id") or comparison_result.get("hypothesis_id")
    our_text = hypothesis.get("text")
 
    if not any([paper_models, paper_baselines, paper_objectives, our_model, our_title, our_text]):
        return ""
 
    def cell(items: List[str], empty: str = "Not reported") -> str:
        return html_lib.escape(", ".join(items)) if items else f"<em>{html_lib.escape(empty)}</em>"
 
    our_approach = ""
    if our_title:
        our_approach = f"<strong>{html_lib.escape(str(our_title))}</strong>"
        if our_id:
            our_approach += f" <span style='opacity:.7'>(ID: {html_lib.escape(str(our_id))})</span>"
    if our_text:
        our_approach += ("<br>" if our_approach else "") + html_lib.escape(short(our_text))
    paper_links = "<br>".join(
        f'<a href="{html_lib.escape(u)}" target="_blank">{html_lib.escape(short(u, 80))}</a>' for u in paper_urls[:3]
    )
    rows = [
        ("Proposed model / system", cell(paper_models), cell(our_model, "Not reported by the experiment")),
        ("Approach / objective", cell([short(o) for o in paper_objectives[:2]]), our_approach or "<em>Not available</em>"),
        ("Baselines", cell(paper_baselines[:8]), "<em>Not applicable</em>"),
        ("Dataset / testbed", cell(paper_datasets), cell(our_dataset)),
        ("Source", paper_links or "<em>Not available</em>", "Rank #1 hypothesis"),
    ]
    body = "".join(f"<tr><th>{html_lib.escape(label)}</th><td>{a}</td><td>{b}</td></tr>" for label, a, b in rows)
    return (
        "<h3>Proposed Models</h3><table><thead><tr><th></th><th>Published (paper evidence)</th>"
        "<th>Automated experiment (Rank #1 hypothesis)</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def format_comparison_html(
    comparison_result: Dict[str, Any],
    full_experiment_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Paper vs automated experiment panel for the Gradio UI.
 
    `full_experiment_result` (optional; the orchestrator's whole experiment result) supplies our Rank #1 hypothesis and the model the
    experiment actually ran, for the "Proposed Models" table.
 
    Only metrics that were actually compared are shown in the main table.
    Metrics that exist on one side only are collapsed into <details> blocks
    (the complete list is always in the saved run report).
    """
    if not comparison_result or not isinstance(comparison_result, dict):
        return ""
 
    def as_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}
 
    # compare() initialises these to None and returns early, so never .get() on them blindly.
    paper_result = as_dict(comparison_result.get("paper_result"))
    experiment_result = as_dict(comparison_result.get("experiment_result"))
    comparability = as_dict(comparison_result.get("comparability"))
    explanation = as_dict(comparison_result.get("explanation"))
    errors = comparison_result.get("errors") or []
    if not isinstance(errors, list):
        errors = [str(errors)]
 
    conclusion = (
        comparison_result.get("conclusion")
        or explanation.get("overall_assessment")
        or comparability.get("reason")
        or "; ".join(str(error) for error in errors)
        or "No comparison conclusion was generated."
    )
    status_label = str(comparison_result.get("status") or "unknown").replace("_", " ").title()
 
    paper_metrics = as_dict(paper_result.get("metrics"))
    experiment_metrics = as_dict(experiment_result.get("metrics"))
    metric_comparison = as_dict(comparison_result.get("metric_comparison"))
    comparison_metrics = as_dict(metric_comparison.get("metrics"))
 
    pct_units = {"%", "percent", "percentage", "percentage_point", "percentage_points"}
 
    def format_metric_value(value: Any, unit: Any = None, percentage: bool = False) -> str:
        if not _cmp_is_number(value):
            return "Not available" if value is None else html_lib.escape(str(value))
        number = float(value)
        if percentage:
            if abs(number) <= 1:
                number *= 100
            return f"{number:.2f}%"
        formatted = f"{number:.4g}"
        return f"{formatted} {unit}" if unit else formatted
 
    def format_reference(value: Any, record: Dict[str, Any], unit: Any) -> str:
        value_type = str(record.get("reference_value_type", "measured_value")).lower()
        relation = str(record.get("reference_relation", "exact")).lower()
        formatted = format_metric_value(value, unit, unit in pct_units)
        if value_type == "upper_bound":
            prefix = {"less_than": "< ", "less_than_or_equal": "<= "}.get(relation, "<= ")
            return html_lib.escape(prefix) + formatted
        if value_type == "lower_bound":
            prefix = {"greater_than": "> ", "greater_than_or_equal": ">= "}.get(relation, ">= ")
            return html_lib.escape(prefix) + formatted
        return formatted
 
    model_section = _model_comparison_html(comparison_result, full_experiment_result)
 
    # Which metrics go in the main table?
    if comparison_metrics:
        compared_names = sorted(comparison_metrics)  # the comparator already paired these
    else:
        compared_names = sorted(
            name
            for name in set(paper_metrics) & set(experiment_metrics)
            if paper_metrics.get(name) is not None and experiment_metrics.get(name) is not None
        )
 
    comparison_rows = []
    for name in compared_names:
        record = as_dict(comparison_metrics.get(name))
        unit = record.get("reference_unit") or record.get("unit")
        percentage = record.get("difference_percentage_points") is not None or unit in pct_units
        paper_value = record.get("paper", paper_metrics.get(name))
        experiment_value = record.get("experiment", experiment_metrics.get(name))
        paper_text = format_reference(paper_value, record, unit)
        experiment_text = format_metric_value(experiment_value, unit, percentage)
        difference_pp = record.get("difference_percentage_points")
        difference = record.get("difference")
        value_type = str(record.get("reference_value_type", "measured_value")).lower()
        if difference_pp is not None:
            difference_text = f"{float(difference_pp):+.2f} pp"
        elif value_type != "measured_value":
            difference_text = record.get("comparison_interpretation") or "Compared with reported constraint"
        elif _cmp_is_number(difference):
            difference_text = format_metric_value(difference, unit, percentage)
            if percentage:
                difference_text = f"{float(difference) * 100:+.2f} pp"
            elif float(difference) > 0:
                difference_text = f"+{difference_text}"
        else:
            difference_text = "Not comparable"
        comparison_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(name).replace('_', ' ').title())}</td>"
            f"<td>{paper_text}</td>"
            f"<td>{experiment_text}</td>"
            f"<td>{html_lib.escape(str(difference_text))}</td>"
            "</tr>"
        )
 
    if comparison_rows:
        metric_section = (
            "<h3>Metric Comparison</h3><table><thead><tr><th>Metric</th><th>Published evidence</th>"
            "<th>Automated experiment</th><th>Difference / interpretation</th></tr></thead>"
            f"<tbody>{''.join(comparison_rows)}</tbody></table>"
        )
    else:
        metric_section = (
            "<h3>Metric Comparison</h3><p>No shared, comparable numerical metrics were available.</p>"
        )
 
    # Metrics present on only one side: collapsed, never in the main table.
    shown = set(compared_names)
 
    def collapsed_metrics(title: str, metrics: Dict[str, Any]) -> str:
        items = [(name, value) for name, value in sorted(metrics.items()) if name not in shown]
        if not items:
            return ""
        body = "".join(
            f"<tr><td>{html_lib.escape(str(name).replace('_', ' ').title())}</td>"
            f"<td>{html_lib.escape(str(value) if not _cmp_is_number(value) else format(value, '.4g'))}</td></tr>"
            for name, value in items
        )
        return (
            f"<details><summary>{html_lib.escape(title)} ({len(items)})</summary>"
            f"<table><tbody>{body}</tbody></table></details>"
        )
 
    scalar_paper = {k: v for k, v in paper_metrics.items() if not isinstance(v, (dict, list, tuple))}
    scalar_experiment = {k: v for k, v in experiment_metrics.items() if not isinstance(v, (dict, list, tuple))}
    collapsed = collapsed_metrics("Published metrics with no matching experiment metric", scalar_paper)
    collapsed += collapsed_metrics("Experiment metrics with no matching published metric", scalar_experiment)
 
    warnings = comparability.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = [warnings]
    warning_html = (
        "<ul>" + "".join(f"<li>{html_lib.escape(str(w))}</li>" for w in warnings) + "</ul>" if warnings else ""
    )
 
    return f"""
<div style="margin-top: 20px; padding: 20px; border: 2px solid #6f42c1; border-radius: 8px;">
<h2>Paper vs Automated Experiment</h2>
<p><strong>Status:</strong> {html_lib.escape(status_label)}</p>
{model_section}
{metric_section}
{collapsed}
<h3>Comparison Conclusion</h3>
<p>{html_lib.escape(str(conclusion))}</p>
{warning_html}
<p>The complete metric lists are saved in the run history report.</p>
</div>
"""


def execute_cycle(
    research_goal: ResearchGoal,
    context: ContextMemory,
    cycle_supervisor: SupervisorAgent,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    run_experiment: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run a cycle against the supplied state and return display-ready results.

    ``run_experiment`` carries the experiment checkbox above Run Cycle; ``None``
    falls back to ``experiment_auto_run`` in config.yaml.
    """
    import datetime

    research_trace: List[Dict[str, Any]] = []

    def capture_progress(event: Dict[str, Any]) -> None:
        normalized = normalize_trace_event(event)
        merge_trace_event(research_trace, normalized)
        if progress_callback is not None:
            try:
                progress_callback(dict(normalized))
            except Exception as exc:
                logger.warning("Research progress callback failed: %s", redact_secrets(str(exc)))

    # Prepare log file
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"app_log_{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"LOGGING FOR THIS GOAL: {research_goal.description}\n")
        f.write("--- Endpoint /run_cycle START ---\n")

    # Set inside the try block; the error path below keeps whatever the agent
    # workflow already finished.
    cycle_details: Optional[Dict[str, Any]] = None
    results_html = ""

    try:
        iteration = context.iteration_number + 1

        # Start timing the cycle execution
        start_time = time.perf_counter()

        # Tavily bills per request, so each cycle accounts for its own spend.
        reset_web_search_usage()

        print("\n" + "=" * 60)
        print("AI CO-SCIENTIST AUTOMATED PIPELINE")
        print("=" * 60)

        print("\n[1/2] Running AI Co-Scientist workflow...")

        logger.info(f"Running cycle {iteration}")

        # Run the cycle
        cycle_details = cycle_supervisor.run(
            research_goal,
            context,
            progress_callback=capture_progress,
        )
        cycle_details.setdefault("research_trace", research_trace)

        # ================================================
        # Automated Experiment Pipeline
        # ================================================

        hypothesis_pipeline_enabled = context.uses_hypothesis_pipeline()

        # The experiment shares one deadline with the agent workflow, so give it
        # only the time that workflow left over. Without this the run keeps
        # training past the cycle limit and the app discards the whole cycle
        # moments before the results land.
        remaining_budget_seconds = execution_remaining_seconds()
        experiment_timeout_seconds = EXPERIMENT_TIMEOUT_SECONDS
        if remaining_budget_seconds is not None:
            experiment_timeout_seconds = int(min(EXPERIMENT_TIMEOUT_SECONDS, remaining_budget_seconds))
        experiment_has_budget = experiment_timeout_seconds >= EXPERIMENT_MIN_BUDGET_SECONDS
        # With the experiment off a cycle ends after the agent workflow, so its
        # hypotheses appear without waiting for code generation and training.
        experiment_auto_run = (
            bool(config.get("experiment_auto_run", True)) if run_experiment is None else bool(run_experiment)
        )
        experiment_enabled = hypothesis_pipeline_enabled and experiment_auto_run and experiment_has_budget
        print("\n" + "=" * 60)
        print("AI CO-SCIENTIST WORKFLOW COMPLETED")
        print("STARTING AUTOMATED EXPERIMENT PIPELINE" if experiment_enabled else "AUTOMATED EXPERIMENT SKIPPED")
        print("=" * 60)

        if experiment_enabled:
            print("\n[2/2] Running automated deep-learning experiment...")
            logger.debug("Starting automated experiment pipeline.")
        elif hypothesis_pipeline_enabled and not experiment_auto_run:
            print("\n[2/2] Skipping the automated experiment: it is turned off for this cycle.")
            logger.info("Skipping the automated experiment because it is turned off for this cycle.")
        elif hypothesis_pipeline_enabled:
            print("\n[2/2] Skipping the automated experiment: not enough cycle budget left.")
            logger.warning(
                "Skipping the automated experiment: %s of the cycle budget left, minimum is %s.",
                format_timeout_duration(max(remaining_budget_seconds or 0.0, 0.0)),
                format_timeout_duration(EXPERIMENT_MIN_BUDGET_SECONDS),
            )
        else:
            print("\n[2/2] Skipping hypothesis-dependent automated experiment.")
            logger.info(
                "Skipping automated experiment for research type %s without hypotheses.",
                context.research_type,
            )

        capture_progress(
            {
                "step": "experiment",
                "status": "running" if experiment_enabled else "completed",
                "title": "Automated Experiment",
                "summary": (
                    "Selecting the best hypothesis and starting the PyTorch experiment."
                    if experiment_enabled
                    else "Skipped: the automated experiment is turned off for this cycle."
                    if hypothesis_pipeline_enabled and not experiment_auto_run
                    else (
                        f"Skipped: only {format_timeout_duration(max(remaining_budget_seconds or 0.0, 0.0))} "
                        f"of the cycle budget was left, and the experiment needs at least "
                        f"{format_timeout_duration(EXPERIMENT_MIN_BUDGET_SECONDS)}."
                        if hypothesis_pipeline_enabled
                        else f"Skipped for {context.research_type}: this research plan has no hypothesis candidate to test."
                    )
                ),
                "details": [],
            }
        )

        comparison_result = {
            "success": False,
            "status": "not_started",
            "conclusion": "Comparison was not started.",
            "errors": [],
        }

        if experiment_enabled:
            print(
                f"Dataset configuration: "
                f"{EXPERIMENT_DATASET_NAME}"
            )

            print(
                f"Dataset path override: "
                f"{EXPERIMENT_DATASET_PATH or '<DatasetManager default>'}"
            )

            print(
                f"Experiment device: "
                f"{EXPERIMENT_DEVICE}"
            )

            print(
                f"Experiment timeout: "
                f"{format_timeout_duration(experiment_timeout_seconds)}"
            )

            logger.info(
                "Starting automated experiment pipeline: "
                "dataset=%s, path=%s, device=%s, timeout=%s",
                EXPERIMENT_DATASET_NAME,
                EXPERIMENT_DATASET_PATH or "<DatasetManager default>",
                EXPERIMENT_DEVICE,
                format_timeout_duration(
                    experiment_timeout_seconds
                ),
            )

            experiment_orchestrator = ExperimentOrchestrator(
                dataset_name=EXPERIMENT_DATASET_NAME,
                dataset_path=EXPERIMENT_DATASET_PATH,
                device=EXPERIMENT_DEVICE,
            )

            experiment_result = (
                experiment_orchestrator.run_experiment(
                    context=context,
                    research_goal=research_goal,
                    execute_generated_code=True,
                    timeout_seconds=experiment_timeout_seconds,
                )
            )

            if not isinstance(
                experiment_result,
                dict,
            ):
                raise TypeError(
                    "ExperimentOrchestrator.run_experiment() "
                    "must return a dictionary."
                )

            # The ExperimentOrchestrator now owns the complete
            # paper-vs-experiment comparison.
            # Expected structure:
            # experiment_result["comparison"]
            # Do not call ExperimentComparator directly from app.py.

            comparison_result = (
                experiment_result.get(
                    "comparison"
                )
            )

            if not isinstance(
                comparison_result,
                dict,
            ):
                comparison_result = {
                    "success": False,
                    "status": "comparison_not_available",
                    "conclusion": (
                        "The experiment completed, "
                        "but no comparison result was returned "
                        "by ExperimentOrchestrator."
                    ),
                    "errors": [],
                }

                logger.warning(
                    "ExperimentOrchestrator returned no "
                    "comparison result."
                )
        elif hypothesis_pipeline_enabled and not experiment_auto_run:
            experiment_result = {
                "success": False,
                "status": "skipped_auto_run_disabled",
                "skipped": True,
                "research_type": context.research_type,
                "reason": (
                    "The automated experiment is turned off for this cycle "
                    '(the "Run automated experiment after the cycle" checkbox above Run Cycle). '
                    "The hypotheses above are complete."
                ),
                "errors": [],
            }
        elif hypothesis_pipeline_enabled:
            experiment_result = {
                "success": False,
                "status": "skipped_no_cycle_budget",
                "skipped": True,
                "research_type": context.research_type,
                "reason": (
                    f"Only {format_timeout_duration(max(remaining_budget_seconds or 0.0, 0.0))} of the cycle "
                    f"budget was left and the experiment needs at least "
                    f"{format_timeout_duration(EXPERIMENT_MIN_BUDGET_SECONDS)}. The hypotheses above are complete; "
                    "raise CO_SCIENTIST_CYCLE_TIMEOUT_SECONDS to run the experiment in the same cycle."
                ),
                "errors": [],
            }
        else:
            experiment_result = {
                "success": False,
                "status": "skipped_for_research_type",
                "skipped": True,
                "research_type": context.research_type,
                "reason": "No hypothesis candidate exists in this research mode.",
                "errors": [],
            }

        if not experiment_enabled:
            comparison_result = {
                "success": False,
                "status": experiment_result.get("status", "skipped_for_research_type"),
                "conclusion": (
                    "Paper comparison was skipped because the experiment did not run: "
                    f"{experiment_result.get('reason', '')}".strip()
                ),
                "errors": [],
            }

        cycle_details["experiment_result"] = experiment_result
        cycle_details["comparison_result"] = comparison_result

        if str(experiment_result.get("status", "")).startswith("skipped"):
            pass
        elif experiment_result.get(
            "success",
            False,
        ):
            print("\nAUTOMATED EXPERIMENT COMPLETED")

            capture_progress(
                {
                    "step": "experiment",
                    "status": "completed",
                    "title": "Automated Experiment",
                    "summary": ("PyTorch experiment completed successfully."),
                    "details": [],
                }
            )

        else:
            experiment_errors = experiment_result.get(
                "errors",
                [],
            )

            print("\nAUTOMATED EXPERIMENT FAILED")

            for error in experiment_errors:
                first_line = str(error).strip().splitlines()[0] if str(error).strip() else str(error)
                print(f"  - {first_line}")

            execution_logs = experiment_result.get("execution") or {}
            stdout_log = execution_logs.get("stdout_path")
            stderr_log = execution_logs.get("stderr_path")
            if stdout_log or stderr_log:
                print(f"  (full output: {stdout_log or '-'} , {stderr_log or '-'})")

            capture_progress(
                {
                    "step": "experiment",
                    "status": "error",
                    "title": "Automated Experiment",
                    "summary": ("The experiment pipeline completed with errors."),
                    "details": [str(error) for error in experiment_errors],
                }
            )

        print("\n" + "=" * 60)
        print("PAPER VS AUTOMATED EXPERIMENT COMPARISON")
        print("=" * 60)

        explanation = (
            comparison_result.get("explanation")
            or {}
        )

        if isinstance(
            explanation,
            dict,
        ):
            print(
                explanation.get(
                    "overall_assessment",
                    comparison_result.get(
                        "conclusion",
                        "No comparison conclusion was generated.",
                    ),
                )
            )
        else:
            print(
                comparison_result.get(
                    "conclusion",
                    "No comparison conclusion was generated.",
                )
            )

        print("=" * 60)

        print("\n" + "=" * 60)
        print("COMPLETE AUTOMATED PIPELINE FINISHED")
        print("=" * 60)

        # Log execution time
        total_time = time.perf_counter() - start_time
        formatted_time = format_execution_time(total_time)

        cycle_details["execution_time"] = total_time
        cycle_details["execution_time_formatted"] = formatted_time

        logger.info(f"Cycle execution time: {formatted_time}")

        web_usage = web_search_usage()
        cycle_details["web_search_usage"] = web_usage
        logger.info(
            "Tavily usage this cycle: %d search call(s), %d extract call(s), %d served from cache, %d skipped on budget.",
            web_usage.get("search_calls", 0),
            web_usage.get("extract_calls", 0),
            web_usage.get("search_cache_hits", 0) + web_usage.get("extract_cache_hits", 0),
            web_usage.get("budget_skips", 0),
        )

        # Log all steps and hypotheses
        steps = cycle_details.get("steps", {})
        with open(log_file, "a", encoding="utf-8") as f:
            for step_name, step_data in steps.items():
                hypos = step_data.get("hypotheses", [])
                f.write(f"Step: {step_name} | {len(hypos)} hypotheses\n")
                for h in hypos:
                    f.write(f"  - ID: {h.get('id')} | Title: {h.get('title')} | Elo: {h.get('elo_score', 'N/A')}\n")

        # Format results for display (also logs final rankings)
        results_html = format_cycle_results(cycle_details, log_file=log_file)

        experiment_result = cycle_details.get(
            "experiment_result",
            {},
        )

        experiment_results_html = format_experiment_results_html(experiment_result, comparison_result)

        results_html += experiment_results_html

        # Get references
        references_html = get_references_html(cycle_details, research_goal=research_goal)

        # Status message: surface the real cause when generation failed, instead
        # of reporting success over an empty result (issue llnl#36).
        errors = cycle_details.get("errors", [])
        produced_any = bool(cycle_details.get("steps", {}).get("generation", {}).get("hypotheses"))
        finalization = cycle_details.get("finalization", {})
        if errors:
            categories = sorted({classify_llm_error(e) for e in errors})
            cause = "; ".join(categories)
            if produced_any:
                status_msg = f"⚠️ Cycle {iteration} completed with errors ({cause}).\n\n{to_bold('Execution Time:')} {formatted_time}.\n{to_bold('Log:')} {log_file}"
            else:
                status_msg = (
                    f"⚠️ Cycle {iteration} could not generate hypotheses — {cause}.\n\n{to_bold('Execution Time:')} {formatted_time}.\n"
                    f"See the results panel for details. {to_bold('Log:')} {log_file}"
                )
        elif finalization and not finalization.get("ready", False):
            unmet = "; ".join(finalization.get("reasons", [])) or "final quality requirements were not met"
            finalization_status = str(finalization.get("status", ""))
            if finalization_status == "generation_budget_exhausted":
                headline = (
                    f"⚠️ Cycle {iteration} completed its bounded Generation/Evolution work "
                    "but did not pass the final quality gate"
                )
            else:
                headline = f"⚠️ Cycle {iteration} reached its compute budget before finalization"
            status_msg = (
                f"{headline} ({unmet}).\n\n{to_bold('Execution Time:')} {formatted_time}\n{to_bold('Log:')} {log_file}"
            )
        elif cycle_details.get("warnings"):
            status_msg = (
                f"⚠️ Cycle {iteration} completed with recovery warnings.\n\n"
                "Evidence retrieval or generation was degraded; see the results panel for details.\n"
                f"{to_bold('Execution Time:')} {formatted_time}\n"
                f"{to_bold('Log:')} {log_file}"
            )
        else:
            status_msg = (
                f"✅ Cycle {iteration} completed successfully!\n\n"
                f"{to_bold('Execution Time:')} {formatted_time}\n"
                f"{to_bold('Log:')} {log_file}"
            )

        return {
            "status": status_msg,
            "results_html": results_html,
            "references_html": references_html,
            "cycle_details": cycle_details,
            "log_file": log_file,
        }

    except Exception as e:
        error_msg = f"❌ Error during cycle execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        capture_progress(
            {
                "step": "cycle_error",
                "status": "error",
                "title": "Research cycle stopped",
                "summary": error_msg,
                "details": [],
            }
        )
        # A failure after the agent workflow finished, e.g. while building the
        # experiment report, must not discard the hypotheses it produced.
        finished_steps = (cycle_details or {}).get("steps") or {}
        if any(isinstance(step, dict) and step.get("hypotheses") for step in finished_steps.values()):
            try:
                if not results_html:
                    results_html = format_cycle_results(cycle_details, log_file=log_file)
                references_html = get_references_html(cycle_details, research_goal=research_goal)
            except Exception:
                logger.exception("The finished hypotheses could not be shown after the cycle error.")
            else:
                import html as html_lib

                reason = redact_secrets(str(e))
                cycle_details.setdefault("errors", []).append(error_msg)
                return {
                    "status": (
                        f"⚠️ Cycle {iteration} kept its hypotheses, but a later step failed: {reason}\n\n"
                        f"{to_bold('Log:')} {log_file}"
                    ),
                    "results_html": results_html
                    + f"""
                    <div style="margin-top: 20px; padding: 15px; border: 2px solid #e67e22; border-radius: 8px;">
                        <h3>⚠️ A later step failed</h3>
                        <p>The hypotheses above are complete. The cycle stopped after them: {html_lib.escape(reason)}</p>
                    </div>
                    """,
                    "references_html": references_html,
                    "cycle_details": cycle_details,
                    "log_file": log_file,
                }
        return {
            "status": error_msg,
            "results_html": "",
            "references_html": "",
            "cycle_details": {
                "iteration": context.iteration_number + 1,
                "steps": {},
                "errors": [error_msg],
                "research_trace": research_trace,
            },
            "log_file": log_file,
        }


def persist_cycle_result(
    research_goal: ResearchGoal,
    cycle_result: Dict[str, Any],
    context: Optional[ContextMemory] = None,
) -> Tuple[str, str, str]:
    """Persist an accepted cycle result and return Gradio output values."""
    state_warning = ""
    if context is not None and bool((config.get("research_state", {}) or {}).get("enabled", True)):
        context.research_id = research_goal.research_id
        try:
            research_state_store.save(research_goal, context)
        except (ResearchStateError, OSError, ValueError) as exc:
            state_warning = "\n⚠️ The run report was saved, but its resumable research checkpoint could not be updated."
            logger.warning(
                "Research-state checkpoint could not be saved: %s",
                redact_secrets(str(exc)),
            )
    saved_run = save_run(
        research_goal=research_goal,
        cycle_details=cycle_result["cycle_details"],
        status=cycle_result["status"],
        references_html=cycle_result["references_html"],
        results_html=cycle_result["results_html"],
        log_file=cycle_result["log_file"],
        experiment_result=(cycle_result["cycle_details"].get("experiment_result")),
        comparison_result=(cycle_result["cycle_details"].get("comparison_result")),
    )
    report_path = write_report(saved_run)
    status_msg = (
        f"{cycle_result['status']}{state_warning}\n"
        f"{to_bold('Run ID:')} {saved_run['run_id']}\n"
        f"{to_bold('Report:')} {report_file_url(report_path)}"
    )
    return status_msg, cycle_result["results_html"], cycle_result["references_html"]


def run_cycle() -> Tuple[str, str, str]:
    """Run a single research cycle with detailed step logging for debugging."""
    global current_research_goal, global_context, supervisor

    if not current_research_goal:
        return "❌ Error: No research goal set. Please set a research goal first.", "", ""

    return persist_cycle_result(
        current_research_goal,
        execute_cycle(current_research_goal, global_context, supervisor),
        global_context,
    )


def format_timeout_duration(timeout_seconds: float) -> str:
    if timeout_seconds < 60:
        return f"{timeout_seconds:.2f} sec"
    minutes = timeout_seconds / 60
    if minutes.is_integer():
        return f"{int(minutes)} mins"
    return f"{minutes:.2f} mins"


def timeout_results_html(timeout_seconds: float) -> str:
    timeout_duration = format_timeout_duration(timeout_seconds)
    return f"""
    <div style="margin: 20px 0; padding: 15px; border: 2px solid #e67e22; border-radius: 8px; background-color: #fff8ee;">
        <h3>Cycle stopped at the time limit</h3>
        <p>The run exceeded the {timeout_duration} upper limit before the app received a completed cycle.</p>
        <p>Try fewer hypotheses, a different model, or a later retry if the model provider is slow.</p>
    </div>
    """


def format_evidence_sources_html(
    hypothesis: Dict,
    generation_sources: List[Dict],
) -> str:
    """Render validated evidence as useful links to the original source."""
    import html as html_lib

    available_sources = {}
    for source in generation_sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id") or f"arXiv:{source.get('arxiv_id')}")
        if source_id:
            available_sources[source_id] = source
    evidence_source_ids = hypothesis.get("evidence_source_ids", [])
    if not isinstance(evidence_source_ids, list):
        evidence_source_ids = []

    links = []
    for source_id in dict.fromkeys(evidence_source_ids):
        if not isinstance(source_id, str) or source_id not in available_sources:
            continue
        source = available_sources[source_id]
        href = str(source.get("url") or source.get("arxiv_url") or source.get("pdf_url") or "").strip()
        if not href and source_id.startswith("arXiv:"):
            arxiv_id = source_id.removeprefix("arXiv:")
            href = f"https://arxiv.org/abs/{quote(arxiv_id, safe='/.-')}"
        if not href.startswith(("https://", "http://")):
            continue
        label = str(source.get("title") or source_id)
        links.append(
            f'<a href="{html_lib.escape(href, quote=True)}" '
            'target="_blank" rel="noopener noreferrer">'
            f"{html_lib.escape(label)}</a>"
        )

    rendered_sources = ", ".join(links) if links else "None recorded"
    evidence_refs = hypothesis.get("evidence_refs", [])
    if not isinstance(evidence_refs, list):
        evidence_refs = []
    rendered_refs = ", ".join(
        f"<code>{html_lib.escape(str(chunk_id))}</code>"
        for chunk_id in dict.fromkeys(evidence_refs)
        if isinstance(chunk_id, str) and chunk_id
    )
    provenance = f"<p><strong>Evidence chunks:</strong> {rendered_refs}</p>" if rendered_refs else ""
    return f"<p><strong>Evidence Sources:</strong> {rendered_sources}</p>{provenance}"


def format_ranking_confidence(value: Any) -> str:
    """Render current 1-10 and legacy 0-1 ranking confidence values."""
    if isinstance(value, bool):
        return "Not available"

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "Not available"

    if isinstance(value, int) and 1 <= value <= 10:
        score = float(value)
    elif 0 <= confidence <= 1:
        score = confidence * 10
    elif 1 <= confidence <= 10:
        score = confidence
    else:
        return "Not available"

    return f"{score:g}/10 ({score * 10:.0f}%)"


def _ordered_ranking_step_names(steps: Dict[str, Any]) -> List[str]:
    """Return fixed and dynamically numbered ranking steps newest first."""

    ranked = []
    for index, step_name in enumerate(steps):
        match = re.fullmatch(r"ranking(?:_?(\d+)|_final)?", step_name)
        if not match:
            continue
        priority = float("inf") if step_name == "ranking_final" else int(match.group(1) or 0)
        ranked.append((priority, index, step_name))
    return [step_name for _, _, step_name in sorted(ranked, reverse=True)]


def run_cycle_with_progress(
    timeout_seconds: int = CYCLE_TIMEOUT_SECONDS,
    poll_seconds: float = CYCLE_PROGRESS_INTERVAL_SECONDS,
    run_experiment: Optional[bool] = None,
):
    """Run a cycle in the background and stream its research-process trace."""
    global global_context

    if not current_research_goal:
        yield (
            "❌ Error: No research goal set. Please set a research goal first.",
            "",
            "",
            format_research_trace_html([]),
        )
        return

    if not _cycle_run_lock.acquire(blocking=False):
        busy_status = "⚠️ A previous cycle is still stopping. Wait for it to release the model before starting again."
        yield (
            busy_status,
            "<p>A previous cycle is still stopping.</p>",
            "",
            format_research_trace_html([]),
        )
        return

    run_goal = current_research_goal
    run_context = deepcopy(global_context)
    run_supervisor = SupervisorAgent()
    result: Dict[str, Dict[str, Any]] = {}
    progress_events: Queue[Dict[str, Any]] = Queue()
    live_trace: List[Dict[str, Any]] = []

    def drain_progress_events() -> None:
        while True:
            try:
                event = progress_events.get_nowait()
            except Empty:
                return
            merge_trace_event(live_trace, event)

    started = time.monotonic()
    deadline = started + timeout_seconds
    cancel_event = threading.Event()
    deadline_timer = threading.Timer(timeout_seconds, cancel_event.set)
    deadline_timer.daemon = True

    def worker():
        try:
            with execution_budget(deadline, cancel_event):
                result["value"] = execute_cycle(
                    run_goal,
                    run_context,
                    run_supervisor,
                    progress_callback=progress_events.put,
                    run_experiment=run_experiment,
                )
        finally:
            deadline_timer.cancel()
            _cycle_run_lock.release()

    thread = threading.Thread(target=worker, daemon=True)
    deadline_timer.start()
    thread.start()
    iteration = global_context.iteration_number + 1

    def timeout_update(elapsed: float):
        timeout_duration = format_timeout_duration(timeout_seconds)
        timeout_status = (
            f"⚠️ Cycle {iteration} timed out after {timeout_duration}. "
            "The app cancelled remaining agent work before accepting another run."
        )
        timeout_html = timeout_results_html(timeout_seconds)
        merge_trace_event(
            live_trace,
            {
                "step": "timeout",
                "status": "error",
                "title": "Cycle time limit reached",
                "summary": timeout_status,
                "details": [],
                "elapsed_seconds": elapsed,
            },
        )
        saved_run = save_run(
            research_goal=run_goal,
            cycle_details={
                "iteration": iteration,
                "steps": {},
                "errors": [timeout_status],
                "research_trace": live_trace,
            },
            status=timeout_status,
            references_html="",
            results_html=timeout_html,
            log_file="",
        )
        report_path = write_report(saved_run)
        return (
            f"{timeout_status}\n{to_bold('Run ID:')} {saved_run['run_id']}\n{to_bold('Report:')} {report_file_url(report_path)}",
            timeout_html,
            "",
            format_research_trace_html(live_trace, elapsed_seconds=elapsed),
        )

    while thread.is_alive():
        drain_progress_events()
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            cancel_event.set()
            yield timeout_update(elapsed)
            return

        active_event = next(
            (event for event in reversed(live_trace) if event.get("status") == "running"),
            live_trace[-1] if live_trace else None,
        )
        active_title = (
            active_event.get("title") if active_event else "Generating, reviewing, ranking, and evolving hypotheses"
        )
        latest_summary = active_event.get("summary") if active_event else "The agent workflow is starting."
        status = (
            f"⏳ Cycle {iteration} is running.\n"
            f"Elapsed: {format_timeout_duration(elapsed)}.\n"
            f"Active work: {active_title}.\n"
            f"Latest update: {latest_summary}\n"
            f"Upper limit: {format_timeout_duration(timeout_seconds)}."
        )
        yield (
            status,
            "<p>Cycle is still running. Results will appear when the cycle completes.</p>",
            "",
            format_research_trace_html(live_trace, running=True, elapsed_seconds=elapsed),
        )
        thread.join(timeout=min(poll_seconds, max(timeout_seconds - elapsed, 0.1)))

    drain_progress_events()
    elapsed = time.monotonic() - started
    if cancel_event.is_set() and elapsed >= timeout_seconds:
        yield timeout_update(elapsed)
        return

    cycle_result = result.get("value")
    if not cycle_result:
        merge_trace_event(
            live_trace,
            {
                "step": "cycle_error",
                "status": "error",
                "title": "Cycle ended without a result",
                "summary": "The background worker stopped without returning cycle data.",
                "details": [],
            },
        )
        yield (
            "❌ Error: Cycle ended without a result.",
            "",
            "",
            format_research_trace_html(live_trace, elapsed_seconds=time.monotonic() - started),
        )
        return
    final_trace = list(live_trace)
    for event in cycle_result.get("cycle_details", {}).get("research_trace", []):
        merge_trace_event(final_trace, event)
    cycle_result.setdefault("cycle_details", {})["research_trace"] = final_trace
    if current_research_goal is run_goal:
        global_context = run_context
    status, results, references = persist_cycle_result(run_goal, cycle_result, run_context)
    total_elapsed = cycle_result.get("cycle_details", {}).get("execution_time", time.monotonic() - started)
    yield status, results, references, format_research_trace_html(final_trace, elapsed_seconds=total_elapsed)


def format_cycle_results(cycle_details: Dict, log_file: str = None) -> str:
    """Format cycle results as HTML with expandable sections. Optionally log final rankings to log_file."""
    import html as html_lib

    html = f"<h2>🔬 Iteration {cycle_details.get('iteration', 'Unknown')}</h2>"

    # Surface generation errors up front with an actionable category, so a failed
    # run explains itself instead of silently showing empty rankings (issue llnl#36).
    errors = cycle_details.get("errors", [])
    if errors:
        items = ""
        for e in errors:
            category = classify_llm_error(e)
            items += f"<li><strong>{html_lib.escape(category)}:</strong> {html_lib.escape(str(e))}</li>"
        html += f"""
        <div style="margin: 20px 0; padding: 15px; border: 2px solid #e74c3c; border-radius: 8px; background-color: #fff5f5;">
            <h3>⚠️ Generation could not complete</h3>
            <p>The Generation pipeline reported the following, so some or all hypotheses were not generated:</p>
            <ul style="color: #c0392b;">{items}</ul>
        </div>
        """

    warnings = cycle_details.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        warning_items = "".join(
            f"<li>{html_lib.escape(str(warning))}</li>" for warning in warnings if str(warning).strip()
        )
        if warning_items:
            html += f"""
            <div style="margin: 20px 0; padding: 15px; border: 2px solid #e67e22; border-radius: 8px; background-color: #fffaf2;">
                <h3>⚠️ Generation completed with recovery warnings</h3>
                <ul style="color: #a65f00;">{warning_items}</ul>
            </div>
            """

    steps = cycle_details.get("steps", {})
    finalization = cycle_details.get("finalization", {})
    hypothesis_pipeline_enabled = not (
        isinstance(finalization, dict) and finalization.get("hypothesis_pipeline_enabled") is False
    )
    generation_plan = steps.get("generation", {}).get("query_plan", {})
    research_type = (
        (finalization.get("research_type") if isinstance(finalization, dict) else None)
        or (generation_plan.get("research_type") if isinstance(generation_plan, dict) else None)
        or "hypothesis_testing"
    )
    # Process steps in order
    generation_sources = steps.get("generation", {}).get("sources", [])
    if not isinstance(generation_sources, list):
        generation_sources = []
    # Display steps in the order they appear in the steps dict (preserves backend execution order)
    for step_name, step_data in steps.items():
        step_title = {
            "generation": "🎯 Generation",
            "reflection": "🔍 Reflection",
            "ranking": "📊 Ranking",
            "evolution": "🧬 Evolution",
            "reflection_evolved": "🔍 Reflection (Evolved)",
            "ranking_final": "📊 Final Ranking",
            "proximity": "🔗 Proximity Analysis",
            "meta_review": "📋 Meta-Review",
        }.get(step_name, step_name.title())

        html += f"""
        <details style="margin: 15px 0; border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
            <summary style="font-weight: bold; font-size: 1.1em; cursor: pointer; padding: 5px;">
                {step_title}
            </summary>
            <div style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        """

        # Step-specific content
        if step_name == "generation":
            hypotheses = step_data.get("hypotheses", [])
            generation_stages = step_data.get("stages", {})
            if isinstance(generation_stages, dict) and generation_stages:
                stage_labels = {
                    "evidence_retrieval": "Evidence retrieval",
                    "literature_synthesis": "Literature synthesis",
                    "hypothesis_generation": "Hypothesis generation",
                }
                html += "<p><strong>Generation stage diagnostics:</strong></p><ul>"
                for stage_name, stage_label in stage_labels.items():
                    stage = generation_stages.get(stage_name, {})
                    if not isinstance(stage, dict):
                        continue
                    status = html_lib.escape(str(stage.get("status") or "unknown").replace("_", " "))
                    detail = html_lib.escape(str(stage.get("detail") or ""))
                    suffix = f" — {detail}" if detail else ""
                    html += f"<li><strong>{stage_label}:</strong> {status}{suffix}</li>"
                html += "</ul>"
            search_stats = step_data.get("search_stats", [])
            evidence_funnel = step_data.get("evidence_funnel", {})
            if not isinstance(evidence_funnel, dict):
                evidence_funnel = {}
            evidence_pipeline = step_data.get("evidence_pipeline", [])
            if not isinstance(evidence_pipeline, list):
                evidence_pipeline = []
            query_plan = step_data.get("query_plan", {})
            if not isinstance(query_plan, dict):
                query_plan = {}
            provisional_hypotheses = query_plan.get(
                "provisional_hypotheses",
                [],
            )
            planned_queries = query_plan.get("queries", [])
            query_fidelity = step_data.get("query_fidelity", [])
            has_search_details = any(
                isinstance(items, list) and items
                for items in (
                    search_stats,
                    provisional_hypotheses,
                    planned_queries,
                    query_fidelity,
                    evidence_pipeline,
                )
            )
            if has_search_details:
                html += """
                <details style="margin: 5px 0 10px;">
                    <summary style="cursor: pointer; font-size: 0.9em;">Search details</summary>
                """
                if isinstance(provisional_hypotheses, list) and provisional_hypotheses:
                    html += "<p><strong>Provisional retrieval hypotheses (not evidence):</strong></p><ul>"
                    for provisional in provisional_hypotheses:
                        if not isinstance(provisional, dict):
                            continue
                        role = html_lib.escape(str(provisional.get("role", "unknown")))
                        statement = html_lib.escape(str(provisional.get("statement", "")))
                        html += f"<li>{role}: {statement}</li>"
                    html += "</ul>"
                if isinstance(planned_queries, list) and planned_queries:
                    html += "<p><strong>Planned queries:</strong></p><ul>"
                    for query in planned_queries:
                        if not isinstance(query, dict):
                            continue
                        query_text = html_lib.escape(str(query.get("query", "")))
                        intent = html_lib.escape(str(query.get("search_intent", "goal")))
                        source_type = html_lib.escape(str(query.get("source_type", "all")))
                        html += f"<li>{intent} · {source_type}: {query_text}</li>"
                    html += "</ul>"
                if isinstance(query_fidelity, list) and query_fidelity:
                    checked_queries = [
                        item for item in query_fidelity if isinstance(item, dict) and item.get("kind") == "query"
                    ]
                    if checked_queries:
                        accepted = sum(item.get("accepted") is True for item in checked_queries)
                        html += f"<p><strong>Query fidelity:</strong> {accepted}/{len(checked_queries)} accepted</p>"
                if isinstance(search_stats, list) and search_stats:
                    html += "<p><strong>Search providers called:</strong></p><ul>"
                for stat in search_stats if isinstance(search_stats, list) else []:
                    if not isinstance(stat, dict):
                        continue
                    provider = html_lib.escape(str(stat.get("source", "Unknown")))
                    status = html_lib.escape(str(stat.get("status", "unknown")))
                    html += (
                        f"<li>Round {int(stat.get('round', 0))}: {provider} — "
                        f"{int(stat.get('queries_completed', 0))}/{int(stat.get('queries_requested', 0))} "
                        f"queries, {int(stat.get('results', 0))} results, "
                        f"{int(stat.get('elapsed_ms', 0))} ms ({status})</li>"
                    )
                if isinstance(search_stats, list) and search_stats:
                    html += "</ul>"
                if evidence_funnel:
                    html += (
                        "<p><strong>Evidence funnel:</strong> "
                        f"{int(evidence_funnel.get('raw_search_hits', 0))} raw search hits → "
                        f"{int(evidence_funnel.get('unique_candidates', 0))} unique candidates → "
                        f"{int(evidence_funnel.get('selected_sources', 0))} selected sources → "
                        f"{int(evidence_funnel.get('acquisition_attempts', 0))} acquisition attempts → "
                        f"{int(evidence_funnel.get('committed_sources', 0))} COMMITTED sources → "
                        f"{int(evidence_funnel.get('retrieved_passages', 0))} passages → "
                        f"{int(evidence_funnel.get('coverage_approved_sources', 0))} coverage-approved → "
                        f"{int(evidence_funnel.get('generation_consumed_sources', 0))} generation-consumed.</p>"
                    )
                if evidence_pipeline:
                    html += "<p><strong>Evidence loss diagnostics:</strong></p><ul>"
                    for item in evidence_pipeline:
                        if not isinstance(item, dict):
                            continue
                        source_id = html_lib.escape(str(item.get("candidate_source_id") or "unknown"))
                        requirement_id = html_lib.escape(str(item.get("requirement_id") or "unscoped"))
                        acquisition = html_lib.escape(str(item.get("acquisition_result") or "not_attempted"))
                        gate_reason = html_lib.escape(str(item.get("strict_gate_rejection_reason") or "not_evaluated"))
                        html += (
                            f"<li>{requirement_id}: {source_id} — acquisition={acquisition}, "
                            f"index={html_lib.escape(str(item.get('index_status') or 'MISSING'))}, "
                            f"passages={len(item.get('selected_chunk_ids') or [])}, gate={gate_reason}</li>"
                        )
                    html += "</ul>"
                html += "</details>"
            if hypotheses:
                html += f"<p><strong>Generated {len(hypotheses)} new hypotheses:</strong></p>"
            elif not hypothesis_pipeline_enabled:
                html += (
                    "<p><strong>Mode-specific evidence synthesis completed without "
                    "fabricating hypotheses.</strong> "
                    f"Research type: {html_lib.escape(str(research_type))}.</p>"
                )
            else:
                html += "<p><strong>Generated 0 new hypotheses.</strong></p>"
            for i, hypo in enumerate(hypotheses):
                audit = hypo.get("audit_report", {})
                audit_html = ""
                if isinstance(audit, dict) and audit:
                    score = audit.get("weighted_score", "N/A")
                    verdict = html_lib.escape(str(audit.get("verdict", "UNREVIEWED")))
                    prior_art = audit.get("closest_prior_art", [])
                    prior_art_ids = ", ".join(
                        html_lib.escape(str(item.get("source_id", "")))
                        for item in prior_art
                        if isinstance(item, dict) and item.get("source_id")
                    )
                    audit_html = (
                        "<p><strong>Generation quality gate:</strong> "
                        f"{verdict} · {score}/100"
                        + (f" · Closest prior art: {prior_art_ids}" if prior_art_ids else "")
                        + "</p>"
                    )
                html += f"""
                <div style="border-left: 3px solid #28a745; padding: 10px; margin: 10px 0; border-radius: 15px;">
                    <h5>#{i + 1}: {hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p style="white-space: pre-line;">{hypo.get("text")}</p>
                    {audit_html}
                    {format_evidence_sources_html(hypo, generation_sources)}
                </div>
                """

            audits = step_data.get("audits", [])
            if isinstance(audits, list) and audits:
                html += """
                <details style="margin: 10px 0;">
                    <summary style="cursor: pointer; font-size: 0.9em;">Quality audit details</summary>
                    <ol>
                """
                for audit in audits:
                    if not isinstance(audit, dict):
                        continue
                    verdict = html_lib.escape(str(audit.get("verdict", "UNREVIEWED")))
                    score = html_lib.escape(str(audit.get("weighted_score", "N/A")))
                    messages = [
                        str(message).strip()
                        for key in ("hard_failures", "warnings")
                        for message in audit.get(key, [])
                        if isinstance(message, str) and message.strip()
                    ]
                    message_html = (
                        "<ul>" + "".join(f"<li>{html_lib.escape(message)}</li>" for message in messages) + "</ul>"
                        if messages
                        else ""
                    )
                    html += f"<li><strong>{verdict} · {score}/100</strong>{message_html}</li>"
                html += "</ol></details>"
        elif step_name in ["reflection", "reflection_evolved"]:
            hypotheses = step_data.get("hypotheses", [])
            html += f"<p><strong>Reviewed {len(hypotheses)} hypotheses:</strong></p>"
            for hypo in hypotheses:
                html += f"""
                <div style="border-left: 3px solid #17a2b8; padding: 10px; margin: 10px 0; border-radius: 15px;">
                    <h5>{hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                       <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
                    {f"<p><strong>Comments:</strong> {hypo.get('comments', 'No comments')}</p>" if hypo.get("comments") else ""}
                    {format_evidence_sources_html(hypo, generation_sources)}
                </div>
                """

        elif step_name.startswith("ranking"):
            hypotheses = step_data.get("hypotheses", [])
            tournament_results = step_data.get("tournament_results", [])
            title_map = {h.get("id"): h.get("title", "Untitled") for h in hypotheses}
            if hypotheses:
                sorted_hypotheses = sorted(hypotheses, key=lambda h: h.get("elo_score", 0), reverse=True)
                html += f"<p><strong>Ranking results ({len(hypotheses)} hypotheses):</strong></p>"
                html += "<ol>"
                for hypo in sorted_hypotheses:
                    html += f"""
                    <li style="margin:5px 0;">
                        <strong>{hypo.get("title", "Untitled")}</strong>
                        (ID: {hypo.get("id", "Unknown")})
                        - Elo: {hypo.get("elo_score", 0):.2f}
                        {format_evidence_sources_html(hypo, generation_sources)}
                    </li>
                    """
                html += "</ol>"
            if tournament_results:
                html += "<h4>⚔️ Tournament Debate Results</h4>"
                count = 0  # initialize match counter
                for match in tournament_results:
                    count += 1  # increment for each match counter
                    title_a = title_map.get(match["hypothesis_a"], match["hypothesis_a"])
                    title_b = title_map.get(match["hypothesis_b"], match["hypothesis_b"])
                    if match["outcome"] == "A":
                        winner = title_a
                    elif match["outcome"] == "B":
                        winner = title_b
                    elif match["outcome"] == "TIE":
                        winner = "Tie"
                    else:
                        winner = "Abstain"
                    html += f"""
                    <details open style="
                        border:1px solid #ddd;
                        border-radius:10px;
                        padding:15px;
                        margin:15px 0;
                        background:#f8f9fa;">
                        <summary style="cursor: pointer;">
                            <strong>⚔️ Tournament Match {count}:</strong> {title_a} <strong>(ID: {match.get("hypothesis_a", "Unknown")})</strong> vs {title_b} <strong>(ID: {match.get("hypothesis_b", "Unknown")})</strong>
                        </summary>
                        <div style="margin-top: 10px; spacing: 5px;">
                            <p><b>🅰 Hypothesis A</b><br>
                            {title_a} <strong>(ID: {match.get("hypothesis_a", "Unknown")})</strong></p>

                            <p><b>🅱 Hypothesis B</b><br>
                            {title_b} <strong>(ID: {match.get("hypothesis_b", "Unknown")})</strong></p>

                            <p><b>🏆 Winner</b><br>
                            {winner}</p>

                            <p><b>🎯 Confidence</b><br>
                            {format_ranking_confidence(match.get("confidence"))}</p>

                            <p><b>💡 Why it won</b><br>
                            {match.get("reasoning") or "No reason was provided by the ranking judge."}</p>

                            <p><b>📌 Decisive Criteria</b></p>

                            <ul>
                                {"".join(f"<li>{c}</li>" for c in match.get("criteria", []))}
                            </ul>
                        </div>
                    </details>
                    """
            else:
                if step_name == "ranking2":
                    explanation = (
                        "Ranking 2 only compares newly evolved hypotheses that passed reflection, "
                        "and no eligible new pair was available."
                    )
                else:
                    explanation = (
                        "Fewer than two eligible hypotheses were available, or no new hypothesis "
                        "required another comparison."
                    )
                html += f"<p><strong>No tournament debates were run.</strong> {explanation}</p>"

        elif step_name == "evolution":
            hypotheses = step_data.get("hypotheses", [])
            html += f"<p><strong>Evolved {len(hypotheses)} new hypotheses by combining top performers:</strong></p>"
            for hypo in hypotheses:
                html += f"""
                <div style="border-left: 3px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 15px;">
                    <h5>{hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p style="white-space: pre-line;">{hypo.get("text")}</p>
                    {format_evidence_sources_html(hypo, generation_sources)}
                </div>
                """

        elif step_name == "proximity":
            adjacency_graph = step_data.get("adjacency_graph", {})
            nodes = step_data.get("nodes", [])
            edges = step_data.get("edges", [])

            logger.debug(
                "Proximity data - adjacency_graph keys: %s",
                list(adjacency_graph.keys()) if adjacency_graph else "None",
            )
            logger.debug("Proximity data - nodes count: %d", len(nodes) if nodes else 0)
            logger.debug("Proximity data - edges count: %d", len(edges) if edges else 0)

            if adjacency_graph:
                num_hypotheses = len(adjacency_graph)
                html += "<p><strong>Similarity Analysis:</strong></p>"
                html += f"<p>Analyzed relationships between {num_hypotheses} hypotheses</p>"

                # Calculate and display average similarity
                all_similarities = []
                for hypo_id, connections in adjacency_graph.items():
                    for conn in connections:
                        all_similarities.append(conn.get("similarity", 0))

                if all_similarities:
                    avg_sim = sum(all_similarities) / len(all_similarities)
                    html += f"<p>Average similarity: {avg_sim:.3f}</p>"
                    html += f"<p>Total connections analyzed: {len(all_similarities)}</p>"

                # Show top similar pairs
                similarity_pairs = []
                for hypo_id, connections in adjacency_graph.items():
                    for conn in connections:
                        similarity_pairs.append((hypo_id, conn.get("other_id"), conn.get("similarity", 0)))

                # Sort by similarity and show top 5
                similarity_pairs.sort(key=lambda x: x[2], reverse=True)
                if similarity_pairs:
                    html += "<h6>Top Similar Hypothesis Pairs:</h6><ul>"
                    for i, (id1, id2, sim) in enumerate(similarity_pairs[:5]):
                        html += f"<li>{id1} ↔ {id2}: {sim:.3f}</li>"
                    html += "</ul>"
                else:
                    html += "<p>No proximity data available.</p>"

        elif step_name == "meta_review":
            logger.debug("meta_review step_data = %s", step_data)
            assert isinstance(step_data, dict), "meta_review step_data is not a dict"
            # Accept both direct dict or nested under 'meta_review'
            if "meta_review" in step_data and isinstance(step_data["meta_review"], dict):
                meta_review = step_data["meta_review"]
            else:
                meta_review = step_data
            assert "meta_review_critique" in meta_review, f"meta_review_critique missing in meta_review: {meta_review}"
            assert "research_overview" in meta_review, f"research_overview missing in meta_review: {meta_review}"
            # Critique section
            if meta_review.get("meta_review_critique"):
                html += "<h5>Critique:</h5><ul>"
                for critique in meta_review["meta_review_critique"]:
                    html += f"<li>{critique}</li>"
                html += "</ul>"
            # Top ranked hypotheses section
            top_hypos = meta_review.get("research_overview", {}).get("top_ranked_hypotheses", [])
            assert isinstance(top_hypos, list), f"top_ranked_hypotheses is not a list: {top_hypos}"
            if top_hypos:
                html += "<h5>Top Ranked Hypotheses:</h5>"
                for i, hypo in enumerate(top_hypos):
                    html += f"""
                    <div style="border-left: 3px solid #28a745; padding: 10px; margin: 10px 0; border-radius: 15px;">
                        <h6>#{i + 1}: {hypo.get("title", "Untitled")}</h6>
                        <p><strong>ID:</strong> {hypo.get("id", "Unknown")} | 
                           <strong>Elo Score:</strong> {hypo.get("elo_score", 0):.2f}</p>
                        <p style="white-space: pre-line;"><strong>Description:</strong> {hypo.get("text")}</p>
                        <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                           <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
                        {format_evidence_sources_html(hypo, generation_sources)}
                    </div>
                    """
            # Suggested next steps section
            if meta_review.get("research_overview", {}).get("suggested_next_steps"):
                html += "<h5>Suggested Next Steps:</h5><ul>"
                for step in meta_review["research_overview"]["suggested_next_steps"]:
                    html += f"<li>{step}</li>"
                html += "</ul>"

        # Add timing information if available
        if step_data.get("duration"):
            html += f"<p><em>Duration: {step_data['duration']:.2f}s</em></p>"

        html += "</div></details>"

    # Final summary section - always expanded
    # Prefer ranking steps, else fallback to step with most hypotheses
    final_hypotheses = []
    final_step = None
    step_order = _ordered_ranking_step_names(steps)
    for step_name in step_order:
        if step_name in steps and steps[step_name].get("hypotheses"):
            final_hypotheses = steps[step_name]["hypotheses"]
            final_step = step_name
            break

    # Fallback: no ranking step ran, so show the most recent reviewed batch.
    # Steps are recorded in execution order, and picking the largest batch
    # instead surfaced the pre-Reflection Generation candidates as "final".
    if not final_hypotheses:
        for sname, sdata in reversed(list(steps.items())):
            hypos = sdata.get("hypotheses", [])
            if hypos:
                final_hypotheses = hypos
                final_step = sname
                break

    # Assertions: final list should not be empty and no duplicate IDs (only for ranking steps)
    ranking_steps = set(step_order)
    if final_hypotheses:
        ids = [h.get("id") for h in final_hypotheses]
        if final_step in ranking_steps:
            assert len(ids) == len(set(ids)), "Duplicate hypothesis IDs found in final rankings!"
        assert len(final_hypotheses) > 0, "Final hypothesis list is empty!"

        # Sort by Elo score if present, else by ID
        if any("elo_score" in h for h in final_hypotheses):
            final_hypotheses = sorted(final_hypotheses, key=lambda h: h.get("elo_score", 0), reverse=True)
        else:
            final_hypotheses = sorted(final_hypotheses, key=lambda h: h.get("id", ""))

        html += """
        <div style="margin: 20px 0; padding: 15px; border: 2px solid #28a745; border-radius: 8px; background-color: #f8fff8;">
            <h3>🏆 Final Rankings - Top Hypotheses</h3>
        """
        if final_step not in ranking_steps:
            html += '<p style="color: #e67e22;">Warning: No ranking step found. Showing hypotheses from the latest available step ("{}"). These may not be ranked.</p>'.format(
                final_step
            )

        # Log final rankings if log_file is provided
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- Final Rankings Section (step: {final_step}) ---\n")
                for i, hypo in enumerate(final_hypotheses[:10]):
                    f.write(
                        f"  #{i + 1}: ID: {hypo.get('id')} | Title: {hypo.get('title')} | Elo: {hypo.get('elo_score', 'N/A')}\n"
                    )

        for i, hypo in enumerate(final_hypotheses[:10]):  # Show top 10
            comments = hypo.get("review_comments") or []
            comments_html = "".join(f"<li>{_escape(comment)}</li>" for comment in comments)
            rank_color = "#28a745" if i < 3 else "#17a2b8" if i < 6 else "#6c757d"
            html += f"""
            <div style="border-left: 4px solid {rank_color}; padding: 15px; margin: 10px 0; background-color: white; border-radius: 5px;">
                <h4>#{i + 1}: {hypo.get("title", "Untitled")}</h4>
                <p><strong>ID:</strong> {hypo.get("id", "Unknown")} | 
                   <strong>Elo Score:</strong> {hypo.get("elo_score", 0):.2f}</p>
                <p style="white-space: pre-line;"><strong>Description:</strong><br /> {(hypo.get("text"))}</p>
                <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                   <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
                        <p><strong>Reviewer Comments</strong></p>
                        <ul>{comments_html}</ul>
                {format_evidence_sources_html(hypo, generation_sources)}
            </div>
            """

        html += "</div>"
    else:
        if not hypothesis_pipeline_enabled:
            meta_review = steps.get("meta_review", {})
            overview = meta_review.get("research_overview", {}) if isinstance(meta_review, dict) else {}
            next_steps = overview.get("suggested_next_steps", []) if isinstance(overview, dict) else []
            next_steps_html = "".join(
                f"<li>{html_lib.escape(str(item))}</li>" for item in next_steps if str(item).strip()
            )
            html += f"""
            <div style="margin: 20px 0; padding: 15px; border: 2px solid #17a2b8; border-radius: 8px; background-color: #f4fbfd;">
                <h3>📚 Mode-Specific Research Synthesis</h3>
                <p>The {html_lib.escape(str(research_type))} plan completed without a hypothesis-ranking stage.</p>
                {f"<p><strong>Suggested next steps:</strong></p><ul>{next_steps_html}</ul>" if next_steps_html else ""}
            </div>
            """
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"--- Mode-specific synthesis complete ({research_type}); hypothesis ranking not applicable. ---\n"
                    )
        elif errors:
            cause = "; ".join(sorted({classify_llm_error(e) for e in errors}))
            no_rank_msg = (
                f"No hypotheses available for final ranking because generation failed: {html_lib.escape(cause)}. "
                "See the details above."
            )
        else:
            no_rank_msg = "No hypotheses available for final ranking. This may indicate an error in the workflow."

        if hypothesis_pipeline_enabled:
            html += f"""
            <div style="margin: 20px 0; padding: 15px; border: 2px solid #e74c3c; border-radius: 8px; background-color: #fff5f5;">
                <h3>🏆 Final Rankings - Top Hypotheses</h3>
                <p style="color: #e74c3c;">{no_rank_msg}</p>
            </div>
            """
            # Log missing final rankings if log_file is provided
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write("--- Final Rankings Section: No hypotheses available for final ranking. ---\n")

    return html


def get_references_html(cycle_details: Dict, research_goal: Optional[ResearchGoal] = None) -> str:
    """Render validated sources and how the active research mode consumed them."""
    import html as html_lib

    generation_step = cycle_details.get("steps", {}).get("generation", {})
    finalization = cycle_details.get("finalization", {})
    hypothesis_pipeline_enabled = not (
        isinstance(finalization, dict) and finalization.get("hypothesis_pipeline_enabled") is False
    )
    sources = generation_step.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return "<p>No retrieved evidence was used for generation.</p>"

    evidence_consumed = generation_step.get("evidence_consumed")
    if evidence_consumed is False:
        html = (
            "<h3>📚 Evidence Retrieval Completed</h3>"
            "<p>Validated evidence was retrieved, but hypothesis generation did not execute.</p>"
        )
    elif not hypothesis_pipeline_enabled:
        html = "<h3>📚 Retrieved Evidence Used for Research Synthesis</h3>"
    else:
        html = "<h3>📚 Retrieved Evidence Used for Generation</h3>"
    for source in sources:
        if not isinstance(source, dict):
            continue

        title = html_lib.escape(str(source.get("title") or "Untitled"))
        authors = html_lib.escape(", ".join(str(author) for author in source.get("authors", [])[:5]))
        source_id = html_lib.escape(str(source.get("source_id") or source.get("arxiv_id") or "Unknown"))
        source_type = str(source.get("source_type") or "academic")
        provider = html_lib.escape(str(source.get("provider") or source.get("source") or "arxiv"))
        published = html_lib.escape(str(source.get("published_at") or source.get("published") or "Unknown"))
        summary = html_lib.escape(str(source.get("summary") or source.get("abstract") or "No summary")[:300])
        source_url = html_lib.escape(
            str(source.get("url") or source.get("arxiv_url") or "#"),
            quote=True,
        )
        raw_pdf_url = str(source.get("pdf_url") or "").strip()
        pdf_link = ""
        if raw_pdf_url.startswith(("https://", "http://")):
            pdf_url = html_lib.escape(raw_pdf_url, quote=True)
            pdf_link = f' | <a href="{pdf_url}" target="_blank">📁 Download PDF</a>'
        evidence_status = str(source.get("evidence_status") or "abstract_only")
        if source_type == "web" and not source.get("full_text_indexed"):
            library_status = "Retrieved web content used directly"
        elif source.get("full_text_indexed"):
            chunks_used = int(source.get("full_text_chunks_used") or 0)
            library_status = f"Indexed in local ChromaDB; {chunks_used} relevant full-text chunk(s) used"
        elif evidence_status == "full_text_failed":
            library_status = "Full-text acquisition failed; abstract-only evidence used"
        else:
            library_status = "Abstract-only evidence"
        content_label = "Web content" if source_type == "web" else "Abstract"
        author_line = f"<p><strong>Authors:</strong> {authors}</p>" if authors else ""
        evidence_refs = source.get("evidence_refs", [])
        full_text_refs = (
            [ref for ref in evidence_refs if isinstance(ref, dict) and ref.get("evidence_type") == "full_text"]
            if isinstance(evidence_refs, list)
            else []
        )
        sections = ", ".join(dict.fromkeys(str(ref.get("section") or "Unknown") for ref in full_text_refs))
        pages = ", ".join(dict.fromkeys(str(ref.get("page")) for ref in full_text_refs if ref.get("page") is not None))
        chunk_ids = ", ".join(
            f"<code>{html_lib.escape(str(ref.get('chunk_id')))}</code>" for ref in full_text_refs if ref.get("chunk_id")
        )
        provenance_html = ""
        if full_text_refs:
            provenance_html = (
                f"<p><strong>Sections:</strong> {html_lib.escape(sections)} | "
                f"<strong>Pages:</strong> {html_lib.escape(pages)}</p>"
                f"<p><strong>Evidence chunks:</strong> {chunk_ids}</p>"
            )
        html += f"""
        <div style="border: 1px solid #e0e0e0; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #fafafa;">
            <h4>{title}</h4>
            {author_line}
            <p><strong>Source:</strong> {provider} |
               <strong>Type:</strong> {html_lib.escape(source_type)} |
               <strong>Source ID:</strong> {source_id} |
               <strong>Published:</strong> {published}</p>
            <p><strong>{content_label}:</strong> {summary}...</p>
            <p><strong>Evidence storage:</strong> {library_status}</p>
            {provenance_html}
            <p>
                <a href="{source_url}" target="_blank">📄 View source</a>{pdf_link}
            </p>
        </div>
        """

    return html


HEADER_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "guard5g-logo.png"


def header_html(logo_path: Path = HEADER_LOGO_PATH) -> str:
    """Return the page title, with the logo inlined as a data URI when available."""
    title = "5G Guard AI Co-Scientist"
    try:
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except OSError:
        return f"<h1>🔬 {title}</h1>"
    logo = (
        f'<img src="data:image/png;base64,{encoded}" alt="" '
        'style="height:1.6em;width:auto;vertical-align:middle;margin-right:0.35em;">'
    )
    return f'<h1 style="display:flex;align-items:center;">{logo}{title}</h1>'


def create_gradio_interface():
    """Create the Gradio interface."""

    # Fetch models on startup
    fetch_available_models()

    # Get deployment status
    status_text, status_color = get_deployment_status()

    # Define custom theme and CSS for launch()
    theme = gr.themes.Soft()
    css = """
        .status-box {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .orange { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .blue { background-color: #d1ecf1; border: 1px solid #bee5eb; }

        #research-history-sidebar {
            background: var(--block-background-fill) !important;
            border-right: 1px solid var(--border-color-primary);
        }
        #research-history-sidebar .sidebar-history-copy {
            color: var(--body-text-color-subdued);
            font-size: 0.9rem;
        }
        #sidebar-run-list {
            background: var(--block-background-fill) !important;
            border: 0;
            box-shadow: none;
            padding: 0;
        }
        #sidebar-run-list > .wrap:not([data-testid]) {
            align-items: stretch;
            background: var(--block-background-fill) !important;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        #sidebar-run-list label {
            background: var(--block-background-fill) !important;
            border: 0;
            border-radius: 10px;
            box-shadow: none !important;
            color: var(--body-text-color) !important;
            cursor: pointer;
            display: block;
            margin: 0;
            padding: 10px 12px;
            transition: background-color 120ms ease;
            width: 100%;
        }
        #sidebar-run-list label:hover {
            background: color-mix(in srgb, var(--body-text-color) 6%, transparent) !important;
        }
        #sidebar-run-list label:has(input:checked) {
            background: color-mix(in srgb, var(--body-text-color) 10%, transparent) !important;
            color: var(--body-text-color) !important;
        }
        #sidebar-run-list label:has(input:checked) span {
            color: var(--body-text-color) !important;
        }
        #sidebar-run-list input[type="radio"] {
            opacity: 0;
            pointer-events: none;
            position: absolute;
        }
        #sidebar-run-list label span {
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .dark #research-history-sidebar,
        .dark #sidebar-run-list,
        .dark #sidebar-run-list > .wrap:not([data-testid]) {
            background: var(--block-background-fill) !important;
        }
        .dark #sidebar-run-list label:not(:hover):not(:has(input:checked)) {
            background: var(--block-background-fill) !important;
        }
        .dark #sidebar-run-list label:hover,
        .dark #sidebar-run-list label:hover span {
            background: rgba(255, 255, 255, 0.08) !important;
            color: var(--body-text-color) !important;
        }
        .dark #sidebar-run-list label:has(input:checked),
        .dark #sidebar-run-list label:has(input:checked) span {
            background: rgba(255, 255, 255, 0.12) !important;
            color: var(--body-text-color) !important;
        }

        /* Let the browser handle theme matching natively */
        :root {
            color-scheme: light dark;
        }

        /* Universal Fix for Dark Mode: Targets absolutely everything inside the custom HTML block */
        .dark div[id^="html-"],
        .dark div[id^="html-"] * {
            /* 1. Force all text to stay perfectly white */
            color: #ffffff !important;
        }

        /* Universal Background Fix: Automatically converts any forced light/white panels to dark */
        .dark div[id^="html-"] div,
        .dark div[id^="html-"] details,
        .dark div[id^="html-"] section {
            background-color: var(--block-background-fill) !important;
            border-color: var(--border-color-primary) !important;
        }

        /* Accent Fix: Keeps specific highlight containers readable (like things with heavy green borders) */
        .dark div[id^="html-"] div[style*="#28a745"] {
            background-color: #064e3b !important; /* Soft deep emerald instead of blinding light green */
            border-color: #28a745 !important;
        }

        /* Keep Activity source links legible instead of rendering white-on-white. */
        .dark div[id^="html-"] .activity-drawer .source-chip {
            background: rgba(255, 255, 255, 0.10) !important;
            color: var(--body-text-color) !important;
        }
        .dark div[id^="html-"] .activity-drawer .source-chip:hover {
            background: rgba(255, 255, 255, 0.18) !important;
            color: var(--body-text-color) !important;
        }
        """

    with gr.Blocks(title="5G Guard AI Co-Scientist") as demo:
        with gr.Sidebar(open=False, width=320, elem_id="research-history-sidebar"):
            gr.Markdown("## Research history")
            gr.Markdown(
                "Saved research goals remain available after a refresh. Select one to restore its results.",
                elem_classes="sidebar-history-copy",
            )
            sidebar_refresh_btn = gr.Button("Refresh history", size="sm")
            sidebar_delete_btn = gr.Button("Delete", variant="stop", size="sm")
            sidebar_history = gr.Radio(
                choices=sidebar_run_choices(),
                value=None,
                label="Recent research goals",
                interactive=True,
                elem_id="sidebar-run-list",
                buttons=[sidebar_delete_btn],
            )
            sidebar_delete_status = gr.Markdown()

        # Header
        gr.HTML(header_html())
        gr.Markdown("Scientific Hypothesis-driven Investigation, Evidence-based Learning and Defence.")

        # Deployment status
        gr.HTML(f'<div class="status-box {status_color}">🔧 Deployment Status: {status_text}</div>')

        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                # Research goal input
                research_goal_input = gr.Textbox(
                    label="Research Goal",
                    placeholder="Enter your research goal (e.g., 'Develop new methods for increasing the efficiency of solar panels')",
                    lines=3,
                )

                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    default_model = get_default_model_choice()
                    model_dropdown = gr.Dropdown(
                        choices=get_model_dropdown_choices(),
                        value=default_model,
                        label=f"LLM Model (default: {default_model})",
                        info="Select a model currently loaded or available in LM Studio.",
                        interactive=True,
                    )

                    with gr.Row():
                        num_hypotheses = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=config.get("num_hypotheses", 4),
                            step=1,
                            label="Hypotheses per Cycle",
                        )
                        top_k_hypotheses = gr.Slider(minimum=2, maximum=5, value=2, step=1, label="Top K for Evolution")

                    with gr.Row():
                        generation_temp = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Generation Temperature (Creativity)"
                        )
                        reflection_temp = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Reflection Temperature (Analysis)"
                        )

                    elo_k_factor = gr.Slider(
                        minimum=1, maximum=100, value=32, step=1, label="Elo K-Factor (Ranking Sensitivity)"
                    )

                # Kept beside Run Cycle, since it changes how long a cycle runs.
                run_experiment_toggle = gr.Checkbox(
                    value=bool(config.get("experiment_auto_run", True)),
                    label="Run automated experiment after the cycle",
                    info=(
                        "Off: the cycle ends after ranking and shows its hypotheses right away; "
                        "no experiment code is generated or run."
                    ),
                )

                # Single action button
                with gr.Row():
                    run_cycle_btn = gr.Button("🔄 Run Cycle", variant="primary")

                # Status display
                status_output = gr.Textbox(
                    label="Status",
                    value="Enter a research goal and click 'Run Cycle' to begin.",
                    interactive=False,
                    lines=3,
                )

            with gr.Column(scale=1):
                # Instructions
                # gr.Markdown("""
                # ### 📖 Instructions

                # 1. **Enter Research Goal**: Describe what you want to research.
                # 2. **Adjust Settings** (optional): Customize model and parameters.
                # 3. **Click "Run Cycle"**: The system will set your goal and immediately generate, review, rank, and evolve hypotheses in one step.

                # ### 💡 Tips
                # - Start LM Studio's local server before running a cycle
                # - Load a model in LM Studio, then select it in Advanced Settings
                # - Higher generation temperature = more creative ideas
                # - Lower reflection temperature = more analytical reviews
                # - Each cycle builds on previous results

                # **Note:** Runtime depends on your local model size and hardware.
                # """)
                gr.HTML("""
                <div style="
                    border: 1px solid #e2e8f0; 
                    padding: 20px;
                    border-radius: 8px; 
                    background-color: #f8fafc; 
                    color: #334155;
                ">
                    <h4 style="margin: 0 0 10px 0; color: #0f172a; font-size: 1.1em;">📖 Instructions</h4>
                    <ol style="margin: 0 0 15px 0; padding-left: 20px; line-height: 1.5;">
                        <li style="margin-bottom: 6px;"><strong>Enter Research Goal</strong>: Describe what you want to research.</li>
                        <li style="margin-bottom: 6px;"><strong>Adjust Settings</strong> (optional): Customize model and parameters.</li>
                        <li style="margin-bottom: 0;"><strong>Click "Run Cycle"</strong>: The system will set your goal and immediately generate, review, rank, and evolve hypotheses in one step.</li>
                    </ol>
                    
                    <h4 style="margin: 15px 0 10px 0; color: #0f172a; font-size: 1.1em;">💡 Tips</h4>
                    <ul style="margin: 0 0 15px 0; padding-left: 20px; line-height: 1.5;">
                        <li style="margin-bottom: 6px;">Start LM Studio's local server before running a cycle.</li>
                        <li style="margin-bottom: 6px;">Load a model in LM Studio, then select it in Advanced Settings.</li>
                        <li style="margin-bottom: 6px;">Higher generation temperature = more creative ideas.</li>
                        <li style="margin-bottom: 6px;">Lower reflection temperature = more analytical reviews.</li>
                        <li style="margin-bottom: 0;">Each cycle builds on previous results.</li>
                    </ul>
                    
                    <p style="margin: 15px 0 0 0; font-size: 0.95em; color: #64748b;">
                        <strong>Note:</strong> Runtime depends on your local model size and hardware.
                    </p>
                </div>
                """)

        with gr.Tabs(selected="current-run") as run_tabs:
            with gr.Tab("Current Run", id="current-run"):
                with gr.Row():
                    with gr.Column():
                        research_trace_output = gr.HTML(
                            label="Research Process",
                            value=format_research_trace_html([]),
                        )

                with gr.Row():
                    with gr.Column():
                        results_output = gr.HTML(
                            label="Results", value="<p>Results will appear here after running cycles.</p>"
                        )

                with gr.Row():
                    with gr.Column():
                        references_output = gr.HTML(
                            label="References", value="<p>Related research papers will appear here.</p>"
                        )

            with gr.Tab("Run History", id="run-history") as run_history_tab:
                gr.Markdown("Saved runs load automatically. Use refresh if runs were changed outside this page.")
                refresh_history_btn = gr.Button("Refresh History")
                history_output = gr.HTML(label="Saved Runs", value=history_html())
                with gr.Row():
                    delete_run_dropdown = gr.Dropdown(
                        choices=history_run_choices(),
                        label="Saved Run to Delete",
                        interactive=True,
                    )
                    delete_history_btn = gr.Button("Delete Selected Run", variant="stop")
                delete_history_status = gr.Markdown()

        # Event handler: single button sets research goal and runs cycle
        def run_full_cycle(
            research_goal,
            llm_model,
            num_hypotheses,
            generation_temp,
            reflection_temp,
            elo_k_factor,
            top_k_hypotheses,
            run_experiment,
        ):
            # Set research goal
            status_msg, _ = set_research_goal(
                research_goal,
                llm_model,
                num_hypotheses,
                generation_temp,
                reflection_temp,
                elo_k_factor,
                top_k_hypotheses,
            )
            experiment_note = "" if run_experiment else " The automated experiment is off for this cycle."
            yield (
                f"{status_msg}\n\nStarting cycle with a {format_timeout_duration(CYCLE_TIMEOUT_SECONDS)} limit."
                f"{experiment_note}",
                format_research_trace_html([], running=True),
                "<p>Starting cycle...</p>",
                "",
                history_html(),
                gr.update(choices=history_run_choices(), value=None),
                gr.update(choices=sidebar_run_choices(), value=None),
            )
            for status, results, references, research_trace in run_cycle_with_progress(
                run_experiment=bool(run_experiment)
            ):
                yield (
                    f"{status_msg}\n\n{status}",
                    research_trace,
                    results,
                    references,
                    history_html(),
                    gr.update(choices=history_run_choices(), value=None),
                    gr.update(choices=sidebar_run_choices(), value=None),
                )

        run_cycle_btn.click(
            fn=run_full_cycle,
            inputs=[
                research_goal_input,
                model_dropdown,
                num_hypotheses,
                generation_temp,
                reflection_temp,
                elo_k_factor,
                top_k_hypotheses,
                run_experiment_toggle,
            ],
            outputs=[
                status_output,
                research_trace_output,
                results_output,
                references_output,
                history_output,
                delete_run_dropdown,
                sidebar_history,
            ],
        )

        sidebar_history.select(
            fn=load_history_run,
            inputs=[sidebar_history],
            outputs=[
                research_goal_input,
                status_output,
                research_trace_output,
                results_output,
                references_output,
                model_dropdown,
                num_hypotheses,
                generation_temp,
                reflection_temp,
                elo_k_factor,
                top_k_hypotheses,
                run_tabs,
            ],
            show_progress="minimal",
        )

        demo.load(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, sidebar_history, delete_history_status],
        )
        run_history_tab.select(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, sidebar_history, delete_history_status],
        )
        refresh_history_btn.click(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, sidebar_history, delete_history_status],
        )
        sidebar_refresh_btn.click(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, sidebar_history, delete_history_status],
        )
        sidebar_delete_btn.click(
            fn=delete_history_run,
            inputs=[sidebar_history],
            outputs=[sidebar_delete_status, history_output, delete_run_dropdown, sidebar_history],
        )
        delete_history_btn.click(
            fn=delete_history_run,
            inputs=[delete_run_dropdown],
            outputs=[delete_history_status, history_output, delete_run_dropdown, sidebar_history],
        )

        # Example inputs
        gr.Examples(
            examples=[
                ["Develop a closed-loop multi-agent AI framework to dynamically allocate 5G slice bandwidth during traffic spikes"],
                ["Create a machine learning orchestrator that injects post-quantum cryptographic keys into active 5G network slices without increasing latency"],
                ["Improve the existing 5G intrusion detection model to recognise previously unseen attack types and reduce the misclassification of unseen attacks as normal/benign traffic"],
                ["Improve 5G battery life by optimizing device wake-up sensors"],
                ["Automate the root-cause diagnosis of 5G tower failures by deploying AI agents to read logs and execute patches"],
                ["Develop a new 5G intrusion detection model that can detect all known attack types in the 5G-NIDD dataset, including attack types that the existing model fails to recognise"],
            ],
            inputs=[research_goal_input],
            label="Example Research Goals",
        )

        # GitHub icon and link at the bottom
        gr.HTML(
            """
            <div style="text-align:center; margin-top: 30px;">
                <a href="https://github.com/chunhualiao/ai-co-scientist" target="_blank" style="text-decoration:none; display:inline-flex; align-items:center; gap:8px;">
                    <svg height="32" width="32" viewBox="0 0 16 16" fill="currentColor" style="vertical-align:middle;">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
                        -.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2
                        -3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64
                        -.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08
                        2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                        1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                    <span style="font-size: 1.1em; vertical-align:middle;">View on GitHub</span>
                </a>
            </div>
            """
        )

    demo.theme = theme
    demo.css = css

    return demo


if __name__ == "__main__":
    runtime_log = configure_runtime_logging()
    logger.info("Runtime diagnostics are saved to %s", runtime_log)
    # Create and launch the Gradio app
    logger.info("Using LM Studio API at %s", get_lmstudio_base_url())
    demo = create_gradio_interface()

    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    allowed_paths = get_gradio_allowed_paths()

    logger.info(
        "Gradio allowed file paths: %s",
        allowed_paths,
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=allowed_paths,
        theme=getattr(demo, "theme", None),
        css=getattr(demo, "css", None),
    )
