import logging
import os
import threading
import time
from copy import deepcopy
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import gradio as gr
from numpy.ma import count  # noqa: F401

from app.agents import SupervisorAgent
from app.config import config
from app.models import ContextMemory, ResearchGoal
from app.research_trace import format_research_trace_html, merge_trace_event, normalize_trace_event
from app.run_store import (
    _escape,
    delete_run,
    get_reports_dir,
    history_html,
    list_runs,
    load_run,
    report_file_url,
    save_run,
    write_report,
)
from app.utils import (
    classify_llm_error,
    fetch_lmstudio_models,
    get_lmstudio_base_url,
    get_lmstudio_model,
    logger,
    redact_secrets,
)

# Global state for the Gradio app
global_context = ContextMemory()
supervisor = SupervisorAgent()
current_research_goal: Optional[ResearchGoal] = None
available_models: List[str] = []
CONFIGURED_LLM_MODEL = get_lmstudio_model()
SAFE_FALLBACK_LLM_MODEL = CONFIGURED_LLM_MODEL or "-- Select Model --"
CYCLE_TIMEOUT_SECONDS = int(os.getenv("CO_SCIENTIST_CYCLE_TIMEOUT_SECONDS", "1800"))
CYCLE_PROGRESS_INTERVAL_SECONDS = 5

# Configure logging for Gradio
logging.basicConfig(level=logging.INFO)


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
        constraints=goal_data.get("constraints") or {},
        llm_model=goal_data.get("llm_model"),
        query_rewrite_model=goal_data.get("query_rewrite_model"),
        num_hypotheses=goal_data.get("num_hypotheses"),
        generation_temperature=goal_data.get("generation_temperature"),
        reflection_temperature=goal_data.get("reflection_temperature"),
        elo_k_factor=goal_data.get("elo_k_factor"),
        top_k_hypotheses=goal_data.get("top_k_hypotheses"),
    )
    model_choices = list(available_models)
    if loaded_goal.llm_model and loaded_goal.llm_model not in model_choices:
        model_choices.append(loaded_goal.llm_model)

    stored_status = run.get("status") or "No status was recorded for this run."
    status = f"Loaded saved run {run.get('run_id', selected_run_id)}.\n\n{stored_status}"
    cycle_details = run.get("cycle_details") or {}
    trace = cycle_details.get("research_trace") or []
    return (
        description,
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
        # Create research goal with settings
        current_research_goal = ResearchGoal(
            description=description.strip(),
            constraints={},
            llm_model=llm_model if llm_model and llm_model != "-- Select Model --" else None,
            num_hypotheses=num_hypotheses,
            generation_temperature=generation_temperature,
            reflection_temperature=reflection_temperature,
            elo_k_factor=elo_k_factor,
            top_k_hypotheses=top_k_hypotheses,
        )

        # Reset context
        global_context = ContextMemory()

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


def execute_cycle(
    research_goal: ResearchGoal,
    context: ContextMemory,
    cycle_supervisor: SupervisorAgent,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run a cycle against the supplied state and return display-ready results."""
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

    try:
        iteration = context.iteration_number + 1

        # Start timing the cycle execution
        start_time = time.perf_counter()

        logger.info(f"Running cycle {iteration}")

        # Run the cycle
        cycle_details = cycle_supervisor.run(
            research_goal,
            context,
            progress_callback=capture_progress,
        )
        cycle_details.setdefault("research_trace", research_trace)

        # Log execution time
        total_time = time.perf_counter() - start_time
        formatted_time = format_execution_time(total_time)

        cycle_details["execution_time"] = total_time
        cycle_details["execution_time_formatted"] = formatted_time

        logger.info(f"Cycle execution time: {formatted_time}")

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
            status_msg = (
                f"⚠️ Cycle {iteration} reached its compute budget before finalization ({unmet}).\n\n"
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


def persist_cycle_result(research_goal: ResearchGoal, cycle_result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Persist an accepted cycle result and return Gradio output values."""
    saved_run = save_run(
        research_goal=research_goal,
        cycle_details=cycle_result["cycle_details"],
        status=cycle_result["status"],
        references_html=cycle_result["references_html"],
        results_html=cycle_result["results_html"],
        log_file=cycle_result["log_file"],
    )
    report_path = write_report(saved_run)
    status_msg = f"{cycle_result['status']}\n{to_bold('Run ID:')} {saved_run['run_id']}\n{to_bold('Report:')} {report_file_url(report_path)}"
    return status_msg, cycle_result["results_html"], cycle_result["references_html"]


def run_cycle() -> Tuple[str, str, str]:
    """Run a single research cycle with detailed step logging for debugging."""
    global current_research_goal, global_context, supervisor

    if not current_research_goal:
        return "❌ Error: No research goal set. Please set a research goal first.", "", ""

    return persist_cycle_result(
        current_research_goal,
        execute_cycle(current_research_goal, global_context, supervisor),
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
    return f"<p><strong>Evidence Sources:</strong> {rendered_sources}</p>"


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


def run_cycle_with_progress(
    timeout_seconds: int = CYCLE_TIMEOUT_SECONDS,
    poll_seconds: float = CYCLE_PROGRESS_INTERVAL_SECONDS,
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

    def worker():
        result["value"] = execute_cycle(
            run_goal,
            run_context,
            run_supervisor,
            progress_callback=progress_events.put,
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started = time.monotonic()
    iteration = global_context.iteration_number + 1

    while thread.is_alive():
        drain_progress_events()
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            timeout_duration = format_timeout_duration(timeout_seconds)
            timeout_status = (
                f"⚠️ Cycle {iteration} timed out after {timeout_duration}. "
                "The app stopped waiting for the model provider instead of leaving the run spinning."
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
            yield (
                f"{timeout_status}\n{to_bold('Run ID:')} {saved_run['run_id']}\n{to_bold('Report:')} {report_file_url(report_path)}",
                timeout_html,
                "",
                format_research_trace_html(live_trace, elapsed_seconds=elapsed),
            )
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
    status, results, references = persist_cycle_result(run_goal, cycle_result)
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
            <p>The model/API reported the following, so some or all hypotheses were not generated:</p>
            <ul style="color: #c0392b;">{items}</ul>
        </div>
        """

    # Process steps in order
    steps = cycle_details.get("steps", {})
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
            search_stats = step_data.get("search_stats", [])
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
                html += "</details>"
            html += f"<p><strong>Generated {len(hypotheses)} new hypotheses:</strong></p>"
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

            # Debug logging
            logger.info(
                f"Proximity data - adjacency_graph keys: {list(adjacency_graph.keys()) if adjacency_graph else 'None'}"
            )
            logger.info(f"Proximity data - nodes count: {len(nodes) if nodes else 0}")
            logger.info(f"Proximity data - edges count: {len(edges) if edges else 0}")

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
            # Debug: log the actual meta_review data structure
            import sys

            print("DEBUG: meta_review step_data =", step_data, file=sys.stderr)
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
    step_order = ["ranking_final", "ranking2", "ranking", "ranking1"]
    for step_name in step_order:
        if step_name in steps and steps[step_name].get("hypotheses"):
            final_hypotheses = steps[step_name]["hypotheses"]
            final_step = step_name
            break

    # Fallback: use step with most hypotheses if no ranking step exists
    if not final_hypotheses:
        max_count = 0
        for sname, sdata in steps.items():
            hypos = sdata.get("hypotheses", [])
            if len(hypos) > max_count:
                final_hypotheses = hypos
                final_step = sname
                max_count = len(hypos)

    # Assertions: final list should not be empty and no duplicate IDs (only for ranking steps)
    ranking_steps = ["ranking_final", "ranking2", "ranking", "ranking1"]
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
        if errors:
            cause = "; ".join(sorted({classify_llm_error(e) for e in errors}))
            no_rank_msg = (
                f"No hypotheses available for final ranking because generation failed: {html_lib.escape(cause)}. "
                "See the details above."
            )
        else:
            no_rank_msg = "No hypotheses available for final ranking. This may indicate an error in the workflow."
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
    """Render the exact sources supplied to the Generation Agent."""
    import html as html_lib

    sources = cycle_details.get("steps", {}).get("generation", {}).get("sources", [])
    if not isinstance(sources, list) or not sources:
        return "<p>No retrieved evidence was used for generation.</p>"

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
        if source_type == "web" and not source.get("full_text_indexed"):
            library_status = "Retrieved web content used directly"
        elif source.get("full_text_indexed"):
            chunks_used = int(source.get("full_text_chunks_used") or 0)
            library_status = f"Indexed in local ChromaDB; {chunks_used} relevant full-text chunk(s) used"
        else:
            library_status = "Abstract-only evidence"
        content_label = "Web content" if source_type == "web" else "Abstract"
        author_line = f"<p><strong>Authors:</strong> {authors}</p>" if authors else ""
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
            <p>
                <a href="{source_url}" target="_blank">📄 View source</a>{pdf_link}
            </p>
        </div>
        """

    return html


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

    with gr.Blocks(title="AI Co-Scientist - Hypothesis Evolution System") as demo:
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
        gr.Markdown("# 🔬 AI Co-Scientist - Hypothesis Evolution System")
        gr.Markdown("Generate, review, rank, and evolve research hypotheses using AI agents.")

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
            research_goal, llm_model, num_hypotheses, generation_temp, reflection_temp, elo_k_factor, top_k_hypotheses
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
            yield (
                f"{status_msg}\n\nStarting cycle with a {format_timeout_duration(CYCLE_TIMEOUT_SECONDS)} limit.",
                format_research_trace_html([], running=True),
                "<p>Starting cycle...</p>",
                "",
                history_html(),
                gr.update(choices=history_run_choices(), value=None),
                gr.update(choices=sidebar_run_choices(), value=None),
            )
            for status, results, references, research_trace in run_cycle_with_progress():
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
                [
                    "Develop a closed-loop multi-agent AI framework to dynamically allocate 5G slice bandwidth during traffic spikes"
                ],
                [
                    "Create a machine learning orchestrator that injects post-quantum cryptographic keys into active 5G network slices without increasing latency"
                ],
                [
                    "Develop a real-time anomaly detector for Open-RAN architectures that spots and blocks malicious, rogue network apps"
                ],
                ["Improve 5G battery life by optimizing device wake-up sensors"],
                [
                    "Automate the root-cause diagnosis of 5G tower failures by deploying AI agents to read logs and execute patches"
                ],
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
    # Create and launch the Gradio app
    logger.info("Using LM Studio API at %s", get_lmstudio_base_url())
    demo = create_gradio_interface()

    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[str(reports_dir.resolve())],
        theme=getattr(demo, "theme", None),
        css=getattr(demo, "css", None),
    )
