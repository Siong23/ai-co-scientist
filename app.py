import logging
import os
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import gradio as gr

from app.agents import SupervisorAgent

# Import the existing app components
from app.config import config
from app.models import ContextMemory, ResearchGoal
from app.run_store import delete_run, get_reports_dir, history_html, list_runs, report_file_url, save_run, write_report
from app.utils import (
    classify_llm_error,
    fetch_lmstudio_models,
    get_lmstudio_base_url,
    get_lmstudio_model,
    logger,
)

# Global state for the Gradio app
global_context = ContextMemory()
supervisor = SupervisorAgent()
current_research_goal: Optional[ResearchGoal] = None
available_models: List[str] = []
CONFIGURED_LLM_MODEL = get_lmstudio_model()
SAFE_FALLBACK_LLM_MODEL = CONFIGURED_LLM_MODEL or "-- Select Model --"
CYCLE_TIMEOUT_SECONDS = int(os.getenv("CO_SCIENTIST_CYCLE_TIMEOUT_SECONDS", "900"))
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


def refresh_history_view() -> Tuple[str, Dict[str, Any], str]:
    """Refresh the history table and delete-run dropdown."""
    return history_html(), gr.update(choices=history_run_choices(), value=None), ""


def delete_history_run(selected_run_id: Optional[str]) -> Tuple[str, str, Dict[str, Any]]:
    """Delete the selected saved run and refresh the history display."""
    if not selected_run_id:
        return "Select a saved run to delete.", history_html(), gr.update(choices=history_run_choices(), value=None)

    deleted = delete_run(selected_run_id)
    message = f"Deleted saved run {selected_run_id}." if deleted else f"Saved run {selected_run_id} was not found."
    return message, history_html(), gr.update(choices=history_run_choices(), value=None)


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

        status_msg = f"✅ Research goal set successfully!\n\n**Goal:** {description}\n**Model:** {current_research_goal.llm_model or 'Default'}\n**Hypotheses per cycle:** {num_hypotheses}"

        return status_msg, "Ready to run first cycle. Click 'Run Cycle' to begin."

    except Exception as e:
        error_msg = f"❌ Error setting research goal: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""


def execute_cycle(
    research_goal: ResearchGoal,
    context: ContextMemory,
    cycle_supervisor: SupervisorAgent,
) -> Dict[str, Any]:
    """Run a cycle against the supplied state and return display-ready results."""
    import datetime

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
        logger.info(f"Running cycle {iteration}")

        # Run the cycle
        cycle_details = cycle_supervisor.run_cycle(research_goal, context)

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
        if errors:
            categories = sorted({classify_llm_error(e) for e in errors})
            cause = "; ".join(categories)
            if produced_any:
                status_msg = f"⚠️ Cycle {iteration} completed with errors ({cause}). Log: {log_file}"
            else:
                status_msg = (
                    f"⚠️ Cycle {iteration} could not generate hypotheses — {cause}. "
                    f"See the results panel for details. Log: {log_file}"
                )
        else:
            status_msg = f"✅ Cycle {iteration} completed successfully! Log: {log_file}"

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
        return {
            "status": error_msg,
            "results_html": "",
            "references_html": "",
            "cycle_details": {"iteration": context.iteration_number + 1, "steps": {}, "errors": [error_msg]},
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
    status_msg = f"{cycle_result['status']}\nRun ID: {saved_run['run_id']}\nReport: {report_file_url(report_path)}"
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
        return f"{timeout_seconds:g} seconds"
    minutes = timeout_seconds / 60
    if minutes.is_integer():
        return f"{int(minutes)} minutes"
    return f"{minutes:.1f} minutes"


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
    """Render validated hypothesis source IDs as clickable arXiv links."""
    import html as html_lib

    available_source_ids = {
        str(source.get("source_id") or f"arXiv:{source.get('arxiv_id')}")
        for source in generation_sources
        if source.get("source_id") or source.get("arxiv_id")
    }
    evidence_source_ids = hypothesis.get("evidence_source_ids", [])
    if not isinstance(evidence_source_ids, list):
        evidence_source_ids = []

    links = []
    for source_id in dict.fromkeys(evidence_source_ids):
        if not isinstance(source_id, str) or source_id not in available_source_ids:
            continue
        arxiv_id = source_id.removeprefix("arXiv:")
        href = f"https://arxiv.org/abs/{quote(arxiv_id, safe='/.-')}"
        links.append(
            f'<a href="{html_lib.escape(href, quote=True)}" '
            'target="_blank" rel="noopener noreferrer">'
            f"{html_lib.escape(source_id)}</a>"
        )

    rendered_sources = ", ".join(links) if links else "None recorded"
    return (
        "<p><strong>Evidence Sources:</strong> "
        f"{rendered_sources}</p>"
    )


def run_cycle_with_progress(
    timeout_seconds: int = CYCLE_TIMEOUT_SECONDS,
    poll_seconds: float = CYCLE_PROGRESS_INTERVAL_SECONDS,
):
    """Run a cycle in the background and show results once it finishes."""
    global global_context

    if not current_research_goal:
        yield "❌ Error: No research goal set. Please set a research goal first.", "", ""
        return

    run_goal = current_research_goal
    run_context = deepcopy(global_context)
    run_supervisor = SupervisorAgent()
    result: Dict[str, Dict[str, Any]] = {}

    def worker():
        result["value"] = execute_cycle(run_goal, run_context, run_supervisor)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started = time.monotonic()
    iteration = global_context.iteration_number + 1

    while thread.is_alive():
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            timeout_duration = format_timeout_duration(timeout_seconds)
            timeout_status = (
                f"⚠️ Cycle {iteration} timed out after {timeout_duration}. "
                "The app stopped waiting for the model provider instead of leaving the run spinning."
            )
            timeout_html = timeout_results_html(timeout_seconds)
            saved_run = save_run(
                research_goal=run_goal,
                cycle_details={
                    "iteration": iteration,
                    "steps": {},
                    "errors": [timeout_status],
                },
                status=timeout_status,
                references_html="",
                results_html=timeout_html,
                log_file="",
            )
            report_path = write_report(saved_run)
            yield (
                f"{timeout_status}\nRun ID: {saved_run['run_id']}\nReport: {report_file_url(report_path)}",
                timeout_html,
                "",
            )
            return

        status = (
            f"⏳ Cycle {iteration} is running.\n"
            f"Elapsed: {format_timeout_duration(elapsed)}.\n"
            "Active work: generating, reviewing, ranking, and evolving hypotheses.\n"
            f"Upper limit: {format_timeout_duration(timeout_seconds)}."
        )
        yield status, "<p>Cycle is still running. Results will appear when the cycle completes.</p>", ""
        thread.join(timeout=min(poll_seconds, max(timeout_seconds - elapsed, 0.1)))

    cycle_result = result.get("value")
    if not cycle_result:
        yield "❌ Error: Cycle ended without a result.", "", ""
        return
    if current_research_goal is run_goal:
        global_context = run_context
    yield persist_cycle_result(run_goal, cycle_result)


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
            html += f"<p><strong>Generated {len(hypotheses)} new hypotheses:</strong></p>"
            for i, hypo in enumerate(hypotheses):
                html += f"""
                <div style="border-left: 3px solid #28a745; padding-left: 10px; margin: 10px 0;">
                    <h5>#{i + 1}: {hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p style="white-space: pre-line;">{hypo.get("text", "No description")}</p>
                    {format_evidence_sources_html(hypo, generation_sources)}
                </div>
                """

        elif step_name in ["reflection", "reflection_evolved"]:
            hypotheses = step_data.get("hypotheses", [])
            html += f"<p><strong>Reviewed {len(hypotheses)} hypotheses:</strong></p>"
            for hypo in hypotheses:
                html += f"""
                <div style="border-left: 3px solid #17a2b8; padding-left: 10px; margin: 10px 0;">
                    <h5>{hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                       <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
                    {f"<p><strong>Comments:</strong> {hypo.get('comments', 'No comments')}</p>" if hypo.get("comments") else ""}
                    {format_evidence_sources_html(hypo, generation_sources)}
                </div>
                """

        elif step_name.startswith("ranking"):
            hypotheses = step_data.get("hypotheses", [])
            if hypotheses:
                # Sort by Elo score
                sorted_hypotheses = sorted(hypotheses, key=lambda h: h.get("elo_score", 0), reverse=True)
                html += f"<p><strong>Ranking results ({len(hypotheses)} hypotheses):</strong></p>"
                html += "<ol>"
                for hypo in sorted_hypotheses:
                    html += f"""
                    <li style="margin: 5px 0;">
                        <strong>{hypo.get("title", "Untitled")}</strong> (ID: {hypo.get("id", "Unknown")}) 
                        - Elo: {hypo.get("elo_score", 0):.2f}
                        {format_evidence_sources_html(hypo, generation_sources)}
                    </li>
                    """
                html += "</ol>"

        elif step_name == "evolution":
            hypotheses = step_data.get("hypotheses", [])
            html += f"<p><strong>Evolved {len(hypotheses)} new hypotheses by combining top performers:</strong></p>"
            for hypo in hypotheses:
                html += f"""
                <div style="border-left: 3px solid #ffc107; padding-left: 10px; margin: 10px 0;">
                    <h5>{hypo.get("title", "Untitled")} (ID: {hypo.get("id", "Unknown")})</h5>
                    <p>{hypo.get("text", "No description")}</p>
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
                    <div style="border-left: 3px solid #28a745; padding-left: 10px; margin: 10px 0;">
                        <h6>#{i + 1}: {hypo.get("title", "Untitled")}</h6>
                        <p><strong>ID:</strong> {hypo.get("id", "Unknown")} | 
                           <strong>Elo Score:</strong> {hypo.get("elo_score", 0):.2f}</p>
                        <p><strong>Description:</strong> {hypo.get("text", "No description")}</p>
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
            rank_color = "#28a745" if i < 3 else "#17a2b8" if i < 6 else "#6c757d"
            html += f"""
            <div style="border-left: 4px solid {rank_color}; padding: 15px; margin: 10px 0; background-color: white; border-radius: 5px;">
                <h4>#{i + 1}: {hypo.get("title", "Untitled")}</h4>
                <p><strong>ID:</strong> {hypo.get("id", "Unknown")} | 
                   <strong>Elo Score:</strong> {hypo.get("elo_score", 0):.2f}</p>
                <p><strong>Description:</strong> {hypo.get("text", "No description")}</p>
                <p><strong>Novelty:</strong> {hypo.get("novelty_review", "Not assessed")} | 
                   <strong>Feasibility:</strong> {hypo.get("feasibility_review", "Not assessed")}</p>
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
        arxiv_id = html_lib.escape(str(source.get("arxiv_id") or "Unknown"))
        published = html_lib.escape(str(source.get("published") or "Unknown"))
        abstract = html_lib.escape(str(source.get("abstract") or "No abstract")[:300])
        arxiv_url = html_lib.escape(
            str(source.get("arxiv_url") or "#"),
            quote=True,
        )
        pdf_url = html_lib.escape(
            str(source.get("pdf_url") or "#"),
            quote=True,
        )
        html += f"""
        <div style="border: 1px solid #e0e0e0; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #fafafa;">
            <h4>{title}</h4>
            <p><strong>Authors:</strong> {authors}</p>
            <p><strong>arXiv ID:</strong> {arxiv_id} |
               <strong>Published:</strong> {published}</p>
            <p><strong>Abstract:</strong> {abstract}...</p>
            <p>
                <a href="{arxiv_url}" target="_blank">📄 View on arXiv</a> |
                <a href="{pdf_url}" target="_blank">📁 Download PDF</a>
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

    with gr.Blocks(
        title="Open AI Co-Scientist - Hypothesis Evolution System",
        theme=gr.themes.Soft(),
        css="""
        .status-box {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .orange { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .blue { background-color: #d1ecf1; border: 1px solid #bee5eb; }

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
        """,
    ) as demo:
        # Header
        gr.Markdown("# 🔬 Open AI Co-Scientist - Hypothesis Evolution System")
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
                gr.Markdown("""
                ### 📖 Instructions

                1. **Enter Research Goal**: Describe what you want to research.
                2. **Adjust Settings** (optional): Customize model and parameters.
                3. **Click "Run Cycle"**: The system will set your goal and immediately generate, review, rank, and evolve hypotheses in one step.

                ### 💡 Tips
                - Start LM Studio's local server before running a cycle
                - Load a model in LM Studio, then select it in Advanced Settings
                - Higher generation temperature = more creative ideas
                - Lower reflection temperature = more analytical reviews
                - Each cycle builds on previous results

                **Note:** Runtime depends on your local model size and hardware.
                """)

        with gr.Tabs():
            with gr.Tab("Current Run"):
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

            with gr.Tab("Run History") as run_history_tab:
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
                "<p>Starting cycle...</p>",
                "",
                history_html(),
                gr.update(choices=history_run_choices(), value=None),
            )
            for status, results, references in run_cycle_with_progress():
                yield (
                    f"{status_msg}\n\n{status}",
                    results,
                    references,
                    history_html(),
                    gr.update(choices=history_run_choices(), value=None),
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
            outputs=[status_output, results_output, references_output, history_output, delete_run_dropdown],
        )

        demo.load(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, delete_history_status],
        )
        run_history_tab.select(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, delete_history_status],
        )
        refresh_history_btn.click(
            fn=refresh_history_view,
            inputs=[],
            outputs=[history_output, delete_run_dropdown, delete_history_status],
        )
        delete_history_btn.click(
            fn=delete_history_run,
            inputs=[delete_run_dropdown],
            outputs=[delete_history_status, history_output, delete_run_dropdown],
        )

        # Example inputs
        gr.Examples(
            examples=[
                ["Develop new methods for increasing the efficiency of solar panels"],
                ["Create novel approaches to treat Alzheimer's disease"],
                ["Design sustainable materials for construction"],
                ["Improve machine learning model interpretability"],
                ["Develop new quantum computing algorithms"],
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
    )
