"""Workflow supervisor agent.

Orchestrates multi-agent hypothesis generation, reflection, tournament ranking,
evolution, proximity clustering, and meta-review. Supports both deterministic
sequential cycle pipelines and dynamic LLM-driven planning cycles.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..research_trace import merge_trace_event, normalize_trace_event
from ._compat import _legacy
from .evolution import EvolutionAgent
from .generation import GenerationAgent
from .meta_review import MetaReviewAgent
from .proximity import ProximityAgent
from .ranking import RankingAgent
from .reflection import ReflectionAgent
from .supervisor_planner import (
    SupervisorPlanner,
)

ProgressCallback = Callable[[Dict[str, Any]], None]


def _shorten(value: Any, limit: int = 240) -> str:
    text = " ".join(_legacy.redact_secrets(str(value)).split())
    if len(text) > limit:
        return f"{text[: limit - 3].rstrip()}..."
    return text


def _source_details(sources: List[Mapping[str, Any]]) -> List[str]:
    details = []
    for source in sources[:3]:
        source_id = source.get("source_id") or source.get("id") or "unknown source"
        title = source.get("title") or source_id
        details.append(f"Evidence: {_shorten(title, 150)} ({_shorten(source_id, 80)})")
    return details


def _reflection_details(hypotheses: List[Any]) -> List[str]:
    details = []
    for hypothesis in hypotheses[:4]:
        assessment = (
            f"{hypothesis.hypothesis_id}: novelty {hypothesis.novelty_review or 'UNREVIEWED'}, "
            f"feasibility {hypothesis.feasibility_review or 'UNREVIEWED'}"
        )
        if hypothesis.review_comments:
            assessment += f". {_shorten(hypothesis.review_comments[-1], 220)}"
        details.append(assessment)
    return details


def _reflection_routing(hypotheses: List[Any]) -> Dict[str, List[Any]]:
    """Partition hypotheses by the action requested by Reflection."""
    routed: Dict[str, List[Any]] = {
        "accepted": [],
        "revise": [],
        "unreviewed": [],
    }
    for hypothesis in hypotheses:
        report = getattr(hypothesis, "reflection_report", None)
        recommendation = str(getattr(report, "recommendation", "UNREVIEWED")).strip().upper()
        if recommendation == "ACCEPT":
            routed["accepted"].append(hypothesis)
        elif recommendation == "REVISE":
            routed["revise"].append(hypothesis)
        else:
            routed["unreviewed"].append(hypothesis)
    return routed


def _reflection_routing_summary(routed: Mapping[str, List[Any]]) -> Dict[str, List[str]]:
    """Serialize routing decisions without duplicating full hypotheses."""
    return {name: [hypothesis.hypothesis_id for hypothesis in hypotheses] for name, hypotheses in routed.items()}


def _ranking_details(results: List[Mapping[str, Any]]) -> List[str]:
    details = []
    for result in results[:4]:
        comparison = f"{result.get('hypothesis_a', '?')} vs {result.get('hypothesis_b', '?')}"
        outcome = result.get("outcome", "unknown")
        rationale = _shorten(result.get("reasoning") or "No rationale supplied.", 240)
        details.append(f"{comparison}: {outcome}. {rationale}")
    return details


def _evolution_details(attempts: List[Mapping[str, Any]]) -> List[str]:
    details = []
    for attempt in attempts[:4]:
        strategy = attempt.get("strategy") or "unknown strategy"
        status = attempt.get("status") or "unknown status"
        reason = _shorten(attempt.get("reason") or "No reason supplied.", 200)
        details.append(f"{strategy}: {status} ({reason})")
    return details


def _meta_review_details(overview: Mapping[str, Any]) -> List[str]:
    critiques = list(overview.get("meta_review_critique") or [])[:2]
    next_steps = list((overview.get("research_overview") or {}).get("suggested_next_steps") or [])[:2]
    details = [f"Critique: {_shorten(item, 240)}" for item in critiques]
    details.extend(f"Next step: {_shorten(item, 240)}" for item in next_steps)
    return details


class SupervisorAgent:
    """Orchestrates the Open AI Co-Scientist workflow."""

    def __init__(self, mode: str = "sequential"):
        self.mode = mode
        self.generation_agent = GenerationAgent()
        self.reflection_agent = ReflectionAgent()
        self.ranking_agent = RankingAgent()
        self.evolution_agent = EvolutionAgent()
        self.proximity_agent = ProximityAgent()
        self.meta_review_agent = MetaReviewAgent()
        self.planner = SupervisorPlanner()

    # -------------------------------------------------------------------------
    # Modular Step Methods
    # -------------------------------------------------------------------------

    def step_generation(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        publish: Callable[..., None],
        cycle_details: Dict[str, Any],
    ) -> List[Hypothesis]:
        """Execute evidence discovery and hypothesis generation."""
        _legacy.logger.info("Supervisor Step: Generation")
        phase_started = time.perf_counter()
        publish(
            "generation",
            "running",
            "Discovering evidence and generating hypotheses",
            f"Searching the literature and preparing up to {research_goal.num_hypotheses} evidence-grounded candidates.",
        )
        new_hypotheses, generation_errors = self.generation_agent.generate_new_hypotheses(research_goal, context)
        for nh in new_hypotheses:
            context.add_hypothesis(nh)

        generation_sources = list(context.last_retrieved_sources)
        generation_audits = list(context.last_hypothesis_audits)
        query_plan = getattr(
            self.generation_agent.rag_retriever,
            "last_query_plan",
            None,
        )
        if not isinstance(getattr(query_plan, "queries", None), (list, tuple)):
            query_plan = None
        query_fidelity = getattr(
            self.generation_agent.rag_retriever,
            "last_query_fidelity",
            [],
        )
        if not isinstance(query_fidelity, list):
            query_fidelity = []

        query_plan_details = {
            "provisional_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "role": item.role,
                    "statement": item.statement,
                    "goal_quote": item.goal_quote,
                }
                for item in (query_plan.provisional_hypotheses if query_plan else ())
            ],
            "queries": [
                {
                    "query": item.query,
                    "purpose": item.purpose,
                    "sub_question": item.sub_question,
                    "source_type": item.source_type,
                    "evidence_requirement_id": item.evidence_requirement_id,
                    "hypothesis_id": item.hypothesis_id,
                    "search_intent": item.search_intent,
                }
                for item in (query_plan.queries if query_plan else ())
            ],
        }
        cycle_details.setdefault("steps", {})["generation"] = {
            "hypotheses": [h.to_dict() for h in new_hypotheses],
            "sources": generation_sources,
            "audits": generation_audits,
            "search_stats": list(getattr(self.generation_agent.rag_retriever, "last_search_stats", [])),
            "query_plan": query_plan_details,
            "query_fidelity": list(query_fidelity),
        }

        audit_counts: Dict[str, int] = {}
        for audit in generation_audits:
            verdict = str(audit.get("verdict") or audit.get("status") or "unknown").upper()
            audit_counts[verdict] = audit_counts.get(verdict, 0) + 1
        generation_details = _source_details(generation_sources)
        for hypothesis in query_plan_details["provisional_hypotheses"]:
            generation_details.append(
                f"Provisional {hypothesis['role']} retrieval hypothesis: {hypothesis['statement']}"
            )
        if audit_counts:
            audit_summary = ", ".join(f"{name}: {count}" for name, count in sorted(audit_counts.items()))
            generation_details.append(f"Candidate audits: {audit_summary}")

        publish(
            "generation",
            "warning" if generation_errors else "completed",
            "Discovering evidence and generating hypotheses",
            f"Generated {len(new_hypotheses)} candidate hypotheses from {len(generation_sources)} evidence sources.",
            details=generation_details,
            elapsed_seconds=time.perf_counter() - phase_started,
            sources=generation_sources,
        )

        if generation_errors:
            cycle_details["errors"] = generation_errors

        return new_hypotheses

    def step_reflection(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        publish: Callable[..., None],
        cycle_details: Dict[str, Any],
        target_hypos: Optional[List[Hypothesis]] = None,
        step_name: str = "reflection",
    ) -> Dict[str, List[Hypothesis]]:
        """Execute quality reflection and routing."""
        _legacy.logger.info("Supervisor Step: Reflection (%s)", step_name)
        phase_started = time.perf_counter()
        active_hypos = target_hypos if target_hypos is not None else context.get_active_hypotheses()

        publish(
            step_name,
            "running",
            "Reviewing scientific quality",
            f"Checking novelty, feasibility, and evidence quality for {len(active_hypos)} hypotheses.",
        )
        self.reflection_agent.review_hypotheses(active_hypos, context, research_goal)
        reflection_routing = _reflection_routing(active_hypos)
        rankable_hypos = reflection_routing["accepted"]

        cycle_details.setdefault("steps", {})[step_name] = {
            "hypotheses": [h.to_dict() for h in active_hypos],
            "routing": _reflection_routing_summary(reflection_routing),
        }
        publish(
            step_name,
            "warning" if reflection_routing["unreviewed"] else "completed",
            "Reviewing scientific quality",
            f"Accepted {len(rankable_hypos)} of {len(active_hypos)} reviewed hypotheses for ranking.",
            details=_reflection_details(active_hypos),
            elapsed_seconds=time.perf_counter() - phase_started,
        )
        return reflection_routing

    def step_ranking(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        publish: Callable[..., None],
        cycle_details: Dict[str, Any],
        target_hypos: Optional[List[Hypothesis]] = None,
        new_hypotheses: Optional[List[Hypothesis]] = None,
        step_name: str = "ranking",
    ) -> List[Dict[str, Any]]:
        """Execute pairwise tournament matches and update Elo ratings."""
        _legacy.logger.info("Supervisor Step: Ranking (%s)", step_name)
        phase_started = time.perf_counter()

        if target_hypos is None:
            active_hypos = context.get_active_hypotheses()
            routing = _reflection_routing(active_hypos)
            rankable_hypos = routing["accepted"]
        else:
            rankable_hypos = target_hypos

        publish(
            step_name,
            "running",
            "Comparing hypotheses in tournament",
            f"Running evidence-aware pairwise comparisons for {len(rankable_hypos)} accepted hypotheses.",
        )
        start = len(context.tournament_results)
        rankable_ids = {h.hypothesis_id for h in rankable_hypos}
        rankable_new = [h for h in (new_hypotheses or []) if h.hypothesis_id in rankable_ids]

        self.ranking_agent.run_tournament(
            rankable_hypos,
            context,
            research_goal,
            new_hypotheses=rankable_new,
        )
        ranking_results = context.tournament_results[start:]
        cycle_details.setdefault("steps", {})[step_name] = {
            "hypotheses": [h.to_dict() for h in rankable_hypos],
            "tournament_results": ranking_results,
        }
        abstentions = sum(result.get("outcome") == "ABSTAIN" for result in ranking_results)
        publish(
            step_name,
            "completed",
            "Comparing hypotheses in tournament",
            f"Completed {len(ranking_results)} comparisons with {abstentions} abstentions.",
            details=_ranking_details(ranking_results),
            elapsed_seconds=time.perf_counter() - phase_started,
        )
        return ranking_results

    def step_evolution(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        publish: Callable[..., None],
        cycle_details: Dict[str, Any],
    ) -> List[Hypothesis]:
        """Execute mutation and refinement strategies on top candidates."""
        _legacy.logger.info("Supervisor Step: Evolution")
        phase_started = time.perf_counter()
        publish(
            "evolution",
            "running",
            "Evolving promising hypotheses",
            "Applying configured refinement strategies to the strongest candidates.",
        )
        evolved_hypotheses = self.evolution_agent.evolve_hypotheses(context, research_goal)
        evolution_attempts = list(context.last_evolution_attempts)

        if evolved_hypotheses:
            for eh in evolved_hypotheses:
                context.add_hypothesis(eh)
            cycle_details.setdefault("steps", {})["evolution"] = {
                "hypotheses": [h.to_dict() for h in evolved_hypotheses],
                "attempts": evolution_attempts,
            }
            publish(
                "evolution",
                "completed",
                "Evolving promising hypotheses",
                f"Created {len(evolved_hypotheses)} evolved hypotheses from {len(evolution_attempts)} attempts.",
                details=_evolution_details(evolution_attempts),
                elapsed_seconds=time.perf_counter() - phase_started,
            )
        else:
            cycle_details.setdefault("steps", {})["evolution"] = {
                "hypotheses": [],
                "attempts": evolution_attempts,
            }
            publish(
                "evolution",
                "completed",
                "Evolving promising hypotheses",
                f"No evolved hypotheses were accepted from {len(evolution_attempts)} attempts.",
                details=_evolution_details(evolution_attempts),
                elapsed_seconds=time.perf_counter() - phase_started,
            )
        return evolved_hypotheses

    def step_proximity(
        self,
        context: ContextMemory,
        publish: Callable[..., None],
        cycle_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map hypothesis similarity topology and deactivate near-duplicates."""
        _legacy.logger.info("Supervisor Step: Proximity Analysis")
        phase_started = time.perf_counter()
        publish(
            "proximity",
            "running",
            "Mapping hypothesis relationships",
            "Measuring similarity and organizing related hypotheses into a proximity graph.",
        )
        proximity_result = self.proximity_agent.build_proximity_graph(context)
        near_duplicates = proximity_result.get("near_duplicates", [])

        # Deactivate confirmed near-duplicates, preserving the higher-Elo hypothesis
        deactivated_ids: List[str] = []
        for dup_pair in near_duplicates:
            id_a = dup_pair["id_a"]
            id_b = dup_pair["id_b"]
            hypo_a = context.hypotheses.get(id_a)
            hypo_b = context.hypotheses.get(id_b)
            if hypo_a is None or hypo_b is None:
                continue
            if not hypo_a.is_active or not hypo_b.is_active:
                continue
            loser = hypo_b if hypo_a.elo_score >= hypo_b.elo_score else hypo_a
            loser.is_active = False
            deactivated_ids.append(loser.hypothesis_id)
            _legacy.logger.info(
                "Proximity: deactivated near-duplicate %s (kept %s, confidence %.2f)",
                loser.hypothesis_id,
                id_a if loser is hypo_b else id_b,
                dup_pair.get("confidence", 0.0),
            )

        cycle_details.setdefault("steps", {})["proximity"] = {
            "adjacency_graph": proximity_result.get("adjacency_graph", {}),
            "nodes": proximity_result.get("nodes", []),
            "edges": proximity_result.get("edges", []),
            "clusters": proximity_result.get("clusters", {}),
            "cluster_labels": proximity_result.get("cluster_labels", {}),
            "outliers": proximity_result.get("outliers", []),
            "exemplars": proximity_result.get("exemplars", []),
            "near_duplicates": near_duplicates,
            "deactivated_near_duplicates": deactivated_ids,
            "diversity_score": proximity_result.get("diversity_score", 0.0),
        }
        dup_summary = f" Deactivated {len(deactivated_ids)} near-duplicate(s)." if deactivated_ids else ""
        publish(
            "proximity",
            "completed",
            "Mapping hypothesis relationships",
            (
                f"Mapped {len(proximity_result.get('nodes', []))} hypotheses and "
                f"{len(proximity_result.get('edges', []))} relationships across "
                f"{len(proximity_result.get('clusters', {}))} cluster(s).{dup_summary}"
            ),
            elapsed_seconds=time.perf_counter() - phase_started,
        )
        return proximity_result

    def step_meta_review(
        self,
        context: ContextMemory,
        publish: Callable[..., None],
        cycle_details: Dict[str, Any],
        proximity_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize cross-hypothesis critique and global research overview."""
        _legacy.logger.info("Supervisor Step: Meta-Review")
        phase_started = time.perf_counter()
        publish(
            "meta_review",
            "running",
            "Synthesizing the research review",
            "Summarizing the strongest candidates, remaining weaknesses, and suggested next steps.",
        )
        adjacency = (proximity_result or {}).get("adjacency_graph", {})
        overview = self.meta_review_agent.summarize_and_feedback(
            context,
            adjacency,
            proximity_data=proximity_result,
        )
        cycle_details["meta_review"] = overview
        cycle_details.setdefault("steps", {})["meta_review"] = overview
        publish(
            "meta_review",
            "completed",
            "Synthesizing the research review",
            "Completed the cross-hypothesis critique and research overview.",
            details=_meta_review_details(overview),
            elapsed_seconds=time.perf_counter() - phase_started,
        )
        return overview

    # -------------------------------------------------------------------------
    # Pipeline Orchestration
    # -------------------------------------------------------------------------

    def run_cycle(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        progress_callback: Optional[ProgressCallback] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs a cycle using the configured execution mode."""
        active_mode = (mode or self.mode or "sequential").lower()
        if active_mode == "dynamic":
            return self.run_dynamic_cycle(research_goal, context, progress_callback=progress_callback)

        return self._run_sequential_cycle(research_goal, context, progress_callback=progress_callback)

    def _run_sequential_cycle(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Runs a single deterministic sequential cycle of hypothesis refinement."""
        _legacy.logger.info("--- Starting Cycle %d (Sequential) ---", context.iteration_number + 1)
        research_trace: List[Dict[str, Any]] = []
        cycle_details: Dict[str, Any] = {
            "iteration": context.iteration_number + 1,
            "steps": {},
            "meta_review": {},
            "research_trace": research_trace,
        }

        def publish(
            step: str,
            status: str,
            title: str,
            summary: str,
            *,
            details: Optional[List[str]] = None,
            elapsed_seconds: Optional[float] = None,
            sources: Optional[List[Mapping[str, Any]]] = None,
        ) -> None:
            event: Dict[str, Any] = {
                "step": step,
                "status": status,
                "title": title,
                "summary": summary,
                "details": details or [],
                "sources": sources or [],
                "source_count": len(sources or []),
            }
            if elapsed_seconds is not None:
                event["elapsed_seconds"] = elapsed_seconds
            normalized = normalize_trace_event(event)
            merge_trace_event(research_trace, normalized)
            if progress_callback is not None:
                try:
                    progress_callback(dict(normalized))
                except Exception as exc:
                    _legacy.logger.warning(
                        "Research progress callback failed: %s",
                        _legacy.redact_secrets(str(exc)),
                    )

        # 1. Generation
        new_hypotheses = self.step_generation(research_goal, context, publish, cycle_details)
        active_hypos = context.get_active_hypotheses()

        # 2. Reflection
        reflection_routing = self.step_reflection(
            research_goal, context, publish, cycle_details, target_hypos=active_hypos, step_name="reflection"
        )
        rankable_hypos = reflection_routing["accepted"]

        # 3. Ranking 1
        rankable_ids = {h.hypothesis_id for h in rankable_hypos}
        rankable_new = [h for h in new_hypotheses if h.hypothesis_id in rankable_ids]
        self.step_ranking(
            research_goal,
            context,
            publish,
            cycle_details,
            target_hypos=rankable_hypos,
            new_hypotheses=rankable_new,
            step_name="ranking1",
        )

        # 4. Evolution
        evolved_hypotheses = self.step_evolution(research_goal, context, publish, cycle_details)

        if evolved_hypotheses:
            # 4a. Review Evolved
            self.step_reflection(
                research_goal,
                context,
                publish,
                cycle_details,
                target_hypos=evolved_hypotheses,
                step_name="reflection_evolved",
            )

        # 5. Ranking 2
        active_hypos = context.get_active_hypotheses()
        final_routing = _reflection_routing(active_hypos)
        final_rankable = final_routing["accepted"]
        final_ids = {h.hypothesis_id for h in final_rankable}
        rankable_evolved = [h for h in (evolved_hypotheses or []) if h.hypothesis_id in final_ids]
        self.step_ranking(
            research_goal,
            context,
            publish,
            cycle_details,
            target_hypos=final_rankable,
            new_hypotheses=rankable_evolved,
            step_name="ranking2",
        )

        # 6. Proximity
        proximity_result = self.step_proximity(context, publish, cycle_details)

        # 7. Meta-review
        self.step_meta_review(context, publish, cycle_details, proximity_result=proximity_result)

        context.iteration_number += 1
        _legacy.logger.info("--- Cycle %d Complete ---", context.iteration_number)
        return cycle_details

    def run_dynamic_cycle(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        progress_callback: Optional[ProgressCallback] = None,
        max_steps: int = 8,
        planner_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Runs a dynamic cycle where the Supervisor actively plans and schedules actions."""
        _legacy.logger.info("--- Starting Cycle %d (Dynamic Planning) ---", context.iteration_number + 1)
        research_trace: List[Dict[str, Any]] = []
        supervisor_decisions: List[Dict[str, Any]] = []
        cycle_details: Dict[str, Any] = {
            "iteration": context.iteration_number + 1,
            "steps": {},
            "meta_review": {},
            "research_trace": research_trace,
            "supervisor_decisions": supervisor_decisions,
        }

        def publish(
            step: str,
            status: str,
            title: str,
            summary: str,
            *,
            details: Optional[List[str]] = None,
            elapsed_seconds: Optional[float] = None,
            sources: Optional[List[Mapping[str, Any]]] = None,
        ) -> None:
            event: Dict[str, Any] = {
                "step": step,
                "status": status,
                "title": title,
                "summary": summary,
                "details": details or [],
                "sources": sources or [],
                "source_count": len(sources or []),
            }
            if elapsed_seconds is not None:
                event["elapsed_seconds"] = elapsed_seconds
            normalized = normalize_trace_event(event)
            merge_trace_event(research_trace, normalized)
            if progress_callback is not None:
                try:
                    progress_callback(dict(normalized))
                except Exception as exc:
                    _legacy.logger.warning(
                        "Research progress callback failed: %s",
                        _legacy.redact_secrets(str(exc)),
                    )

        proximity_result: Optional[Dict[str, Any]] = None
        last_new_hypotheses: List[Hypothesis] = []
        last_evolved_hypotheses: List[Hypothesis] = []

        step_count = 0
        while step_count < max_steps:
            steps_remaining = max_steps - step_count

            # Assess state and plan next action
            decision = self.planner.plan_next_action(
                context,
                research_goal,
                history=supervisor_decisions,
                steps_remaining=steps_remaining,
                planner_mode=planner_mode,
                proximity_data=proximity_result,
            )
            decision_dict = decision.to_dict()
            supervisor_decisions.append(decision_dict)

            publish(
                "supervisor_planning",
                "completed",
                f"Supervisor Decision (Step {step_count + 1})",
                f"Selected {decision.action}: {decision.reasoning}",
                details=[
                    f"Action: {decision.action}",
                    f"Rationale: {decision.reasoning}",
                    f"Target IDs: {', '.join(decision.target_hypothesis_ids) or 'All active'}",
                    f"Steps remaining in budget: {steps_remaining - 1}",
                ],
            )

            if decision.action == "FINALIZE":
                _legacy.logger.info("Supervisor decided to finalize the session.")
                break

            elif decision.action == "GENERATE":
                last_new_hypotheses = self.step_generation(research_goal, context, publish, cycle_details)

            elif decision.action == "REFLECT":
                # Check target hypotheses or unreviewed
                target_hypos = None
                if decision.target_hypothesis_ids:
                    target_hypos = [
                        h for h in context.get_active_hypotheses() if h.hypothesis_id in decision.target_hypothesis_ids
                    ]
                self.step_reflection(
                    research_goal, context, publish, cycle_details, target_hypos=target_hypos, step_name="reflection"
                )

            elif decision.action == "RANK":
                active_hypos = context.get_active_hypotheses()
                routing = _reflection_routing(active_hypos)
                rankable_hypos = routing["accepted"]
                target_hypos = rankable_hypos
                if decision.target_hypothesis_ids:
                    filtered_targets = [
                        h for h in rankable_hypos if h.hypothesis_id in decision.target_hypothesis_ids
                    ]
                    if len(filtered_targets) >= 2:
                        target_hypos = filtered_targets

                self.step_ranking(
                    research_goal,
                    context,
                    publish,
                    cycle_details,
                    target_hypos=target_hypos,
                    new_hypotheses=last_new_hypotheses or last_evolved_hypotheses,
                    step_name=f"ranking_{step_count + 1}",
                )

            elif decision.action == "EVOLVE":
                last_evolved_hypotheses = self.step_evolution(research_goal, context, publish, cycle_details)

            elif decision.action == "PROXIMITY":
                proximity_result = self.step_proximity(context, publish, cycle_details)

            elif decision.action == "META_REVIEW":
                self.step_meta_review(context, publish, cycle_details, proximity_result=proximity_result)

            step_count += 1

        # If meta-review was never run, perform a final synthesis
        if "meta_review" not in cycle_details.get("steps", {}):
            if proximity_result is None and len(context.get_active_hypotheses()) >= 2:
                proximity_result = self.step_proximity(context, publish, cycle_details)
            self.step_meta_review(context, publish, cycle_details, proximity_result=proximity_result)

        context.iteration_number += 1
        _legacy.logger.info("--- Cycle %d (Dynamic) Complete ---", context.iteration_number)
        return cycle_details
