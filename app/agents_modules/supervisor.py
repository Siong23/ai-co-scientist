"""Workflow supervisor agent."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..models import ContextMemory, ResearchGoal
from ..research_trace import merge_trace_event, normalize_trace_event
from ._compat import _legacy
from .evolution import EvolutionAgent
from .generation import GenerationAgent
from .meta_review import MetaReviewAgent
from .proximity import ProximityAgent
from .ranking import RankingAgent
from .reflection import ReflectionAgent

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
    return {
        name: [hypothesis.hypothesis_id for hypothesis in hypotheses]
        for name, hypotheses in routed.items()
    }


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

    def __init__(self):
        self.generation_agent = GenerationAgent()
        self.reflection_agent = ReflectionAgent()
        self.ranking_agent = RankingAgent()
        self.evolution_agent = EvolutionAgent()
        self.proximity_agent = ProximityAgent()
        self.meta_review_agent = MetaReviewAgent()

    def run_cycle(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict:
        """Runs a single cycle of hypothesis generation and refinement."""
        _legacy.logger.info("--- Starting Cycle %d ---", context.iteration_number + 1)
        research_trace: List[Dict[str, Any]] = []
        cycle_details = {
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
        _legacy.logger.info("Step 1: Generation")
        phase_started = time.perf_counter()
        publish(
            "generation",
            "running",
            "Discovering evidence and generating hypotheses",
            f"Searching the literature and preparing up to {research_goal.num_hypotheses} evidence-grounded candidates.",
        )
        new_hypotheses, generation_errors = self.generation_agent.generate_new_hypotheses(research_goal, context)
        for nh in new_hypotheses:
            context.add_hypothesis(nh)  # Add to central context
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
        cycle_details["steps"]["generation"] = {
            "hypotheses": [h.to_dict() for h in new_hypotheses],
            "sources": generation_sources,
            "audits": generation_audits,
            "search_stats": list(self.generation_agent.rag_retriever.last_search_stats),
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
                "Provisional "
                f"{hypothesis['role']} retrieval hypothesis: "
                f"{hypothesis['statement']}"
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

        # Propagate LLM errors to top-level errors field for frontend display, so a
        # generation failure surfaces its real cause instead of an empty ranking.
        if generation_errors:
            cycle_details["errors"] = generation_errors

        # Get all active hypotheses for subsequent steps
        active_hypos = context.get_active_hypotheses()

        # 2. Reflection
        _legacy.logger.info("Step 2: Reflection")
        phase_started = time.perf_counter()
        publish(
            "reflection",
            "running",
            "Reviewing scientific quality",
            f"Checking novelty, feasibility, and evidence quality for {len(active_hypos)} active hypotheses.",
        )
        self.reflection_agent.review_hypotheses(active_hypos, context, research_goal)  # Pass research_goal
        reflection_routing = _reflection_routing(active_hypos)
        rankable_hypos = reflection_routing["accepted"]
        cycle_details["steps"]["reflection"] = {
            "hypotheses": [h.to_dict() for h in active_hypos],
            "routing": _reflection_routing_summary(reflection_routing),
        }
        publish(
            "reflection",
            "warning" if reflection_routing["unreviewed"] else "completed",
            "Reviewing scientific quality",
            f"Accepted {len(rankable_hypos)} of {len(active_hypos)} reviewed hypotheses for ranking.",
            details=_reflection_details(active_hypos),
            elapsed_seconds=time.perf_counter() - phase_started,
        )

        # 3. Ranking (Tournament 1)
        _legacy.logger.info("Step 3: Ranking 1")
        phase_started = time.perf_counter()
        publish(
            "ranking1",
            "running",
            "Comparing initial candidates",
            "Running evidence-aware pairwise comparisons and updating tournament scores.",
        )
        start = len(context.tournament_results)
        rankable_ids = {hypothesis.hypothesis_id for hypothesis in rankable_hypos}
        rankable_new_hypotheses = [
            hypothesis for hypothesis in new_hypotheses if hypothesis.hypothesis_id in rankable_ids
        ]
        self.ranking_agent.run_tournament(
            rankable_hypos,
            context,
            research_goal,
            new_hypotheses=rankable_new_hypotheses,
        )
        ranking1_results = context.tournament_results[start:]
        cycle_details["steps"]["ranking1"] = {
            "hypotheses": [h.to_dict() for h in rankable_hypos],
            "tournament_results": ranking1_results,
        }
        ranking1_abstentions = sum(result.get("outcome") == "ABSTAIN" for result in ranking1_results)
        publish(
            "ranking1",
            "completed",
            "Comparing initial candidates",
            f"Completed {len(ranking1_results)} comparisons with {ranking1_abstentions} abstentions.",
            details=_ranking_details(ranking1_results),
            elapsed_seconds=time.perf_counter() - phase_started,
        )

        # 4. Evolution
        _legacy.logger.info("Step 4: Evolution")
        phase_started = time.perf_counter()
        publish(
            "evolution",
            "running",
            "Evolving promising hypotheses",
            "Applying configured refinement strategies to the strongest candidates.",
        )
        evolved_hypotheses = self.evolution_agent.evolve_hypotheses(context, research_goal)  # Pass research_goal
        evolution_attempts = list(context.last_evolution_attempts)
        if evolved_hypotheses:
            for eh in evolved_hypotheses:
                context.add_hypothesis(eh)
            cycle_details["steps"]["evolution"] = {
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
            _legacy.logger.info("Step 4a: Reviewing Evolved Hypotheses")
            phase_started = time.perf_counter()
            publish(
                "reflection_evolved",
                "running",
                "Reviewing evolved hypotheses",
                f"Checking {len(evolved_hypotheses)} evolved candidates before the final tournament.",
            )
            self.reflection_agent.review_hypotheses(evolved_hypotheses, context, research_goal)  # Pass research_goal
            active_hypos = context.get_active_hypotheses()  # Update active list
            evolved_routing = _reflection_routing(evolved_hypotheses)
            # Add explicit step for reviewing evolved hypotheses AFTER evolution
            cycle_details["steps"]["reflection_evolved"] = {
                "hypotheses": [h.to_dict() for h in evolved_hypotheses],
                "routing": _reflection_routing_summary(evolved_routing),
            }
            publish(
                "reflection_evolved",
                "warning" if evolved_routing["unreviewed"] else "completed",
                "Reviewing evolved hypotheses",
                f"Accepted {len(evolved_routing['accepted'])} of {len(evolved_hypotheses)} evolved hypotheses for ranking.",
                details=_reflection_details(evolved_hypotheses),
                elapsed_seconds=time.perf_counter() - phase_started,
            )
        else:
            cycle_details["steps"]["evolution"] = {
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

        # 5. Ranking (Tournament 2 - includes evolved)
        # Recompute the gate after Evolution because active_hypos deliberately
        # retains REVISE candidates for refinement.  Only ACCEPT reports may
        # participate in Elo; otherwise rejected content could leak back into
        # the final tournament merely because it is still active.
        final_routing = _reflection_routing(active_hypos)
        rankable_hypos = final_routing["accepted"]
        rankable_ids = {hypothesis.hypothesis_id for hypothesis in rankable_hypos}
        rankable_evolved_hypotheses = [
            hypothesis for hypothesis in evolved_hypotheses if hypothesis.hypothesis_id in rankable_ids
        ]
        _legacy.logger.info("Step 5: Ranking 2")
        phase_started = time.perf_counter()
        publish(
            "ranking2",
            "running",
            "Running the final tournament",
            f"Comparing {len(rankable_hypos)} Reflection-accepted hypotheses after evolution.",
        )
        start = len(context.tournament_results)
        self.ranking_agent.run_tournament(
            rankable_hypos,
            context,
            research_goal,
            new_hypotheses=rankable_evolved_hypotheses,
        )
        ranking2_results = context.tournament_results[start:]
        cycle_details["steps"]["ranking2"] = {
            "hypotheses": [h.to_dict() for h in rankable_hypos],
            "tournament_results": ranking2_results,
        }
        ranking2_abstentions = sum(result.get("outcome") == "ABSTAIN" for result in ranking2_results)
        publish(
            "ranking2",
            "completed",
            "Running the final tournament",
            f"Completed {len(ranking2_results)} comparisons with {ranking2_abstentions} abstentions.",
            details=_ranking_details(ranking2_results),
            elapsed_seconds=time.perf_counter() - phase_started,
        )

        # 6. Proximity Analysis
        _legacy.logger.info("Step 6: Proximity Analysis")
        phase_started = time.perf_counter()
        publish(
            "proximity",
            "running",
            "Mapping hypothesis relationships",
            "Measuring similarity and organizing related hypotheses into a proximity graph.",
        )
        proximity_result = self.proximity_agent.build_proximity_graph(context)  # Pass context
        cycle_details["steps"]["proximity"] = {
            "adjacency_graph": proximity_result["adjacency_graph"],
            "nodes": proximity_result["nodes"],
            "edges": proximity_result["edges"],
        }
        publish(
            "proximity",
            "completed",
            "Mapping hypothesis relationships",
            f"Mapped {len(proximity_result['nodes'])} hypotheses and {len(proximity_result['edges'])} relationships.",
            elapsed_seconds=time.perf_counter() - phase_started,
        )

        # 7. Meta-review
        _legacy.logger.info("Step 7: Meta-Review")
        phase_started = time.perf_counter()
        publish(
            "meta_review",
            "running",
            "Synthesizing the research review",
            "Summarizing the strongest candidates, remaining weaknesses, and suggested next steps.",
        )
        overview = self.meta_review_agent.summarize_and_feedback(context, proximity_result["adjacency_graph"])
        cycle_details["meta_review"] = overview
        # Add meta-review to steps for consistency
        cycle_details["steps"]["meta_review"] = overview
        publish(
            "meta_review",
            "completed",
            "Synthesizing the research review",
            "Completed the cross-hypothesis critique and research overview.",
            details=_meta_review_details(overview),
            elapsed_seconds=time.perf_counter() - phase_started,
        )

        # Increment iteration number at the end of the cycle
        context.iteration_number += 1
        _legacy.logger.info("--- Cycle %d Complete ---", context.iteration_number)
        return cycle_details
