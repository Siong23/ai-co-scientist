"""Dynamic planning and scheduling engine for the Supervisor Agent.

Implements state assessment, dynamic action selection (via LLM and heuristic fallback),
and task orchestration modeled after the AI Co-Scientist architecture.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ..config import config
from ..models import ContextMemory, ResearchGoal
from ..utils import logger, redact_secrets
from .generation_helpers import _call_llm

SupervisorAction = Literal[
    "GENERATE",
    "REFLECT",
    "RANK",
    "EVOLVE",
    "PROXIMITY",
    "META_REVIEW",
    "FINALIZE",
]

SUPERVISOR_ACTIONS: tuple[SupervisorAction, ...] = (
    "GENERATE",
    "REFLECT",
    "RANK",
    "EVOLVE",
    "PROXIMITY",
    "META_REVIEW",
    "FINALIZE",
)

_ACTION_DESCRIPTIONS: dict[str, str] = {
    "GENERATE": "Discover literature evidence and generate new candidate hypotheses.",
    "REFLECT": "Perform scientific reflection (novelty, feasibility, safety) and route hypotheses.",
    "RANK": "Conduct pairwise tournament matches to establish or refine Elo rankings.",
    "EVOLVE": "Apply strategic mutations and recombinations to top-ranked hypotheses.",
    "PROXIMITY": "Compute hypothesis similarity topology, cluster topics, and prune near-duplicates.",
    "META_REVIEW": "Synthesize global cross-hypothesis critique and strategic research directions.",
    "FINALIZE": "Conclude the exploration cycle and finalize research outputs.",
}


@dataclass
class SupervisorDecision:
    """Action decision made by the Supervisor planner."""

    action: str
    reasoning: str
    target_hypothesis_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "target_hypothesis_ids": list(self.target_hypothesis_ids),
            "confidence": self.confidence,
        }


def assess_supervisor_state(
    context: ContextMemory,
    research_goal: ResearchGoal,
    history: Sequence[Mapping[str, Any]] = (),
    steps_remaining: int = 5,
    proximity_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess the current state of hypothesis exploration to inform planning."""
    active_hypos = context.get_active_hypotheses()
    all_hypos = list(context.hypotheses.values())

    accepted_hypos = []
    revise_hypos = []
    unreviewed_hypos = []

    for h in active_hypos:
        report = getattr(h, "reflection_report", None)
        rec = str(getattr(report, "recommendation", "UNREVIEWED")).strip().upper()
        if rec == "ACCEPT":
            accepted_hypos.append(h)
        elif rec == "REVISE":
            revise_hypos.append(h)
        else:
            unreviewed_hypos.append(h)

    elo_scores = [h.elo_score for h in active_hypos]
    elo_spread = (max(elo_scores) - min(elo_scores)) if len(elo_scores) >= 2 else 0.0

    actions_taken = [str(item.get("action", "")).upper() for item in history if item.get("action")]

    tournament_hypos = set()
    for match in context.tournament_results:
        if isinstance(match, dict) and match.get("outcome") in {"A", "B", "TIE"}:
            if "hypothesis_a" in match:
                tournament_hypos.add(match["hypothesis_a"])
            if "hypothesis_b" in match:
                tournament_hypos.add(match["hypothesis_b"])

    unranked_accepted = [h for h in accepted_hypos if h.hypothesis_id not in tournament_hypos]

    diversity_score = None
    cluster_count = 0
    if proximity_data:
        diversity_score = proximity_data.get("diversity_score")
        clusters = proximity_data.get("clusters")
        if isinstance(clusters, dict):
            cluster_count = len(clusters)

    supervisor_state = getattr(context, "supervisor_state", {}) or {}
    snapshots = list(supervisor_state.get("elo_snapshots", []))
    convergence_config = config.get("supervisor", {}).get("convergence", {})
    min_snapshots = max(2, int(convergence_config.get("min_snapshots", 2)))
    convergence_threshold = max(0.0, float(convergence_config.get("top_elo_delta", 2.0)))
    recent_snapshots = snapshots[-min_snapshots:]
    top_elo_delta = None
    ratings_converged = False
    if len(recent_snapshots) >= min_snapshots:
        candidate_sets = [set((snapshot.get("ratings") or {}).keys()) for snapshot in recent_snapshots]
        if candidate_sets and all(candidate_set == candidate_sets[0] for candidate_set in candidate_sets[1:]):
            top_scores = [float(snapshot.get("top_elo", 1200.0)) for snapshot in recent_snapshots]
            top_elo_delta = max(top_scores) - min(top_scores)
            ratings_converged = top_elo_delta <= convergence_threshold

    return {
        "total_hypotheses": len(all_hypos),
        "active_hypotheses_count": len(active_hypos),
        "accepted_count": len(accepted_hypos),
        "revise_count": len(revise_hypos),
        "unreviewed_count": len(unreviewed_hypos),
        "unranked_accepted_count": len(unranked_accepted),
        "tournament_comparisons": sum(
            match.get("outcome") in {"A", "B", "TIE"} for match in context.tournament_results
        ),
        "elo_spread": round(elo_spread, 2),
        "evolution_attempts": len(getattr(context, "last_evolution_attempts", [])),
        "actions_taken": actions_taken,
        "steps_remaining": steps_remaining,
        "accepted_ids": [h.hypothesis_id for h in accepted_hypos],
        "unreviewed_ids": [h.hypothesis_id for h in unreviewed_hypos],
        "unranked_accepted_ids": [h.hypothesis_id for h in unranked_accepted],
        "diversity_score": diversity_score,
        "cluster_count": cluster_count,
        "ranking_snapshot_count": len(snapshots),
        "top_elo_delta": round(top_elo_delta, 3) if top_elo_delta is not None else None,
        "ratings_converged": ratings_converged,
        "top_hypothesis_ids": [
            h.hypothesis_id for h in sorted(accepted_hypos, key=lambda x: x.elo_score, reverse=True)[:3]
        ],
    }


def evaluate_finalization_readiness(
    context: ContextMemory,
    research_goal: ResearchGoal,
) -> dict[str, Any]:
    """Return an auditable quality gate for concluding a research cycle."""
    gate_config = config.get("supervisor", {}).get("finalization", {})
    min_accepted = max(1, int(gate_config.get("min_accepted_hypotheses", 2)))
    min_accepted = min(min_accepted, max(1, int(research_goal.num_hypotheses)))
    min_matches = max(0, int(gate_config.get("min_completed_matches", 1)))
    require_evidence = bool(gate_config.get("require_evidence", True))
    min_overall_confidence = float(gate_config.get("min_overall_confidence", 5.0))
    min_claim_confidence = float(gate_config.get("min_claim_confidence", 4.0))
    min_audit_score = float(gate_config.get("min_audit_score", 70.0))
    require_successful_evolution = bool(gate_config.get("require_successful_evolution", True))
    min_hypothesis_clusters = max(1, int(gate_config.get("min_hypothesis_clusters", 2)))

    accepted = []
    for hypothesis in context.get_active_hypotheses():
        report = getattr(hypothesis, "reflection_report", None)
        if str(getattr(report, "recommendation", "")).strip().upper() == "ACCEPT":
            accepted.append(hypothesis)
    accepted.sort(key=lambda item: (item.elo_score, item.hypothesis_id), reverse=True)
    finalists = accepted[:min_accepted]

    completed_matches = [
        match
        for match in context.tournament_results
        if isinstance(match, dict) and match.get("outcome") in {"A", "B", "TIE"}
    ]
    ranked_ids = {
        str(match.get(key)) for match in completed_matches for key in ("hypothesis_a", "hypothesis_b") if match.get(key)
    }
    missing_evidence = []
    low_confidence_finalists = []
    unsupported_claim_finalists = []
    failed_audit_finalists = []
    for hypothesis in finalists:
        evidence_ids = {str(source_id) for source_id in getattr(hypothesis, "evidence_source_ids", [])}
        stored_source_ids = {
            str(source.get("source_id"))
            for source in (getattr(hypothesis, "evidence_sources", []) or [])
            if isinstance(source, dict) and source.get("source_id")
        }
        if require_evidence and (not evidence_ids or not evidence_ids.intersection(stored_source_ids)):
            missing_evidence.append(hypothesis.hypothesis_id)

        report = getattr(hypothesis, "reflection_report", None)
        claims = list(getattr(report, "claims", []) or [])
        overall_confidence = float(getattr(report, "overall_confidence", 1.0) or 1.0)
        claim_confidences = [float(getattr(claim, "confidence", 1.0)) for claim in claims]
        if (
            overall_confidence < min_overall_confidence
            or not claim_confidences
            or min(claim_confidences) < min_claim_confidence
        ):
            low_confidence_finalists.append(hypothesis.hypothesis_id)
        if require_evidence and any(not getattr(claim, "supporting_evidence", []) for claim in claims):
            unsupported_claim_finalists.append(hypothesis.hypothesis_id)

        audit_verdict = str(getattr(hypothesis, "audit_verdict", "") or "").upper()
        audit_score = getattr(hypothesis, "audit_score", None)
        if audit_verdict and audit_verdict not in {"PASS", "PASS_WITH_WARNINGS"}:
            failed_audit_finalists.append(hypothesis.hypothesis_id)
        elif audit_score is not None and float(audit_score) < min_audit_score:
            failed_audit_finalists.append(hypothesis.hypothesis_id)
    unranked_finalists = [
        hypothesis.hypothesis_id for hypothesis in finalists if hypothesis.hypothesis_id not in ranked_ids
    ]

    reasons = []
    if len(accepted) < min_accepted:
        reasons.append(f"Need at least {min_accepted} accepted hypotheses; found {len(accepted)}.")
    if len(completed_matches) < min_matches:
        reasons.append(f"Need at least {min_matches} completed tournament match(es); found {len(completed_matches)}.")
    if unranked_finalists:
        reasons.append("Finalists missing completed tournament comparisons: " + ", ".join(unranked_finalists) + ".")
    if missing_evidence:
        reasons.append("Finalists missing verified evidence citations: " + ", ".join(missing_evidence) + ".")
    if low_confidence_finalists:
        reasons.append("Finalists below claim-confidence thresholds: " + ", ".join(low_confidence_finalists) + ".")
    if unsupported_claim_finalists:
        reasons.append(
            "Finalists have core claims without supporting evidence: " + ", ".join(unsupported_claim_finalists) + "."
        )
    if failed_audit_finalists:
        reasons.append("Finalists failed the generation quality audit: " + ", ".join(failed_audit_finalists) + ".")

    evolution_attempts = list(getattr(context, "last_evolution_attempts", []) or [])
    successful_evolution = any(
        isinstance(attempt, dict) and attempt.get("status") == "accepted" for attempt in evolution_attempts
    )
    if require_successful_evolution and not successful_evolution:
        reasons.append("Need at least one successful Evolution candidate before finalization.")

    proximity = getattr(context, "proximity_analysis", {}) or {}
    clusters = proximity.get("clusters", {}) if isinstance(proximity, dict) else {}
    cluster_count = len(clusters) if isinstance(clusters, (dict, list)) else 0
    if len(finalists) >= 2 and cluster_count < min_hypothesis_clusters:
        reasons.append(f"Need at least {min_hypothesis_clusters} hypothesis clusters; found {cluster_count}.")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "accepted_count": len(accepted),
        "required_accepted_count": min_accepted,
        "completed_matches": len(completed_matches),
        "required_completed_matches": min_matches,
        "finalist_ids": [hypothesis.hypothesis_id for hypothesis in finalists],
        "unranked_finalist_ids": unranked_finalists,
        "missing_evidence_ids": missing_evidence,
        "low_confidence_finalist_ids": low_confidence_finalists,
        "unsupported_claim_finalist_ids": unsupported_claim_finalists,
        "failed_audit_finalist_ids": failed_audit_finalists,
        "successful_evolution": successful_evolution,
        "cluster_count": cluster_count,
        "required_cluster_count": min_hypothesis_clusters,
    }


def build_supervisor_planning_prompt(
    state: Mapping[str, Any],
    research_goal: ResearchGoal,
    available_actions: Sequence[str] = SUPERVISOR_ACTIONS,
) -> str:
    """Build prompt for LLM-based supervisor action planning."""
    action_list = "\n".join(f"- {act}: {_ACTION_DESCRIPTIONS.get(act, '')}" for act in available_actions)

    diversity_info = ""
    if state.get("diversity_score") is not None:
        diversity_info = (
            f"- Semantic diversity score: {state.get('diversity_score'):.2f} "
            f"across {state.get('cluster_count', 1)} cluster(s)\n"
        )

    return (
        "You are the autonomous Supervisor Agent for the AI Co-Scientist research system.\n"
        "Your task is to analyze the current hypothesis exploration state and decide the SINGLE next best action.\n\n"
        f"RESEARCH GOAL:\n{research_goal.description}\n\n"
        "CURRENT EXPLORATION STATE:\n"
        f"- Active hypotheses: {state.get('active_hypotheses_count', 0)} (Total created: {state.get('total_hypotheses', 0)})\n"
        f"- Reflection accepted: {state.get('accepted_count', 0)}, Need revision: {state.get('revise_count', 0)}, Unreviewed: {state.get('unreviewed_count', 0)}\n"
        f"- Unranked accepted candidates: {state.get('unranked_accepted_count', 0)}\n"
        f"- Tournament comparisons completed: {state.get('tournament_comparisons', 0)} (Elo spread: {state.get('elo_spread', 0)})\n"
        f"- Ranking snapshots: {state.get('ranking_snapshot_count', 0)}; ratings converged: "
        f"{state.get('ratings_converged', False)}; top Elo delta: {state.get('top_elo_delta')}\n"
        f"{diversity_info}"
        f"- Evolution attempts: {state.get('evolution_attempts', 0)}\n"
        f"- Actions already executed in this session: {', '.join(state.get('actions_taken', [])) or 'None'}\n"
        f"- Steps remaining in compute budget: {state.get('steps_remaining', 1)}\n\n"
        f"AVAILABLE ACTIONS:\n{action_list}\n\n"
        "DECISION GUIDELINES:\n"
        "1. If there are 0 active hypotheses, choose GENERATE.\n"
        "2. If unreviewed hypotheses exist, choose REFLECT before ranking or evolving them.\n"
        "3. If accepted hypotheses exist but haven't been compared in tournaments, choose RANK.\n"
        "4. If accepted candidates exist and diversity is low (diversity_score < 0.35), choose GENERATE or EVOLVE.\n"
        "5. If strong accepted hypotheses exist and haven't been evolved, choose EVOLVE.\n"
        "6. If hypotheses are evolved, REFLECT and RANK them to determine their quality.\n"
        "7. If multiple hypotheses exist and proximity hasn't run, choose PROXIMITY to prune duplicates.\n"
        "8. When budget is almost exhausted (steps remaining <= 1) or research is well explored, choose META_REVIEW.\n"
        "9. If META_REVIEW has finished and budget is done, choose FINALIZE.\n\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "```json\n"
        "{\n"
        '  "action": "<ONE_OF_THE_AVAILABLE_ACTIONS>",\n'
        '  "reasoning": "<Concise 1-2 sentence rationale for this decision>",\n'
        '  "target_hypothesis_ids": ["<id1>", "<id2>"]\n'
        "}\n"
        "```"
    )


def parse_supervisor_decision(
    raw_response: str,
    available_actions: Sequence[str] = SUPERVISOR_ACTIONS,
) -> SupervisorDecision | None:
    """Parse JSON supervisor decision from LLM response text."""
    text = raw_response.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        obj_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if obj_match:
            text = obj_match.group(0).strip()

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        action = str(data.get("action", "")).strip().upper()
        if action not in available_actions:
            return None
        reasoning = str(data.get("reasoning", "")).strip() or f"Executing {action} step."
        target_ids = data.get("target_hypothesis_ids")
        if not isinstance(target_ids, list):
            target_ids = []
        target_ids = [str(tid).strip() for tid in target_ids if str(tid).strip()]
        return SupervisorDecision(
            action=action,
            reasoning=reasoning,
            target_hypothesis_ids=target_ids,
            confidence=1.0,
        )
    except Exception:
        return None


def decide_action_heuristically(
    state: Mapping[str, Any],
    research_goal: ResearchGoal,
) -> SupervisorDecision:
    """Heuristic fallback policy for selecting the next supervisor action."""
    active_count = state.get("active_hypotheses_count", 0)
    unreviewed = state.get("unreviewed_count", 0)
    unreviewed_ids = list(state.get("unreviewed_ids", []))
    accepted = state.get("accepted_count", 0)
    accepted_ids = list(state.get("accepted_ids", []))
    unranked_accepted = int(state.get("unranked_accepted_count", 0))
    actions = state.get("actions_taken", [])
    steps_left = state.get("steps_remaining", 1)
    target_count = getattr(research_goal, "num_hypotheses", 2)
    ranking_snapshots = int(state.get("ranking_snapshot_count", 0))
    ratings_converged = bool(state.get("ratings_converged", False))
    convergence_config = config.get("supervisor", {}).get("convergence", {})
    max_ranking_batches = max(1, int(convergence_config.get("max_ranking_batches_before_evolution", 2)))

    # 1. No hypotheses -> Generate
    if active_count == 0:
        return SupervisorDecision(
            action="GENERATE",
            reasoning="No active hypotheses found in context; initiating literature discovery and generation.",
        )

    # 2. Unreviewed hypotheses exist -> Reflect
    if unreviewed > 0:
        return SupervisorDecision(
            action="REFLECT",
            reasoning=f"Found {unreviewed} unreviewed candidate hypothesis(es); performing quality and feasibility reflection.",
            target_hypothesis_ids=unreviewed_ids,
        )

    # 3. If accepted count is too low and we haven't just generated -> Generate more
    if accepted < max(1, target_count // 2) and (not actions or actions[-1] != "GENERATE"):
        return SupervisorDecision(
            action="GENERATE",
            reasoning=f"Accepted hypotheses ({accepted}) below target; expanding literature search to discover new candidates.",
        )

    # 4. If accepted >= 2 and haven't ranked yet -> Rank
    if accepted >= 2 and "RANK" not in actions:
        return SupervisorDecision(
            action="RANK",
            reasoning="Running tournament comparisons to establish initial Elo rankings among accepted candidates.",
            target_hypothesis_ids=accepted_ids,
        )

    # 5. Refine tournament scores before evolution. Once ratings plateau (or
    # the bounded batch budget is reached), invest compute in new candidates.
    if accepted >= 2 and "EVOLVE" not in actions:
        ranking_batches = actions.count("RANK")
        if ranking_snapshots and ranking_batches < max_ranking_batches and not ratings_converged:
            return SupervisorDecision(
                action="RANK",
                reasoning="Elo ratings have not converged; running another bounded tournament batch before evolution.",
                target_hypothesis_ids=accepted_ids,
            )
        return SupervisorDecision(
            action="EVOLVE",
            reasoning=(
                "Elo ratings have plateaued; evolving the leading hypotheses to escape the current quality plateau."
                if ratings_converged
                else "Tournament batch budget reached; evolving top-ranked hypotheses to explore grounded refinements."
            ),
        )

    # 6. If evolved hypotheses were accepted and need ranking -> Rank
    if "EVOLVE" in actions and unranked_accepted > 0:
        return SupervisorDecision(
            action="RANK",
            reasoning="Conducting tournament matches to evaluate newly evolved hypotheses against prior champions.",
            target_hypothesis_ids=accepted_ids,
        )

    # 7. If haven't mapped proximity and deduplicated -> Proximity
    if "PROXIMITY" not in actions and active_count >= 2:
        return SupervisorDecision(
            action="PROXIMITY",
            reasoning="Constructing proximity graph to cluster related topics and deactivate near-duplicates.",
        )

    # 8. If near end of budget or haven't done meta-review -> Meta-Review
    if steps_left <= 1 or "META_REVIEW" not in actions:
        return SupervisorDecision(
            action="META_REVIEW",
            reasoning="Synthesizing research overview, cross-hypothesis critiques, and actionable next steps.",
        )

    # 9. Conclude
    return SupervisorDecision(
        action="FINALIZE",
        reasoning="Exploration goals achieved and review finalized; completing research cycle.",
    )


class SupervisorPlanner:
    """Dynamic planning controller for the Supervisor Agent."""

    def __init__(self, default_mode: str = "auto"):
        self.default_mode = default_mode

    def plan_next_action(
        self,
        context: ContextMemory,
        research_goal: ResearchGoal,
        history: Sequence[Mapping[str, Any]] = (),
        steps_remaining: int = 5,
        planner_mode: str | None = None,
        proximity_data: Mapping[str, Any] | None = None,
    ) -> SupervisorDecision:
        """Decide the next supervisor action using LLM with heuristic fallback."""
        mode = (planner_mode or self.default_mode or "auto").lower()
        state = assess_supervisor_state(
            context,
            research_goal,
            history=history,
            steps_remaining=steps_remaining,
            proximity_data=proximity_data,
        )

        if mode == "heuristic":
            return decide_action_heuristically(state, research_goal)

        # Attempt LLM-based planning
        try:
            prompt = build_supervisor_planning_prompt(state, research_goal)
            response = _call_llm(prompt, temperature=0.2)
            decision = parse_supervisor_decision(response)
            if decision is not None:
                return decision
        except Exception as exc:
            logger.debug(
                "Supervisor LLM planner fallback triggered: %s",
                redact_secrets(str(exc)),
            )

        return decide_action_heuristically(state, research_goal)
