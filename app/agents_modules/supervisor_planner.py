"""Dynamic planning and scheduling engine for the Supervisor Agent.

Implements state assessment, dynamic action selection (via LLM and heuristic fallback),
and task orchestration modeled after the AI Co-Scientist architecture.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ..models import ContextMemory, ResearchGoal
from ._compat import _legacy

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
        if isinstance(match, dict):
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

    return {
        "total_hypotheses": len(all_hypos),
        "active_hypotheses_count": len(active_hypos),
        "accepted_count": len(accepted_hypos),
        "revise_count": len(revise_hypos),
        "unreviewed_count": len(unreviewed_hypos),
        "unranked_accepted_count": len(unranked_accepted),
        "tournament_comparisons": len(context.tournament_results),
        "elo_spread": round(elo_spread, 2),
        "evolution_attempts": len(getattr(context, "last_evolution_attempts", [])),
        "actions_taken": actions_taken,
        "steps_remaining": steps_remaining,
        "accepted_ids": [h.hypothesis_id for h in accepted_hypos],
        "unreviewed_ids": [h.hypothesis_id for h in unreviewed_hypos],
        "unranked_accepted_ids": [h.hypothesis_id for h in unranked_accepted],
        "diversity_score": diversity_score,
        "cluster_count": cluster_count,
        "top_hypothesis_ids": [
            h.hypothesis_id for h in sorted(accepted_hypos, key=lambda x: x.elo_score, reverse=True)[:3]
        ],
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
    unranked_count = state.get("unranked_accepted_count", 0)
    unranked_ids = list(state.get("unranked_accepted_ids", []))
    actions = state.get("actions_taken", [])
    steps_left = state.get("steps_remaining", 1)
    target_count = getattr(research_goal, "num_hypotheses", 2)

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

    # 5. If ranked and haven't evolved yet -> Evolve
    if accepted >= 2 and "EVOLVE" not in actions:
        return SupervisorDecision(
            action="EVOLVE",
            reasoning="Evolving top-ranked hypotheses to explore mutations, combinations, and grounded refinements.",
        )

    # 6. If evolved hypotheses were accepted and need ranking -> Rank
    if "EVOLVE" in actions and actions.count("RANK") < 2:
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
            response = _legacy.call_llm(prompt, temperature=0.2)
            decision = parse_supervisor_decision(response)
            if decision is not None:
                return decision
        except Exception as exc:
            _legacy.logger.debug(
                "Supervisor LLM planner fallback triggered: %s",
                _legacy.redact_secrets(str(exc)),
            )

        return decide_action_heuristically(state, research_goal)
