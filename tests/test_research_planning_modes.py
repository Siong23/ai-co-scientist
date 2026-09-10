"""Offline coverage for research-mode planning and downstream routing."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from app.agents_modules.generation_helpers import call_llm_for_search_queries
from app.agents_modules.meta_review import MetaReviewAgent
from app.agents_modules.supervisor import SupervisorAgent
from app.agents_modules.supervisor_planner import SupervisorDecision, SupervisorPlanner
from app.models import ContextMemory, ResearchGoal


def _hypotheses(goal: str) -> list[dict[str, str]]:
    return [
        {
            "hypothesis_id": "primary_hypothesis",
            "role": "primary",
            "statement": "The focal relationship is supported by the available evidence.",
            "goal_quote": goal,
        },
        {
            "hypothesis_id": "alternative_hypothesis",
            "role": "alternative",
            "statement": "A different explanation accounts for the focal relationship.",
            "goal_quote": goal,
        },
        {
            "hypothesis_id": "null_hypothesis",
            "role": "null",
            "statement": "The available evidence does not support the focal relationship.",
            "goal_quote": goal,
        },
    ]


def _planner_payload(mode: str, goal: str, *, hypotheses: list[dict[str, str]] | None = None) -> str:
    payload = {
        "research_goal": goal,
        "research_type": mode,
        "key_entities": ["target"],
        "constraints": [],
        "sub_questions": ["What evidence addresses the target?"],
        "evidence_requirements": ["Source-grounded target evidence"],
        "freshness_requirement": "No special freshness constraint",
        "ambiguities": [],
        "search_strategy": "Search primary and scholarly sources.",
        "provisional_hypotheses": hypotheses or [],
        "competing_candidates": [],
        "competing_explanations": [],
        "comparison_dimensions": [],
        "research_questions": [],
        "topic_dimensions": [],
        "themes": [],
        "controversies": [],
        "evidence_dimensions": [],
        "areas_of_agreement": [],
        "areas_of_disagreement": [],
        "literature_gaps": [],
        "claims": [],
        "risks": [],
        "counterclaims": [],
        "primary_source_checks": [],
        "missing_evidence": [],
    }
    if mode == "comparative":
        payload.update(
            competing_candidates=["candidate A", "candidate B"],
            comparison_dimensions=["effectiveness", "failure modes"],
        )
    elif mode == "exploratory":
        payload.update(
            research_questions=["Which mechanisms warrant investigation?"],
            topic_dimensions=["mechanisms", "boundary conditions"],
            missing_evidence=["Direct measurements of the target"],
        )
    elif mode == "literature_review":
        payload.update(
            themes=["reported effects"],
            controversies=["measurement validity"],
            evidence_dimensions=["study design", "population"],
            areas_of_agreement=["Map findings supported across studies"],
            areas_of_disagreement=["Map conflicting effect estimates"],
            literature_gaps=["Underrepresented populations"],
        )
    elif mode == "due_diligence":
        payload.update(
            claims=["The target performs as represented"],
            risks=["Selection bias"],
            counterclaims=["Reported performance is not reproducible"],
            primary_source_checks=["Verify the underlying study records"],
            missing_evidence=["Independent replication"],
        )
    return json.dumps(payload)


def _query_payload(goal: str, *, with_hypothesis_ids: bool) -> str:
    hypothesis_ids = (
        ["primary_hypothesis", "null_hypothesis", "primary_hypothesis"] if with_hypothesis_ids else [None, None, None]
    )
    return json.dumps(
        {
            "queries": [
                {
                    "query": f"{goal} empirical supporting evidence",
                    "sub_question": "What evidence supports relevant claims?",
                    "purpose": "Support retrieval",
                    "source_type": "academic",
                    "preferred_domains": [],
                    "freshness": None,
                    "evidence_requirement_id": "target_scope",
                    "hypothesis_id": hypothesis_ids[0],
                    "search_intent": "support",
                },
                {
                    "query": f"{goal} contradictory evidence limitations",
                    "sub_question": "What evidence challenges relevant claims?",
                    "purpose": "Counterevidence retrieval",
                    "source_type": "academic",
                    "preferred_domains": [],
                    "freshness": None,
                    "evidence_requirement_id": "target_scope",
                    "hypothesis_id": hypothesis_ids[1],
                    "search_intent": "counterevidence",
                },
                {
                    "query": f"{goal} prior work existing literature",
                    "sub_question": "What prior work addresses the target?",
                    "purpose": "Prior-art retrieval",
                    "source_type": "academic",
                    "preferred_domains": [],
                    "freshness": None,
                    "evidence_requirement_id": "target_scope",
                    "hypothesis_id": hypothesis_ids[2],
                    "search_intent": "prior_art",
                },
            ],
            "required_terms": ["target"],
            "explicit_requirements": [
                {
                    "id": "target_scope",
                    "goal_quote": goal,
                    "evidence_need": "Source-grounded evidence addressing the target",
                }
            ],
            "exploration_directions": ["Inspect boundary conditions"],
        }
    )


@pytest.mark.parametrize(
    ("mode", "mode_field"),
    [
        ("hypothesis_testing", None),
        ("causal", None),
        ("comparative", "comparison_dimensions"),
        ("exploratory", "research_questions"),
        ("literature_review", "themes"),
        ("due_diligence", "claims"),
    ],
)
def test_mode_specific_plans_preserve_retrieval_controls(mode, mode_field):
    goal = "Assess target evidence"
    requires_hypotheses = mode in {"hypothesis_testing", "causal"}
    planner = _planner_payload(
        mode,
        goal,
        hypotheses=_hypotheses(goal) if requires_hypotheses else None,
    )
    fidelity = Mock(return_value=(True, "goal anchored"))
    with patch(
        "app.agents.call_llm",
        side_effect=[planner, _query_payload(goal, with_hypothesis_ids=requires_hypotheses)],
    ):
        plan, error = call_llm_for_search_queries(
            goal,
            research_type=mode,
            query_count=3,
            query_fidelity_validator=fidelity,
        )

    assert error is None
    assert plan is not None
    assert plan.research_type == mode
    assert plan.research_plan is not None
    assert plan.research_plan.research_goal == goal
    assert {query.search_intent for query in plan.queries} == {
        "support",
        "counterevidence",
        "prior_art",
    }
    assert [hypothesis.role for hypothesis in plan.provisional_hypotheses] == (
        ["primary", "alternative", "null"] if requires_hypotheses else []
    )
    assert plan.hypothesis_pipeline_enabled is requires_hypotheses
    if mode_field:
        assert getattr(plan.research_plan, mode_field)
    fidelity.assert_called_once_with(plan)


def test_comparative_plan_accepts_competing_explanations_without_named_candidates():
    goal = "Compare target explanations"
    payload = json.loads(_planner_payload("comparative", goal))
    payload["competing_candidates"] = []
    payload["competing_explanations"] = ["explanation A", "explanation B"]
    with patch(
        "app.agents.call_llm",
        side_effect=[json.dumps(payload), _query_payload(goal, with_hypothesis_ids=False)],
    ):
        plan, error = call_llm_for_search_queries(goal, research_type="comparative", query_count=3)

    assert error is None
    assert plan is not None and plan.research_plan is not None
    assert plan.research_plan.competing_candidates == ()
    assert plan.research_plan.competing_explanations == ("explanation A", "explanation B")
    assert not plan.hypothesis_pipeline_enabled


def test_exploratory_planner_repairs_fabricated_hypotheses():
    goal = "Assess target evidence"
    fabricated = _planner_payload("exploratory", goal, hypotheses=_hypotheses(goal))
    corrected = _planner_payload("exploratory", goal)
    with patch(
        "app.agents.call_llm",
        side_effect=[fabricated, corrected, _query_payload(goal, with_hypothesis_ids=False)],
    ) as llm:
        plan, error = call_llm_for_search_queries(goal, research_type="exploratory", query_count=3)

    assert error is None
    assert plan is not None
    assert plan.provisional_hypotheses == ()
    assert not plan.hypothesis_pipeline_enabled
    assert llm.call_count == 3
    assert "must not fabricate provisional hypotheses" in llm.call_args_list[1].args[0]


def _complete_evidence_mode(context: ContextMemory, mode: str = "exploratory") -> None:
    context.research_type = mode
    context.research_plan = {
        "research_goal": "Assess target evidence",
        "research_type": mode,
        "research_questions": ["What evidence addresses the target?"],
        "topic_dimensions": ["mechanisms"],
        "missing_evidence": ["Independent replication"],
        "hypothesis_pipeline_enabled": False,
    }
    context.last_generation_diagnostics = {
        "evidence_retrieval": {"status": "completed"},
        "literature_synthesis": {"status": "completed"},
    }
    context.last_literature_synthesis = {
        "established_findings": [
            {"claim": "One finding is source-grounded.", "source_ids": ["source:1"], "evidence_refs": []}
        ],
        "contradictions": [],
        "knowledge_gaps": ["Independent replication"],
        "analytical_rationale": "The evidence supports further investigation.",
        "warnings": [],
    }


def test_evidence_led_meta_review_summarizes_mode_outputs_not_hypotheses():
    context = ContextMemory(research_type="exploratory")
    _complete_evidence_mode(context)

    result = MetaReviewAgent().summarize_and_feedback(
        context,
        {},
        research_goal=ResearchGoal("Assess target evidence", research_type="exploratory"),
    )

    overview = result["research_overview"]
    assert result["synthesis_mode"] == "mode_aware"
    assert overview["top_ranked_hypotheses"] == []
    assert overview["mode_summary"]["research_type"] == "exploratory"
    assert overview["mode_summary"]["established_findings"][0]["source_ids"] == ["source:1"]
    assert any("exploratory gap" in step for step in overview["suggested_next_steps"])
    assert all("Refine top hypotheses" not in step for step in overview["suggested_next_steps"])


def test_sequential_supervisor_skips_hypothesis_only_agents_for_exploratory_mode():
    context = ContextMemory()
    goal = ResearchGoal("Assess target evidence", research_type="exploratory")
    supervisor = SupervisorAgent(mode="sequential")

    def generate(_goal, ctx, _publish, _details):
        _complete_evidence_mode(ctx)
        return []

    supervisor.step_generation = Mock(side_effect=generate)
    supervisor.step_reflection = Mock()
    supervisor.step_ranking = Mock()
    supervisor.step_evolution = Mock()
    supervisor.step_proximity = Mock()
    supervisor.step_meta_review = Mock(return_value={})

    result = supervisor.run_cycle(goal, context)

    supervisor.step_meta_review.assert_called_once()
    supervisor.step_reflection.assert_not_called()
    supervisor.step_ranking.assert_not_called()
    supervisor.step_evolution.assert_not_called()
    supervisor.step_proximity.assert_not_called()
    assert result["finalization"]["ready"] is True
    assert context.iteration_number == 1


def test_dynamic_supervisor_rewrites_hypothesis_action_when_pipeline_is_disabled():
    context = ContextMemory(research_type="exploratory")
    _complete_evidence_mode(context)
    goal = ResearchGoal("Assess target evidence", research_type="exploratory")
    supervisor = SupervisorAgent(mode="dynamic")
    supervisor.planner.plan_next_action = Mock(
        return_value=SupervisorDecision(action="REFLECT", reasoning="invalid for this mode")
    )
    supervisor.step_reflection = Mock()
    supervisor.step_ranking = Mock()
    supervisor.step_evolution = Mock()
    supervisor.step_proximity = Mock()

    def meta_review(_context, _publish, details, **_kwargs):
        details.setdefault("steps", {})["meta_review"] = {}
        return {}

    supervisor.step_meta_review = Mock(side_effect=meta_review)

    result = supervisor.run_dynamic_cycle(goal, context, max_steps=1, planner_mode="llm")

    assert result["supervisor_decisions"][0]["action"] == "META_REVIEW"
    assert result["supervisor_decisions"][0]["requested_action"] == "REFLECT"
    supervisor.step_reflection.assert_not_called()
    supervisor.step_ranking.assert_not_called()
    supervisor.step_evolution.assert_not_called()
    supervisor.step_proximity.assert_not_called()


def test_restored_state_forces_evidence_refresh_before_llm_planning():
    context = ContextMemory(research_type="exploratory")
    _complete_evidence_mode(context)
    context.resume_state["requires_evidence_refresh"] = True
    planner = SupervisorPlanner()

    with patch("app.agents.call_llm") as llm:
        decision = planner.plan_next_action(
            context,
            ResearchGoal("Assess target evidence", research_type="exploratory"),
            planner_mode="llm",
        )

    assert decision.action == "GENERATE"
    assert "fresh transient evidence" in decision.reasoning
    llm.assert_not_called()
