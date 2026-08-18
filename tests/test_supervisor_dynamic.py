"""Offline unit tests for the Dynamic Supervisor Agent and Planner."""

from __future__ import annotations

from unittest.mock import Mock, patch

from app.agents import (
    ContextMemory,
    Hypothesis,
    ResearchGoal,
    SupervisorAgent,
    SupervisorPlanner,
    assess_supervisor_state,
    decide_action_heuristically,
    parse_supervisor_decision,
)
from app.models import ReflectionReport


def _sample_hypothesis(hypothesis_id: str, rec: str | None = None, elo: float = 1200.0) -> Hypothesis:
    h = Hypothesis(hypothesis_id, f"Title {hypothesis_id}", f"Statement for {hypothesis_id}")
    h.elo_score = elo
    if rec is not None:
        h.reflection_report = ReflectionReport(
            alignment_score=8,
            novelty_score=8,
            feasibility_score=8,
            plausibility_score=8,
            testability_score=8,
            evidence_quality_score=8,
            expected_research_value_score=8,
            recommendation=rec,
        )
    return h


def test_assess_supervisor_state():
    context = ContextMemory()
    h1 = _sample_hypothesis("H1", "ACCEPT", elo=1250.0)
    h2 = _sample_hypothesis("H2", "REVISE", elo=1150.0)
    h3 = _sample_hypothesis("H3", None, elo=1200.0)

    context.add_hypothesis(h1)
    context.add_hypothesis(h2)
    context.add_hypothesis(h3)

    goal = ResearchGoal(description="Test goal", num_hypotheses=3)
    state = assess_supervisor_state(context, goal, history=[], steps_remaining=4)

    assert state["total_hypotheses"] == 3
    assert state["active_hypotheses_count"] == 3
    assert state["accepted_count"] == 1
    assert state["revise_count"] == 1
    assert state["unreviewed_count"] == 1
    assert state["elo_spread"] == 100.0
    assert state["steps_remaining"] == 4
    assert state["accepted_ids"] == ["H1"]
    assert state["unreviewed_ids"] == ["H3"]


def test_parse_supervisor_decision():
    # Valid markdown json
    raw = '```json\n{\n  "action": "RANK",\n  "reasoning": "Compare candidates",\n  "target_hypothesis_ids": ["H1", "H2"]\n}\n```'
    decision = parse_supervisor_decision(raw)
    assert decision is not None
    assert decision.action == "RANK"
    assert decision.reasoning == "Compare candidates"
    assert decision.target_hypothesis_ids == ["H1", "H2"]

    # Direct json
    raw2 = '{"action": "GENERATE", "reasoning": "Need more ideas", "target_hypothesis_ids": []}'
    decision2 = parse_supervisor_decision(raw2)
    assert decision2 is not None
    assert decision2.action == "GENERATE"
    assert decision2.reasoning == "Need more ideas"

    # Invalid action
    raw_invalid = '{"action": "INVALID_ACTION", "reasoning": "foo"}'
    assert parse_supervisor_decision(raw_invalid) is None

    # Broken JSON
    assert parse_supervisor_decision("not json at all") is None


def test_decide_action_heuristically():
    goal = ResearchGoal(description="Goal", num_hypotheses=2)

    # Empty context -> GENERATE
    context = ContextMemory()
    state = assess_supervisor_state(context, goal, history=[])
    d1 = decide_action_heuristically(state, goal)
    assert d1.action == "GENERATE"

    # Unreviewed hypotheses -> REFLECT
    context.add_hypothesis(_sample_hypothesis("H1", None))
    state = assess_supervisor_state(context, goal, history=[{"action": "GENERATE"}])
    d2 = decide_action_heuristically(state, goal)
    assert d2.action == "REFLECT"

    # Accepted hypotheses -> RANK
    context.hypotheses.clear()
    context.add_hypothesis(_sample_hypothesis("H1", "ACCEPT"))
    context.add_hypothesis(_sample_hypothesis("H2", "ACCEPT"))
    state = assess_supervisor_state(context, goal, history=[{"action": "GENERATE"}, {"action": "REFLECT"}])
    d3 = decide_action_heuristically(state, goal)
    assert d3.action == "RANK"

    # After ranking -> EVOLVE
    state = assess_supervisor_state(
        context, goal, history=[{"action": "GENERATE"}, {"action": "REFLECT"}, {"action": "RANK"}]
    )
    d4 = decide_action_heuristically(state, goal)
    assert d4.action == "EVOLVE"

    # After evolution -> PROXIMITY
    state = assess_supervisor_state(
        context,
        goal,
        history=[
            {"action": "GENERATE"},
            {"action": "REFLECT"},
            {"action": "RANK"},
            {"action": "EVOLVE"},
            {"action": "RANK"},
        ],
    )
    d5 = decide_action_heuristically(state, goal)
    assert d5.action == "PROXIMITY"

    # After proximity -> META_REVIEW
    state = assess_supervisor_state(
        context,
        goal,
        history=[
            {"action": "GENERATE"},
            {"action": "REFLECT"},
            {"action": "RANK"},
            {"action": "EVOLVE"},
            {"action": "RANK"},
            {"action": "PROXIMITY"},
        ],
    )
    d6 = decide_action_heuristically(state, goal)
    assert d6.action == "META_REVIEW"

    # All done -> FINALIZE
    state = assess_supervisor_state(
        context,
        goal,
        history=[
            {"action": "GENERATE"},
            {"action": "REFLECT"},
            {"action": "RANK"},
            {"action": "EVOLVE"},
            {"action": "RANK"},
            {"action": "PROXIMITY"},
            {"action": "META_REVIEW"},
        ],
    )
    d7 = decide_action_heuristically(state, goal)
    assert d7.action == "FINALIZE"


def test_supervisor_planner_llm_and_fallback():
    planner = SupervisorPlanner()
    context = ContextMemory()
    goal = ResearchGoal(description="Goal", num_hypotheses=2)

    # Mock successful LLM decision
    llm_payload = '{"action": "GENERATE", "reasoning": "LLM decided to generate", "target_hypothesis_ids": []}'
    with patch("app.agents.call_llm", return_value=llm_payload):
        decision = planner.plan_next_action(context, goal, planner_mode="llm")
        assert decision.action == "GENERATE"
        assert decision.reasoning == "LLM decided to generate"

    # Mock failed LLM -> fallback to heuristic
    with patch("app.agents.call_llm", side_effect=RuntimeError("API error")):
        decision_fb = planner.plan_next_action(context, goal, planner_mode="auto")
        assert decision_fb.action == "GENERATE"
        assert "initiating literature discovery" in decision_fb.reasoning


def test_supervisor_run_dynamic_cycle():
    supervisor = SupervisorAgent()
    supervisor.generation_agent = Mock()
    h1 = _sample_hypothesis("H1", "ACCEPT")
    h2 = _sample_hypothesis("H2", "ACCEPT")
    supervisor.generation_agent.generate_new_hypotheses.return_value = ([h1, h2], [])
    supervisor.generation_agent.rag_retriever.last_query_plan = None
    supervisor.generation_agent.rag_retriever.last_query_fidelity = []
    supervisor.generation_agent.rag_retriever.last_search_stats = []

    supervisor.reflection_agent = Mock()
    supervisor.ranking_agent = Mock()
    supervisor.evolution_agent = Mock()
    supervisor.evolution_agent.evolve_hypotheses.return_value = []
    supervisor.proximity_agent = Mock()
    prox_mock_result = {
        "graph": {
            "adjacency_graph": {},
            "nodes": [],
            "edges": [],
        },
        "adjacency_graph": {},
        "nodes": [],
        "edges": [],
        "clusters": {},
        "near_duplicates": [],
    }
    supervisor.proximity_agent.get_proximity_analysis.return_value = prox_mock_result
    supervisor.proximity_agent.build_proximity_graph.return_value = prox_mock_result
    supervisor.meta_review_agent = Mock()
    supervisor.meta_review_agent.summarize_and_feedback.return_value = {
        "meta_review_critique": ["Quality critique"],
        "research_overview": {"top_ranked_hypotheses": [], "suggested_next_steps": []},
    }

    progress_events = []
    context = ContextMemory()
    goal = ResearchGoal(description="Test goal", num_hypotheses=2)

    details = supervisor.run_dynamic_cycle(
        goal,
        context,
        progress_callback=progress_events.append,
        max_steps=6,
        planner_mode="heuristic",
    )

    assert "supervisor_decisions" in details
    assert len(details["supervisor_decisions"]) > 0
    assert details["iteration"] == 1
    assert "meta_review" in details["steps"]
    assert any(event.get("step") == "supervisor_planning" for event in details["research_trace"])


def test_assess_supervisor_state_with_proximity_data():
    context = ContextMemory()
    h1 = _sample_hypothesis("H1", "ACCEPT", elo=1300.0)
    h2 = _sample_hypothesis("H2", "ACCEPT", elo=1200.0)
    context.add_hypothesis(h1)
    context.add_hypothesis(h2)

    goal = ResearchGoal(description="Test goal", num_hypotheses=2)
    proximity_data = {
        "diversity_score": 0.42,
        "clusters": {0: ["H1"], 1: ["H2"]},
    }
    state = assess_supervisor_state(context, goal, history=[], proximity_data=proximity_data)

    assert state["diversity_score"] == 0.42
    assert state["cluster_count"] == 2
    assert state["unranked_accepted_count"] == 2
    assert state["unranked_accepted_ids"] == ["H1", "H2"]


def test_supervisor_dynamic_cycle_safe_rank_filtering():
    supervisor = SupervisorAgent()
    supervisor.generation_agent = Mock()
    h_accept1 = _sample_hypothesis("H_acc1", "ACCEPT")
    h_accept2 = _sample_hypothesis("H_acc2", "ACCEPT")
    h_revise = _sample_hypothesis("H_rev", "REVISE")
    supervisor.generation_agent.generate_new_hypotheses.return_value = ([h_accept1, h_accept2, h_revise], [])
    supervisor.generation_agent.rag_retriever.last_query_plan = None
    supervisor.generation_agent.rag_retriever.last_query_fidelity = []
    supervisor.generation_agent.rag_retriever.last_search_stats = []

    supervisor.reflection_agent = Mock()
    supervisor.ranking_agent = Mock()
    supervisor.evolution_agent = Mock()
    supervisor.evolution_agent.evolve_hypotheses.return_value = []
    supervisor.proximity_agent = Mock()
    prox_mock_result = {
        "graph": {
            "adjacency_graph": {},
            "nodes": [],
            "edges": [],
        },
        "adjacency_graph": {},
        "nodes": [],
        "edges": [],
        "clusters": {},
        "near_duplicates": [],
    }
    supervisor.proximity_agent.get_proximity_analysis.return_value = prox_mock_result
    supervisor.proximity_agent.build_proximity_graph.return_value = prox_mock_result
    supervisor.meta_review_agent = Mock()
    supervisor.meta_review_agent.summarize_and_feedback.return_value = {
        "meta_review_critique": [],
        "research_overview": {"top_ranked_hypotheses": [], "suggested_next_steps": []},
    }

    context = ContextMemory()
    goal = ResearchGoal(description="Test goal", num_hypotheses=2)

    # In planner, force decision to rank H_rev and H_acc1
    decision_with_invalid_target = '{"action": "RANK", "reasoning": "Test ranking with invalid target", "target_hypothesis_ids": ["H_rev", "H_acc1"]}'
    with patch("app.agents.call_llm", return_value=decision_with_invalid_target):
        details = supervisor.run_dynamic_cycle(
            goal,
            context,
            max_steps=3,
            planner_mode="auto",
        )
    assert "supervisor_decisions" in details

    # Verify that ranking_agent.run_tournament was only called with ACCEPTED hypotheses
    for call in supervisor.ranking_agent.run_tournament.call_args_list:
        ranked_hypos = call.args[0]
        for h in ranked_hypos:
            rec = str(getattr(h.reflection_report, "recommendation", "")).upper()
            assert rec == "ACCEPT", f"Non-accepted hypothesis {h.hypothesis_id} was sent to ranking"
