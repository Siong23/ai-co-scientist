"""Offline tests for paper-aligned hypothesis evolution strategies."""

from copy import deepcopy
from unittest.mock import Mock, patch

from app.agents import (
    EVOLUTION_STRATEGIES,
    EvolutionAgent,
    SupervisorAgent,
    parse_evolution_response,
)
from app.models import ContextMemory, Hypothesis, ResearchGoal


def _goal(*, top_k: int = 2) -> ResearchGoal:
    return ResearchGoal(
        "Find a testable mechanism for treatment resistance.",
        constraints={"budget": "small laboratory"},
        llm_model="offline-model",
        top_k_hypotheses=top_k,
    )


def _context() -> tuple[ContextMemory, Hypothesis, Hypothesis]:
    context = ContextMemory()
    first = Hypothesis("H1", "Transport mechanism", "Transporter X causes treatment resistance.")
    first.elo_score = 1320
    first.review_comments = ["The causal step is underspecified."]
    first.evidence_source_ids = ["paper:1", "paper:2"]
    first.evidence_sources = [
        {"source_id": "paper:1", "title": "Transport study", "abstract": "X increases after treatment."},
        {"source_id": "paper:2", "title": "Resistance study", "abstract": "Blocking X restores sensitivity."},
    ]
    first.references = [{"id": "paper:1"}]

    second = Hypothesis("H2", "Stress mechanism", "Stress pathway Y causes treatment resistance.")
    second.elo_score = 1280
    second.evidence_source_ids = ["paper:2", "paper:3"]
    second.evidence_sources = [
        {"source_id": "paper:2", "title": "Resistance study", "abstract": "Blocking X restores sensitivity."},
        {"source_id": "paper:3", "title": "Stress study", "abstract": "Y is required for survival."},
    ]
    second.references = [{"id": "paper:3"}]

    context.add_hypothesis(first)
    context.add_hypothesis(second)
    return context, first, second


def test_parse_evolution_response_requires_a_complete_json_hypothesis():
    assert parse_evolution_response('```json\n{"title": "Child", "hypothesis": "A testable child."}\n```') == {
        "title": "Child",
        "text": "A testable child.",
    }
    assert parse_evolution_response('{"title": "Missing hypothesis"}') is None
    assert parse_evolution_response("not json") is None
    assert parse_evolution_response("Error: provider unavailable") is None


def test_parse_evolution_response_accepts_reasoning_before_fenced_json():
    response = """
    <think>I should return one testable candidate.</think>
    Here is the requested result:
    ```json
    {"Title": "Child", "Hypothesis": "A testable child."}
    ```
    """

    assert parse_evolution_response(response) == {
        "title": "Child",
        "text": "A testable child.",
    }


def test_near_duplicate_evolution_is_repaired_once():
    context, first, _ = _context()
    agent = EvolutionAgent(
        strategies=("simplification",),
        max_candidates_per_cycle=1,
        quality_repair_attempts=1,
    )
    responses = [
        '{"title": "Rephrased parent", "hypothesis": "Transporter X causes treatment resistance."}',
        '{"title": "Decisive inhibition test", "hypothesis": "Transiently inhibit X before treatment; restored sensitivity would isolate X as the causal resistance mechanism."}',
    ]

    with patch("app.agents.call_llm", side_effect=responses) as call_llm:
        evolved = agent.evolve_hypotheses(context, _goal())

    assert len(evolved) == 1
    assert evolved[0].title == "Decisive inhibition test"
    assert evolved[0].parent_ids == [first.hypothesis_id]
    assert call_llm.call_count == 2
    assert "near_duplicate_parent:H1" in call_llm.call_args_list[1].args[0]
    assert "do not merely rephrase" in call_llm.call_args_list[1].args[0]
    assert context.last_evolution_attempts == [
        {
            "strategy": "simplification",
            "parent_ids": ["H1"],
            "status": "accepted",
            "reason": "accepted_after_quality_repair",
            "quality_rejections": ["near_duplicate_parent:H1"],
        }
    ]


def test_stitched_combination_is_rejected_without_entering_tournament():
    context, _, _ = _context()
    agent = EvolutionAgent(
        strategies=("combination",),
        max_candidates_per_cycle=1,
        quality_repair_attempts=0,
    )
    response = (
        '{"title": "Combined", '
        '"hypothesis": "Combination of:<br>1. Transporter X causes resistance.<br>2. Stress Y causes resistance."}'
    )

    with patch("app.agents.call_llm", return_value=response):
        evolved = agent.evolve_hypotheses(context, _goal())

    assert evolved == []
    assert context.last_evolution_attempts[0]["reason"] == "stitched_combination"
    assert context.last_evolution_attempts[0]["quality_rejections"] == ["stitched_combination"]


def test_evolution_creates_new_children_with_lineage_and_inherited_evidence():
    context, first, second = _context()
    original_first = deepcopy(first.to_dict())
    original_second = deepcopy(second.to_dict())
    agent = EvolutionAgent(
        strategies=("combination", "feasibility", "out_of_box"),
        max_candidates_per_cycle=3,
    )
    responses = [
        '{"title": "Combined child", "hypothesis": "X activates Y, which can be tested by dual inhibition."}',
        '{"title": "Feasible child", "hypothesis": "Inhibit X before treatment and measure restored sensitivity."}',
        '{"title": "Divergent child", "hypothesis": "Transient membrane tension independently drives resistance."}',
    ]

    with patch("app.agents.call_llm", side_effect=responses) as call_llm:
        evolved = agent.evolve_hypotheses(context, _goal())

    assert [child.evolution_strategy for child in evolved] == ["combination", "feasibility", "out_of_box"]
    assert evolved[0].parent_ids == ["H1", "H2"]
    assert evolved[1].parent_ids == ["H1"]
    assert evolved[2].parent_ids == ["H1", "H2"]
    assert evolved[0].evidence_source_ids == ["paper:1", "paper:2", "paper:3"]
    assert [source["source_id"] for source in evolved[0].evidence_sources] == ["paper:1", "paper:2", "paper:3"]
    assert evolved[0].references == [{"id": "paper:1"}, {"id": "paper:3"}]
    assert first.to_dict() == original_first
    assert second.to_dict() == original_second
    assert call_llm.call_count == 3
    assert call_llm.call_args_list[0].kwargs == {
        "temperature": 0.7,
        "model": "offline-model",
        "max_tokens": 2048,
        "reasoning": "off",
    }
    assert "never edit" in call_llm.call_args_list[0].args[0]
    assert "Combination" not in first.title


def test_strategy_library_rotates_across_iterations():
    context, _, _ = _context()
    context.iteration_number = 1
    agent = EvolutionAgent(strategies=EVOLUTION_STRATEGIES, max_candidates_per_cycle=2)
    responses = [
        '{"title": "Simpler", "hypothesis": "A single intervention tests X."}',
        '{"title": "Grounded", "hypothesis": "Existing evidence supports testing X first."}',
    ]

    with patch("app.agents.call_llm", side_effect=responses) as call_llm:
        evolved = agent.evolve_hypotheses(context, _goal())

    assert [child.evolution_strategy for child in evolved] == ["simplification", "grounding"]
    prompts = [call.args[0] for call in call_llm.call_args_list]
    assert "Evolution strategy: simplification" in prompts[0]
    assert "Evolution strategy: grounding" in prompts[1]
    assert "Transport study" in prompts[1]


def test_failed_evolution_calls_keep_parents_without_stitched_fallback():
    context, first, second = _context()
    agent = EvolutionAgent(strategies=("combination", "feasibility"), max_candidates_per_cycle=2)

    with patch("app.agents.call_llm", return_value="Error: provider unavailable"):
        evolved = agent.evolve_hypotheses(context, _goal())

    assert evolved == []
    assert first.is_active
    assert second.is_active
    assert context.last_evolution_attempts == [
        {
            "strategy": "combination",
            "parent_ids": ["H1", "H2"],
            "status": "rejected",
            "reason": "llm_error",
            "transport_retries": 2,
            "response_excerpt": "Error: provider unavailable",
        },
        {
            "strategy": "feasibility",
            "parent_ids": ["H1"],
            "status": "rejected",
            "reason": "llm_error",
            "transport_retries": 2,
            "response_excerpt": "Error: provider unavailable",
        },
    ]


def test_transient_evolution_transport_failure_is_retried():
    context, _, _ = _context()
    agent = EvolutionAgent(
        strategies=("simplification",),
        max_candidates_per_cycle=1,
        transport_retry_attempts=2,
    )

    with patch(
        "app.agents.call_llm",
        side_effect=[
            "Error: provider temporarily unavailable",
            '{"title": "Recovered child", "hypothesis": "A decisive intervention tests X causally."}',
        ],
    ) as call_llm:
        evolved = agent.evolve_hypotheses(context, _goal())

    assert [hypothesis.title for hypothesis in evolved] == ["Recovered child"]
    assert call_llm.call_count == 2
    assert context.last_evolution_attempts[0]["status"] == "accepted"
    assert context.last_evolution_attempts[0]["transport_retries"] == 1


def test_supervisor_handles_nested_proximity_result():
    context, _, _ = _context()
    supervisor = SupervisorAgent()
    supervisor.generation_agent = Mock()
    supervisor.generation_agent.generate_new_hypotheses.return_value = ([], [])
    supervisor.generation_agent.rag_retriever.last_search_stats = []
    supervisor.reflection_agent = Mock()
    supervisor.ranking_agent = Mock()
    supervisor.evolution_agent = Mock()
    supervisor.evolution_agent.evolve_hypotheses.return_value = []
    supervisor.proximity_agent = Mock()
    supervisor.proximity_agent.get_proximity_analysis.return_value = {
        "graph": {
            "adjacency_graph": {},
            "nodes": ["H1", "H2"],
            "edges": [],
        },
        "clusters": {},
        "cluster_members": {},
        "largest_clusters": [],
        "connectivity": {"H1": 0, "H2": 0},
        "highly_connected": [],
        "isolated": ["H1", "H2"],
    }
    supervisor.meta_review_agent = Mock()
    supervisor.meta_review_agent.summarize_and_feedback.return_value = {}

    details = supervisor.run_cycle(_goal(), context)

    assert details["steps"]["proximity"]["nodes"] == ["H1", "H2"]
    assert details["steps"]["proximity"]["edges"] == []


def test_supervisor_exposes_evolution_attempts_in_cycle_details():
    context, _, _ = _context()
    progress_events = []
    supervisor = SupervisorAgent()
    supervisor.generation_agent = Mock()
    supervisor.generation_agent.generate_new_hypotheses.return_value = ([], [])
    supervisor.generation_agent.rag_retriever.last_search_stats = []
    supervisor.reflection_agent = Mock()
    supervisor.ranking_agent = Mock()
    supervisor.evolution_agent = EvolutionAgent(
        strategies=("combination",),
        max_candidates_per_cycle=1,
    )
    supervisor.proximity_agent = Mock()
    supervisor.proximity_agent.get_proximity_analysis.return_value = {
        "graph": {
            "adjacency_graph": {},
            "nodes": [],
            "edges": [],
        },
        "clusters": {},
        "cluster_members": {},
        "largest_clusters": [],
        "connectivity": {},
        "highly_connected": [],
        "isolated": [],
    }
    supervisor.meta_review_agent = Mock()
    supervisor.meta_review_agent.summarize_and_feedback.return_value = {}

    with patch("app.agents.call_llm", return_value="not json"):
        details = supervisor.run_cycle(_goal(), context, progress_callback=progress_events.append)

    assert details["steps"]["evolution"] == {
        "hypotheses": [],
        "attempts": [
            {
                "strategy": "combination",
                "parent_ids": ["H1", "H2"],
                "status": "rejected",
                "reason": "no_json_object",
                "response_excerpt": "not json",
            }
        ],
    }
    assert [event["step"] for event in details["research_trace"]] == [
        "generation",
        "reflection",
        "ranking1",
        "evolution",
        "ranking2",
        "proximity",
        "meta_review",
    ]
    assert all(event["status"] in {"completed", "warning"} for event in details["research_trace"])
    assert any(event["status"] == "running" for event in progress_events)
    assert progress_events[-1]["step"] == "meta_review"
    assert progress_events[-1]["status"] == "completed"


def test_evolution_resolves_parent_evidence_ids_from_context_sources():
    context = ContextMemory()
    parent = Hypothesis("H1", "Seed", "A seed hypothesis.")
    parent.evidence_source_ids = ["paper:1"]
    context.last_retrieved_sources = [
        {
            "source_id": "paper:1",
            "title": "Retrieved evidence",
            "summary": "The mechanism was observed.",
        }
    ]
    context.add_hypothesis(parent)
    agent = EvolutionAgent(strategies=("grounding",), max_candidates_per_cycle=1)

    with patch(
        "app.agents.call_llm",
        return_value='{"title": "Grounded", "hypothesis": "Evidence supports a direct test."}',
    ) as call_llm:
        evolved = agent.evolve_hypotheses(context, _goal(top_k=1))

    assert "Retrieved evidence" in call_llm.call_args.args[0]
    assert "The mechanism was observed." in call_llm.call_args.args[0]
    assert evolved[0].evidence_sources == context.last_retrieved_sources
    assert parent.evidence_sources == []


def test_single_parent_runs_only_unary_refinement_strategies():
    context = ContextMemory()
    parent = Hypothesis("H1", "Seed", "A seed hypothesis.")
    context.add_hypothesis(parent)
    agent = EvolutionAgent(strategies=EVOLUTION_STRATEGIES, max_candidates_per_cycle=3)
    responses = [
        '{"title": "Feasible", "hypothesis": "A feasible hypothesis."}',
        '{"title": "Simple", "hypothesis": "A simple hypothesis."}',
        '{"title": "Grounded", "hypothesis": "A grounded hypothesis."}',
    ]

    with patch("app.agents.call_llm", side_effect=responses):
        evolved = agent.evolve_hypotheses(context, _goal(top_k=1))

    assert [child.evolution_strategy for child in evolved] == ["feasibility", "simplification", "grounding"]
    assert all(child.parent_ids == ["H1"] for child in evolved)


def test_evolution_injects_meta_review_feedback():
    context = ContextMemory()
    parent = Hypothesis("H1", "Seed", "A seed hypothesis.")
    context.add_hypothesis(parent)
    context.meta_review_feedback = [
        {
            "meta_review_critique": ["Explore orthogonal mechanisms."],
            "research_overview": {
                "suggested_next_steps": ["Use out_of_box strategy."],
            },
        }
    ]
    agent = EvolutionAgent(strategies=("feasibility",), max_candidates_per_cycle=1)

    with patch(
        "app.agents.call_llm",
        return_value='{"title": "Feasible", "hypothesis": "A feasible hypothesis addressing critiques."}',
    ) as call_llm:
        agent.evolve_hypotheses(context, _goal(top_k=1))

    prompt = call_llm.call_args.args[0]
    assert "Prior cycle meta-review feedback to address:" in prompt
    assert "Explore orthogonal mechanisms." in prompt
    assert "Use out_of_box strategy." in prompt

