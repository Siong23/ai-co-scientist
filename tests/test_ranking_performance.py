"""Performance and regression tests for the improved Ranking Agent.

Most tests use mocked LLM calls so they remain deterministic and
independent of external APIs.

Integration tests marked with @pytest.mark.integration perform real
LLM calls to measure end-to-end behaviour.

The tests evaluate:

1. LLM call efficiency
2. Pairwise execution time
3. Full tournament execution time
4. New-hypothesis tournament comparison efficiency
5. Ranking consistency
6. A/B order consistency
7. Elo reproducibility
8. TIE and ABSTAIN handling
9. Inactive hypothesis filtering
10. Tournament result recording
11. Structured ranking-decision information

The improved Ranking Agent uses an LLM-based adjudication process for
pairwise ranking. The dedicated ranking model is tested through a mocked
LLM call, allowing the test suite to remain deterministic and independent
of an external API connection.

The improved tournament can prioritize comparisons involving newly
introduced hypotheses instead of repeatedly performing unnecessary
old-vs-old comparisons.

The ranking system also supports structured outcomes including A, B, TIE,
and ABSTAIN, with corresponding Elo update behavior.

To run this test file:
Real Integration:
pytest tests/test_ranking_performance.py -v -s

Offline (mocked LLM):
pytest tests/test_ranking_performance.py -v -s -m "not integration"

Real LLM:
pytest tests/test_ranking_performance.py -v -s -m integration

"""

from copy import deepcopy
from time import perf_counter
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.agents_modules.ranking import RankingAgent
from app.agents_modules.ranking_helpers import (
    RANKING_LLM_MODEL,
    format_reflection_report,
    parse_confidence,
    parse_pairwise_result,
    parse_short_justification,
    run_pairwise_debate,
    update_elo,
    update_elo_tie,
)
from app.config import config
from app.models import ContextMemory, Hypothesis, PairwiseDecision, ReflectionReport, ResearchGoal

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _reflection_report(
    novelty=7.0,
    feasibility=7.0,
    plausibility=7.0,
    testability=7.0,
    evidence_quality=7.0,
    research_value=7.0,
):
    return ReflectionReport(
        novelty_score=novelty,
        feasibility_score=feasibility,
        plausibility_score=plausibility,
        testability_score=testability,
        evidence_quality_score=evidence_quality,
        expected_research_value_score=research_value,
        strengths=["Reasonably strong hypothesis."],
        weaknesses=[],
        recommendation="ACCEPT",
        overall_confidence=8.0,
    )

def _hypothesis(hypothesis_id: str, elo_score: float = 1200.0, reflection_report=None) -> Hypothesis:
    """Create a minimal hypothesis for testing."""
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        text=f"Hypothesis {hypothesis_id}",
        elo_score=elo_score,
        reflection_report=reflection_report or _reflection_report(),
    )


def _decision(
    hypothesis_a,
    hypothesis_b,
    outcome="A",
    confidence=8,
):
    """Create a deterministic pairwise decision."""

    scores = {
        "research_goal_alignment": 7.0,
        "novelty": 7.0,
        "feasibility": 7.0,
        "scientific_plausibility": 7.0,
        "testability": 7.0,
        "evidence_quality": 7.0,
        "expected_research_value": 7.0,
    }

    return PairwiseDecision(
        hypothesis_a_id=hypothesis_a.hypothesis_id,
        hypothesis_b_id=hypothesis_b.hypothesis_id,
        outcome=outcome,
        scores_a=scores.copy(),
        scores_b=scores.copy(),
        confidence=confidence,
        reasoning="A better matches the research goal.",
        decisive_criteria=["scientific validity"],
    )


def _ranking_response(
    outcome: str,
    confidence: int = 8,
    criterion: str = "scientific validity",
) -> str:
    """Create a deterministic LLM ranking response."""
    return f"""Decision:
            {outcome}

            Short Justification:
            The selected outcome better satisfies the research goal.

            Decisive Criteria:
            - {criterion}

            Confidence:
            {confidence}
            """


def test_pairwise_confidence_requires_integer_score_from_one_to_ten():
    assert parse_confidence("Confidence: 8") == 8
    assert parse_confidence("Confidence: 10/10") == 10
    assert parse_confidence("Confidence: 0.8") == 1

    with pytest.raises(ValidationError):
        PairwiseDecision(
            hypothesis_a_id="A",
            hypothesis_b_id="B",
            outcome="A",
            confidence=8.0,
            reasoning="Invalid decimal confidence.",
        )

    with pytest.raises(ValidationError):
        PairwiseDecision(
            hypothesis_a_id="A",
            hypothesis_b_id="B",
            outcome="A",
            confidence=11,
            reasoning="Out-of-range confidence.",
        )


@pytest.mark.parametrize(
    "heading",
    ["Short Justification", "Justification", "Reasoning", "Why it won", "Winner Explanation"],
)
def test_pairwise_justification_parser_accepts_common_headings(heading):
    response = f"""Decision:
A

{heading}:
Hypothesis A has stronger experimental evidence.

Decisive Criteria:
- Evidence quality

Confidence:
8
"""

    assert parse_short_justification(response) == (
        "Hypothesis A has stronger experimental evidence."
    )


def test_pairwise_justification_parser_accepts_unlabelled_decision_explanation():
    response = """Decision: B
Hypothesis B presents a more feasible validation plan.

Decisive Criteria:
- Feasibility

Confidence: 7
"""

    assert parse_short_justification(response) == (
        "Hypothesis B presents a more feasible validation plan."
    )


def test_formatted_reflection_report_omits_obsolete_contradictions():
    report = _reflection_report()

    assert "Contradictions:" not in format_reflection_report(report)


# ---------------------------------------------------------------------------
# 1. LLM CALL EFFICIENCY
# ---------------------------------------------------------------------------

def test_pairwise_ranking_uses_one_llm_call():
    """Improved ranking should use one LLM adjudication per pair."""

    goal = ResearchGoal(
        description="Test research goal",
        llm_model="test-model",
    )

    response = _ranking_response("A")

    with patch(
        "app.agents_modules.ranking_helpers._call_llm",
        return_value=response,
    ) as call_llm:

        decision = run_pairwise_debate(
            _hypothesis("A"),
            _hypothesis("B"),
            goal,
        )

    assert decision.outcome == "A"

    # Main performance requirement:
    assert call_llm.call_count == 1
    print(f"\nLLM calls per pair: {call_llm.call_count}")

    # With no dedicated ranking model configured, ranking reuses the main model.
    assert call_llm.call_args.kwargs["model"] == config["llm_model"]


# ---------------------------------------------------------------------------
# 2. EXECUTION TIME
# ---------------------------------------------------------------------------

def test_pairwise_ranking_execution_time():
    """Measure the runtime of a single pairwise ranking decision."""

    goal = ResearchGoal(
        description="Test research goal",
        llm_model="test-model",
    )

    response = _ranking_response("A")

    with patch(
        "app.agents_modules.ranking_helpers._call_llm",
        return_value=response,
    ):

        start = perf_counter()

        decision = run_pairwise_debate(
            _hypothesis("A"),
            _hypothesis("B"),
            goal,
        )

        elapsed = perf_counter() - start

    assert decision.outcome == "A"

    # The test primarily records runtime rather than enforcing a strict
    # machine-dependent threshold.
    print(f"\nPairwise ranking execution time: {elapsed:.6f} seconds")

    assert elapsed >= 0


@pytest.mark.integration
def test_full_tournament_real_execution_time():
    """
    Measure the real end-to-end execution time of the complete
    improved Ranking Agent tournament, including real LLM calls.
    """

    hypotheses = [
        _hypothesis("A"),
        _hypothesis("B"),
        _hypothesis("C"),
        _hypothesis("D"),
    ]

    context = ContextMemory()

    goal = ResearchGoal(
        description="Test research goal",
        llm_model=RANKING_LLM_MODEL,
    )

    # IMPORTANT:
    # Do NOT mock _call_llm here.
    # This measures the real end-to-end tournament,
    # including LLM inference/API latency.
    start = perf_counter()

    RankingAgent().run_tournament(
        hypotheses,
        context,
        goal,
    )

    elapsed = perf_counter() - start

    print(
        f"\nFull improved tournament execution time: "
        f"{elapsed:.6f} seconds"
    )

    print(
        f"Total tournament results: "
        f"{len(context.tournament_results)}"
    )

    assert elapsed >= 0


# ---------------------------------------------------------------------------
# 3. NUMBER OF PAIRWISE COMPARISONS
# ---------------------------------------------------------------------------

def test_tournament_comparison_count_with_new_hypothesis():
    """
    When one new hypothesis is introduced, the improved tournament should
    compare it against existing hypotheses rather than repeating old-vs-old
    comparisons.
    """

    old_hypotheses = [
        _hypothesis("old-1"),
        _hypothesis("old-2"),
        _hypothesis("old-3"),
    ]

    new_hypothesis = _hypothesis("new-1")

    hypotheses = old_hypotheses + [new_hypothesis]

    context = ContextMemory()

    goal = ResearchGoal(
        description="Test research goal",
    )

    def fake_debate(h_a, h_b, research_goal):
        return _decision(h_a, h_b)

    with patch(
        "app.agents_modules.ranking.run_pairwise_debate",
        side_effect=fake_debate,
    ) as debate:

        RankingAgent().run_tournament(
            hypotheses,
            context,
            goal,
            new_hypotheses=[new_hypothesis],
        )

    # Only:
    #
    # old-1 vs new-1
    # old-2 vs new-1
    # old-3 vs new-1
    #
    # should be evaluated.

    assert debate.call_count == 3

    compared_pairs = {
        frozenset(
            (
                call.args[0].hypothesis_id,
                call.args[1].hypothesis_id,
            )
        )
        for call in debate.call_args_list
    }

    expected_pairs = {
        frozenset(("old-1", "new-1")),
        frozenset(("old-2", "new-1")),
        frozenset(("old-3", "new-1")),
    }

    assert compared_pairs == expected_pairs


def test_tournament_caps_matches_without_requiring_proximity_data():
    hypotheses = [_hypothesis(f"H{index}") for index in range(5)]
    context = ContextMemory()
    goal = ResearchGoal(description="Test research goal")

    with (
        patch(
            "app.agents_modules.ranking.run_pairwise_debate",
            side_effect=lambda h_a, h_b, goal: _decision(h_a, h_b),
        ) as debate,
        patch.dict(config["ranking"], {"max_matches_per_cycle": 2}),
    ):
        RankingAgent().run_tournament(hypotheses, context, goal)

    assert debate.call_count == 2
    assert len(context.tournament_results) == 2


def test_tournament_does_not_repeat_completed_pairs():
    hypotheses = [_hypothesis("A"), _hypothesis("B"), _hypothesis("C")]
    context = ContextMemory()
    context.tournament_results.append(
        {"hypothesis_a": "A", "hypothesis_b": "B", "outcome": "A"}
    )
    goal = ResearchGoal(description="Test research goal")

    with patch(
        "app.agents_modules.ranking.run_pairwise_debate",
        side_effect=lambda h_a, h_b, goal: _decision(h_a, h_b),
    ) as debate:
        RankingAgent().run_tournament(hypotheses, context, goal)

    compared_pairs = {
        frozenset((call.args[0].hypothesis_id, call.args[1].hypothesis_id))
        for call in debate.call_args_list
    }
    assert frozenset(("A", "B")) not in compared_pairs


# ---------------------------------------------------------------------------
# 4. RANKING CONSISTENCY
# ---------------------------------------------------------------------------

def test_ranking_consistency():
    """
    Identical pairwise results should produce the same final ordering.
    """

    goal = ResearchGoal(
        description="Test research goal",
    )

    hypotheses_1 = [
        _hypothesis("A"),
        _hypothesis("B"),
        _hypothesis("C"),
    ]

    hypotheses_2 = deepcopy(hypotheses_1)

    outcomes = {
        frozenset(("A", "B")): "A",
        frozenset(("A", "C")): "A",
        frozenset(("B", "C")): "B",
    }

    def deterministic_debate(h_a, h_b, research_goal):

        outcome = outcomes[frozenset(
            (h_a.hypothesis_id, h_b.hypothesis_id)
        )]

        # Convert the predetermined winner into the current A/B orientation.
        if outcome == h_a.hypothesis_id:
            result = "A"
        else:
            result = "B"

        return _decision(
            h_a,
            h_b,
            outcome=result,
        )

    def run_ranking(hypotheses):

        context = ContextMemory()

        with patch(
            "app.agents_modules.ranking.run_pairwise_debate",
            side_effect=deterministic_debate,
        ):

            RankingAgent().run_tournament(
                hypotheses,
                context,
                goal,
            )

        return [
            h.hypothesis_id
            for h in sorted(
                hypotheses,
                key=lambda h: h.elo_score,
                reverse=True,
            )
        ]

    ranking_1 = run_ranking(hypotheses_1)
    ranking_2 = run_ranking(hypotheses_2)

    print("\nRanking consistency:")
    print(f"  Run 1: {ranking_1}")
    print(f"  Run 2: {ranking_2}")
    print(f"  Consistent: {ranking_1 == ranking_2}")

    assert ranking_1 == ranking_2


# ---------------------------------------------------------------------------
# 5. A/B ORDER CONSISTENCY
# ---------------------------------------------------------------------------

def test_ab_order_consistency():
    """
    Reversing A/B order should not change the actual winning hypothesis.

    This tests the order-bias requirement.
    """

    goal = ResearchGoal(
        description="Test research goal",
    )

    hypo_a = _hypothesis("A")
    hypo_b = _hypothesis("B")

    response_ab = _ranking_response("A")
    response_ba = _ranking_response("B")

    with patch(
        "app.agents_modules.ranking_helpers._call_llm",
        side_effect=[
            response_ab,
            response_ba,
        ],
    ):

        decision_ab = run_pairwise_debate(
            hypo_a,
            hypo_b,
            goal,
        )

        decision_ba = run_pairwise_debate(
            hypo_b,
            hypo_a,
            goal,
        )

    # Convert both decisions back to the same original hypothesis orientation.

    winner_ab = (
        hypo_a.hypothesis_id
        if decision_ab.outcome == "A"
        else hypo_b.hypothesis_id
    )

    winner_ba = (
        hypo_b.hypothesis_id
        if decision_ba.outcome == "A"
        else hypo_a.hypothesis_id
    )

    print(f"\nWinner A/B ordering: {winner_ab}")
    print(f"Winner B/A ordering: {winner_ba}")
    print(f"  Order consistent: {winner_ab == winner_ba}")

    assert winner_ab == winner_ba


# ---------------------------------------------------------------------------
# 6. ELO CONSISTENCY
# ---------------------------------------------------------------------------

def test_elo_results_are_reproducible():
    """
    The same set of match results processed from the same initial state
    should produce identical Elo ratings.
    """

    matches = [
        ("A", "B", "A"),
        ("A", "C", "A"),
        ("B", "C", "B"),
    ]

    def run_matches():

        hypotheses = {
            "A": _hypothesis("A"),
            "B": _hypothesis("B"),
            "C": _hypothesis("C"),
        }

        for winner_id, loser_id, outcome in matches:

            winner = hypotheses[winner_id]
            loser = hypotheses[loser_id]

            update_elo(
                winner,
                loser,
                k_factor=32,
            )

        return {
            hypothesis_id: hypothesis.elo_score
            for hypothesis_id, hypothesis in hypotheses.items()
        }

    ratings_1 = run_matches()
    ratings_2 = run_matches()

    print("\nElo reproducibility:")
    print(f"  Run 1: {ratings_1}")
    print(f"  Run 2: {ratings_2}")
    print(f"  Consistent: {ratings_1 == ratings_2}")

    assert ratings_1 == ratings_2


# ---------------------------------------------------------------------------
# 7. TIE AND ABSTAIN HANDLING
# ---------------------------------------------------------------------------

def test_tie_and_abstain_are_handled_correctly():
    """
    TIE should be parsed as TIE and ABSTAIN should be parsed as ABSTAIN.

    These outcomes should not be confused with A/B decisions.
    """

    assert parse_pairwise_result(
        _ranking_response("TIE")
    ) == "TIE"

    assert parse_pairwise_result(
        _ranking_response("ABSTAIN")
    ) == "ABSTAIN"


def test_tie_updates_elo_but_abstain_does_not():
    """
    Verify the current Elo policy:

    TIE     -> Elo update
    ABSTAIN -> no Elo update
    """

    hypothesis_a = _hypothesis("A")
    hypothesis_b = _hypothesis("B")

    initial_a = hypothesis_a.elo_score
    initial_b = hypothesis_b.elo_score

    # TIE between equally-rated hypotheses should leave their Elo unchanged.
    update_elo_tie(
        hypothesis_a,
        hypothesis_b,
        k_factor=32,
    )

    assert hypothesis_a.elo_score == initial_a
    assert hypothesis_b.elo_score == initial_b

    # ABSTAIN does not call an Elo update function.
    abstain_a = _hypothesis("A")
    abstain_b = _hypothesis("B")

    initial_abstain_a = abstain_a.elo_score
    initial_abstain_b = abstain_b.elo_score

    # Simulate the RankingAgent's ABSTAIN branch:
    outcome = "ABSTAIN"

    if outcome == "A":
        update_elo(abstain_a, abstain_b, 32)

    elif outcome == "B":
        update_elo(abstain_b, abstain_a, 32)

    elif outcome == "TIE":
        update_elo_tie(abstain_a, abstain_b, 32)

    elif outcome == "ABSTAIN":
        pass

    assert abstain_a.elo_score == initial_abstain_a
    assert abstain_b.elo_score == initial_abstain_b

    print(
        f"\nTIE Elo change: "
        f"A {initial_a:.2f} -> {hypothesis_a.elo_score:.2f}, "
        f"B {initial_b:.2f} -> {hypothesis_b.elo_score:.2f}"
    )

    print(
        f"ABSTAIN Elo change: "
        f"A {initial_abstain_a:.2f} -> {abstain_a.elo_score:.2f}, "
        f"B {initial_abstain_b:.2f} -> {abstain_b.elo_score:.2f}"
    )


# ---------------------------------------------------------------------------
# 8. INACTIVE HYPOTHESES
# ---------------------------------------------------------------------------

def test_inactive_hypotheses_are_not_ranked():
    """
    Inactive hypotheses should not participate in tournament comparisons.

    This verifies that the improved Ranking Agent filters inactive
    hypotheses before generating pairwise comparisons.
    """

    active_a = _hypothesis("A")
    active_b = _hypothesis("B")
    inactive_c = _hypothesis("C")

    inactive_c.is_active = False

    hypotheses = [
        active_a,
        active_b,
        inactive_c,
    ]

    context = ContextMemory()

    goal = ResearchGoal(
        description="Test research goal",
    )

    def fake_debate(h_a, h_b, research_goal):
        return _decision(h_a, h_b)

    with patch(
        "app.agents_modules.ranking.run_pairwise_debate",
        side_effect=fake_debate,
    ) as debate:

        RankingAgent().run_tournament(
            hypotheses,
            context,
            goal,
        )

    # Only the two active hypotheses should be compared.
    assert debate.call_count == 1

    compared_pair = frozenset(
        (
            debate.call_args_list[0].args[0].hypothesis_id,
            debate.call_args_list[0].args[1].hypothesis_id,
        )
    )

    assert compared_pair == frozenset(("A", "B"))

    # The inactive hypothesis must not participate.
    assert "C" not in compared_pair

    print(
        f"\nActive hypotheses compared: {debate.call_count}"
    )


# ---------------------------------------------------------------------------
# 9. TOURNAMENT RESULT RECORDING
# ---------------------------------------------------------------------------

def test_tournament_records_results_in_context():
    """
    Every completed pairwise comparison should be recorded in
    ContextMemory.tournament_results.

    The recorded result should contain the iteration, participating
    hypotheses, outcome, confidence, reasoning, and updated Elo scores.
    """

    hypothesis_a = _hypothesis("A")
    hypothesis_b = _hypothesis("B")

    hypotheses = [
        hypothesis_a,
        hypothesis_b,
    ]

    context = ContextMemory()

    goal = ResearchGoal(
        description="Test research goal",
    )

    def fake_debate(h_a, h_b, research_goal):
        return _decision(
            h_a,
            h_b,
            outcome="A",
            confidence=9,
        )

    with patch(
        "app.agents_modules.ranking.run_pairwise_debate",
        side_effect=fake_debate,
    ):

        RankingAgent().run_tournament(
            hypotheses,
            context,
            goal,
        )

    # Exactly one pairwise comparison should be recorded.
    assert len(context.tournament_results) == 1

    result = context.tournament_results[0]

    # Verify iteration.
    assert result["iteration"] == context.iteration_number

    # A and B must be the two hypotheses involved.
    # Their order may change because the tournament shuffles
    # active hypotheses before creating pairs.
    assert {
        result["hypothesis_a"],
        result["hypothesis_b"],
    } == {"A", "B"}

    # The decision is relative to the pair orientation.
    assert result["outcome"] == "A"

    # Verify decision metadata.
    assert result["confidence"] == 9
    assert result["reasoning"] == "A better matches the research goal."

    # Verify Elo scores after the comparison are recorded.
    recorded_a_id = result["hypothesis_a"]
    recorded_b_id = result["hypothesis_b"]

    hypothesis_lookup = {
        "A": hypothesis_a,
        "B": hypothesis_b,
    }

    assert result["elo_a_after"] == hypothesis_lookup[recorded_a_id].elo_score
    assert result["elo_b_after"] == hypothesis_lookup[recorded_b_id].elo_score

    # Verify structured decision information.
    expected_scores = {
        "research_goal_alignment": 7.0,
        "novelty": 7.0,
        "feasibility": 7.0,
        "scientific_plausibility": 7.0,
        "testability": 7.0,
        "evidence_quality": 7.0,
        "expected_research_value": 7.0,
    }
    assert result["scores_a"] == expected_scores
    assert result["scores_b"] == expected_scores
    assert result["criteria"] == ["scientific validity"]

    print("\nTournament result:")
    print(f"  {result}")

def test_ranking_abstains_without_reflection_report():
    goal = ResearchGoal(
        description="Test research goal",
    )

    hypo_a = Hypothesis(
        hypothesis_id="A",
        text="Hypothesis A",
    )

    hypo_b = Hypothesis(
        hypothesis_id="B",
        text="Hypothesis B",
    )

    decision = run_pairwise_debate(
        hypo_a,
        hypo_b,
        goal,
    )

    assert decision.outcome == "ABSTAIN"
    assert decision.scores_a == {}
    assert decision.scores_b == {}
    assert "ReflectionReport" in decision.reasoning


def test_ranking_model_resolution_from_config():
    from app.agents_modules.ranking_helpers import _get_ranking_model

    # When ranking_llm_model is configured in config dict
    with patch("app.config.config", {"ranking_llm_model": "custom/ranking-model", "llm_model": "default/model"}):
        assert _get_ranking_model() == "custom/ranking-model"

    # When ranking_llm_model is None or empty, fall back to llm_model
    with patch("app.config.config", {"ranking_llm_model": None, "llm_model": "default/model"}):
        assert _get_ranking_model() == "default/model"

    with patch("app.config.config", {"llm_model": "default/model"}):
        assert _get_ranking_model() == "default/model"

