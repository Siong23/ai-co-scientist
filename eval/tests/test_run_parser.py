import json
import os

import pytest

from scripts.evaluate_run import (
    RunValidationError,
    configure_local_judge,
    extract_evidence_sources,
    load_run,
    locate_final_hypotheses,
    parse_run,
    select_highest_elo,
    validate_research_goal,
)

FIXED_GOAL = "Improve perovskite humidity stability."


def hypothesis(hypothesis_id, elo, sources=None):
    return {
        "id": hypothesis_id,
        "title": f"Hypothesis {hypothesis_id}",
        "text": "A testable mechanism.",
        "elo_score": elo,
        "evidence_source_ids": [source["source_id"] for source in sources or []],
        "evidence_sources": sources or [],
    }


def run_fixture(steps, goal=FIXED_GOAL):
    return {
        "run_id": "run-test",
        "research_goal": {"description": goal},
        "cycle_details": {"steps": steps},
    }


def test_load_run_reads_json_object(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run_fixture({"generation": {"hypotheses": []}})), encoding="utf-8")

    assert load_run(path)["run_id"] == "run-test"


def test_load_run_rejects_non_object_json(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(RunValidationError, match="root must be an object"):
        load_run(path)


def test_goal_validation_normalizes_whitespace_and_rejects_mismatch():
    run = run_fixture({}, goal="Improve   perovskite\n humidity stability.")

    assert validate_research_goal(run, FIXED_GOAL) == run["research_goal"]["description"]
    with pytest.raises(RunValidationError, match="research goal mismatch"):
        validate_research_goal(run, "A different goal")


def test_ranking_final_takes_precedence_and_is_sorted_by_elo():
    run = run_fixture(
        {
            "ranking_12": {"hypotheses": [hypothesis("old", 1900)]},
            "ranking_final": {
                "hypotheses": [hypothesis("low", 1100), hypothesis("winner", 1300)]
            },
        }
    )

    step, hypotheses = locate_final_hypotheses(run)

    assert step == "ranking_final"
    assert [item["id"] for item in hypotheses] == ["winner", "low"]


def test_highest_numbered_nonempty_ranking_step_is_used():
    run = run_fixture(
        {
            "ranking2": {"hypotheses": [hypothesis("two", 1200)]},
            "ranking_10": {"hypotheses": []},
            "ranking_3": {"hypotheses": [hypothesis("three", 1250)]},
        }
    )

    step, hypotheses = locate_final_hypotheses(run)

    assert step == "ranking_3"
    assert hypotheses[0]["id"] == "three"


def test_first_nonempty_step_is_fallback_when_no_rankings_have_hypotheses():
    run = run_fixture(
        {
            "generation": {"hypotheses": [hypothesis("generated", 1000)]},
            "reflection": {"hypotheses": [hypothesis("reflected", 1500)]},
        }
    )

    step, hypotheses = locate_final_hypotheses(run)

    assert step == "generation"
    assert hypotheses[0]["id"] == "generated"


def test_selects_highest_elo_and_extracts_evidence_sources():
    source = {"source_id": "arxiv:1234", "title": "Humidity barriers"}
    candidates = [hypothesis("low", 1190), hypothesis("high", 1210, [source])]

    selected = select_highest_elo(candidates)

    assert selected["id"] == "high"
    assert extract_evidence_sources(selected) == [source]


def test_parse_run_returns_clear_summary(tmp_path):
    source = {"source_id": "doi:example", "title": "Encapsulation study"}
    run_path = tmp_path / "run.json"
    goal_path = tmp_path / "goal.txt"
    run_path.write_text(
        json.dumps(
            run_fixture(
                {"ranking1": {"hypotheses": [hypothesis("best", 1400, [source])]}}
            )
        ),
        encoding="utf-8",
    )
    goal_path.write_text(f"  {FIXED_GOAL}\n", encoding="utf-8")

    parsed = parse_run(run_path, goal_path)

    assert parsed["goal_matches"] is True
    assert parsed["hypothesis_source_step"] == "ranking1"
    assert parsed["selected_hypothesis"]["id"] == "best"
    assert parsed["evidence_sources"] == [source]


def test_malformed_steps_and_scores_are_rejected():
    with pytest.raises(RunValidationError, match="steps must be an object"):
        locate_final_hypotheses({"cycle_details": {"steps": []}})

    with pytest.raises(RunValidationError, match="non-numeric elo_score"):
        select_highest_elo([hypothesis("bad", "not-a-number")])


def test_local_judge_configuration_requires_environment_key(monkeypatch):
    monkeypatch.delenv("LOCAL_MODEL_API_KEY", raising=False)

    with pytest.raises(RunValidationError, match="LOCAL_MODEL_API_KEY"):
        configure_local_judge("judge-model", "http://localhost:1234/v1/")

    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "placeholder")
    judge = configure_local_judge("judge-model", "http://localhost:1234/v1/")

    assert judge == {
        "provider": "local",
        "model": "judge-model",
        "base_url": "http://localhost:1234/v1/",
    }
    assert os.environ["USE_LOCAL_MODEL"] == "1"
