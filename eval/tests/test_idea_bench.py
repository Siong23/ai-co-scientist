import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("idea_bench", SCRIPTS / "idea_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def fixture_export(tmp_path):
    tasks = []
    for index, goal in enumerate(("Open-RAN detection", "Solar efficiency")):
        run = {
            "run_id": str(index),
            "status": "completed",
            "research_goal": {"description": goal},
            "cycle_details": {
                "steps": {
                    "generation": {"hypotheses": [{"id": "old", "text": "old"}]},
                    "ranking_final": {"hypotheses": [{"id": "new", "text": "A testable idea", "elo_score": 1200}]},
                }
            },
        }
        (tmp_path / f"{index}.json").write_text(json.dumps(run))
        tasks.append({"id": str(index), "goal": goal, "runs": [{"path": f"{index}.json"}]})
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": tasks}))
    return path, bench.export_manifest(path)


def scores():
    return {k: {"score": 4, "reason": "Specific rationale", "source_ids": []} for k in bench.DIMENSIONS}


def test_multiple_goals_missing_fields_and_identity(tmp_path):
    _, exported = fixture_export(tmp_path)
    assert len(exported["records"]) == 2
    record = exported["records"][0]
    assert record["hypothesis_id"] == "new"
    assert record["source_step"] == "ranking_final"
    assert record["proposal"]["experiment_plan"] is None
    assert "experiment_plan" in record["missing_structured_fields"]
    assert "elo_score" not in bench.judge_prompt(record)
    assert "condition" not in bench.judge_prompt(record)


def test_mismatch_preserves_other_tasks(tmp_path):
    path, _ = fixture_export(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["tasks"][0]["goal"] = "Wrong goal"
    path.write_text(json.dumps(manifest))
    exported = bench.export_manifest(path)
    assert len(exported["errors"]) == 1
    assert len(exported["records"]) == 1


def test_no_literature_never_means_novel(tmp_path):
    _, data = fixture_export(tmp_path)
    report = bench.evaluate_export(data, {}, lambda *_: scores())
    assert report["results"][0]["scores"]["novelty"]["score"] is None


@pytest.mark.parametrize("bad", [True, 6, "4", float("nan")])
def test_bad_judge_score_is_error(tmp_path, bad):
    _, data = fixture_export(tmp_path)
    response = scores()
    response["quality"]["score"] = bad
    report = bench.evaluate_export(data, {}, lambda *_: response)
    assert all(r["status"] == "error" for r in report["results"])
    assert all("scores" not in r for r in report["results"])


def test_unknown_citation_rejected(tmp_path):
    _, data = fixture_export(tmp_path)
    response = scores()
    response["novelty"]["source_ids"] = ["invented"]
    with pytest.raises(ValueError):
        bench.validate_scores(response, data["records"][0])


def test_cli_offline_and_no_overwrite(tmp_path):
    path, _ = fixture_export(tmp_path)
    output = tmp_path / "export.json"
    assert bench.main(["export", str(path), "--output", str(output)]) == 0
    before = output.read_bytes()
    assert bench.main(["export", str(path), "--output", str(output)]) == 2
    assert output.read_bytes() == before


def test_error_redacts_environment_key(tmp_path, monkeypatch):
    _, data = fixture_export(tmp_path)
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "fake-test-secret")

    def fail(*_):
        raise RuntimeError("failed fake-test-secret")

    report = bench.evaluate_export(data, {}, fail)
    assert "fake-test-secret" not in json.dumps(report)
