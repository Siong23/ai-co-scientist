"""Parse and display a persisted AI Co-Scientist run."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rubrics.goal_alignment import goals_match  # noqa: E402

DEFAULT_GOAL_PATH = PROJECT_ROOT / "goals" / "goal_001_perovskite_humidity.txt"
RANKING_STEP_PATTERN = re.compile(r"ranking(?:_?(\d+)|_final)?")


class RunValidationError(ValueError):
    """Raised when a run artifact does not have the expected persisted shape."""


def load_run(path: Path) -> dict[str, Any]:
    """Load a saved run JSON object from ``path``."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunValidationError(f"could not read run file {path}: {exc}") from exc

    try:
        run = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunValidationError(f"run file is not valid JSON: {exc}") from exc

    if not isinstance(run, dict):
        raise RunValidationError("run JSON root must be an object")
    return run


def get_research_goal(run: Mapping[str, Any]) -> str:
    """Extract and validate ``research_goal.description``."""
    research_goal = run.get("research_goal")
    if not isinstance(research_goal, Mapping):
        raise RunValidationError("research_goal must be an object")

    description = research_goal.get("description")
    if not isinstance(description, str) or not description.strip():
        raise RunValidationError("research_goal.description must be a non-empty string")
    return description


def validate_research_goal(run: Mapping[str, Any], expected_goal: str) -> str:
    """Return the saved goal, or raise when it differs from the fixed goal."""
    actual_goal = get_research_goal(run)
    if not goals_match(actual_goal, expected_goal):
        raise RunValidationError(
            "research goal mismatch: "
            f"expected {expected_goal.strip()!r}, found {actual_goal.strip()!r}"
        )
    return actual_goal


def _hypotheses_from_step(step_name: str, step_data: Any) -> list[dict[str, Any]]:
    if not isinstance(step_data, Mapping):
        return []
    hypotheses = step_data.get("hypotheses", [])
    if hypotheses is None:
        return []
    if not isinstance(hypotheses, list):
        raise RunValidationError(f"cycle_details.steps.{step_name}.hypotheses must be a list")
    if not all(isinstance(item, dict) for item in hypotheses):
        raise RunValidationError(
            f"cycle_details.steps.{step_name}.hypotheses must contain only objects"
        )
    return hypotheses


def _elo_score(hypothesis: Mapping[str, Any]) -> float:
    score = hypothesis.get("elo_score", 0)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        hypothesis_id = hypothesis.get("id", "<unknown>")
        raise RunValidationError(f"hypothesis {hypothesis_id!r} has a non-numeric elo_score")
    return float(score)


def locate_final_hypotheses(run: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Locate candidates using the same step fallback semantics as the app."""
    cycle_details = run.get("cycle_details")
    if not isinstance(cycle_details, Mapping):
        raise RunValidationError("cycle_details must be an object")
    steps = cycle_details.get("steps")
    if not isinstance(steps, Mapping):
        raise RunValidationError("cycle_details.steps must be an object")

    ranking_steps: list[tuple[float, int, str]] = []
    for index, step_name in enumerate(steps):
        if not isinstance(step_name, str):
            continue
        match = RANKING_STEP_PATTERN.fullmatch(step_name)
        if not match:
            continue
        priority = float("inf") if step_name == "ranking_final" else int(match.group(1) or 0)
        ranking_steps.append((priority, index, step_name))

    for _, _, step_name in sorted(ranking_steps, reverse=True):
        hypotheses = _hypotheses_from_step(step_name, steps[step_name])
        if hypotheses:
            return step_name, sorted(hypotheses, key=_elo_score, reverse=True)

    for step_name, step_data in steps.items():
        hypotheses = _hypotheses_from_step(str(step_name), step_data)
        if hypotheses:
            return str(step_name), hypotheses

    raise RunValidationError("run contains no hypotheses in any step")


def select_highest_elo(hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the first hypothesis with the maximum Elo score."""
    if not hypotheses:
        raise RunValidationError("cannot select a hypothesis from an empty list")
    return max(hypotheses, key=_elo_score)


def extract_evidence_sources(hypothesis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract structured evidence sources serialized on a hypothesis."""
    sources = hypothesis.get("evidence_sources", [])
    if sources is None:
        return []
    if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
        raise RunValidationError("selected hypothesis evidence_sources must be a list of objects")
    return sources


def parse_run(run_path: Path, goal_path: Path) -> dict[str, Any]:
    """Load and deterministically parse a run against a fixed goal file."""
    try:
        expected_goal = goal_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunValidationError(f"could not read goal file {goal_path}: {exc}") from exc
    if not expected_goal.strip():
        raise RunValidationError("fixed goal file must not be empty")

    run = load_run(run_path)
    research_goal = validate_research_goal(run, expected_goal)
    source_step, hypotheses = locate_final_hypotheses(run)
    selected = select_highest_elo(hypotheses)
    evidence_sources = extract_evidence_sources(selected)
    evidence_source_ids = selected.get("evidence_source_ids", [])
    if not isinstance(evidence_source_ids, list) or not all(
        isinstance(source_id, str) for source_id in evidence_source_ids
    ):
        raise RunValidationError("selected hypothesis evidence_source_ids must be a list of strings")

    return {
        "run_id": run.get("run_id"),
        "research_goal": research_goal,
        "goal_matches": True,
        "hypothesis_source_step": source_step,
        "final_hypothesis_count": len(hypotheses),
        "selected_hypothesis": selected,
        "evidence_source_ids": evidence_source_ids,
        "evidence_sources": evidence_sources,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path, help="path to a saved run JSON")
    parser.add_argument(
        "--goal-file",
        type=Path,
        default=DEFAULT_GOAL_PATH,
        help=f"fixed research goal (default: {DEFAULT_GOAL_PATH})",
    )
    parser.add_argument(
        "--llm-metrics",
        action="store_true",
        help="run opt-in DeepEval metrics using a local OpenAI-compatible judge",
    )
    parser.add_argument(
        "--judge-model",
        help="local judge model name (or set LOCAL_MODEL_NAME)",
    )
    parser.add_argument(
        "--judge-base-url",
        help="local OpenAI-compatible endpoint (or set LOCAL_MODEL_BASE_URL)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="passing threshold for every completed LLM metric (default: 0.7)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional path for the complete JSON evaluation report",
    )
    return parser


def configure_local_judge(model: str | None, base_url: str | None) -> dict[str, str]:
    """Configure DeepEval's local adapter without accepting secrets on the CLI."""
    model = model or os.environ.get("LOCAL_MODEL_NAME")
    base_url = base_url or os.environ.get("LOCAL_MODEL_BASE_URL")
    if not model or not base_url:
        raise RunValidationError(
            "LLM metrics require --judge-model and --judge-base-url "
            "(or LOCAL_MODEL_NAME and LOCAL_MODEL_BASE_URL)"
        )
    if not base_url.startswith(("http://", "https://")):
        raise RunValidationError("judge base URL must start with http:// or https://")
    parsed_url = urlsplit(base_url)
    if not parsed_url.hostname or parsed_url.username or parsed_url.password:
        raise RunValidationError("judge base URL must have a host and must not contain credentials")
    if not os.environ.get("LOCAL_MODEL_API_KEY"):
        raise RunValidationError(
            "LLM metrics require LOCAL_MODEL_API_KEY in the environment; "
            "for an unauthenticated LM Studio server, set it to a non-secret placeholder"
        )

    os.environ["USE_LOCAL_MODEL"] = "1"
    os.environ["LOCAL_MODEL_NAME"] = model
    os.environ["LOCAL_MODEL_BASE_URL"] = base_url
    return {"provider": "local", "model": model, "base_url": base_url}


def redact_environment_secrets(message: str) -> str:
    """Remove environment-provided credentials from an error before printing it."""
    redacted = message
    for name, value in os.environ.items():
        upper_name = name.upper()
        if value and len(value) >= 4 and any(
            marker in upper_name for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        parsed = parse_run(args.run_json, args.goal_file)
    except RunValidationError as exc:
        print(f"Evaluation input error: {redact_environment_secrets(str(exc))}", file=sys.stderr)
        return 2

    exit_code = 0
    if args.llm_metrics:
        # Import only in opt-in mode so deterministic parsing remains separate
        # from DeepEval and never initializes a judge provider.
        from rubrics.deepeval_metrics import LLMEvaluationError, evaluate_parsed_run

        try:
            judge = configure_local_judge(args.judge_model, args.judge_base_url)
            llm_report = evaluate_parsed_run(parsed, threshold=args.threshold)
        except (RunValidationError, LLMEvaluationError) as exc:
            print(
                f"Evaluation error: {redact_environment_secrets(str(exc))}",
                file=sys.stderr,
            )
            return 2
        parsed["llm_evaluation"] = {"judge": judge, **llm_report}
        if not llm_report["passed"]:
            exit_code = 1

    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(
                f"Could not write report: {redact_environment_secrets(str(exc))}",
                file=sys.stderr,
            )
            return 2

    print("AI Co-Scientist evaluation report")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
