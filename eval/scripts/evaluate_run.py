"""Parse and display a persisted AI Co-Scientist run."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from rubrics.goal_alignment import goals_match  # noqa: E402
from rubrics.suites import (  # noqa: E402
    DEFAULT_METRIC_SUITE,
    METRIC_SUITES,
    SUITE_ALL,
    SUITE_HYPOTHESIS,
    SUITE_LEGACY,
    SUITE_RAG,
)

DEFAULT_GOAL_PATH = PROJECT_ROOT / "goals" / "goal_001_perovskite_humidity.txt"
RANKING_STEP_PATTERN = re.compile(r"ranking(?:_?(\d+)|_final)?")

#: Where an --llm-metrics run files its audit report when --report is omitted.
REPORTS_DIR = PROJECT_ROOT / "reports"

#: Report filename prefix per suite, matching the reports already on disk.
REPORT_PREFIXES = {
    SUITE_HYPOTHESIS: "hyp",
    SUITE_RAG: "rag",
    SUITE_LEGACY: "legacy",
    SUITE_ALL: "all",
}

UNSAFE_STEM_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

#: A variable whose name carries one of these holds a credential, never a
#: value this command may print.
SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")

#: Judge configuration for repeat runs. Git-ignored; see .env.example.
ENV_FILE = PROJECT_ROOT / ".env"


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
            f"research goal mismatch: expected {expected_goal.strip()!r}, found {actual_goal.strip()!r}"
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
        raise RunValidationError(f"cycle_details.steps.{step_name}.hypotheses must contain only objects")
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
        "--metric-suite",
        choices=METRIC_SUITES,
        default=DEFAULT_METRIC_SUITE,
        help=(
            "which DeepEval metric suite --llm-metrics runs: 'hypothesis' scores the "
            "hypothesis itself, 'rag' scores it against retrieved evidence, 'legacy' runs "
            "the superseded Evidence support metric, and 'all' runs hypothesis plus rag "
            f"(default: {DEFAULT_METRIC_SUITE})"
        ),
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
        help=(
            "where to write the complete JSON evaluation report; with --llm-metrics and no "
            "--report the path is derived as reports/<suite>-<run-id>.json"
        ),
    )
    parser.add_argument(
        "--deepeval-results-folder",
        type=Path,
        help=(
            "optionally also keep DeepEval's own timestamped test-run JSON in this "
            "folder; off by default so that one run produces one report"
        ),
    )
    return parser


def default_report_path(suite: str, run_id: Any, run_json: Path) -> Path:
    """Derive the audit report path for an --llm-metrics run given no --report.

    DeepEval owns the terminal once metrics run, so the report has to land
    somewhere by default: the metric reasons live only in it.
    """
    stem = run_id if isinstance(run_id, str) and run_id.strip() else run_json.stem
    safe_stem = UNSAFE_STEM_CHARS.sub("-", stem.strip()).strip("-.") or "run"
    return REPORTS_DIR / f"{REPORT_PREFIXES[suite]}-{safe_stem}.json"


def display_path(path: Path) -> str:
    """Show a written path relative to the working directory when it is inside it."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def print_llm_footer(llm_report: Mapping[str, Any], report_path: Path | None) -> None:
    """Say where the audit report went and which metrics never reached DeepEval.

    Deliberately small: DeepEval has already printed the scores, and a second
    rendering of the same numbers would only compete with it.
    """
    if report_path is not None:
        print("AI Co-Scientist report written to:")
        print(display_path(report_path))
    skipped = [metric for metric in llm_report["metrics"] if metric["status"] == "skipped"]
    if skipped:
        print("Skipped metrics:")
        for metric in skipped:
            print(f"- {metric['name']}: {metric['reason']}")


class EnvFileLoad(NamedTuple):
    """What an env file contributed, and what the environment already decided."""

    #: Variables this file put into the environment.
    applied: dict[str, str]
    #: File values the environment overrode, by variable name. Reported rather
    #: than silently dropped: a stale shell variable that shadows the file is
    #: otherwise invisible until a judge the file never named starts loading.
    shadowed: dict[str, str]


def load_env_file(path: Path) -> EnvFileLoad:
    """Merge ``KEY=value`` lines from an env file into the process environment.

    A variable already in the environment is never overwritten: the file is a
    default for repeat runs, not an override of what the shell or CI has set.
    Everything after the first ``=`` is the value, so an inline ``#`` is part of
    it rather than a comment.

    DeepEval reads a ``.env`` of its own on import, but relying on that would
    tie this command's configuration to a third-party import side effect and to
    the working directory. This reads the harness's own file, explicitly.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        # No env file is ordinary: the judge can still come from the shell or
        # from --judge-model and --judge-base-url.
        return EnvFileLoad({}, {})

    applied: dict[str, str] = {}
    shadowed: dict[str, str] = {}
    for line in raw.splitlines():
        entry = line.strip().removeprefix("export ").lstrip()
        if not entry or entry.startswith("#"):
            continue
        name, separator, value = entry.partition("=")
        if not separator:
            continue
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not name:
            continue
        if name in os.environ:
            if os.environ[name] != value:
                shadowed[name] = value
            continue
        os.environ[name] = value
        applied[name] = value
    return EnvFileLoad(applied, shadowed)


def configure_local_judge(model: str | None, base_url: str | None) -> dict[str, str]:
    """Configure DeepEval's local adapter without accepting secrets on the CLI."""
    model = model or os.environ.get("LOCAL_MODEL_NAME")
    base_url = base_url or os.environ.get("LOCAL_MODEL_BASE_URL")
    if not model or not base_url:
        raise RunValidationError(
            "LLM metrics require --judge-model and --judge-base-url (or LOCAL_MODEL_NAME and LOCAL_MODEL_BASE_URL)"
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


def names_a_secret(name: str) -> bool:
    """Whether a variable's name marks its value as a credential."""
    upper_name = name.upper()
    return any(marker in upper_name for marker in SECRET_NAME_MARKERS)


def redact_environment_secrets(message: str) -> str:
    """Remove environment-provided credentials from an error before printing it."""
    redacted = message
    for name, value in os.environ.items():
        if value and len(value) >= 4 and names_a_secret(name):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def main(argv: list[str] | None = None) -> int:
    env_file = load_env_file(ENV_FILE)
    args = build_parser().parse_args(argv)
    try:
        parsed = parse_run(args.run_json, args.goal_file)
    except RunValidationError as exc:
        print(f"Evaluation input error: {redact_environment_secrets(str(exc))}", file=sys.stderr)
        return 2

    exit_code = 0
    llm_report: dict[str, Any] | None = None
    report_path: Path | None = args.report
    if args.llm_metrics:
        # Import only in opt-in mode so deterministic parsing remains separate
        # from DeepEval and never initializes a judge provider.
        from rubrics.deepeval_metrics import LLMEvaluationError, evaluate_parsed_run

        # Say so before the judge is contacted. A shadowed LOCAL_MODEL_NAME
        # otherwise shows up as a model the env file never named, minutes later,
        # in whatever the server says when it fails to load it.
        for name, file_value in env_file.shadowed.items():
            # Name the variable always; show what it chose only when the name
            # does not mark it a credential. Both values are the file's and the
            # environment's, and either could be a real key.
            chose = "" if names_a_secret(name) else f" ({os.environ[name]!r} instead of {file_value!r})"
            print(
                redact_environment_secrets(
                    f"Note: {name} from the environment overrides {display_path(ENV_FILE)}{chose}"
                ),
                file=sys.stderr,
            )

        try:
            judge = configure_local_judge(args.judge_model, args.judge_base_url)
            from deepeval.models import LocalModel

            # Pass the provider explicitly: DeepEval may have cached settings
            # before configure_local_judge updated the environment.
            local_model = LocalModel(
                model=judge["model"],
                base_url=judge["base_url"],
                api_key=os.environ["LOCAL_MODEL_API_KEY"],
                # Judging must be as reproducible as a local server allows.
                temperature=0,
            )
            llm_report = evaluate_parsed_run(
                parsed,
                threshold=args.threshold,
                model=local_model,
                suite=args.metric_suite,
                results_folder=(str(args.deepeval_results_folder) if args.deepeval_results_folder else None),
            )
        except (RunValidationError, LLMEvaluationError) as exc:
            print(
                f"Evaluation error: {redact_environment_secrets(str(exc))}",
                file=sys.stderr,
            )
            return 2
        parsed["llm_evaluation"] = {"judge": judge, **llm_report}
        if report_path is None:
            report_path = default_report_path(args.metric_suite, parsed.get("run_id"), args.run_json)
        if llm_report["status"] != "completed":
            # An all-skipped suite is not a pass; say so instead of letting an
            # empty metric list look like success.
            print(f"No metric completed: {llm_report['reason']}", file=sys.stderr)
        if not llm_report["passed"]:
            exit_code = 1

    if report_path is not None:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(
                f"Could not write report: {redact_environment_secrets(str(exc))}",
                file=sys.stderr,
            )
            return 2

    if llm_report is not None:
        # DeepEval has already rendered its native result table. Printing the
        # whole audit JSON underneath it would bury the scores it just showed.
        print_llm_footer(llm_report, report_path)
    else:
        print("AI Co-Scientist evaluation report")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
