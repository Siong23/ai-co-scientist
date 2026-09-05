# Evaluation Harness Instructions

This directory contains the independent evaluation harness for the
AI Co-Scientist system.

## Scope

- Modify files only inside `eval/` and its subdirectories.
- Do not modify files in the parent AI Co-Scientist application.
- Parent repository files may be inspected read-only to understand system
  outputs and schemas.
- Treat `../results/runs/*.json` as read-only evaluation artifacts.

## Dependency Boundary

- DeepEval and evaluation-only dependencies belong only to `eval/pyproject.toml`.
- Never add DeepEval to the parent project's dependencies.
- Never import evaluation code from the production `app/` package.
- Evaluation code should consume persisted artifacts such as run JSON.

## Evaluation Architecture

Use this boundary:

AI Co-Scientist
    -> saved run JSON
    -> eval parser
    -> evaluation metrics
    -> evaluation report

The evaluator must not invoke internal application agents directly.

## Python Environment

This directory is an independent uv project.

Use:

    uv sync
    uv run pytest
    uv run python scripts/evaluate_run.py ...

## Development

- Prefer small, testable parsing functions.
- Validate input schemas defensively.
- Avoid hard-coded absolute Windows paths.
- Use pathlib.Path.
- Add tests for parsing behavior.
- Keep LLM evaluation separate from deterministic artifact validation.