# AI Co-Scientist evaluation harness

This directory is an independent, artifact-based evaluation project. It reads
saved AI Co-Scientist run JSON files and does not import or invoke the
production application.

The evaluator always performs deterministic parsing first:

1. load and validate a saved run JSON;
2. compare `research_goal.description` with a fixed goal;
3. locate final hypotheses using the application's ranking-step fallback;
4. select the hypothesis with the highest Elo score; and
5. print its evidence sources.

DeepEval metrics are opt-in, so ordinary parsing and the pytest suite remain
offline. When enabled, the selected hypothesis is scored for goal alignment,
scientific testability, and evidence support. Evidence support is skipped when
the persisted artifact contains no evidence sources.

## Setup and use

```shell
uv sync
uv run pytest
uv run python scripts/evaluate_run.py ../results/runs/<run-id>.json
```

## DeepEval with LM Studio

Start LM Studio's local OpenAI-compatible server, load a judge model, and run:

```shell
export LOCAL_MODEL_API_KEY=lm-studio
uv run python scripts/evaluate_run.py ../results/runs/<run-id>.json \
  --llm-metrics \
  --judge-model <loaded-model-id> \
  --judge-base-url http://localhost:1234/v1/ \
  --threshold 0.7 \
  --report reports/<run-id>.json
```

On PowerShell, first use `$env:LOCAL_MODEL_API_KEY = "lm-studio"` and replace
the trailing backslashes with backticks. DeepEval's OpenAI-compatible adapter
requires an API-key value even when LM Studio authentication is disabled; in
that case the value is only a non-secret placeholder. The model and base URL
can instead be supplied through `LOCAL_MODEL_NAME` and
`LOCAL_MODEL_BASE_URL`. Real credentials are never accepted as a CLI argument
or written to the report.

The command exits with status 0 when all completed metrics pass, 1 when at
least one metric is below threshold, and 2 for invalid input, configuration,
or judge errors. LLM judging is qualitative and does not independently verify
that a cited paper exists or that the scientific claims are true.

The default fixed goal is
`goals/goal_001_perovskite_humidity.txt`. Supply another goal with
`--goal-file PATH`. A goal mismatch is reported as a validation error and
returns a non-zero exit status.

The selection behavior intentionally mirrors `app.run_store._final_hypotheses`:
`ranking_final` wins, otherwise the highest-numbered ranking step wins, and
otherwise the first step containing hypotheses is used. Empty ranking steps
are skipped. Within the selected candidates, the highest numeric `elo_score`
is chosen (a missing score has the application's default value of zero).
