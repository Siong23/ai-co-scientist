# Claude Code Prompt — Refactor AI Co-Scientist Evaluation to Use DeepEval Native Runner

Please first read and understand the latest code in this repository:

https://github.com/Siong23/ai-co-scientist

Focus especially on:

- `eval/scripts/evaluate_run.py`
- `eval/rubrics/deepeval_metrics.py`
- `eval/rubrics/suites.py`
- `eval/rubrics/retrieval_context.py`
- `eval/rubrics/experimental_readiness.py`
- `eval/tests/test_deepeval_metrics.py`
- `eval/tests/test_run_parser.py`
- `eval/README.md`
- `eval/pyproject.toml`

The current `eval` project pins:

```toml
deepeval==4.2.2
```

Do **not** upgrade DeepEval unless implementation is impossible otherwise, and do not change production application behavior unrelated to evaluation.

---

## Goal

Refactor the evaluation harness so that LLM-backed evaluation uses **DeepEval's native `deepeval.evaluate()` runner and native terminal result display**, instead of manually calling `metric.measure()` and then dumping the entire custom JSON report to the terminal.

The desired UX is:

```powershell
uv run python scripts/evaluate_run.py `
  ../results/runs/run-20260911-073448-611c235c.json `
  --goal-file goals/goal_005_5G-NIDD.txt `
  --llm-metrics `
  --metric-suite hypothesis `
  --threshold 0.7 `
  --judge-model unsloth/qwen3.8-27b `
  --judge-base-url http://100.117.90.5:1234/v1/
```

After evaluation, the terminal should show **DeepEval's own native result output/table**, including metric names, scores, thresholds, and pass/fail status.

Do **not** implement a custom ASCII table, `Format-Table` equivalent, Rich table, or another hand-built renderer. I specifically want DeepEval's native terminal display produced by `deepeval.evaluate()`.

The full AI Co-Scientist audit report must still be saved as JSON, but the whole JSON must no longer be printed to terminal during `--llm-metrics` runs.

---

## 1. Replace standalone `metric.measure()` execution with `deepeval.evaluate()`

Current `eval/rubrics/deepeval_metrics.py` loops through metric specs and calls each metric's:

```python
metric.measure(test_case)
```

Refactor production execution to use:

```python
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
```

Conceptually:

```python
evaluation_result = evaluate(
    test_cases=[test_case],
    metrics=applicable_metrics,
    identifier=<run id>,
    async_config=AsyncConfig(
        run_async=False,
    ),
    display_config=DisplayConfig(
        print_results=True,
        show_indicator=True,
        display="all",
    ),
)
```

Verify the **exact DeepEval 4.2.2 API from the installed package before coding**. Do not assume API fields from another DeepEval version.

`run_async=False` is intentional. This project uses a local LM Studio judge, often a 27B reasoning model, and evaluation should remain sequential rather than concurrently sending several expensive judge requests.

Do not remove the existing metric-level:

```python
async_mode=False
```

unless DeepEval 4.2.2 proves it redundant and tests demonstrate that removing it is safe.

---

## 2. Preserve the current scientific metric definitions exactly

Do **not** change scoring rubrics in this task.

Preserve:

### hypothesis suite

- Goal alignment
- Experimental readiness
- Scientific testability
- Feasibility
- Scientific plausibility
- Novelty vs retrieved prior art

### rag suite

- Answer relevancy
- Faithfulness
- Contextual relevancy

### legacy suite

- Evidence support

Preserve:

- all existing GEval `evaluation_steps`
- Experimental readiness DAG
- `FAITHFULNESS_TRUTHS_LIMIT = 5`
- current thresholds
- current LocalModel configuration
- `temperature=0`
- suite semantics
- `all` excluding legacy Evidence support

This task is about the **runner and output UX**, not changing the evaluation science.

---

## 3. Preserve existing retrieval-context skip semantics

This requirement is critical.

Some metrics require substantive `retrieval_context`.

Examples:

- `Novelty vs retrieved prior art`
- `Faithfulness`
- `Contextual relevancy`
- legacy `Evidence support`

The current code intentionally skips these metrics when persisted evidence contains only citation metadata or no substantive passage text.

Keep that behavior.

Before calling `deepeval.evaluate()`:

1. select the metric specs for the requested suite;
2. inspect `RetrievalContext`;
3. split specs into:
   - runnable specs
   - skipped specs;
4. instantiate and pass **only runnable metrics** into `deepeval.evaluate()`.

Do not send a metric with missing required retrieval context to DeepEval just to let it error.

After DeepEval finishes, merge:

- completed native DeepEval metric results;
- project-specific skipped metric entries;

back into the existing AI Co-Scientist report schema and preserve the original metric order.

A skipped entry must remain:

```json
{
  "name": "Faithfulness",
  "status": "skipped",
  "reason": "..."
}
```

A skipped metric must not fail the run.

If every metric is skipped:

```json
{
  "status": "no_metrics_completed",
  "passed": false
}
```

and the CLI must return exit code `1`, preserving current behavior.

---

## 4. Convert DeepEval `EvaluationResult` back into the current report schema

The project already has a useful artifact/report schema and I do not want to lose it.

`deepeval.evaluate()` returns an `EvaluationResult`.

Inspect the exact DeepEval 4.2.2 result types, especially:

```python
evaluation_result.test_results
test_result.metrics_data
```

Convert each completed DeepEval metric result into the existing shape:

```json
{
  "name": "...",
  "status": "completed",
  "score": 0.7,
  "threshold": 0.7,
  "passed": true,
  "reason": "..."
}
```

Use the actual DeepEval 4.2.2 fields. Do not guess names if the installed objects differ.

Prefer a small adapter/helper function such as:

```python
metric_data_to_report_entry(...)
```

or equivalent.

Keep this translation isolated and testable.

Do not serialize arbitrary DeepEval Python objects into the existing report.

---

## 5. Let DeepEval own terminal rendering

When `--llm-metrics` is used:

DeepEval should render its native evaluation result display.

Remove the current behavior:

```python
print("AI Co-Scientist evaluation report")
print(json.dumps(parsed, indent=2, ensure_ascii=False))
```

for LLM-backed runs.

After DeepEval prints its native result, the AI Co-Scientist CLI may print only a very small footer, for example:

```text
AI Co-Scientist report written to:
reports/hyp-run-20260911-073448-611c235c.json
```

and, when necessary:

```text
Skipped metrics:
- Novelty vs retrieved prior art: no substantive retrieval context
```

Do not duplicate DeepEval's metric table with another custom table.

For parser-only mode without `--llm-metrics`, preserving the existing JSON stdout behavior is acceptable because there is no DeepEval terminal display in that mode.

---

## 6. Make `--report` convenient instead of mandatory

Keep backward compatibility:

```text
--report some/path.json
```

must continue to work exactly.

But when:

```text
--llm-metrics
```

is supplied and `--report` is omitted, automatically derive a project report path.

Suggested mapping:

```text
hypothesis -> reports/hyp-<run_id>.json
rag        -> reports/rag-<run_id>.json
all        -> reports/all-<run_id>.json
legacy     -> reports/legacy-<run_id>.json
```

Example:

```text
reports/hyp-run-20260911-073448-611c235c.json
```

Do not overwrite an explicitly provided `--report`.

Ensure `reports/` is created automatically.

If the run has no usable `run_id`, derive a safe fallback from the input JSON filename.

---

## 7. Keep judge configuration backward compatible

Continue supporting:

```text
--judge-model
--judge-base-url
LOCAL_MODEL_NAME
LOCAL_MODEL_BASE_URL
LOCAL_MODEL_API_KEY
```

Existing precedence must remain:

```text
CLI value first, then environment variable
```

The user should be able to put these in PowerShell/environment configuration:

```powershell
$env:LOCAL_MODEL_API_KEY = "lm-studio"
$env:LOCAL_MODEL_NAME = "unsloth/qwen3.8-27b"
$env:LOCAL_MODEL_BASE_URL = "http://100.117.90.5:1234/v1/"
```

and then run a shorter command:

```powershell
uv run python scripts/evaluate_run.py `
  ../results/runs/run-20260911-073448-611c235c.json `
  --goal-file goals/goal_005_5G-NIDD.txt `
  --llm-metrics `
  --metric-suite hypothesis
```

It should automatically:

- use threshold `0.7`;
- use the configured local judge;
- show DeepEval native terminal results;
- save the AI Co-Scientist JSON report automatically.

---

## 8. Do not accidentally create two confusing report systems

DeepEval 4.2.2 supports `DisplayConfig(results_folder=...)`, which can save native DeepEval test-run JSON.

For this task, the AI Co-Scientist `--report` JSON remains the canonical project report.

Do **not** enable an additional DeepEval `results_folder` by default if that would create a second report file every run and confuse users.

If you think native DeepEval result persistence is useful, make it an explicitly optional CLI feature such as:

```text
--deepeval-results-folder PATH
```

but this is optional and must not be required to complete this task.

The important requirement is DeepEval's **native terminal display**, not duplicate report persistence.

---

## 9. Preserve exit codes

Preserve the current command contract:

```text
0 = every completed metric passed
1 = at least one completed metric failed OR no metric could run
2 = invalid run / goal / judge config / evaluation execution error
```

Skipped metrics by themselves do not fail a run if other completed metrics pass.

---

## 10. Error handling

Preserve secret redaction.

Do not print:

- API keys
- Authorization headers
- environment secrets

Wrap errors from native `deepeval.evaluate()` into `LLMEvaluationError` or the existing project error boundary so the CLI still reports:

```text
Evaluation error: ...
```

and returns exit code `2`.

Do not swallow errors silently.

---

## 11. Tests

Update/add offline tests.

The test suite must not contact:

- LM Studio
- OpenAI
- Confident AI
- any network service

Add tests that verify:

1. production evaluation uses a native DeepEval-style evaluation runner rather than independently calling every metric's `measure()` from project code;
2. exactly one `LLMTestCase` is passed for the selected hypothesis;
3. the requested suite creates the correct metric instances;
4. `AsyncConfig(run_async=False)` is used;
5. `DisplayConfig(print_results=True, display="all")` is used;
6. native result `MetricData` is correctly converted into the existing report schema;
7. score, threshold, reason and success/pass are preserved;
8. metric order remains deterministic;
9. novelty is skipped when substantive prior art is absent;
10. Faithfulness and Contextual relevancy are skipped when substantive retrieval context is absent;
11. skipped metrics do not fail an otherwise passing run;
12. all-skipped suites return `no_metrics_completed`;
13. automatic report filenames are generated correctly;
14. explicit `--report` still overrides automatic naming;
15. parser-only/offline mode still works without importing/initializing an LLM judge;
16. existing goal mismatch validation still works;
17. existing secret redaction still works;
18. existing DeepEval 4.2.2 compatibility tests continue to pass.

Prefer dependency injection or monkeypatching `deepeval.evaluate` / a small project wrapper around it so tests stay offline.

Do not weaken tests merely to make the refactor pass.

---

## 12. Documentation

Update `eval/README.md`.

Explain that LLM-backed execution now uses:

```text
deepeval.evaluate()
```

and DeepEval's native terminal result display.

Add this recommended PowerShell workflow:

```powershell
cd eval

$env:LOCAL_MODEL_API_KEY = "lm-studio"
$env:LOCAL_MODEL_NAME = "unsloth/qwen3.8-27b"
$env:LOCAL_MODEL_BASE_URL = "http://100.117.90.5:1234/v1/"

uv run python scripts/evaluate_run.py `
  ../results/runs/run-20260911-073448-611c235c.json `
  --goal-file goals/goal_005_5G-NIDD.txt `
  --llm-metrics `
  --metric-suite hypothesis
```

Explain that:

- DeepEval prints the native result table automatically;
- the full AI Co-Scientist audit JSON is automatically saved under `reports/`;
- `--report` is only needed to override the destination;
- metric reasons remain available in the JSON report;
- skipped metrics may appear in the project report even if they are not passed to native DeepEval evaluation.

---

## 13. Scope constraints

Do not modify:

- hypothesis generation
- reflection
- ranking/Elo
- evolution
- retrieval behavior
- experiment execution
- metric rubrics

unless required to fix a regression caused by this refactor.

Do not add `rich`, `tabulate`, pandas, or another table library. DeepEval already owns terminal result rendering.

Do not replace `LocalModel` with a different provider abstraction unless required by DeepEval 4.2.2.

---

## 14. Validation before finishing

Run:

```powershell
cd eval
uv sync
uv run pytest
```

Also run/parser-check any existing offline commands that do not require LM Studio.

If an LM Studio server is not available, do not fake a live result. State that live native-terminal verification requires the configured endpoint.

At the end, report:

1. files changed;
2. architectural changes;
3. tests added/updated;
4. exact `uv run pytest` result;
5. final recommended PowerShell command;
6. any behavior that remains intentionally unchanged.

---

## Important implementation note

Do **not** simply replace `_run_metric()` with `evaluate()` and stop there.

The current project has its own:

- skip semantics for missing substantive retrieval context;
- project report schema;
- exit code contract;
- deterministic metric ordering;
- secret redaction;
- parser-only offline mode.

A careless migration could break these behaviors.

The intended responsibility split after the refactor is:

```text
AI Co-Scientist code
├── run artifact parsing
├── goal validation
├── metric-suite selection
├── retrieval-context skip policy
├── DeepEval metric construction
├── mapping DeepEval results back into project JSON schema
├── canonical audit report persistence
└── exit codes / error handling

DeepEval
├── metric execution
├── native terminal result display
└── evaluation result object
```

The final user workflow should be short and repeatable:

```powershell
cd eval

$env:LOCAL_MODEL_API_KEY = "lm-studio"
$env:LOCAL_MODEL_NAME = "unsloth/qwen3.8-27b"
$env:LOCAL_MODEL_BASE_URL = "http://100.117.90.5:1234/v1/"

uv run python scripts/evaluate_run.py `
  ../results/runs/run-20260911-073448-611c235c.json `
  --goal-file goals/goal_005_5G-NIDD.txt `
  --llm-metrics `
  --metric-suite hypothesis
```

Expected behavior:

1. DeepEval renders its native terminal evaluation results.
2. No giant AI Co-Scientist JSON blob is dumped to the terminal.
3. The canonical AI Co-Scientist audit report is saved automatically, e.g.:

```text
reports/hyp-run-20260911-073448-611c235c.json
```

4. Detailed reasons remain available in that JSON report.
