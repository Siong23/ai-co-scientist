You are working in my existing repository:

`Siong23/ai-co-scientist`

Your task is to improve the existing DeepEval-based evaluation harness by adding scientifically useful metrics for the AI Co-Scientist system.

Do not redesign the production application. First inspect the repository and understand the current evaluation architecture before making changes.

Read and obey both:

`AGENTS.md`

and especially:

`eval/AGENTS.md`

The `eval/` directory is intentionally an independent artifact-based evaluation project. Preserve that architecture.

The required boundary is:

```text
AI Co-Scientist production
    -> persisted results/runs/*.json
    -> eval parser
    -> DeepEval metrics
    -> evaluation report
```

Do NOT import production agents into `eval/`.

Do NOT invoke application agents from the evaluator.

Do NOT add DeepEval to the root production dependencies.

Do NOT modify `app/`, `app.py`, production RAG logic, production LLM calls, or production tracing as part of this task.

Parent application files and `results/runs/*.json` may be inspected read-only to understand persisted schemas.

## Current state to inspect

The repository already has:

```text
eval/
├── pyproject.toml
├── README.md
├── rubrics/
│   └── deepeval_metrics.py
├── scripts/
│   └── evaluate_run.py
├── tests/
│   └── test_deepeval_metrics.py
└── ...
```

DeepEval is already integrated.

`eval/rubrics/deepeval_metrics.py` currently evaluates approximately:

```text
Goal Alignment
Scientific Testability
Evidence Support
```

`eval/scripts/evaluate_run.py` already supports:

```text
--llm-metrics
--judge-model
--judge-base-url
--threshold
--report
```

and uses a DeepEval `LocalModel` against LM Studio's OpenAI-compatible endpoint.

Preserve this functionality.

## First: verify DeepEval 4.2.2 API

Before coding, verify the implementation against the current DeepEval 4.2.2 API, not older examples.

Update the eval project's DeepEval dependency/lock from the currently locked 4.2.0 to 4.2.2, while keeping DeepEval exclusively inside `eval/`.

Prefer a stable dependency constraint appropriate for this independent evaluation project, and regenerate the uv lockfile correctly.

DeepEval 4.2.2 behavior that must be respected:

1. Metric scores are 0–1 and higher is better.
2. `GEval` should use either `criteria` or explicit `evaluation_steps`, not both.
3. Prefer explicit `evaluation_steps` for our custom scientific metrics because we want more stable and reproducible judging.
4. Only include `SingleTurnParams` actually referenced by the metric.
5. Continue passing the explicitly configured local judge model to every LLM-backed metric.
6. Configure the local judge with temperature 0 where supported.

Review the existing `GEval` definitions. If they currently pass both `criteria` and `evaluation_steps`, correct them as part of this change.

## Metric architecture

Do NOT run 10–20 metrics indiscriminately in one evaluation.

Implement metric suites so that different failure modes can be evaluated independently.

Add a CLI option similar to:

```text
--metric-suite hypothesis
--metric-suite rag
--metric-suite all
```

`--llm-metrics` must remain the switch that enables LLM judging.

Preserve backward compatibility for the existing command as much as reasonably possible.

### Hypothesis metric suite

Implement these custom scientific metrics using `GEval`.

### 1. Goal Alignment

Keep the existing metric.

Evaluate whether the hypothesis directly addresses the original research goal and its explicit constraints.

Use only:

```text
INPUT
ACTUAL_OUTPUT
```

### 2. Scientific Testability

Keep and improve the existing metric if necessary.

Evaluate whether the hypothesis is falsifiable and operationalizable.

The judge should consider:

```text
clear intervention / independent variable
measurable dependent variable or outcome
comparison/control where appropriate
predicted effect
ability to design an experiment that can falsify the hypothesis
```

Do not reward a hypothesis merely for sounding scientific.

Use only:

```text
INPUT
ACTUAL_OUTPUT
```

### 3. Feasibility

Add a custom `GEval`.

Evaluate whether the proposed research can realistically be executed given the information contained in the hypothesis and research goal.

Consider:

```text
required data
required equipment/resources
experimental complexity
availability of measurements
implementation burden
unrealistic dependencies
```

Do not assume access to unspecified proprietary datasets, unavailable equipment, or impossible resources.

Use:

```text
INPUT
ACTUAL_OUTPUT
```

### 4. Scientific Plausibility

Add a custom `GEval`.

Evaluate whether the proposed mechanism and causal/scientific reasoning are internally coherent and consistent with established scientific reasoning visible in the supplied test case.

Penalize:

```text
unsupported causal jumps
internally contradictory mechanisms
physically/biologically/computationally implausible assumptions
conclusions that do not follow from the proposed mechanism
```

Do not let the judge pretend it has verified external literature that was not supplied.

Use:

```text
INPUT
ACTUAL_OUTPUT
```

### 5. Novelty Relative to Retrieved Prior Art

Add a custom `GEval`, but this metric MUST require substantive persisted evidence/prior-art text.

It must NOT claim scientific novelty from the hypothesis text alone.

Evaluate whether the hypothesis meaningfully differs from approaches represented in the supplied retrieval/prior-art context.

Use:

```text
INPUT
ACTUAL_OUTPUT
RETRIEVAL_CONTEXT
```

If substantive prior-art text is unavailable, mark the metric:

```text
status = "skipped"
```

with a clear reason.

Do NOT ask the judge to use its own parametric knowledge as a substitute for retrieved prior art.

## RAG metric suite

Use DeepEval's built-in metrics instead of rebuilding them with GEval.

### 6. AnswerRelevancyMetric

Add:

```python
AnswerRelevancyMetric
```

For this application, treat the selected final hypothesis as the generated output and the research goal as input.

Required data:

```text
input
actual_output
```

This metric can run even when no evidence text exists.

### 7. FaithfulnessMetric

Add:

```python
FaithfulnessMetric
```

Required data:

```text
input
actual_output
retrieval_context
```

This metric should measure whether claims in the selected hypothesis are supported by the retrieved evidence.

CRITICAL:

Do not run Faithfulness against citation metadata alone.

For example, this is NOT sufficient retrieval context:

```json
{
  "source_id": "doi:...",
  "title": "Some Paper"
}
```

Faithfulness requires substantive evidence text/passages.

Inspect the persisted run JSON schema and the parent application's serializers read-only.

Look for genuine persisted text fields such as actual schema equivalents of:

```text
raw_text
retrieval_text
text
content
passage
excerpt
snippet
```

Do NOT assume these exact names exist.

Use only fields that actually exist in the persisted artifact and represent source evidence.

If the persisted run does not contain substantive evidence text, skip Faithfulness with an explicit reason.

Do NOT modify production code to manufacture the missing data during this task.

### 8. ContextualRelevancyMetric

Add:

```python
ContextualRelevancyMetric
```

Required data:

```text
input
actual_output
retrieval_context
```

Use the same validated substantive retrieval context as Faithfulness.

Evaluate whether the evidence retrieved for the hypothesis is relevant to the research goal.

Skip this metric when there is no substantive retrieval text.

## Existing Evidence Support metric

The existing custom `Evidence Support` GEval overlaps substantially with `FaithfulnessMetric`.

Do not blindly run both in the same default RAG suite and double-count the same failure mode.

Inspect compatibility requirements and tests.

Prefer replacing the default Evidence Support behavior with DeepEval `FaithfulnessMetric`, because it is the system-specific RAG metric designed for this purpose.

If keeping the old metric is necessary for backward compatibility, place it behind a clearly named legacy path/suite rather than evaluating both by default.

Document the decision.

## Metrics NOT to add in this change

Do NOT add `ContextualPrecisionMetric` or `ContextualRecallMetric` yet unless the persisted evaluation artifact contains a genuine independently defined `expected_output` / golden reference.

These DeepEval metrics require:

```text
input
actual_output
expected_output
retrieval_context
```

Never fabricate `expected_output`.

Do NOT use:

```text
research_goal
selected hypothesis
another LLM-generated hypothesis
```

as fake ground truth.

If there is no independent expected output, document these metrics as deferred.

Also do NOT add these agent trajectory metrics in this task:

```text
TaskCompletionMetric
StepEfficiencyMetric
PlanQualityMetric
PlanAdherenceMetric
ToolCorrectnessMetric
ArgumentCorrectnessMetric
```

Trajectory/component-level agent evaluation requires DeepEval tracing or tool-call ground truth, while this repository intentionally keeps `eval/` artifact-based and independent from production agents.

Do not add `@observe` to production code.

Document agentic metrics as a possible future phase only.

## Retrieval-context extraction

Create a small, testable helper responsible for converting persisted evidence into DeepEval `retrieval_context`.

Requirements:

The helper must distinguish:

```text
substantive source text
```

from:

```text
metadata-only evidence
```

Metadata such as:

```text
title
DOI
URL
source_id
authors
publication date
```

does not count as substantive evidence by itself.

Preserve each evidence passage as a separate string in the `retrieval_context` list where possible. Do not concatenate unrelated evidence into one fake document unless the persisted schema forces it.

Do not invent or summarize missing source text with an LLM.

## Metric execution architecture

Refactor `eval/rubrics/deepeval_metrics.py` cleanly rather than creating a long chain of special cases.

Prefer an architecture where metric definitions/suites are declarative and execution is centralized.

Every report entry should retain a consistent shape similar to:

```json
{
  "name": "Faithfulness",
  "status": "completed",
  "score": 0.84,
  "threshold": 0.7,
  "passed": true,
  "reason": "..."
}
```

Skipped metric:

```json
{
  "name": "Faithfulness",
  "status": "skipped",
  "reason": "No substantive persisted evidence passages were available."
}
```

Overall `passed` must be calculated only from completed metrics.

Skipped metrics must not automatically fail the entire run.

If no metric in a requested suite can be completed, handle that situation explicitly rather than silently returning success.

Do not expose environment API keys in:

```text
stdout
stderr
exceptions
reports
```

Continue using the existing secret-redaction approach.

## CLI behavior

Extend `evaluate_run.py` so these examples work:

```bash
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite hypothesis \
  --judge-model <model> \
  --judge-base-url http://127.0.0.1:1234/v1/ \
  --threshold 0.7
```

RAG evaluation:

```bash
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite rag \
  --judge-model <model> \
  --judge-base-url http://127.0.0.1:1234/v1/ \
  --threshold 0.7
```

All currently supported metrics:

```bash
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite all \
  --judge-model <model> \
  --judge-base-url http://127.0.0.1:1234/v1/ \
  --threshold 0.7 \
  --report reports/<run-id>.json
```

Keep LM Studio support local.

Do not add OpenAI as a requirement.

Continue supporting an unauthenticated LM Studio endpoint through a non-secret placeholder API key.

## Tests

All tests for this change must run offline.

Never require a live LM Studio server in the normal test suite.

Use fakes/mocks/monkeypatching around metric execution and model clients.

Add or update tests covering at least:

1. metric-suite selection;
2. all new metric names;
3. existing Goal Alignment behavior;
4. Scientific Testability;
5. Feasibility;
6. Scientific Plausibility;
7. Novelty being skipped without substantive prior-art text;
8. Answer Relevancy working without retrieval context;
9. Faithfulness being skipped for metadata-only evidence;
10. Faithfulness running when real persisted evidence text exists;
11. Contextual Relevancy skip/run behavior;
12. report serialization;
13. threshold pass/fail behavior;
14. all-skipped behavior;
15. explicit `LocalModel` forwarding;
16. no secret leakage;
17. CLI `--metric-suite` validation;
18. GEval construction compatible with DeepEval 4.2.2.

Specifically add a regression test ensuring custom GEval construction does not pass both `criteria` and `evaluation_steps`.

Do not make network calls in these tests.

## Documentation

Update `eval/README.md`.

Document:

```text
metric suites
what each metric measures
required artifact fields
when RAG metrics are skipped
LM Studio setup
example commands
why Contextual Precision/Recall are deferred
why agent trajectory metrics are deferred
limitations of LLM-as-a-judge scientific evaluation
```

Make it explicit that a high score from an LLM judge is not proof that a scientific claim or citation is factually true.

## Dependency update

Update only the independent `eval/` dependency environment.

Target DeepEval:

```text
4.2.2
```

Regenerate the appropriate uv lockfile.

Do not modify the parent production `requirements.txt` for DeepEval.

## Validation

After implementation, run from `eval/`:

```bash
uv sync
uv run pytest
```

Also run appropriate formatting/lint checks available for the eval project.

If the root repository's canonical offline test environment is available, run its required offline test command too, without enabling integration/network tests.

Do not run live LM Studio evaluations unless I explicitly ask you to.

## Scope discipline

Keep the diff focused.

Do not perform unrelated refactors.

Do not change production behavior.

Do not modify persisted run files.

Do not weaken existing validation or secret-handling protections.

Do not create a PR unless I ask.

Follow repository instructions regarding local commits.

## Final response

When finished, give me:

1. files changed;
2. exact metrics added;
3. metric-suite behavior;
4. DeepEval version before/after;
5. tests run and results;
6. any metrics skipped because current persisted run artifacts lack required information;
7. any important limitations discovered in the saved run schema;
8. the exact commands I should use to run `hypothesis`, `rag`, and `all` evaluation suites.

Do not claim a metric is correctly supported unless its required DeepEval test-case fields are genuinely available.