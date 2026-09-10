# AI Co-Scientist evaluation harness

This directory is an independent, artifact-based evaluation project. It reads
saved AI Co-Scientist run JSON files and does not import or invoke the
production application.

```text
AI Co-Scientist production
    -> results/runs/<run-id>.json
    -> eval parser
    -> DeepEval metrics
    -> evaluation report
```

The evaluator always performs deterministic parsing first:

1. load and validate a saved run JSON;
2. compare `research_goal.description` with a fixed goal;
3. locate final hypotheses using the application's ranking-step fallback;
4. select the hypothesis with the highest Elo score; and
5. extract its evidence sources.

DeepEval metrics are opt-in (`--llm-metrics`), so ordinary parsing and the
pytest suite remain offline.

## Setup and use

```shell
uv sync
uv run pytest
uv run python scripts/evaluate_run.py ../results/runs/<run-id>.json
```

DeepEval is pinned to `4.2.2` in `eval/pyproject.toml` and belongs only to this
project. It is never added to the parent application's `requirements.txt`.

## Metric suites

Running every metric on every run mixes unrelated failure modes into one score,
so metrics are grouped into suites selected with `--metric-suite`. `--llm-metrics`
remains the switch that enables LLM judging at all.

| Suite | Metrics | Answers |
| --- | --- | --- |
| `hypothesis` | Goal alignment, Scientific testability, Feasibility, Scientific plausibility, Novelty vs retrieved prior art | Is the idea itself any good? |
| `rag` | Answer relevancy, Faithfulness, Contextual relevancy | Is it grounded in what was retrieved? |
| `all` (default) | `hypothesis` + `rag` | Both of the above. |
| `legacy` | Evidence support | The superseded custom metric (see below). |

`all` deliberately excludes `legacy`.

### `hypothesis` — custom `GEval` metrics

Each is a `GEval` metric defined with explicit `evaluation_steps` (never
`criteria` as well) so that judging is as reproducible as an LLM judge allows.
Scores are DeepEval's 0–1 scale, higher is better.

| Metric | Test-case fields | What it measures |
| --- | --- | --- |
| Goal alignment | `input`, `actual_output` | Whether the hypothesis addresses the research goal and its explicit constraints, rather than a tangential or generic restatement. |
| Scientific testability | `input`, `actual_output` | Whether the hypothesis is falsifiable and operationalizable: a stated intervention, a measurable outcome, a comparison or control where one is needed, and a predicted effect. Scientific-sounding prose earns nothing on its own. |
| Feasibility | `input`, `actual_output` | Whether the work could realistically be executed on the information given — required data, equipment, measurements, complexity, implementation burden — without assuming unnamed proprietary datasets or unavailable hardware. |
| Scientific plausibility | `input`, `actual_output` | Whether the proposed mechanism is internally coherent: no unsupported causal jumps, no self-contradiction, no physically or computationally impossible assumptions, and a conclusion that follows from the mechanism. The judge is told not to claim it verified anything against external literature. |
| Novelty vs retrieved prior art | `input`, `actual_output`, `retrieval_context` | Whether the hypothesis meaningfully differs from the approaches present in the **retrieved** prior art. It never claims novelty from the hypothesis text alone, and it is **skipped** when the artifact holds no substantive prior-art text. |

### `rag` — DeepEval's built-in metrics

These are DeepEval's own implementations rather than GEval rewrites of them.

| Metric | Test-case fields | What it measures |
| --- | --- | --- |
| `AnswerRelevancyMetric` | `input`, `actual_output` | Whether the selected hypothesis actually responds to the research goal. Runs even when no evidence text exists. |
| `FaithfulnessMetric` | `input`, `actual_output`, `retrieval_context` | Whether the hypothesis's claims are supported by the retrieved evidence. **Skipped** without substantive evidence text. |
| `ContextualRelevancyMetric` | `input`, `retrieval_context` | Whether the evidence retrieved for the hypothesis is relevant to the research goal. **Skipped** without substantive evidence text. DeepEval 4.2.2 scores the context against the input only, so `actual_output` is not among this metric's required params. |

### Evidence support (`legacy`) — superseded

The original custom `Evidence support` GEval measures the same failure mode as
`FaithfulnessMetric`: are the hypothesis's claims backed by the retrieved
passages? Running both would double-count one failure. `FaithfulnessMetric` wins
the default slot because it is DeepEval's purpose-built RAG metric — it
decomposes the output into claims and the context into truths rather than asking
for one holistic score — so `Evidence support` moved to `--metric-suite legacy`
and is in no default run. It is retained only so older comparisons can be
reproduced; when it runs, it uses the same validated retrieval context as
Faithfulness, not the raw source JSON it used previously.

## Required artifact fields, and when RAG metrics are skipped

Metrics are only reported as scored when the artifact genuinely contains what
DeepEval needs. The mapping from persisted run JSON to test case is:

| Test-case field | Persisted source |
| --- | --- |
| `input` | `research_goal.description` |
| `actual_output` | selected hypothesis `title` + `text` |
| `retrieval_context` | substantive passages from the selected hypothesis's `evidence_sources` |

`rubrics/retrieval_context.py` builds `retrieval_context` and is the single
place that decides what counts as evidence. It reads, in order:

1. `evidence_sources[].evidence_refs[].text` — the retrieved chunk passages, one
   string per passage; and
2. `evidence_sources[].content`, `.abstract`, `.summary` — source-level prose.

A passage must be at least 200 characters to count. Nothing is concatenated,
summarized, or generated: each persisted passage stays its own
`retrieval_context` entry, and the abstract, which the serializer writes into
both `abstract` and `summary`, is counted once.

Identifying metadata never counts as evidence on its own — `title`, `doi`,
`url`, `canonical_url`, `source_id`, `arxiv_id`, `authors`, `venue`,
`published`, `provider`, and friends. A record like

```json
{ "source_id": "doi:...", "title": "Some Paper" }
```

cannot support or contradict a claim, so any metric needing
`retrieval_context` is reported as

```json
{
  "name": "Faithfulness",
  "status": "skipped",
  "reason": "the persisted evidence sources carry only citation metadata (no source passage of at least 200 characters)"
}
```

A completed metric is reported as

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

Overall `passed` is computed from completed metrics only; a skipped metric never
fails the run by itself. When *no* metric in the requested suite could run, the
report carries `"status": "no_metrics_completed"` with a reason, `passed` is
false, and the command says so on stderr rather than exiting as a silent pass.

The report also records what the extractor found:

```json
"retrieval_context": {
  "substantive": true,
  "evidence_source_count": 2,
  "sources_with_text": 2,
  "passage_count": 10,
  "reason": null
}
```

### Known limits of the current run schema

- Runs that end before any hypothesis is produced contain no hypotheses at all
  and fail deterministic parsing before any metric runs.
- Not every run persists evidence with text. In runs that do, the selected
  hypothesis typically carries 1–2 sources and 5–10 passages.
- `evidence_sources[].content` is persisted but empty in the current artifacts;
  the real passage text lives in `evidence_refs[].text`, and the abstract in
  `abstract`/`summary`.
- `evidence_refs` on a *hypothesis* is a list of reference id strings, which is
  a different field from `evidence_refs` on an *evidence source* (a list of
  chunk objects). Only the latter carries `text`.
- The persisted artifact contains no independent golden answer; see the deferred
  metrics below.

## DeepEval with LM Studio

Start LM Studio's local OpenAI-compatible server and load a judge model. The
judge is configured explicitly and passed to every LLM-backed metric, with
`temperature=0` for reproducibility. No OpenAI account is required or used.

```shell
export LOCAL_MODEL_API_KEY=lm-studio
```

Hypothesis suite:

```shell
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite hypothesis \
  --judge-model <loaded-model-id> \
  --judge-base-url http://127.0.0.1:1234/v1/ \
  --threshold 0.7
```

RAG suite:

```shell
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite rag \
  --judge-model <loaded-model-id> \
  --judge-base-url http://127.0.0.1:1234/v1/ \
  --threshold 0.7
```

All currently supported metrics, written to a report file:

```shell
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite all \
  --judge-model <loaded-model-id> \
  --judge-base-url http://127.0.0.1:1234/v1/ \
  --threshold 0.7 \
  --report reports/<run-id>.json
```

On PowerShell, use `$env:LOCAL_MODEL_API_KEY = "lm-studio"` and replace the
trailing backslashes with backticks. DeepEval's OpenAI-compatible adapter
requires an API-key value even when LM Studio authentication is disabled; in
that case the value is only a non-secret placeholder. The model and base URL can
instead be supplied through `LOCAL_MODEL_NAME` and `LOCAL_MODEL_BASE_URL`. Real
credentials are never accepted as a CLI argument, printed, or written to the
report — error text passes through environment-secret redaction first.

The command exits with status 0 when all completed metrics pass, 1 when at least
one metric is below threshold or no metric could run, and 2 for invalid input,
configuration, or judge errors.

The default fixed goal is `goals/goal_001_perovskite_humidity.txt`. Supply
another goal with `--goal-file PATH`. A goal mismatch is reported as a
validation error and returns a non-zero exit status.

The selection behavior intentionally mirrors `app.run_store._final_hypotheses`:
`ranking_final` wins, otherwise the highest-numbered ranking step wins, and
otherwise the first step containing hypotheses is used. Empty ranking steps are
skipped. Within the selected candidates, the highest numeric `elo_score` is
chosen (a missing score has the application's default value of zero).

## Deferred metrics

### Contextual Precision and Contextual Recall

`ContextualPrecisionMetric` and `ContextualRecallMetric` require an
independently defined `expected_output` alongside `input`, `actual_output`, and
`retrieval_context`. The persisted run artifact contains no golden reference,
and there is no honest substitute: the research goal is the question, not the
answer; the selected hypothesis is the system's own output; and another
LLM-generated hypothesis is just a second output. Scoring retrieval against
fabricated ground truth would measure nothing. These metrics stay deferred until
a human-authored golden set exists for these goals.

### Agent trajectory metrics

`TaskCompletionMetric`, `StepEfficiencyMetric`, `PlanQualityMetric`,
`PlanAdherenceMetric`, `ToolCorrectnessMetric`, and `ArgumentCorrectnessMetric`
evaluate an agent's trajectory, which needs DeepEval tracing (`@observe` inside
the production agents) or recorded tool-call ground truth. This harness is
deliberately artifact-based and independent of the production package, and
production code is not instrumented for evaluation. Trajectory and
component-level evaluation is therefore a possible future phase, not part of
this suite.

## Limitations of LLM-as-a-judge scientific evaluation

**A high score from an LLM judge is not proof that a scientific claim or a
citation is factually true.** Every score here is one language model's
qualitative opinion about text.

- Faithfulness measures agreement between the hypothesis and the *retrieved
  passages*. It does not check whether those passages are correct, whether the
  cited paper exists, or whether the citation was applied to the right claim.
- Novelty is measured only against what this run retrieved. A hypothesis can
  score well simply because the relevant prior work was never retrieved.
- Plausibility and feasibility judge the reasoning that is written down. A
  fluent, confident, well-structured hypothesis can outscore a correct but
  tersely stated one.
- Judges are sensitive to phrasing, model choice, and quantization. Compare
  scores only across runs judged by the same model at the same settings, and
  treat them as a triage signal for human review — not as a verdict.
