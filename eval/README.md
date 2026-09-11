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

## Who does what

An LLM-backed run is executed by DeepEval's own runner, not by this project:

```text
this harness                         DeepEval
─────────────────────────────────    ─────────────────────────────────
parse the run, validate the goal
select the metric suite
skip what the artifact cannot
  ground
build the metrics and the one
  LLMTestCase
                                 ->  measure every metric it is given
                                 ->  print the native result table
                                 <-  return an EvaluationResult
map the results back into the
  audit JSON, and file it
```

The whole suite goes through one `deepeval.evaluate()` call, configured with
`AsyncConfig(run_async=False)` so a local judge is never asked several questions
at once, and `DisplayConfig(print_results=True)`, which is what puts the result
table on the terminal. There is no hand-built score table here: what you see is
DeepEval's own, and the AI Co-Scientist audit JSON goes to a file rather than
being dumped underneath it.

## Setup and use

```shell
uv sync
uv run pytest
uv run python scripts/evaluate_run.py ../results/runs/<run-id>.json
```

DeepEval is pinned to `4.2.2` in `eval/pyproject.toml` and belongs only to this
project. It is never added to the parent application's `requirements.txt`.

The recommended workflow configures the judge once, in `eval/.env`:

```powershell
cd eval
Copy-Item .env.example .env   # then edit it
```

```ini
LOCAL_MODEL_API_KEY=lm-studio
LOCAL_MODEL_NAME=unsloth/qwen3.8-27b
LOCAL_MODEL_BASE_URL=http://100.117.90.5:1234/v1/
```

Every later run is then just the run and the goal:

```powershell
uv run python scripts/evaluate_run.py `
  ../results/runs/run-20260911-073448-611c235c.json `
  --goal-file goals/goal_005_5G-NIDD.txt `
  --llm-metrics `
  --metric-suite hypothesis
```

That run uses the default threshold of `0.7`, shows DeepEval's native result
table, and writes the full audit report to
`reports/hyp-run-20260911-073448-611c235c.json` by itself. `--report` is only
needed to send it somewhere else.

### The env file

`scripts/evaluate_run.py` reads `eval/.env` at start-up, wherever it is invoked
from, and merges `KEY=value` lines into the process environment. Precedence runs
lowest to highest:

```text
eval/.env  ->  the shell environment  ->  --judge-model / --judge-base-url
```

A variable already in the environment is never overwritten, so one-off runs
still work with `$env:LOCAL_MODEL_NAME = "..."` or the CLI flags, and CI keeps
setting its own values. Everything after the first `=` is the value, so an
inline `#` is part of it rather than a comment.

That precedence is easy to forget once a shell variable has been sitting in a
session for an hour, so an `--llm-metrics` run says when one shadows the file:

```text
Note: LOCAL_MODEL_NAME from the environment overrides .env ('qwen/qwen3.8-27b' instead of 'unsloth/qwen3.8-27b')
```

Without it a stale `$env:LOCAL_MODEL_NAME` is invisible until the server fails
to load a model the env file never named. Credentials are named but never
quoted, from either side.

`.env` is git-ignored and `.env.example` is the committed template — keep real
credentials out of the template. `LOCAL_MODEL_API_KEY` is a non-secret
placeholder whenever the LM Studio server has authentication disabled, which is
the usual case here; DeepEval's OpenAI-compatible adapter simply refuses to run
without some value.

The env file is also the natural home for a reasoning judge's timeout
overrides, which would otherwise have to be exported before every run — see
[Judge timeouts](#judge-timeouts).

## Metric suites

Running every metric on every run mixes unrelated failure modes into one score,
so metrics are grouped into suites selected with `--metric-suite`. `--llm-metrics`
remains the switch that enables LLM judging at all.

| Suite | Metrics | Answers |
| --- | --- | --- |
| `hypothesis` | Goal alignment, Experimental readiness, Scientific testability, Feasibility, Scientific plausibility, Novelty vs retrieved prior art | Is the idea itself any good, and could anyone run it? |
| `rag` | Answer relevancy, Faithfulness, Contextual relevancy | Is it grounded in what was retrieved? |
| `all` (default) | `hypothesis` + `rag` | Both of the above. |
| `legacy` | Evidence support | The superseded custom metric (see below). |

`all` deliberately excludes `legacy`.

### `hypothesis` — custom metrics

Scores are DeepEval's 0–1 scale, higher is better. One metric is a `DAGMetric`
(below); the rest are `GEval` metrics defined with explicit `evaluation_steps`
(never `criteria` as well) so that judging is as reproducible as an LLM judge
allows.

#### Experimental readiness — a `DAGMetric`

"Could someone actually run this?" is not a matter of opinion, so it is not
scored like one. A hypothesis is executable exactly when it names the four
things an experimenter needs before touching any equipment, and this metric asks
for them one rung at a time:

| Rung | The question it asks | Score if absent |
| --- | --- | --- |
| intervention | What gets manipulated — the independent variable? | 0.0 |
| measurement | What gets recorded — the dependent variable? | 0.4 |
| comparison | What is the result judged against — a control or baseline? | 0.6 |
| prediction | Which way is the effect expected to go? | 0.8 |
| — | all four present | 1.0 |

The ladder is ordered by dependency, not by importance. A dependent variable
means nothing without an intervention to attribute it to, and a predicted
direction means nothing without something to measure, so a hypothesis that fails
an early rung cannot be rescued by a later one. The gap between a missing
intervention and every other failure is deliberate: without a manipulated
variable there is no experiment to design at all.

**Why a DAG rather than another GEval.** A GEval judge returns one holistic 0–1
opinion that moves between runs and cannot be audited. Here each rung is a
separate yes/no judgement and the score is fixed by the graph, so every lost
point names the slot that was left empty. "0.6, because it states no comparator"
is reviewable in a way that "0.6" is not.

Every rung reads `input` and `actual_output` only, never `retrieval_context`:
readiness is a property of the hypothesis text, and a slot that only the
retrieved evidence fills is still a slot the hypothesis left empty. Each rung
also tells the judge not to supply the missing piece from its own domain
knowledge — judges reliably repair a vague hypothesis by inventing the obvious
metric or the conventional baseline, which is precisely the failure this metric
exists to detect.

The tree short-circuits at the first "no", so a hypothesis naming no
intervention costs one judge call while one that clears every rung costs four.

`Experimental readiness` and `Scientific testability` overlap: both ask whether
the hypothesis can be put to a test. They are kept separate because they fail
differently — readiness reports *which* slot is empty and scores identically on
repeat runs, while testability gives a holistic judgement that also reacts to
how convincingly the test is framed. Read readiness first; treat testability as
corroboration, not as a second independent vote.

#### The `GEval` metrics

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
| `FaithfulnessMetric` | `input`, `actual_output`, `retrieval_context` | Whether the hypothesis's claims are supported by the retrieved evidence. **Skipped** without substantive evidence text. Runs with `truths_extraction_limit=5`; see below. |
| `ContextualRelevancyMetric` | `input`, `retrieval_context` | Whether the evidence retrieved for the hypothesis is relevant to the research goal. **Skipped** without substantive evidence text. DeepEval 4.2.2 scores the context against the input only, so `actual_output` is not among this metric's required params. |

#### Why Faithfulness caps its truth extraction

`FaithfulnessMetric` works in two stages: it distils the retrieval context into
a list of truths, then checks each claim in the hypothesis against that list.
Left uncapped — DeepEval's default — the first stage is unbounded, and on one
real run it produced **141 truths**, among them the paper's title, its authors,
their affiliations, and their e-mail addresses. A hypothesis was then scored on
whether it contradicted a list of e-mail addresses.

That cost more than the noise it added. Measured against this project's local
judge on a run with 3 evidence sources and 11 passages:

| `truths_extraction_limit` | truths | generated tokens | wall clock |
| --- | --- | --- | --- |
| unset (DeepEval default) | 141 | 34,220 | 6 min 20 s |
| 10 | 10 | 2,818 | 38 s |
| **5 (this project)** | **5** | **1,378** | **21 s** |

And that is only the first of the metric's four sequential calls: the truths are
fed back into the verdict prompt in full, so an uncapped run also enlarges the
stage after it — which is where the judge ran out of room and returned the
unparseable JSON that DeepEval reports as `Evaluation LLM outputted an invalid
JSON. Please use a better evaluation model.` The model was not the problem.

Capping also makes the metric more permissive, and that cuts both ways. The
score is the share of the hypothesis's claims that no truth contradicts, so
fewer truths mean fewer chances to be contradicted: a capped run scores at least
as high as an uncapped one on the same hypothesis, never lower. Read a
Faithfulness of 1.0 as "nothing in the five most important retrieved facts
contradicts this", not as "every claim was verified".

The limit is 5 rather than 10 because raising it does not buy breadth. DeepEval
joins every passage into one string before asking, so "per document" has no
effect here: at both 5 and 10, every truth came from whichever source appears
first. **Faithfulness therefore checks claims mainly against the first evidence
source, not evenly across all of them** — a limitation of the metric worth
stating in any write-up that quotes its score.

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
`retrieval_context` is left out of the `deepeval.evaluate()` call and reported as

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
export LOCAL_MODEL_NAME=<loaded-model-id>
export LOCAL_MODEL_BASE_URL=http://127.0.0.1:1234/v1/
```

Hypothesis suite:

```shell
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite hypothesis
```

RAG suite:

```shell
uv run python scripts/evaluate_run.py \
  ../results/runs/<run-id>.json \
  --goal-file goals/<goal>.txt \
  --llm-metrics \
  --metric-suite rag
```

`--judge-model` and `--judge-base-url` override the environment for a single
run, and take precedence over `LOCAL_MODEL_NAME` and `LOCAL_MODEL_BASE_URL`:

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

### Where the results go

An `--llm-metrics` run produces two things, and they do not overlap:

- **the terminal** carries DeepEval's native result display — metric names,
  scores, thresholds, pass/fail, and its own aggregate table. Nothing is printed
  over it;
- **`reports/<prefix>-<run-id>.json`** holds the complete AI Co-Scientist audit
  report, including every metric's `reason`, which is the part worth reading.

The report path is derived from the suite whenever `--report` is omitted:
`hypothesis` writes `reports/hyp-<run-id>.json`, and `rag`, `legacy`, and `all`
write the `rag-`, `legacy-`, and `all-` prefixes. A run artifact with no usable
`run_id` falls back to the run file's own name. `reports/` is created when it
does not exist, and an explicit `--report` always wins.

A metric the artifact cannot ground is never handed to DeepEval, so it does not
appear in DeepEval's table at all. It is still in the JSON report, in its usual
position and marked `"status": "skipped"`, and the command names it under the
report path:

```text
AI Co-Scientist report written to:
reports/rag-run-20260911-073448-611c235c.json
Skipped metrics:
- Faithfulness: the persisted evidence sources carry only citation metadata (no source passage of at least 200 characters)
```

DeepEval can save its own timestamped test-run JSON as well, but that stays off
by default so that one run produces one report. Ask for it with
`--deepeval-results-folder PATH` when you want DeepEval's raw record too.

DeepEval's score cache is switched off, so re-running a suite always re-judges
rather than replaying an earlier verdict. That is deliberate for an audit, and
it also avoids a Windows failure: reading the cache back takes a shared file
lock that needs pywin32, and without it every run after the first died at the
very end, once the judge had already done all the work.

Without `--llm-metrics` nothing is measured and no report path is derived, so
that mode still prints the parsed run as JSON on stdout.

On PowerShell, use `$env:LOCAL_MODEL_API_KEY = "lm-studio"` and replace the
trailing backslashes with backticks. DeepEval's OpenAI-compatible adapter
requires an API-key value even when LM Studio authentication is disabled; in
that case the value is only a non-secret placeholder. Real credentials are never
accepted as a CLI argument, printed, or written to the report — error text
passes through environment-secret redaction first.

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

### Judge timeouts

DeepEval caps every judge call: `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS` defaults
to 88.5 seconds and `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS` to 180. A metric makes
several calls in sequence, so a slow local judge exhausts the task budget and
fails with `RetryError[<Future ... raised TimeoutError>]` rather than a score.

Reasoning models are the usual cause. On one 27B reasoning model served by LM
Studio, a single short judging call took ~16 seconds and spent 368 of its 426
generated tokens on hidden reasoning — the visible verdict was the small
remainder. Raise both budgets before running:

```shell
export DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=2400
export DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=900
```

Or set the same two in `eval/.env`, where they apply to every run without being
re-exported.

A smaller non-reasoning judge is the cheaper fix, at the cost of judging
quality. Whichever you pick, keep it fixed across every run you intend to
compare.

## Step-by-step: evaluating one run

PowerShell, from `eval/`. Substitute your own run id, goal file, judge model,
and endpoint. Forward slashes work throughout.

**1. Install the project.**

```powershell
uv sync
```

**2. Find the goal file matching the run.** The evaluator refuses to score a run
against a goal it was not produced for, so check first:

```powershell
Get-ChildItem ../results/runs/*.json | ForEach-Object { $g = (Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json).research_goal; if ($g -is [string]) { $d = $g } else { $d = $g.description }; [PSCustomObject]@{ Run = $_.Name; Goal = $d.Substring(0, [Math]::Min(60, $d.Length)) } } | Format-Table -AutoSize
```

A wrong choice fails with `research goal mismatch: expected ... found ...`.

**3. Parse the run offline first.** No `--llm-metrics` means no model is called
and nothing is spent:

```powershell
uv run python scripts/evaluate_run.py ../results/runs/<run-id>.json --goal-file goals/<goal>.txt
```

Confirm `goal_matches` is true, the selected hypothesis has text, and
`evidence_sources` carries real passages rather than citation metadata alone.
That last point decides whether the `rag` metrics will run or be skipped.

**4. Configure the judge**, once, in `eval/.env` — copied from
`.env.example`. The API key is required even against an unauthenticated server,
where it is only a non-secret placeholder:

```ini
LOCAL_MODEL_API_KEY=lm-studio
LOCAL_MODEL_NAME=<loaded-model-id>
LOCAL_MODEL_BASE_URL=http://127.0.0.1:1234/v1/
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=2400
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=900
```

For a one-off override, export the same names in the shell instead; the
environment wins over the file.

Check the model actually loads before starting a long run — a server short on
memory answers `Failed to load model` to the first request.

**5. Score the hypothesis.** DeepEval prints its result table as it goes, and
the audit report lands in `reports/hyp-<run-id>.json` on its own:

```powershell
uv run python scripts/evaluate_run.py ../results/runs/<run-id>.json --goal-file goals/<goal>.txt --llm-metrics --metric-suite hypothesis
```

**6. Score its grounding.**

```powershell
uv run python scripts/evaluate_run.py ../results/runs/<run-id>.json --goal-file goals/<goal>.txt --llm-metrics --metric-suite rag
```

Run the suites separately rather than reaching for `--metric-suite all`.
DeepEval's own guidance is no more than about five metrics at a time, and a
mixed report makes a retrieval failure and a reasoning failure look alike.

**7. Read the reasons.** The scores were already on the terminal; what the
report adds is why.

```powershell
$r = Get-Content reports/hyp-<run-id>.json -Raw -Encoding UTF8 | ConvertFrom-Json; $r.llm_evaluation.metrics | ForEach-Object { "$($_.name): $($_.reason)" }
```

The reasons matter more than the numbers. `Experimental readiness` in particular
names the missing slot, which is the part that tells you what to fix. The
numbers are still there when you want them back:

```powershell
$r.llm_evaluation.metrics | Format-Table name, status, score, threshold, passed -AutoSize
```

One run scores one hypothesis and settles nothing on its own. For comparisons
across goals, conditions, and repeats, use `scripts/idea_bench.py` — see
`IDEA_BENCH_GUIDE.md`.

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
