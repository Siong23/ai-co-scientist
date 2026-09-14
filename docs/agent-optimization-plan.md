# Agent optimization plan

Created: 2026-09-14. Replaces the 2026-09-06 `docs/quality-gaps.md` audit removed
in `84ab63b`; that file described mechanisms already shipped, this one describes
the work between the agents and the reference paper.

Status: all four items below shipped on 2026-09-14, in the order listed. The
offline suite passes after each. Whether they move the failing metrics is not
yet measured — that needs a fresh eval run against a live LM Studio server.

Sources: `references/nature-ai-co-scientist.pdf` (Fig. 1b and the "specialized
agents" section) and `references/nature-ai-co-scientist-supplement.pdf`
(section 3, quantitative ablation and sensitivity analysis).

Paper mechanisms describe the reference system running Gemini against web search.
They are not mandatory numerical thresholds for this local LM Studio
implementation, and Elo remains a relative ranking proxy, not scientific
validation. Every item below is justified by both a paper mechanism and a local
measurement.

## Measured starting point

DeepEval hypothesis-suite scores from the four most recent local runs in
`eval/reports/`. Threshold is 0.7 for every metric.

| Metric | all-run-...2dc89266 | hyp-baseline-611c235c | hyp-run-...611c235c | hyp-run-...7fe71a20 |
| --- | --- | --- | --- | --- |
| Goal alignment | 0.9 | 0.7 | 0.9 | 0.7 |
| Experimental readiness | 1.0 | 1.0 | 1.0 | 1.0 |
| Scientific testability | 0.9 | 1.0 | 1.0 | 0.9 |
| **Feasibility** | **0.3** | **0.6** | 0.7 | **0.2** |
| **Scientific plausibility** | **0.4** | **0.6** | 0.9 | **0.6** |
| **Novelty vs prior art** | **0.5** | **0.3** | **0.2** | 0.8 |

Alignment, readiness and testability already pass consistently. Feasibility,
plausibility and novelty fail in three runs out of four, and each maps to a paper
mechanism that is missing or unwired below.

## P0-1 — Ranking: multi-turn debate is dead code, and A/B order is biased

Shipped in `c9f0965`.

**Paper.** Top-ranked hypotheses are compared through multi-turn scientific
debates; lower-ranked ones use single-turn comparisons. The ablation reports the
debate prompt is significantly more effective on high-Elo matches and "almost
eliminated" the positional bias (second-player advantage) that the simple
comparison prompt exhibits.

**Current.** `generate_debate_argument` (`app/agents_modules/ranking_helpers.py:303`)
and `judge_debate` (`:377`) have no call sites anywhere in `app/` or `tests/`.
`4a0eb21` replaced that path with the single-shot `judge_hypotheses` (`:738`) and
left the two functions orphaned. Separately, `app/agents_modules/ranking.py:37`
sorts active hypotheses by `-elo_score` before pairing, so slot A is always the
higher-rated hypothesis. Any positional bias therefore accumulates in favour of
the current leader instead of averaging out.

**Fix.**

1. Route matches by rank: pairs where both hypotheses sit in the current top-k
   run the existing multi-turn debate; the rest keep the single-shot judge. Add
   `ranking.debate_enabled` and `ranking.debate_top_k` to `config.yaml` so the
   cost stays bounded on a local model.
2. Randomize or alternate A/B assignment per match, seeded deterministically from
   the hypothesis IDs so runs remain reproducible.

**Tests.** Offline: the debate path is selected above the threshold and not below
it; A/B assignment is stable for a given ID pair and not always Elo-ordered; the
existing abstention and Elo-safety gates are unchanged.

## P0-2 — Reflection: three review types missing, novelty score ungrounded

Shipped in `5e80e54` (deep verification) and `9990511` (prior-art hand-off).

**Paper.** Six review strategies: initial, full (with search), deep verification,
observation, simulation, recurrent/tournament. Deep verification decomposes a
hypothesis into assumptions and sub-assumptions, decontextualizes each, and
evaluates it independently to catch subtle reasoning errors. The ablation shows
that without search the reviewer rates ideas as highly novel (6.14, "very likely
novel") when the correct rating is 2.38 ("trivially modified").

**Current.** `call_llm_for_reflection` (`app/agents_modules/reflection_helpers.py:208`)
performs one combined scoring pass. `evaluate_claims` adds claim-level supporting
and contradictory evidence retrieval. `_format_recurring_review_guidance` (`:176`)
implements the recurrent review. Deep verification, observation review and
simulation review do not exist — `deep verification`, `observation review`,
`simulation review` and `subassumption` return no matches across `app/`.

`novelty_score` is judged only against the hypothesis's own `evidence_sources`,
which are the documents that produced it. The only prior-art retrieval lives in
the Generation novelty audit (`app/agents_modules/generation_helpers.py:2045`,
rejecting below 5/10), and its verdict is never passed to Reflection, so
Reflection re-judges novelty blind.

**Fix.**

1. Deep verification review: decompose the hypothesis into assumptions, evaluate
   each independently, and record whether a failed assumption is fundamental to
   the hypothesis or repairable downstream. Store the result on the reflection
   report so Ranking and Evolution can both read it. Targets plausibility.
2. Pass the Generation prior-art audit verdict (score, closest prior art,
   remaining novelty) into the Reflection prompt as review data, so the novelty
   score is anchored to retrieved prior art instead of the hypothesis's own
   supporting sources. Targets novelty.
3. Deferred to a later pass: observation review and simulation review. Both are
   extra model calls per hypothesis and neither maps to a currently failing
   metric.

**Tests.** Offline, mocking at `app.agents.call_llm`: assumption decomposition
parses and survives malformed output; a fundamental failed assumption changes the
recommendation while a non-fundamental one does not; the prior-art verdict reaches
the prompt, and its absence leaves current behaviour unchanged.

## P1-1 — Evolution: strategy rotation ignores the reviews

Shipped in `56edfdf`.

**Paper.** The Evolution agent refines top-ranked hypotheses with grounding,
coherence/feasibility, inspiration, combination, simplification and out-of-box
strategies. The ablation attributes a precision gain of 70.9% to 75.4% on GPQA
diamond to this refinement.

**Current.** All six strategies exist and are enabled
(`app/agents_modules/evolution_helpers.py`, `config.yaml:59`). But
`_strategies_for_cycle` (`app/agents_modules/evolution.py:58`) rotates through
them by `iteration_number` alone and never reads the parents' reflection reports.
With `max_candidates_per_cycle: 3` over six strategies, a given strategy is
reached roughly every other cycle regardless of what is actually wrong. When
Feasibility scores 0.2, the `feasibility` strategy is not preferentially applied.

**Fix.** Weight strategy selection by the parents' weakest reflection dimensions
(feasibility or plausibility low → `feasibility`; evidence quality low →
`grounding`; testability low → `simplification`; novelty low → `out_of_box`), and
treat an assumption the deep verification review contradicted as a feasibility
deficit. Rotation stays the baseline order and the tie-breaker, so the strategy
library still gets explored. Targets feasibility.

**Tests.** Offline: a parent with a low feasibility score selects `feasibility`
first; with no reflection reports the current rotation order is preserved; the
parent-count guard for `combination`/`inspiration`/`out_of_box` still holds.

## P1-2 — Meta-review: the research overview is a stub

Shipped in `c206745`.

**Paper.** At the end of computation the Meta-review agent synthesizes top-ranked
hypotheses into a research overview: research areas and directions, why each
matters, and specific experiments within each. The overview is also an input to
the Generation agent in subsequent iterations.

**Current.** `research_overview` (`app/agents_modules/meta_review.py:477`) is the
top three hypotheses serialized with `to_dict()` plus a list of next steps.
Generation consumes only the critique list
(`app/agents_modules/generation.py:491`); the overview never reaches it, so that
feedback loop is open. The critique synthesis itself
(`app/agents_modules/meta_review.py:18`) already reads both reviews and tournament
debates and matches the paper — it does not need changing.

**Fix.** Produce a structured overview (areas, rationale, suggested experiments)
from the existing bounded synthesis call, and feed it to Generation alongside the
critiques for research expansion.

**Tests.** Offline: overview structure is validated and malformed model output
falls back to the current heuristic shape; the Generation prompt includes the
overview when present and is unchanged when absent.

## Out of scope — accepted trade-offs

These differ from the paper by choice and are not planned work.

- `config.yaml:196` sets `generation_debate_rounds: 0`, disabling the Generation
  agent's simulated scientific debate. The comment records why: the
  post-generation novelty auditor already performs a targeted critique, and the
  debate costs three extra large-model calls. The paper's ablation values strategy
  diversity, but the local cost constraint is real.
- The Supervisor runs one action per step (`app/agents_modules/supervisor_planner.py`)
  rather than the paper's asynchronous worker queue with weighted agent sampling.
  Summary statistics, convergence detection and terminal-state evaluation are all
  present; only the concurrency model differs, and it does not affect output
  quality.

## What shipped

| Item | Commit | New configuration |
| --- | --- | --- |
| P0-1 Ranking | `c9f0965` | `ranking.debate_enabled`, `ranking.debate_top_k` |
| P0-2 deep verification | `5e80e54` | `reflection.deep_verification_enabled`, `reflection.max_assumptions` |
| P0-2 prior-art hand-off | `9990511` | none |
| P1-1 Evolution | `56edfdf` | none |
| P1-2 Meta-review | `c206745` | none |

Regression coverage: `tests/test_deep_verification.py` (new), plus additions to
`tests/test_ranking_performance.py`, `tests/test_evolution.py` and
`tests/test_agent_review_integrity.py`. The offline suite passes with no API key
and no network.

Cost per cycle rose by two ranking-model calls for each top-k match and one
reflection call per hypothesis. Both are bounded by the new configuration keys
and can be switched off.

## Remaining

- Observation review and simulation review, deferred from P0-2. Each is another
  model call per hypothesis and neither maps to a currently failing metric.
- Evolved children carry no prior-art audit, so the novelty anchor added in
  `9990511` does not reach them. Their novelty score is still judged against
  their parents' evidence alone.
- Measure the effect. The four items target plausibility, novelty and
  feasibility; none of that is confirmed until an eval run against a live
  LM Studio server reproduces the table above. Compare against
  `eval/reports/all-run-20260914-063317-2dc89266.json`.
