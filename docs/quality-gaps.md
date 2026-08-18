# Hypothesis Generation Quality Gaps

> Audit date: 2026-08-18
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist"
> Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)
> Supplementary: [Supplementary Notes (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf)

This document tracks quality defects in the hypothesis-generation pipeline
discovered by comparing the codebase against the published Co-Scientist paper
and its supplementary pseudocode/prompts.

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

---

## 🔴 High Priority (directly degrade output quality)

### QG-01  Meta-Review feedback loop is broken
- **Status:** ✅ fixed
- **Impact:** The paper's core "self-improving loop" relies on each iteration
  learning from the previous meta-review. Currently each generation/evolution
  round starts from scratch, ignoring prior-cycle critique.
- **Root cause:** `MetaReviewAgent.summarize_and_feedback` appends feedback to
  `context.meta_review_feedback` (`meta_review.py:190`), but neither
  `GenerationAgent` nor `EvolutionAgent` ever reads it.
- **Fix:** Inject `context.meta_review_feedback[-1]` into the generation prompt
  (around `generation.py:1180`) and into `build_evolution_prompt`.
- **Test:** Add an offline test asserting that generation prompts contain
  meta-review feedback when `context.meta_review_feedback` is non-empty.

### QG-02  Reflection has no REJECT verdict
- **Status:** ✅ fixed
- **Impact:** Low-quality hypotheses persist in the system forever (remain
  `is_active=True`), consuming Elo bandwidth and potentially becoming evolution
  parents.
- **Root cause:** `_recommendation_from_scores` (`reflection_helpers.py:38-43`)
  only returns `ACCEPT` (all scores ≥4) or `REVISE` (any score <4).
- **Fix:** Add 3-tier policy: `ACCEPT` (all ≥5), `REVISE` (any 3-4),
  `REJECT` (any <3 → set `is_active=False`).
- **Test:** Unit test that a hypothesis with a score of 2 gets `REJECT` and
  `is_active=False`.

### QG-03  Ranking model is hardcoded
- **Status:** ✅ fixed
- **Impact:** If `qwen/qwen3.6-35b-a3b` is not loaded in LM Studio, the entire
  ranking stage fails silently or errors out, ignoring the user's selected model.
- **Root cause:** `RANKING_LLM_MODEL = "qwen/qwen3.6-35b-a3b"` in
  `ranking_helpers.py:21` bypasses `research_goal.llm_model` and `config.yaml`.
- **Fix:** Read from `config.yaml` → `ranking_llm_model` with fallback to
  `research_goal.llm_model`.
- **Test:** Assert `judge_hypotheses` respects the configured model.

### QG-04  Balanced audit mode is too lenient
- **Status:** ⬜ open
- **Impact:** In default `balanced` mode, the auditor's own `REJECT` verdict,
  novelty <5/10, and weighted score <70/100 are downgraded to warnings instead
  of hard rejections.
- **Root cause:** `generation_helpers.py:1592-1643` converts most audit failures
  to `audit_warnings` rather than hard blocks.
- **Fix:** Promote auditor `REJECT` to a hard rejection; tighten the balanced
  thresholds (e.g. novelty <4 hard-reject, weighted <60 hard-reject).
- **Test:** Assert that a hypothesis with an auditor `REJECT` verdict is
  excluded from final output even in balanced mode.

---

## 🟡 Medium Priority (affect robustness and fairness)

### QG-05  Evolution near-duplicate detection is lexical-only
- **Status:** ⬜ open
- **Impact:** Paraphrased duplicates pass the `SequenceMatcher ≥ 0.92` gate.
  The check also only compares against immediate parents, not all active
  hypotheses.
- **Root cause:** `evolution_helpers.py:33, 133-137` uses `SequenceMatcher`
  without embedding similarity, scoped only to `parents`.
- **Fix:** Add cosine embedding similarity ≥ 0.85 check; compare against all
  active hypotheses in `context.hypotheses`.
- **Test:** Construct a paraphrase that passes SequenceMatcher but fails
  embedding similarity; assert rejection.

### QG-06  Ranking tournament has position bias
- **Status:** ⬜ open
- **Impact:** Hypothesis A is always presented first in pairwise debates,
  creating systematic scoring skew.
- **Root cause:** `judge_hypotheses` (`ranking_helpers.py:518-544`) always puts
  hypothesis_a before hypothesis_b with no randomization.
- **Fix:** Randomize A/B assignment per match, or run dual-order evaluation and
  average results.
- **Test:** Property test asserting that swapping A↔B does not change the
  expected winner.

### QG-07  `call_llm_for_hypothesis_revision` is orphaned
- **Status:** ✅ fixed
- **Impact:** `REVISE` verdicts have no effect; hypotheses keep their original
  text unchanged.
- **Root cause:** `reflection_helpers.py:344-391` implements the revision
  function, but it is never called in the supervisor or reflection flow.
- **Fix:** In the supervisor's post-reflection step, call
  `call_llm_for_hypothesis_revision` on each `REVISE` hypothesis.
- **Test:** Assert that a hypothesis with `REVISE` verdict has its text
  modified after the revision call.

### QG-08  No retry on full audit rejection
- **Status:** ⬜ open
- **Impact:** If all generated candidates fail the audit, the system returns an
  empty list without retrying.
- **Root cause:** `generation.py:1280-1282` immediately returns
  `([], ["All generated hypotheses were rejected..."])`.
- **Fix:** Implement 1 retry cycle using aggregated audit feedback to guide
  regeneration.
- **Test:** Mock the LLM to fail audit on the first call but succeed on retry;
  assert non-empty output.

---

## 🟢 Low Priority (polish and correctness)

### QG-09  Quality thresholds are hardcoded
- **Status:** ⬜ open
- **Impact:** Users cannot tune quality sensitivity without editing source code.
- **Affected values:**

  | Parameter | Value | Location |
  |---|---|---|
  | Audit score weights | 7 dims (10-20%) | `generation_helpers.py:1049-1057` |
  | Audit grounding cutoff | 4 / 5 | `generation_helpers.py:1318` |
  | Reflection accept cutoff | all ≥4 | `reflection_helpers.py:40` |
  | Near-dup threshold | 0.92 | `evolution_helpers.py:33` |
  | Meta-review diversity | <0.35 / >0.75 | `meta_review.py:82, 92` |
  | Ranking thread workers | 3 | `ranking.py:92` |

- **Fix:** Move all values to `config.yaml` under appropriate sections.

### QG-10  Meta-Review emits a false deactivation claim
- **Status:** ⬜ open
- **Impact:** The critique text claims "the lower-Elo duplicate was
  automatically deactivated", but no deactivation actually occurs.
- **Root cause:** `meta_review.py:157-159` generates the claim, but neither
  `ProximityAgent` nor `MetaReviewAgent` sets `is_active=False`.
- **Fix:** Either implement the deactivation, or remove the misleading text.

---

## 📊 Test Coverage Gaps

These are not bugs but missing test coverage that prevents catching regressions:

| Gap | Risk |
|---|---|
| No end-to-end pipeline integration test | Agent interface drift undetected |
| No automated quality benchmark (`metric-version: pre-0`) | Prompt/model changes can silently degrade quality |
| No property-based tests (Elo conservation, RRF monotonicity) | Boundary bugs missed |
| No adversarial input tests (prompt injection via retrieved text) | Security/quality risk |
| No multi-cycle diversity collapse test | Mode collapse after 5+ iterations |
| No extreme input tests (unicode, very long context, empty evidence) | Edge case crashes |

---

## Implementation Order

Phase 1 — close the quality loop (QG-01, QG-02, QG-03, QG-07):
```
meta_review feedback → generation/evolution prompts
reflection 3-tier verdicts + hypothesis deactivation
ranking model from config
connect revision function
```

Phase 2 — harden robustness (QG-04, QG-05, QG-06, QG-08):
```
tighten balanced audit mode
semantic dedup in evolution
ranking position randomization
audit-failure retry
```

Phase 3 — infrastructure (QG-09, QG-10, test gaps):
```
externalize thresholds to config.yaml
fix false deactivation claim
build benchmark harness (docs/loop/GOALS.md §7.1)
add E2E + property-based tests
```
