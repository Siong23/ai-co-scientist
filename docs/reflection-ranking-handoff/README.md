# Reflection to Ranking Integration Fix

## Confirmed problem

The current Reflection and Ranking modules do not use the same data contract.

`ReflectionAgent` currently writes only:

```python
hypothesis.novelty_review
hypothesis.feasibility_review
hypothesis.review_comments
hypothesis.references
```

However, `run_pairwise_debate()` and `score_hypothesis()` read:

```python
hypothesis.reflection_report
```

No current code assigns a value to `hypothesis.reflection_report`. This is
confirmed by the latest run output:

```json
{
  "reflection_report": null,
  "scores_a": {},
  "scores_b": {}
}
```

As a result, Ranking receives `No reflection report available.` instead of a
structured scientific review. The existing Ranking tests also treat empty
score dictionaries as expected behavior, so they do not detect this broken
integration.

Reflection should establish the data contract first. Ranking should then be
updated against that finalized contract.

## Phase 1: Reflection changes

### 1. Populate `ReflectionReport`

Reflection must construct and save the existing `ReflectionReport` model:

```python
hypothesis.reflection_report = ReflectionReport(...)
```

The structured LLM response should contain at least:

```json
{
  "claims": [
    {
      "claim": "A factual premise or proposed mechanism",
      "status": "SUPPORTED",
      "supporting_source_ids": ["source-id"],
      "contradictory_source_ids": []
    }
  ],
  "novelty_score": 7,
  "feasibility_score": 6,
  "plausibility_score": 7,
  "testability_score": 8,
  "evidence_quality_score": 5,
  "expected_research_value_score": 7,
  "strengths": [],
  "weaknesses": [],
  "recommendation": "REVISE",
  "overall_confidence": 8
}
```

Allowed claim statuses:

```text
SUPPORTED
CONTRADICTED
MIXED
NOT_FOUND
UNVERIFIED
```

All quality scores must be between 0 and 10. Confidence must be between 0 and
1. Malformed or out-of-range reports must not be accepted.

### 2. Use hypothesis-specific evidence

Reflection should primarily review against:

```python
hypothesis.evidence_sources
hypothesis.evidence_source_ids
```

It must not automatically treat every source in
`context.last_retrieved_sources` as evidence for every hypothesis.

If `hypothesis.evidence_sources` is empty, Reflection may resolve the
hypothesis's `evidence_source_ids` against `context.last_retrieved_sources` as
a compatibility fallback. This is particularly important for evolved
hypotheses, which inherit a specific subset of evidence from their parents.

### 3. Validate Source IDs

The LLM must return only Source IDs from the supplied evidence. The parser
must:

1. Build an allowlist from the supplied sources.
2. Remove unknown or fabricated Source IDs.
3. Resolve valid IDs back into the original source dictionaries.
4. Never trust model-generated titles, URLs, abstracts, or citations.

The resolved dictionaries should populate:

```python
ClaimAssessment.supporting_evidence
ClaimAssessment.contradictory_evidence
```

### 4. Distinguish evidence from novel proposals

Reflection must distinguish between:

- Established factual premises, prior results, algorithms, and performance
  numbers, which require evidence.
- Newly proposed mechanisms or experimental directions, which do not need to
  be previously proven but must be plausible, testable, and falsifiable.

A new mechanism should not automatically be rejected because no source has
already demonstrated it. However, it must not be presented as an established
result.

Unsupported exact values such as accuracy improvements, latency thresholds,
sample sizes, or statistical conclusions should be marked `UNVERIFIED` unless
supported by the supplied evidence.

### 5. Do not add direct Web Search yet

Reflection should not perform unrestricted Web Search in this phase. When
evidence is insufficient, it should return `NOT_FOUND` or `UNVERIFIED`.

A later Supervisor-level step may use those evidence gaps to trigger one
bounded supplementary search.

### 6. Preserve legacy fields

Meta-review and the current UI still use the older categorical fields.
Reflection should derive them from the structured report:

```python
hypothesis.novelty_review = score_to_level(report.novelty_score)
hypothesis.feasibility_review = score_to_level(report.feasibility_score)
```

Suggested mapping:

```text
8-10 -> HIGH
5-7  -> MEDIUM
0-4  -> LOW
```

The structured `ReflectionReport` should remain the authoritative result.

### 7. Fix the `references` type mismatch

`Hypothesis.references` is declared as `List[Dict]`, but Reflection currently
appends Source ID strings. Use one consistent representation:

- Resolve Source IDs into the original source dictionaries before adding
  them; or
- Stop duplicating them in `references` and use `evidence_sources` and claim
  assessments as the source of truth.

Do not mix strings and dictionaries in the same list.

### 8. Handle Reflection failures safely

If the first response is malformed, perform at most one format-repair call.
If the LLM call or repair fails, use:

```python
hypothesis.reflection_report = None
hypothesis.novelty_review = "UNREVIEWED"
hypothesis.feasibility_review = "UNREVIEWED"
```

Do not create a `ReflectionReport` containing all-zero scores. Otherwise,
Ranking may interpret an API or parsing failure as a genuinely poor scientific
hypothesis.

## Reflection acceptance tests

Add offline tests with the LLM boundary mocked. Cover at least:

- A valid response creates and assigns a `ReflectionReport`.
- Both generated and evolved hypotheses can be reviewed.
- Reflection uses only hypothesis-specific evidence.
- Fabricated Source IDs are removed.
- Valid Source IDs are resolved into the original source dictionaries.
- Unsupported numerical claims are marked `UNVERIFIED`.
- A proposed mechanism is not rejected merely because it has not already been
  demonstrated.
- Invalid output receives one format-repair attempt.
- Two failed responses leave `reflection_report` as `None`.
- Legacy `HIGH`, `MEDIUM`, and `LOW` fields are derived correctly.
- Re-reviewing does not accumulate duplicate comments or references.

## Phase 2: Ranking changes

Ranking should be updated after the Reflection report contract is implemented.

### 1. Include claim assessments

`format_reflection_report()` currently omits the report's `claims`. It should
include:

- Claim text
- Assessment status
- Supporting Source IDs
- Contradictory Source IDs
- Reflection confidence

This allows the Ranking judge to distinguish evidence-supported premises from
unverified additions.

### 2. Include actual evidence

The active `judge_hypotheses()` prompt does not directly include the
hypotheses' evidence sources. Add a bounded evidence section for both
hypotheses.

The evidence formatter must support the actual source fields used by the
project:

```text
content
abstract
summary
title
source_id
url
```

It should not depend only on `finding` and `limitation`, because those fields
are often absent from the retrieved source dictionaries.

### 3. Handle missing Reflection reports

If either hypothesis lacks a valid `reflection_report`, Ranking should either:

- Return `ABSTAIN`; or
- Skip the pair and record the reason.

It must not update Elo using an evidence-based judgment when the required
Reflection data is missing.

### 4. Update existing tests

The current expectation:

```python
assert result["scores_a"] == {}
assert result["scores_b"] == {}
```

must no longer represent a successful, fully reviewed comparison. For
hypotheses with valid Reflection reports, both dictionaries must be populated.

### 5. Add a Reflection-to-Ranking integration test

Add an offline integration test that verifies:

- Reflection assigns a valid report.
- Ranking receives that report.
- The Ranking prompt does not contain `No reflection report available.`
- `scores_a` and `scores_b` are non-empty.
- Claim support status and evidence quality reach the Ranking prompt.
- Missing Reflection reports result in `ABSTAIN` or a skipped comparison.

## Definition of done

The work is complete when:

- Every successfully reviewed hypothesis has a structured
  `ReflectionReport`.
- Ranking consumes the report and hypothesis-specific evidence.
- Tournament score dictionaries are populated.
- Fabricated citations cannot enter the report.
- Reflection failures cannot incorrectly reduce Elo.
- Generated and evolved hypotheses follow the same review contract.
- All offline tests pass without network access or API keys.

