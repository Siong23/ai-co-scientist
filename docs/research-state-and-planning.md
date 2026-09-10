# Research state and planning architecture

Phase E separates durable evidence, evolving research sessions, and audit
outputs. The three stores have different lifetimes and must not be substituted
for one another.

```text
                         shared across research sessions
                                      │
                                      v
┌──────────────────────────────────────────────────────────────────────┐
│ Global evidence store                                                │
│ PDFs + source/version registry + structured chunks + embeddings      │
│ canonical paper/version IDs + parser/chunker/embedding signatures    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ retrieval returns exact source/chunk IDs
                               v
┌──────────────────────────────────────────────────────────────────────┐
│ Research state store                                                 │
│ goal + mode + plan + questions + hypotheses/lineage + reviews        │
│ Elo/tournaments + claim→chunk→source relations + supervisor state    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ each accepted cycle emits a new snapshot
                               v
┌──────────────────────────────────────────────────────────────────────┐
│ Run reports                                                          │
│ immutable run JSON + reproducible HTML audit/report view             │
└──────────────────────────────────────────────────────────────────────┘
```

## Storage boundaries

The global evidence layer remains the Phase A–D paper library and source
registry. It owns downloaded source documents, canonical paper versions,
structured chunks, lexical data, and embedding collections. Its contents are
reusable by unrelated research sessions.

The research-state layer is an atomic local JSON implementation of the
`ResearchStateStore` interface. By default it writes one current checkpoint per
research ID under `results/research_state/`; when `CO_SCIENTIST_RUNS_DIR` is
set, the default follows that relocated results directory. The
`research_state.directory` setting can select a different location. Each file
contains schema/compatibility metadata, its research ID, creation/update
timestamps, and a SHA-256 integrity digest.

A checkpoint contains research interpretation and workflow state:

- the goal, resolved research type, cycle number, plan, sub-questions, and
  evidence requirements;
- hypotheses, parent IDs, scores, reviews, audit state, Elo/tournament data,
  meta-review feedback, and proximity state;
- exact research-specific evidence relations with claim, chunk, and source
  identifiers; and
- generation diagnostics, supervisor state, and suspended work descriptions.

It intentionally excludes retrieved document bodies, chunk text, embeddings,
and the transient `last_retrieved_sources` payload. Evidence references are
reduced to provenance fields such as source ID, chunk ID, version, section,
page, and evidence type. The global evidence store remains the sole owner of
the underlying evidence.

Every completed or failed cycle accepted by the UI updates the current
research checkpoint and then creates a new run JSON. A run ID cannot overwrite
an existing run JSON. HTML reports can be deterministically regenerated from
that immutable JSON. Deleting a report does not implicitly delete its research
checkpoint or global evidence.

## Resume semantics

Selecting a saved run with a compatible research checkpoint reconstructs the
current session associated with that run's research ID: its `ResearchGoal`,
`ContextMemory`, hypotheses, reflection reports, lineage, ratings, tournaments,
meta-review/proximity data, and supervisor history. Because the checkpoint is
the session's current mutable state, it can be newer than the selected immutable
per-cycle report. An older run without a research ID or checkpoint remains
display-only.

Resume is conservative:

1. The reader rejects unsupported schema versions, mismatched research IDs,
   and failed integrity checks instead of treating them as current state.
2. Retrieved source bodies are not reconstructed from the checkpoint. A
   resumed session with evidence-dependent state is marked as requiring an
   evidence refresh, so the Supervisor routes through Generation/retrieval
   before later evidence-dependent work.
3. Previously pending actions are recorded as suspended, removed from the live
   queue, and replanned. No in-flight LLM, network, or worker call is replayed.
4. A UI cycle that reaches its timeout is saved only as a run report. Its
   private worker copy is not promoted to global context or checkpointed.

This is partial workflow reconstruction, not process continuation at an exact
instruction boundary. Refreshing evidence may produce a new generation batch;
it does not invent missing documents or silently assert that stale evidence is
available.

## Research planning modes

The Research Planner resolves one of six modes and passes it, its structured
plan, and the resulting retrieval plan through Generation and the Supervisor.

| Research type | Required planning emphasis | Provisional hypotheses |
| --- | --- | --- |
| `hypothesis_testing` | primary, alternative, and falsifying/null accounts | required |
| `causal` | competing causal accounts plus a falsifying/null account | required |
| `comparative` | competing candidates/explanations and comparison dimensions | optional |
| `exploratory` | research questions, topic dimensions, and evidence gaps | not fabricated |
| `literature_review` | themes, controversies, evidence dimensions, agreement/disagreement, and literature gaps | optional |
| `due_diligence` | claims, risks, counterclaims, primary-source checks, and missing evidence | normally omitted |

Query fidelity remains anchored to verbatim goal spans and explicit evidence
requirements. Supporting-evidence, counterevidence, and closest-prior-art
search intents remain available in every mode; when there is no provisional
hypothesis, those queries are anchored to the research goal and mode-specific
claims/questions instead.

Hypothesis-driven sessions retain the established pipeline:

```text
Generation → Reflection → Ranking/Elo → Evolution → Proximity → Meta-review
```

When a plan has no scientifically appropriate hypotheses, Generation retains
the evidence review and literature synthesis, and the Supervisor routes around
Reflection, Ranking, Evolution, and Proximity:

```text
mode-aware planning → evidence retrieval → literature synthesis
                  → mode-aware meta-review → finalization gate
```

Comparative mode uses the hypothesis pipeline only when the planner actually
returns valid optional hypothesis scaffolds. No placeholder hypothesis is
created merely to satisfy a downstream agent.

## Orchestration semantics

Execution is **dynamic orchestration with bounded parallel execution**. The
Supervisor selects one action, executes it to completion, observes the updated
state, and then selects the next action. Individual actions can use bounded
concurrency for provider search, passage retrieval, Reflection reviews,
Ranking matches, and Evolution strategies. There is no distributed worker
system or durable asynchronous task queue.

## Phase A–E evidence pipeline

```text
Research goal + requested/auto research type
  │
  ├─ Phase E: mode-aware Research Planner
  │      plan/questions/requirements/(optional) retrieval hypotheses
  │
  ├─ Query Rewriter + query-fidelity validation
  │      goal/support/counterevidence/prior-art routes
  │
  ├─ Provider search + provider-level reciprocal-rank fusion
  │
  ├─ Phase A: abstract-first relevance gate
  │      bounded PDF acquisition; explicit abstract-only fallback
  │
  ├─ Phase B: structured scientific ingestion
  │      raw text ≠ retrieval text; sections/elements/provenance
  │
  ├─ Phase D: canonical source/version registry
  │      cached parse artifacts + content-addressed incremental indexing
  │
  ├─ Phase C: dense + BM25 passage retrieval
  │      passage RRF + bounded parent/neighbor expansion
  │
  ├─ evidence coverage + chunk-grounded literature synthesis/audit
  │
  ├─ mode-compatible Supervisor route
  │      hypothesis pipeline OR synthesis-only review
  │
  └─ Phase E persistence
         atomic current research checkpoint + immutable per-cycle run report
```

No neural reranker is part of Phases A–E. Dense and lexical rankings are fused
with deterministic reciprocal-rank fusion and remain separately diagnosed.
