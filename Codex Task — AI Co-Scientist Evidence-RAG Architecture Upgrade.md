## Objective

Improve the existing evidence acquisition, scientific-document ingestion, persistence, and retrieval architecture based on the current implementation.

Do NOT redesign the project from scratch.

Preserve the existing Co-Scientist architecture:

- SupervisorAgent
- GenerationAgent
- ReflectionAgent
- RankingAgent
- EvolutionAgent
- ProximityAgent
- MetaReviewAgent

Preserve existing hypothesis generation, Elo ranking, reflection/evolution behavior, evidence coverage gates, corrective retrieval, run reporting, and existing provider integrations unless a change is required for correctness.

Do NOT add a reranker model in this task.
Specifically, do NOT add Qwen3-Reranker, CrossEncoder, Cohere Rerank, Jina rerankers, or another external reranking model yet.

Before editing code, inspect the repository and confirm which of the requirements below are already implemented, partially implemented, or missing. Reuse existing abstractions whenever possible.

---

# Priority 1 — Fix the arXiv / academic paper acquisition funnel

The intended pipeline should be:

Research Goal
→ Research Plan
→ Search Queries
→ arXiv / academic metadata + abstract
→ cheap candidate ranking
→ LLM abstract-level screening
→ ACCEPT / MAYBE / REJECT
→ only then download selected full text
→ parse
→ chunk
→ index
→ retrieve evidence passages

Currently, candidate papers can reach PDF acquisition before the LLM relevance filter has explicitly made a download decision.

Refactor this so that academic PDF/full-text acquisition occurs only after an abstract-level screening decision.

## Required behavior

Introduce a structured abstract-screening result, for example:

```python
class AbstractScreeningResult:
    source_id: str
    decision: Literal["accept", "maybe", "reject"]
    relevance_score: float
    reason: str
    evidence_requirement_ids: list[str]
    hypothesis_ids: list[str]
    full_text_needed: bool
    full_text_questions: list[str]
```

Exact implementation may differ if an existing model can be extended cleanly.

The screener must evaluate:

- relevance to the original research goal
- relevance to explicit evidence requirements
- possible usefulness for primary/alternative/null hypotheses
- whether the abstract contains enough evidence already
- whether full text is actually needed
- potential support, counterevidence, or prior-art value

The screener must NOT treat an abstract as final scientific evidence when methods/results/full details are required.

Only ACCEPT papers, and optionally high-value MAYBE papers when required for evidence coverage, should consume the PDF acquisition budget.

Cached, already-committed full-text sources should remain reusable without redownloading.

Add diagnostics to the research trace / evidence funnel:

- abstract_candidates
- abstract_screened
- abstract_accepted
- abstract_maybe
- abstract_rejected
- full_text_requested
- full_text_cache_hits
- full_text_downloads

---

# Priority 2 — Upgrade scientific PDF parsing and chunking

Current PDF handling is based primarily on:

pypdf
→ page text
→ character-size chunks
→ section="Unknown"

This is insufficient for scientific evidence retrieval.

Refactor the ingestion layer so the architecture supports structured scientific-document elements.

Do not necessarily introduce a heavy third-party parser if the dependency cost is unjustified. Build an abstraction first and provide a reliable fallback.

Target logical model:

```text
Paper
  ├── Title
  ├── Abstract
  ├── Introduction
  ├── Methods
  │    ├── Dataset
  │    └── Experimental Setup
  ├── Results
  │    ├── Main Results
  │    └── Ablations
  ├── Discussion
  ├── Limitations
  ├── Conclusion
  ├── Tables
  └── References
```

Introduce document elements or equivalent structures such as:

```python
DocumentElement(
    element_type=...,
    text=...,
    page=...,
    section=...,
    subsection=...,
    section_path=...,
)
```

Chunking priority should be:

1. section boundaries
2. paragraph boundaries
3. sentence boundaries
4. hard token/character boundary only as fallback

Avoid splitting:

- tables
- code blocks
- equations where practical
- captions from the object they describe where practical

Retain a pypdf fallback when structural parsing is unavailable.

Do not make the application unusable merely because advanced structure recovery fails.

---

# Priority 3 — Separate raw evidence text from retrieval text

Do not make a single text representation serve both evidence fidelity and embedding retrieval.

Introduce the conceptual separation:

```text
raw_text
    = faithful source evidence

retrieval_text
    = document-intrinsic context + raw evidence

display_text
    = source/provenance context + raw evidence for LLM/citation
```

For example:

```text
Paper: <paper title>
Section: Results > Ablation Study
Authors: ...
Publication: ...

<raw chunk>
```

may be used as retrieval_text.

Important:

Do NOT insert research-specific conclusions such as:

"H1 is correct"
"This supports hypothesis H2"

into persistent retrieval_text.

Persistent embeddings must remain reusable across unrelated future research projects.

Research-specific support/contradiction judgments belong in research state / evidence-link metadata, not permanent chunk embeddings.

Preserve raw_text exactly enough for citation/evidence verification.

---

# Priority 4 — Improve chunk metadata schema

Extend the current metadata model without breaking existing records.

A scientific chunk should ideally support:

```text
source_id
document_id
chunk_id
paper_version
title
authors
doi
arxiv_id
published_at
updated_at

section
subsection
section_path
page_start
page_end
chunk_index
chunk_count
element_type

content_sha256
retrieval_text_sha256

parser_version
chunking_version
retrieval_template_version
embedding_model
schema_version

source_type
document_type
evidence_type
```

Where appropriate also support:

```text
parent_id
previous_chunk_id
next_chunk_id
```

Do not force every field into Chroma metadata if it belongs in a durable source registry instead.

Keep Chroma metadata focused on:

- retrieval filtering
- provenance
- citation
- index integrity

---

# Priority 5 — Add parent / neighbor context retrieval

Small chunks improve retrieval precision but can lose scientific context.

Add a mechanism whereby a selected chunk can optionally expand to:

- its parent section, or
- previous/next sibling chunks

before being sent to downstream evidence analysis.

Do not blindly concatenate every neighbor.

Expansion must be bounded by a configurable prompt/context budget.

Recommended flow:

```text
Dense candidate retrieval
→ selected evidence chunks
→ bounded parent/neighbor expansion
→ deduplication
→ evidence coverage / LLM
```

Preserve the original selected chunk ID for provenance.

---

# Priority 6 — Add lexical/BM25 retrieval capability

Do NOT add an external neural reranker yet.

Add lexical retrieval so scientific identifiers and exact terms are not dependent entirely on dense embeddings.

Examples:

- error/model identifiers
- gene/protein names
- dataset names
- benchmark names
- exact metric names
- paper-specific terminology

Target architecture:

```text
Dense retrieval
        +
Lexical/BM25 retrieval
        ↓
Rank fusion
        ↓
candidate passages
```

Reuse the project's existing Reciprocal Rank Fusion concepts where sensible, but keep these concepts distinct:

1. provider/query RRF
2. dense + lexical passage-level RRF

Do not falsely label dense similarity search as a reranker.

Names and diagnostics should clearly distinguish:

- provider_rrf_score
- dense_score
- lexical_score
- hybrid_score

Keep the implementation modular so a neural reranker can be added later after hybrid retrieval is stable.

---

# Priority 7 — Implement version-aware arXiv/source refresh

The project already captures arXiv:

- arxiv_id
- published
- updated

and already has:

- document SHA256
- chunk SHA256
- parser_version
- chunking_version
- schema_version
- embedding-model-aware collection signatures
- manifest verification

Extend this into proper source-version tracking.

Create or extend a durable source registry so the system can determine:

```text
same source + same version + same pipeline
→ reuse index

same source + new remote version
→ fetch new source

same source + same raw document + new parser
→ reparse/rechunk as necessary

same chunks + different embedding model
→ re-embed without redownloading/reparsing

new research goal
→ reuse existing global evidence without re-indexing
```

For arXiv, distinguish canonical paper identity from paper version.

Example:

```text
canonical paper:
arxiv:2608.12345

versions:
v1
v2
v3
```

Do not allow canonical deduplication to hide the fact that the remote paper has changed.

---

# Priority 8 — Implement chunk-level incremental indexing

Avoid full re-embedding when only a subset of a paper changed.

Use deterministic content identities.

Conceptually:

```python
chunk_content_hash = sha256(
    normalized_section_path
    + normalized_raw_text
)

embedding_cache_key = sha256(
    retrieval_text_hash
    + embedding_model
)
```

If a new source version produces:

```text
A unchanged
B unchanged
C modified
D unchanged
E new
```

the desired behavior is:

```text
A reuse
B reuse
C re-embed
D reuse
E embed
deleted chunks → tombstone/delete
```

Do not assume chunk ordinal/index is a stable semantic identity.

Keep chunk ID and embedding-cache identity conceptually separate.

Add tests proving unchanged chunks avoid unnecessary embedding calls.

---

# Priority 9 — Improve persistent research state

Currently ContextMemory is primarily in-memory, while run snapshots are persisted separately.

Introduce a clean persistence boundary for research state.

The system should eventually be able to reconstruct enough state to resume a research session, including:

```text
research goal
iteration
hypotheses
hypothesis lineage
reflection reports
Elo scores
tournament results
evidence relationships
meta-review feedback
proximity state
supervisor state
pending tasks
generation diagnostics
```

Do not tightly couple this to Chroma.

Recommended separation:

```text
Global Evidence Store
    → papers/chunks/embeddings

Research State Store
    → goals/questions/hypotheses/evidence relationships

Run Reports
    → immutable reporting/audit snapshots
```

Keep existing JSON run reports working.

If introducing SQLite/PostgreSQL would cause an unnecessarily large migration, first define a repository/storage interface and provide a local implementation compatible with the current project.

---

# Priority 10 — Make research planning mode-aware

The Research Planner currently always generates:

- primary hypothesis
- alternative hypothesis
- null hypothesis

This is appropriate for hypothesis-driven research but not every research task.

Use `research_type` to decide planning behavior.

Examples:

```text
hypothesis_testing
causal
→ primary / alternative / null hypotheses

comparative
→ competing explanations or candidates

exploratory
→ questions + topic dimensions; hypotheses optional

literature_review
→ themes / controversies / evidence dimensions

due_diligence
→ claims / risks / counterclaims / missing evidence
```

Do not remove hypothesis support from the Co-Scientist workflow.

Instead, make provisional retrieval hypotheses optional/conditional when the research task genuinely does not require them.

Preserve query-fidelity safeguards.

---

# Priority 11 — Clarify orchestration semantics

Do NOT attempt a full distributed asynchronous worker rewrite in this task.

The current architecture already contains useful bounded parallelism:

- provider search
- reflection
- ranking matches
- evolution
- retrieval

Keep this.

However, ensure documentation and naming do not claim a fully asynchronous worker architecture if execution remains:

```text
Supervisor decision
→ execute action
→ wait
→ next decision
```

Document it accurately as dynamic orchestration with bounded parallel execution.

Create interfaces that would permit future task-queue execution without requiring it now.

---

# Existing functionality that must be preserved

Do not regress:

- Research Planner
- Query Rewriter
- query fidelity validation
- primary/alternative/null retrieval scaffolds
- support/counterevidence/prior-art queries
- arXiv integration
- Semantic Scholar integration
- Springer integration
- Elsevier integration
- Tavily integration
- provider RRF
- EvidenceCoverage
- corrective retrieval
- evidence-gap searching
- FIND_COUNTEREVIDENCE
- SEARCH_PRIMARY_SOURCE
- claim verification
- Chroma persistence
- PDF cache
- manifest integrity checking
- stale-record cleanup
- strict evidence gate
- evidence diagnostics/funnel
- hypothesis grounding/novelty audit
- Reflection
- Ranking/Elo
- Evolution
- Proximity
- Meta-review
- Dynamic Supervisor
- run persistence/report generation
- existing tests unless behavior was intentionally superseded

---

# Implementation constraints

1. Inspect before modifying.
2. Prefer extending current classes over creating duplicate pipelines.
3. Keep backwards compatibility where reasonable.
4. No silent fallback that converts a failed scientific validation into success.
5. Do not mark abstract evidence as full-text evidence.
6. Do not treat model-generated interpretation as source evidence.
7. Maintain exact provenance from claim → chunk → source.
8. Do not re-download cached unchanged PDFs.
9. Do not recompute embeddings unnecessarily.
10. Avoid large new dependencies unless justified.
11. Any new optional dependency must degrade gracefully.
12. Keep configuration explicit in `config.yaml`.
13. Add type hints for new public interfaces.
14. Add logging/diagnostics for all new evidence funnel stages.
15. Update README/architecture documentation to match actual behavior.

---

# Testing requirements

Add or update tests for at least:

### Abstract screening
- rejected abstract never downloads a PDF
- accepted abstract can download/index
- MAYBE can be promoted when an uncovered evidence requirement requires it
- cached full text is reused

### Structured chunking
- section metadata survives chunking
- chunks do not cross section boundaries unnecessarily
- pypdf fallback still works

### Raw/retrieval text
- raw evidence remains faithful
- retrieval_text includes intrinsic context
- research hypothesis text is not permanently embedded

### Persistence
- committed unchanged source is reused
- parser-version change invalidates only necessary stages
- embedding-model change does not redownload PDF

### Incremental indexing
- unchanged chunks reuse existing index
- modified chunks are re-embedded
- new chunks are added
- removed chunks are deleted/tombstoned

### arXiv versions
- v1 and v2 share canonical paper identity
- remote updated/version change triggers source refresh

### Hybrid retrieval
- exact-token BM25 candidate can survive when dense retrieval misses it
- dense and lexical rankings are fused deterministically

### Neighbor expansion
- neighbor expansion respects context budget
- provenance remains anchored to selected chunks

### Planning modes
- exploratory research does not require fabricated hypotheses
- hypothesis-testing research still generates primary/alternative/null scaffolds

### Regression
- existing Generation → Reflection → Ranking → Evolution → Meta-review workflow still passes

---

# Work strategy

Do not make one giant unreviewable rewrite.

Implement in coherent phases.

Recommended sequence:

Phase A:
- abstract-first acquisition gate
- diagnostics
- tests

Phase B:
- structured document/chunk model
- raw_text vs retrieval_text
- metadata improvements
- tests

Phase C:
- lexical/BM25 + hybrid RRF
- parent/neighbor expansion
- tests

Phase D:
- source/version registry
- chunk-level incremental indexing
- tests

Phase E:
- persistent research-state abstraction
- research-type-aware planning
- documentation

After each phase:

1. run focused tests
2. run the full test suite
3. report failures
4. fix regressions before proceeding

---

# Deliverables

At completion provide:

1. Summary of architecture changes.
2. Files changed.
3. New data models/interfaces.
4. Config options added.
5. Migration/backwards-compatibility notes.
6. Tests added.
7. Full test results.
8. Remaining technical debt.
9. Any audit item intentionally deferred and why.
10. A concise before/after pipeline diagram.

Do not add a neural reranker in this task.