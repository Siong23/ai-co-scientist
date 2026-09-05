# Agent implementation audit

Updated: 2026-09-06. This replaces the six outdated `*-agent-to-fix.md` lists.

Sources: [Nature article](https://www.nature.com/articles/s41586-026-10644-y)
and [Supplementary Information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf),
especially sections 8 (agent pseudocode) and 9.1-9.5 (prompts).
Paper mechanisms describe the reference system, not mandatory numerical thresholds
for this local implementation. In particular, Elo is not scientific validation.

## Fixed in this audit

- Supervisor no longer rewrites REVISE hypotheses under existing IDs while keeping
  stale reviews and Elo. Existing Evolution creates children for independent
  Reflection. The standalone revision helper also returns fresh children with
  lineage, initial Elo and no inherited review/audit verdict.
- Ranking records missing-score abstentions and retries previously abstained pairs.
  Only A/B/TIE outcomes suppress completed pairs. Planner participation counts and
  convergence snapshots likewise exclude abstentions; failed calls cannot imply
  successful ranking or stable hypothesis quality.
- Meta-review now synthesizes actual strengths, weaknesses, review comments and
  tournament reasoning with the selected research model. Rejected hypotheses remain
  available as lessons even when no active candidates survive. Context is bounded
  to the latest 20 reviews and 20 matches. Structured output is validated; malformed
  output or model errors fall back to heuristic feedback. `synthesis_mode` records
  which path ran. Set `meta_review.llm_enabled: false` for heuristic-only operation.
- Ranking exception logs redact secrets. Existing import/unused-import lint errors
  were corrected without changing those modules' behavior.

Regression coverage: `tests/test_agent_review_integrity.py`, plus the updated
parallel revision test in `tests/test_agent_parallelism.py`.

## Latency improvements

- Full-text evidence queries use up to three concurrent read-only searches via
  `paper_library.retrieval_workers` (set to 1 for serial execution). Query
  deduplication, source filters, result limits and deterministic reciprocal-rank
  fusion are preserved. Store initialization and indexing remain sequential.
- Proximity defaults to the shared configured embedding provider instead of
  separately loading/downloading a Hugging Face model. An explicit
  `SimilarityConfig.embedding_model_name` still selects a separate local model.
- Meta-review uses bounded output and disables native reasoning for this summary
  call; scientific generation and evidence gates retain their existing settings.
- Fixed Proximity's high-similarity branch unpacking a single `max()` result into
  two hypotheses, which crashed before selecting the stronger duplicate.

Controlled benchmark: six mocked 80 ms evidence searches took 0.483 s serially
and 0.164 s with three workers, with identical fused results. This measures only
query scheduling, not live model throughput or total hypothesis generation time.
Regression tests cover concurrent overlap, deterministic output, empty queries,
error propagation, shared embedding reuse and explicit-model compatibility.

## Current status and remaining work

| Agent | Implemented | Remaining work / trade-off |
| --- | --- | --- |
| Generation | Literature retrieval, assumption analysis, optional multi-turn debate, provenance checks and candidate audits | Legacy audit path treats the model verdict as advisory; grounded audit path rejects explicit REJECT. Unify these policies deliberately rather than assume both paths behave alike. No candidate regeneration after a completely rejected batch. Debate remains configurable for local latency. |
| Reflection | Seven quality dimensions, claim assessment, retrieved-source ID checks, ACCEPT/REVISE/REJECT routing | Add independently retrieved counter-evidence and staged screening/deep verification for promising candidates. The paper's observation-by-observation causal analysis is not fully implemented. |
| Ranking | Elo, review prerequisite, bounded proximity-guided pairs, abstention recovery | Same completed pair is evaluated once; this is a latency optimization, not the paper's repeated tournament refinement. Add an explicit repeat budget and counterbalanced A/B judging before claiming convergence equivalent to the paper. |
| Evolution | Six strategies, parent lineage, critique feedback, deterministic output validation, cluster exemplars | Grounding uses inherited sources instead of fresh targeted retrieval. Validate newly introduced claims against explicitly selected sources; inherited citations alone do not establish support. Dedup checks compare lexical similarity to parents, not semantic similarity to all active/sibling candidates. |
| Proximity | Semantic graph, clusters, exemplars and similarity-based duplicate pruning | Similarity is a candidate relationship, not proof that two causal hypotheses are identical. Add scientific confirmation before automatic similarity-based pruning; alternate semantic-topology implementation needs an explicit supported role before removal. |
| Meta-review | LLM thematic feedback from reviews and debates with bounded deterministic fallback | Final overview still lists ranked hypotheses and next steps; a detailed evidence-linked research report and experimental protocols remain separate work. Bounded context is not a synthesis of the entire unbounded run history. |
| Supervisor | Dynamic bounded planner, accepted-only ranking, finalization evidence gate and Elo snapshots | Still orchestrates stages rather than a persistent asynchronous worker queue. Scientist steering and structured natural-language plan editing are incomplete. |

## Validation scope

Use the canonical offline pytest suite without provider credentials or external
network. Live LM Studio integration and scientific output quality require separate
runs and cannot be inferred from unit tests. The Windows environment may need a
writable `--basetemp` and `-p no:cacheprovider`; when GNU Make is unavailable, run
`.venv/Scripts/python.exe -m pytest` with the same default marker exclusions.

Final application validation: 420 passed, 4 skipped, 9 deselected in 15.50 s.
Independent eval validation: 14 passed with its own uv environment.
Repository-wide `ruff check .` passes. Existing untouched formatting differences
remain outside this behavioral audit.
