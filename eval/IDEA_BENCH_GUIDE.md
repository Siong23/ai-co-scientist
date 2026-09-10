# Evaluate multiple research goals

This independent tool reads saved run JSON files without invoking production agents or changing application configuration. Existing evaluate_run.py and DeepEval behavior remain unchanged. The new tool needs no extra dependencies.

The rubric adapts relevance, novelty, significance, quality, feasibility and clarity from [AI Idea Bench 2025](https://github.com/yansheng-qiu/AI_Idea_Bench_2025). These are custom absolute ratings, not its original pairwise competition, MCQ, paper matching or official benchmark scores. No upstream dataset, GROBID or PDF parser is required.

## Configure tasks

Copy configs/tasks.example.json to configs/tasks.json inside eval. Replace each goal with the exact goal in its saved run. Add domain-specific requirements to criteria. Task IDs must be unique. Each runs list can contain repeated runs and different condition labels. Paths are relative to the manifest directory; replace placeholder paths with real files.

Add a fixed reference collection to literature. Each source should have id, title, url and excerpt fields containing actual paper metadata and text. Without excerpts, novelty is forced to null (unverified). Keep literature fixed across systems. The evaluator does not retrieve literature or independently verify citations.

## Export offline

Run from the eval directory:

```powershell
uv run python scripts/idea_bench.py export configs/tasks.json --output reports/proposals.json
```

The exporter selects ranking_final, otherwise the latest nonempty ranking stage, otherwise the first stage containing hypotheses. It exports all hypotheses in the selected stage, preserving source_step, run_status and is_active. A fallback stage does not imply successful completion.

Invalid runs and goal mismatches appear in errors; other runs continue. Original text is preserved without LLM rewriting. Missing structured fields are null, but the original text may still contain an experimental plan. Records include saved goal settings and the source file SHA256. Settings absent from old runs cannot be recovered.

## Evaluate with an independent judge

Start a judge model in LM Studio and specify its actual model ID and URL:

```powershell
$env:LOCAL_MODEL_API_KEY = "lm-studio"
uv run python scripts/idea_bench.py evaluate reports/proposals.json --output reports/scores.json --judge-model YOUR_LOADED_MODEL_ID --judge-base-url http://localhost:1234/v1
```

The key above is a placeholder for a server without authentication. Real credentials belong only in environment variables. Only evaluate sends the goal, proposal, criteria and literature to the specified endpoint and consumes inference resources.

Judge prompts exclude internal Elo, reviews, condition labels and generating model identity. Each dimension receives an integer score from 1 to 5 or null, a brief justification and literature IDs. Invalid scores, invented citations and judge failures produce errors rather than fabricated scores.

Existing output files cannot be overwritten; use a new filename for each run. Exit code 0 means processing completed; 2 means input, export or judge errors occurred. There is no universal passing threshold.

Compare results within each goal. Do not mix domains or incomplete and completed runs into one overall score. Reports retain task, run and condition identifiers for analysis. This tool does not generate baselines, run experiments or establish actual detector performance. For fair comparisons, fix literature, final candidate count and inference budget, repeat runs independently and obtain expert review of samples.

## Offline tests

```powershell
uv run pytest
```

Tests cover multiple goals, goal mismatches, stage selection, missing fields, absent literature, invalid scores, invented citations, judge failures, credential redaction and overwrite protection.
