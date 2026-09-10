---
title: Open AI Co-Scientist
emoji: 📊
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: Open-source implementation of Google's AI Co-Scientist
---

# Open AI Co-Scientist - Hypothesis Evolution System

Open AI Co-Scientist is an AI-powered system for generating, reviewing, ranking, and evolving research hypotheses using a multi-agent architecture and Large Language Models (LLMs). The user interface is built with Gradio for rapid prototyping and interactive research. The system helps researchers explore research spaces and identify promising hypotheses through iterative refinement.

## 🚀 Features

- **Multi-Agent System:** Iteratively generates, reviews, ranks, and evolves research hypotheses using specialized agents (Generation, Reflection, Ranking, Evolution, Proximity, Meta-Review).
- **Local LLM Integration:** Uses LM Studio's OpenAI-compatible local API, with runtime model selection in the UI.
- **Interactive Gradio UI:** Easy-to-use interface for research goal input, advanced settings, and results visualization.
- **References & Literature:** Integrated arXiv search for related papers.
- **Mode-Aware Research:** Plans hypothesis testing, causal, comparative,
  exploratory, literature-review, and due-diligence work without fabricating
  hypotheses for modes that do not need them.
- **Resumable Research State:** Checkpoints evolving session state separately
  from the shared evidence index and immutable per-cycle run reports.
- **Private Inference:** Prompts and model responses stay on the configured local LM Studio server.
- **Logging:** Each run is logged to a timestamped file in the `results/` directory.

## AI Transparency Statement

In accordance with LLNL policy on Generative Artificial Intelligence, this project contains AI-assisted code and documentation. Various AI models (including OpenAI and Claude) were used to draft components and fix errors. The development process involved switching between models when encountering limitations with a particular model. All AI-generated content has been reviewed and verified by human developers to ensure accuracy, security, and alignment with project requirements.

## 💡 Example Research Goals

- Develop new methods for increasing the efficiency of solar panels.
- Create novel approaches to treat Alzheimer's disease.
- Design sustainable materials for construction.
- Improve machine learning model interpretability.
- Develop new quantum computing algorithms.

## Quick Start

1. **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start LM Studio:**
    - Install LM Studio and download or load a chat model.
    - Start its local OpenAI-compatible server.
    - The app defaults to `http://127.0.0.1:1234/v1`. Copy `.env.example` to
      `.env` and set `LMSTUDIO_BASE_URL` or `LMSTUDIO_MODEL` when needed.

4. **Run the Gradio app:**
    ```bash
    python app.py
    ```
    Or, using the Makefile:
    ```bash
    make run
    ```

5. **Access the web interface:**
    - Open your browser and go to [http://localhost:7860](http://localhost:7860)

## 🎯 How to Use

1. **Enter a research goal** in the provided textbox.
2. **(Optional) Adjust advanced settings** such as LLM model, number of hypotheses, temperatures, etc.
3. **Click "Run Cycle"** to generate, review, and evolve hypotheses.
4. **View results, meta-review, and related literature** in the web interface.
5. **Iterate** by running additional cycles to refine hypotheses.

## ⚙️ Configuration

- Default settings can be adjusted in `config.yaml`.
- `LMSTUDIO_BASE_URL` overrides the local API address.
- `LMSTUDIO_MODEL` overrides the configured default model.
- `LMSTUDIO_API_KEY` is optional and only needed when LM Studio authentication is enabled.
- Many settings can be overridden in the Gradio UI under "Advanced Settings".

## 🧠 How It Works

The system uses a multi-agent approach. Hypothesis-driven plans run the full
pipeline below; hypothesis-optional plans retain evidence retrieval and
literature synthesis, then route directly to a mode-aware meta-review and
finalization gate.

1. **Generation Agent:** Creates new research hypotheses.
2. **Reflection Agent:** Reviews and assesses hypotheses for novelty and feasibility.
3. **Ranking Agent:** Uses Elo rating system to rank hypotheses.
4. **Evolution Agent:** Combines top hypotheses to create improved versions.
5. **Proximity Agent:** Analyzes similarity between hypotheses.
6. **Meta-Review Agent:** Provides overall critique and suggests next steps.

The Supervisor provides **dynamic orchestration with bounded parallel
execution**: it chooses and completes one workflow action before replanning,
while provider searches, reviews, tournament matches, retrieval, and evolution
may use bounded concurrency inside that action. It is not a distributed task
queue.

## 📚 Literature Integration

- Plans the research and searches configured academic providers using the
  original goal plus focused rewritten queries.
- Uses title, abstract, and metadata as a high-recall candidate-paper gate, so
  clearly irrelevant search results are removed before PDF acquisition.
- Downloads only the bounded relevant shortlist into `app/paper/`, reuses
  cached PDFs, and stores versioned full-text evidence chunks in `chroma_db/`.
- Parses PDFs through a scientific-document interface with a dependency-light
  pypdf fallback. The fallback conservatively recovers sections, subsections,
  paragraphs, tables, captions, equations, and code blocks, then chunks at
  section, paragraph, and sentence boundaries before using a hard size limit.
- Stores source-faithful `raw_text` separately from document-intrinsic
  `retrieval_text`. Embeddings include paper/section/publication context but
  never research-goal conclusions or hypothesis judgments; prompt/citation
  display is rebuilt from provenance plus raw evidence.
- Retrieves focused method, result, comparison, and limitation passages before
  literature synthesis. Dense similarity and dependency-free BM25 search each
  produce a broader passage ranking; deterministic passage-level reciprocal
  rank fusion deduplicates them by `chunk_id`. This `hybrid_score` remains
  distinct from `dense_score`, `lexical_score`, and source-level
  `provider_rrf_score` diagnostics.
- Expands selected passages only through their Phase B parent/previous/next
  relationships, within separate expansion and total prompt budgets. Every
  neighbor remains an independent evidence passage with its own `chunk_id`,
  while `selected_anchor_chunk_id` records why it was included.
- Each passage carries document/chunk identity, section path, page range,
  element type, content/retrieval hashes, parser/chunker/template versions, and
  source provenance.
- Tracks canonical papers and observable remote versions in a durable JSON
  registry shared by all model-specific collections. arXiv IDs such as `v1`
  and `v2` remain separate versions beneath one versionless paper identity;
  changed version or `updated` metadata forces a source refresh.
- Caches parser/chunker output by document hash and pipeline version, so an
  embedding-model change reuses the cached PDF and chunk artifact. Within one
  embedding collection, deterministic content and retrieval hashes let
  unchanged chunks reuse stored vectors while modified/new chunks are embedded
  and removed chunks are deleted after read-after-write verification.
- Keeps unavailable papers as explicitly limited `abstract_only` evidence;
  one failed PDF does not abort a research cycle.
- Provides passage-level coverage and strict chunk-grounded audit helpers for
  scientific validation. The production workflow retains its configured
  balanced/strict audit policy and concurrent candidate reviews; strict
  chunk-grounded validation is available through
  `call_llm_for_grounded_hypothesis_audit`.

The three gates have intentionally different jobs:

1. **Abstract candidate filter:** decides which papers are worth downloading.
2. **Full-text evidence coverage:** decides what the retrieved literature can
   establish.
3. **Hypothesis grounding audit:** decides what the system is allowed to
   publish as a validated hypothesis.

Important `config.yaml` groups are `rag` (paper discovery and corrective
search), `paper_library` (download budget, PDF cache, Chroma schema and prompt
limits), `evidence_retrieval` (focused-query limits and query-side embedding
instructions), `research_state` (durable session checkpoints), and `validation` (numeric/entailment checks and per-candidate
audit context). Collections are isolated by embedding model, index schema, and
retrieval template. Parser/chunker changes invalidate the shared chunk artifact
and update the current collection incrementally; embedding-model changes reuse
the artifact but populate a model-specific collection. Hybrid/BM25 ranking and
context-expansion settings do not alter stored embeddings.

## 💾 Evidence, Research State, and Reports

Phase E gives each persistence layer one responsibility:

```text
Global evidence store  → papers, versions, chunks, embeddings, source registry
Research-state store   → goal, plan, questions, hypotheses, reviews, ratings,
                         evidence relationships, supervisor state
Run reports            → immutable per-cycle JSON audit snapshots and HTML views
```

Research checkpoints use versioned, integrity-checked atomic JSON under
`results/research_state/` by default. They retain exact claim → chunk → source
IDs but never duplicate document bodies, chunk text, or embeddings. On resume,
compatible state is reconstructed; evidence-dependent sessions refresh global
evidence first, and formerly pending actions are suspended and replanned rather
than silently replayed. Older saved runs without a checkpoint remain
display-only.

See [Research state and planning architecture](docs/research-state-and-planning.md)
for the six planning modes, resume limitations, storage schema, routing rules,
and the complete Phase A–E pipeline.

## ⚙️ Technical Details

- **Models:** Uses any chat model exposed by the configured LM Studio server.
- **Model discovery:** Reads LM Studio's local `/models` endpoint when the UI starts.
- **Offline tests:** Mock the LM Studio boundary and never require a running server.
- **Iterative Process:** Each cycle builds on previous results for continuous improvement.

## 📖 Research Paper

Based on the AI Co-Scientist research: https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf

## 🤝 Contributing

This is an open-source project. Feel free to contribute improvements, bug fixes, or new features. 

See CONTRIBUTING.md for details. 

## ⚠️ Note

LM Studio must be running and have a compatible chat model loaded before a
research cycle can complete.


## Acknowledgements

- Based on the idea of Google's AI Co-Scientist system.
- Uses [Gradio](https://gradio.app/) for the user interface.
- Local LLM access via [LM Studio](https://lmstudio.ai/).

---

## Release

LLNL-CODE-2010270

SPDX-License-Identifier: MIT

# CI/CD test -deploy_test
