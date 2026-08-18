# Evolution Agent Gaps & Improvements (Based on Nature Paper)

> Audit Date: 2026-08-18  
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist", Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)  
> Supplementary Notes: [Supplementary Material (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf) (Notes 8, 9.4)

---

## 📌 Status Summary

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

| ID | Improvement Area | Priority | Paper Reference | Status |
|---|---|---|---|---|
| **EV-01** | Grounding Strategy Active Literature Retrieval | 🔴 High | Main text §Evolution Agent, p. 126087 | ⬜ open |
| **EV-02** | Semantic Embedding Dedup Against All Active Hypotheses | 🔴 High | Code audit & Diversity preservation | ⬜ open |
| **EV-03** | Citation & Source ID Verification on Newly Introduced Mechanisms | 🟡 Medium | Grounding safety guidelines | ⬜ open |
| **EV-04** | Prompt Alignment with Supplementary Note 9.4 (Feasibility & Out-of-Box) | 🟡 Medium | Supp Note 9.4 Prompt Specs | ⬜ open |
| **EV-05** | Parent Sampling via Proximity Diversity (Cross-Cluster Parents) | 🟢 Low | Proximity & Tournament interaction | ⬜ open |

---

## 🔍 Detailed Gap Analysis & Technical Specs

### 🔴 EV-01: Grounding Strategy Active Literature Retrieval
* **Paper Specification (Main text §Evolution Agent, p. 126087)**:
  > *"Enhancement through grounding. Here the agent attempts to improve hypotheses by identifying weaknesses, generating search queries, retrieving and reading articles, suggesting improvements and elaborating on details to fill reasoning gaps."*
* **Current Code Problem**:
  - In `app/agents_modules/evolution_helpers.py`, the `grounding` strategy prompt only passes already-retrieved evidence attached to the parent hypotheses (`resolve_parent_evidence`).
  - It does **not** execute fresh web/academic search queries or Chroma PDF retrievals to fill new reasoning gaps or find evidence for newly proposed components.
* **Implementation Plan**:
  1. For `strategy == "grounding"`, extract the key critique points and weaknesses from the parent's `ReflectionReport`.
  2. Generate 1–2 targeted search queries via `ResearchRetriever`.
  3. Retrieve and append newly discovered evidence before calling the evolution LLM.

---

### 🔴 EV-02: Semantic Embedding Dedup Against All Active Hypotheses
* **Current Code Problem**:
  - `validate_evolution_candidate` (`evolution_helpers.py:133-137`) only performs `SequenceMatcher(a, b).ratio() >= 0.92` against the immediate parent hypotheses.
  - A candidate that paraphrases a parent using synonyms or reproduces an existing sibling hypothesis from another branch bypasses the lexical check.
* **Implementation Plan**:
  1. Compute embedding cosine similarity using the shared embedding model.
  2. Compare against **all active hypotheses** in `context.hypotheses`, not just immediate parents.
  3. Rejection threshold: cosine similarity $> 0.88$ (configurable).

---

### 🟡 EV-03: Citation Verification on Evolved Mechanisms
* **Current Code Problem**:
  - The evolution output schema is currently `{"title": "...", "hypothesis": "..."}`.
  - Evolved hypotheses inherit all parent source IDs en bloc (`create_evolved_hypothesis`), but newly introduced entities or claims have no verifiable citations.
* **Implementation Plan**:
  1. Require evolution LLM to output `source_ids` selected from available evidence.
  2. Run `_resolve_retrieved_source_ids` on the evolved candidate.

---

### 🟡 EV-04: Prompt Alignment with Supplementary Note 9.4
* **Paper Grounding**: Supplementary Note 9.4 provides exact prompts for Feasibility Improvement and Out-of-Box generation.
* **Implementation Plan**:
  - Update `_STRATEGY_INSTRUCTIONS["feasibility"]` and `_STRATEGY_INSTRUCTIONS["out_of_box"]` to faithfully mirror the 4-step structure in Note 9.4 (Scientific domain overview $\to$ pertinent findings $\to$ technology enablement argument $\to$ core contribution).

---

## 📜 Paper Prompt References (Supplementary Note 9.4)

### Prompt for Hypothesis Feasibility Improvement
```text
You are an expert in scientific research and technological feasibility analysis. Your task is to refine the provided conceptual idea, enhancing its practical implementability by leveraging contemporary technological capabilities. Ensure the revised concept retains its novelty, logical coherence, and specific articulation.

Goal: {goal}
Guidelines:
1. Begin with an introductory overview of the relevant scientific domain.
2. Provide a concise synopsis of recent pertinent research findings and related investigations, highlighting successful methodologies and established precedents.
3. Articulate a reasoned argument for how current technological advancements can facilitate the realization of the proposed concept.
4. CORE CONTRIBUTION: Develop a detailed, innovative, and technologically viable alternative to achieve the objective, emphasizing simplicity and practicality.

Evaluation Criteria: {preferences}
Original Conceptualization: {hypothesis}
Response:
```

### Prompt for Hypothesis Generation through Out-of-the-Box Thinking
```text
You are an expert researcher tasked with generating a novel, singular hypothesis inspired by analogous elements from provided concepts.

Goal: {goal}
Instructions:
1. Provide a concise introduction to the relevant scientific domain.
2. Summarize recent findings and pertinent research, highlighting successful approaches.
3. Identify promising avenues for exploration that may yield innovative hypotheses.
4. CORE HYPOTHESIS: Develop a detailed, original, and specific single hypothesis for achieving the stated goal, leveraging analogous principles from the provided ideas. This should not be a mere aggregation of existing methods or entities. Think out-of-the-box.

Criteria for a robust hypothesis: {preferences}
Inspiration may be drawn from the following concepts (utilize analogy and inspiration, not direct replication): {hypotheses}
Response:
```
