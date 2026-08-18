# Meta-Review Agent Gaps & Improvements (Based on Nature Paper)

> Audit Date: 2026-08-18  
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist", Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)  
> Supplementary Notes: [Supplementary Material (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf) (Notes 8, 9.5, 10.10)

---

## 📌 Status Summary

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

| ID | Improvement Area | Priority | Paper Reference | Status |
|---|---|---|---|---|
| **MR-01** | LLM-Driven Deep Thematic Synthesis of Critiques | 🔴 High | Supp Notes 8, 9.5, 10.10 | ⬜ open |
| **MR-02** | Tournament Debate Transcript Integration | 🔴 High | Supp Note 8 (`GenerateSystemFeedback`) | ⬜ open |
| **MR-03** | Final Executive Research Overview Synthesis | 🟡 Medium | Supp Note 8 (`GenerateFinalResearchOverview`) | ⬜ open |
| **MR-04** | Structured 4-Pillar Critique Taxonomy | 🟡 Medium | Supp Note 10.10 Spec | ⬜ open |

---

## 🔍 Detailed Gap Analysis & Technical Specs

### 🔴 MR-01: LLM-Driven Thematic Critique Synthesis
* **Paper Specification (Supp Notes 8, 9.5, 10.10)**:
  > *"The Meta-review agent gathers all reviews and tournament debate transcripts from SharedMemory and prompts an LLM: 'Analyze all these critiques. What are the most common weaknesses and strengths? Summarize this as feedback for the whole system.'"*
* **Current Code Problem**:
  - `app/agents_modules/meta_review.py` currently uses hardcoded heuristic counting (`low_novelty_count`, `low_feasibility_count`, `diversity_score < 0.35`) and deterministic template strings.
  - It does not invoke an LLM to uncover recurring scientific blind spots (e.g. *failure to demonstrate primary causality vs downstream effect*, *unspecified model concentrations*).
* **Implementation Plan**:
  1. Aggregate text comments, strengths, weaknesses, and debate arguments across active hypotheses.
  2. Implement `call_llm_for_meta_review` adhering to Supplementary Note 9.5 prompt.
  3. Combine deterministic topological insights with LLM-generated thematic critique.

---

### 🔴 MR-02: Tournament Debate Transcript Integration
* **Paper Grounding**: Pairwise debates expose subtle failure modes and trade-offs that single-hypothesis reflection reviews miss.
* **Implementation Plan**:
  - Extract the `reasoning` and `decisive_criteria` from recent `context.tournament_results` and feed them into the meta-review prompt.

---

### 🟡 MR-03: Final Executive Research Overview Synthesis
* **Paper Specification (Supp Note 8 / 9.5)**:
  - When completing a multi-cycle run or finalizing, synthesize top-ranked candidates into a coherent, executive-level research overview for scientists.
* **Implementation Plan**:
  - Implement `generate_final_research_overview(context, research_goal)` and present it in the Gradio UI report tab.

---

## 📜 Paper Prompt References (Supplementary Note 9.5)

### Prompt for Meta-Review Generation
```text
You are an expert in scientific research and meta-analysis. Synthesize a comprehensive meta-review of provided reviews pertaining to the following research goal:

Goal: {goal}
Preferences: {preferences}
Additional instructions: {instructions}

Provided reviews for meta-analysis:
{reviews}

Instructions:
* Generate a structured meta-analysis report of the provided reviews.
* Focus on identifying recurring critique points and common issues raised by reviewers.
* The generated meta-analysis should provide actionable insights for researchers developing future proposals.
* Refrain from evaluating individual proposals or reviews; focus on producing a synthesized meta-analysis.

Response:
```
