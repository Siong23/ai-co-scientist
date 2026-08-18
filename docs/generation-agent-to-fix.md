# Generation Agent Gaps & Improvements (Based on Nature Paper)

> Audit Date: 2026-08-18  
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist", Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)  
> Supplementary Notes: [Supplementary Material (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf) (Notes 1, 2, 8, 9.1, 10.1, 10.2)

---

## 📌 Status Summary

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

| ID | Improvement Area | Priority | Paper Reference | Status |
|---|---|---|---|---|
| **GN-01** | Multi-Turn Simulated Scientific Debate (Self-Play Generation) | 🔴 High | Main text p. 118779 & Supp Note 9.1 | ⬜ open |
| **GN-02** | Audit Rejection Retry & Balanced Mode Tightening | 🔴 High | Main text §Quality Gates & Code audit | ⬜ open |
| **GN-03** | Structured Experimental Protocol Specification (Cell models, readouts, controls) | 🟡 Medium | Supp Note 10.2 | ⬜ open |
| **GN-04** | Multi-Hop Conditional Reasoning for Assumption Chains | 🟡 Medium | Main text p. 118779 & Supp Note 8 | ⬜ open |
| **GN-05** | Configurable Audit Weights & Grounding Thresholds | 🟢 Low | Code audit (hardcoded weights) | ⬜ open |

---

## 🔍 Detailed Gap Analysis & Technical Specs

### 🔴 GN-01: Multi-Turn Simulated Scientific Debate for Generation
* **Paper Specification (Main text p. 118779 & Supp Note 9.1)**:
  > *"The Generation agent simulates scientific debates among experts using self-critique and self-play techniques. These debates typically involve multiple turns of conversations (3-5 turns, up to 10) leading to a refined hypothesis generated at the end."*
  
  The collaborative discourse follows a strict procedure:
  1. **Turn 1 (Initial Contribution)**: Propose 3 distinct hypotheses addressing the research goal.
  2. **Subsequent Turns (2–4)**:
     - Clarifying questions on ambiguities.
     - Critical evaluation against criteria (correctness, utility, specificity, novelty).
     - Identifying weaknesses/limitations.
     - Proposing concrete improvements and iterative refinements.
  3. **Termination**: Once sufficient depth is reached, conclude by writing `"HYPOTHESIS"` followed by a concise, self-contained final proposal.

* **Current Code Problem**:
  - In `config.yaml:101`, `generation_debate_rounds: 0` is disabled.
  - The older debate helper in `generation.py` was a generic single-turn text transformation rather than the multi-turn collaborative self-play specified in Supplementary Note 9.1.

* **Implementation Plan**:
  1. Implement `call_llm_for_generation_debate` faithful to Supplementary Note 9.1.
  2. Maintain a conversation transcript across 3 turns.
  3. Extract the final polished hypothesis from the `"HYPOTHESIS"` block.
  4. Make debate rounds optionally configurable (0 to bypass for faster runs, 3 for maximum depth).

---

### 🔴 GN-02: Audit Failure Retry & Balanced Mode Tightening
* **Paper Grounding**: Novelty and grounding gates prevent hallucinated mechanisms and ensure falsifiability.
* **Current Code Problem**:
  - In `app/agents_modules/generation.py:1280-1282`, if all candidates fail the `HypothesisAuditor`, generation immediately returns `([], ["All generated hypotheses were rejected..."])` without attempting a re-generation with audit feedback.
  - In `app/agents_modules/generation_helpers.py:1592-1643` (balanced mode), novelty $< 5/10$ and auditor verdict `REJECT` are downgraded to warnings.
* **Implementation Plan**:
  1. If all candidates fail the audit, trigger a 1-cycle regeneration loop injecting the auditor's rejection diagnostics into the prompt.
  2. Promote auditor `REJECT` verdicts to hard rejections in balanced mode.

---

### 🟡 GN-03: Structured Experimental Protocol Specification
* **Paper Specification (Supp Note 10.2)**:
  Generated proposals must include detailed empirical validation designs:
  - **Biological/Physical Model Systems**: Specific cell lines (e.g. *MOLM-13, iPSC motor neurons*), model organisms, or materials.
  - **Perturbations & Controls**: Pharmacological agents, concentrations, knockouts, vehicle controls.
  - **Measurement Readouts**: Assays (e.g. *IC50, Western blot, mass spectrometry*), timepoints, and expected positive/negative outcomes.
* **Current Code Problem**:
  - Current generation outputs a single `feasibility` text string, which models often keep vague (*"Can be tested using standard in vitro assays"*).
* **Implementation Plan**:
  1. Enhance the generation prompt schema to require specific testing parameters: `test_model`, `perturbation`, `measurable_readout`.
  2. The HypothesisAuditor checks for concrete experimental parameters.

---

### 🟡 GN-04: Multi-Hop Conditional Reasoning for Assumption Chains
* **Paper Specification (Main text p. 118779)**:
  > *"Plausible assumptions and their subassumptions are identified through conditional reasoning hops and subsequently aggregated into complete hypotheses."*
* **Current Code Problem**:
  - `call_llm_for_assumption_analysis` generates flat assumptions in a single round.
* **Implementation Plan**:
  - Support chaining 2-hop conditional assumptions ($A \implies B \implies C$) during agentic research retrieval.

---

### 🟢 GN-05: Externalize Hardcoded Audit Thresholds & Weights
* **Current Code Problem**:
  - `AUDIT_SCORE_WEIGHTS` (`generation_helpers.py:1049-1057`), grounding pass cutoff (`line 1318`), and weighted score warning (`line 1622`) are hardcoded.
* **Implementation Plan**:
  - Move weights and cutoffs to `config.yaml` under the `rag:` section.

---

## 📜 Paper Prompt References (Supplementary Note 9.1)

### Prompt for Hypothesis Generation after Scientific Debate (Self-Play)
```text
You are an expert participating in a collaborative discourse concerning the generation of a {idea_attributes} hypothesis. You will engage in a simulated discussion with other experts. The overarching objective of this discourse is to collaboratively develop a novel and robust {idea_attributes} hypothesis.

Goal: {goal}
Criteria for a high-quality hypothesis: {preferences}
Instructions: {instructions}
Review Overview: {reviews_overview}

Procedure:
Initial contribution (if initiating the discussion): Propose three distinct {idea_attributes} hypotheses.

Subsequent contributions (continuing the discussion):
* Pose clarifying questions if ambiguities or uncertainties arise.
* Critically evaluate the hypotheses proposed thus far, addressing:
    - Adherence to {idea_attributes} criteria.
    - Utility and practicality.
    - Level of detail and specificity.
* Identify any weaknesses or potential limitations.
* Propose concrete improvements and refinements to address identified weaknesses.
* Conclude your response with a refined iteration of the hypothesis.

General guidelines:
* Exhibit boldness and creativity in your contributions.
* Maintain a helpful and collaborative approach.
* Prioritize the generation of a high-quality {idea_attributes} hypothesis.

Termination condition:
When sufficient discussion has transpired (typically 3-5 conversational turns, with a maximum of 10 turns) and all relevant questions and points have been thoroughly addressed and clarified, conclude the process by writing "HYPOTHESIS" (in all capital letters) followed by a concise and self-contained exposition of the finalized idea.

#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:
```

### Prompt for Hypothesis Generation after Literature Review
```text
You are an expert tasked with formulating a novel and robust hypothesis to address the following objective. Describe the proposed hypothesis in detail, including specific entities, mechanisms, and anticipated outcomes. This description is intended for an audience of domain experts. You have conducted a thorough review of relevant literature and developed a logical framework for addressing the objective. The articles consulted, along with your analytical reasoning, are provided below.

Goal: {goal}
Criteria for a strong hypothesis: {preferences}
Existing hypothesis (if applicable): {source_hypothesis}
{instructions}

Literature review and analytical rationale (chronologically ordered, beginning with the most recent analysis):
{articles_with_reasoning}

Proposed hypothesis (detailed description for domain experts):
```
