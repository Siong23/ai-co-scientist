# Reflection Agent Gaps & Improvements (Based on Nature Paper)

> Audit Date: 2026-08-18  
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist", Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)  
> Supplementary Notes: [Supplementary Material (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf) (Notes 8, 9.2, 10.3, 10.4, 10.5, 10.6, 10.7)

---

## 📌 Status Summary

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

| ID | Improvement Area | Priority | Paper Reference | Status |
|---|---|---|---|---|
| **RF-01** | Observation Causal Analysis & Disproof Detection | 🔴 High | Main text §Reflection Agent & Supp Note 9.2 | ⬜ open |
| **RF-02** | Deep Verification Review via Assumption Decomposition | 🔴 High | Main text p. 121589 & Supp Notes 10.6, 10.7 | ⬜ open |
| **RF-03** | Structured Novelty Review (Prior Art vs. Novel Mechanisms) | 🟡 Medium | Supp Note 10.3 | ⬜ open |
| **RF-04** | Tiered Reflection Pipeline (Fast Initial Screen $\to$ Deep Verification) | 🟡 Medium | Main text §Reflection & Supp Note 8 | ⬜ open |
| **RF-05** | Enriched Revision Prompt with Assumption & Causal Diagnostic Context | 🟢 Low | Supp Notes 9.4, 10.4 | ⬜ open |

---

## 🔍 Detailed Gap Analysis & Technical Specs

### 🔴 RF-01: Observation Causal Analysis & Disproof Detection
* **Paper Specification (Main Text §Reflection & Supp Note 9.2)**:
  > *"The Reflection agent analyzes the relationship between a hypothesis and empirical observations from scientific literature. Specifically, it determines whether the hypothesis provides a novel causal explanation for under-explained observations or is contradicted/disproved by them."*
  
  The paper specifies a 5-class causal verdict:
  1. `missing piece`: The hypothesis provides a novel, plausible causal explanation for unexplained observations.
  2. `already explained`: Consistent with observations, but the cause is already established in literature (no novelty).
  3. `other explanations more likely`: Hypothesis could explain it, but standard/superior explanations already exist.
  4. `neutral`: Observations neither support nor contradict.
  5. `disproved`: Literature observations directly contradict or invalidate the hypothesis.

* **Current Code Problem**:
  - `app/agents_modules/reflection_helpers.py` only outputs 7 generic integer scores (1–10).
  - There is no explicit causal classification. If literature observations disprove a hypothesis, it might only receive a low `plausibility_score` rather than a decisive `disproved` flag triggering immediate rejection.

* **Implementation Plan**:
  1. Implement `call_llm_for_observation_causal_analysis` following Supplementary Note 9.2 prompt structure.
  2. If verdict is `disproved`, automatically mark hypothesis as `REJECT` (`is_active = False`).
  3. If verdict is `missing piece`, boost novelty and research value weightings.

---

### 🔴 RF-02: Deep Verification Review via Assumption Decomposition
* **Paper Specification (Main Text p. 121589 & Supp Notes 10.6, 10.7)**:
  > *"The Reflection agent conducts a deep verification review, decomposing the hypothesis into constituent assumptions. Each assumption is further broken down into fundamental subassumptions, decontextualized and independently evaluated for correctness to identify invalidating elements for subsequent filtering."*
  
  The agent probes assumptions with targeted counter-questions (e.g. *Is this mechanism primary or a downstream consequence? Are specific cell types inherently more susceptible?*).

* **Current Code Problem**:
  - Although the `GenerationAgent` has an internal `AssumptionAssessment` during retrieval, the **Reflection Agent** evaluates hypotheses as monolithic text blocks without breaking down or verifying their constituent assumptions against evidence.

* **Implementation Plan**:
  1. Add `AssumptionDecomposer` helper to extract explicit and implicit assumptions from the hypothesis.
  2. Independently verify each assumption against retrieved full-text chunks in `ChromaPaperLibrary`.
  3. Flag critical assumptions with status `CONTRADICTED` or `UNSUPPORTED` in the `ReflectionReport`.

---

### 🟡 RF-03: Structured Novelty Review (Prior Art vs. Novel Mechanisms)
* **Paper Specification (Supp Note 10.3)**:
  - Reflection generates an explicit two-column comparison:
    - **Aspects already explored**: Known mechanisms documented in specific cited literature `[1, 2, 3]`.
    - **Novel aspects**: Specific new causal links, unstudied PTMs, new targets, or new combination therapies.
* **Current Code Problem**:
  - Current reflection prompt requests a single `novelty_score` (1–10) and a brief `comment`.
  - The model does not explicitly articulate *what parts are old vs. what parts are genuinely new*.
* **Implementation Plan**:
  1. Add structured fields `aspects_already_explored: List[str]` and `novel_aspects: List[str]` to `ReflectionReport`.
  2. Instruct the Reflection LLM to separate prior art from novel inferences.

---

### 🟡 RF-04: Tiered Reflection Pipeline (Fast Initial Screen $\to$ Deep Verification)
* **Paper Specification (Supp Note 8)**:
  - Not all generated hypotheses need expensive deep verification.
  - Initial generation uses a fast review screen; hypotheses passing the screen or advancing to tournament finals undergo deep multi-step verification.
* **Current Code Problem**:
  - Full 7-dimensional evaluation is executed for every hypothesis, with no staged progression.
* **Implementation Plan**:
  1. **Tier 1 (Initial Screening)**: Quick alignment, formatting, and causal disproof check.
  2. **Tier 2 (Full Scientific Review)**: 7-criteria scoring + structured novelty breakdown.
  3. **Tier 3 (Deep Verification for Top Candidates)**: Assumption decomposition and targeted literature verification.

---

### 🟢 RF-05: Enriched Revision Prompt with Assumption & Causal Diagnostic Context
* **Paper Specification (Supp Notes 9.4, 10.4)**:
  - Revisions should directly target the specific invalid assumptions or causal gaps identified during review.
* **Current Code Problem**:
  - `call_llm_for_hypothesis_revision` provides the general review comment and strengths/weaknesses, but lacks granular assumption-level breakdown.
* **Implementation Plan**:
  1. Pass the deep verification assumption reports and causal analysis verdicts into `call_llm_for_hypothesis_revision`.

---

## 📜 Paper Prompt References (Supplementary Note 9.2)

### Prompt for Generating Observations & Causal Analysis
```text
You are an expert in scientific hypothesis evaluation. Your task is to analyze the relationship between a provided hypothesis and observations from a scientific article. Specifically, determine if the hypothesis provides a novel causal explanation for the observations, or if they contradict it.

Instructions:
1. Observation extraction: list relevant observations from the article.
2. Causal analysis (individual): for each observation:
   a. State if its cause is already established.
   b. Assess if the hypothesis could be a causal factor (hypothesis => observation).
      Start with: "would we see this observation if the hypothesis was true:".
   c. Explain if it's a novel explanation. If not, or if a better explanation exists, state: "not a missing piece."
3. Causal analysis (summary): determine if the hypothesis offers a novel explanation for a subset of observations. Include reasoning.
   Start with: "would we see some of the observations if the hypothesis was true:".
4. Disproof analysis: determine if any observations contradict the hypothesis.
   Start with: "does some observations disprove the hypothesis:".
5. Conclusion: state: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>".

Scoring:
* Already explained: hypothesis consistent, but causes are known. No novel explanation.
* Other explanations more likely: hypothesis *could* explain, but better explanations exist.
* Missing piece: hypothesis offers a novel, plausible explanation.
* Neutral: hypothesis neither explains nor is contradicted.
* Disproved: observations contradict the hypothesis.

Important: if observations are expected regardless of the hypothesis, and don't disprove it, it's neutral.

Article: {article}
Hypothesis: {hypothesis}
Response: (provide reasoning. end with: "hypothesis: <already explained, other explanations more likely, missing piece, neutral, or disproved>")
```
