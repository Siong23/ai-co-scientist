# Supervisor Agent Gaps & Improvements (Based on Nature Paper)

> Audit Date: 2026-08-18  
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist", Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)  
> Supplementary Notes: [Supplementary Material (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf) (Notes 1, 8, 10.1)

---

## 📌 Status Summary

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

| ID | Improvement Area | Priority | Paper Reference | Status |
|---|---|---|---|---|
| **SV-01** | Scientist-in-the-Loop Interactive Steering & Idea Seeding | 🔴 High | Main text §Scientist-in-the-loop, p. 65027 | ⬜ open |
| **SV-02** | Rating Convergence & Plateaux-Driven Dynamic Planning | 🔴 High | Supp Note 8 (`DecideNextSteps`) | ⬜ open |
| **SV-03** | Finalization Quality Verification Gate | 🟡 Medium | Code audit & Reliability | ⬜ open |
| **SV-04** | Natural Language Goal Parsing into Structured Configuration | 🟡 Medium | Supp Note 10.1 | ⬜ open |

---

## 🔍 Detailed Gap Analysis & Technical Specs

### 🔴 SV-01: Scientist-in-the-Loop Interactive Steering
* **Paper Grounding (Main text p. 65027)**:
  > *"Co-Scientist is purpose-built for a 'scientist-in-the-loop' collaborative paradigm. Scientists can actively interact with and steer the system, including directly suggesting initial ideas and hypotheses for exploration, refining generated ideas or providing feedback through natural language chat."*
* **Current Code Problem**:
  - `SupervisorAgent.run_cycle` and `run_dynamic_cycle` execute as a continuous batch in a worker thread. There are no pause/steer hooks where a user can inject constraints, seed a specific hypothesis, or prune a branch before the next stage.
* **Implementation Plan**:
  1. Add UI controls in Gradio allowing scientists to provide mid-run natural language feedback and pin/prune hypotheses.
  2. The supervisor dynamically incorporates user steering directives into `context.user_guidance`.

---

### 🔴 SV-02: Rating Convergence & Plateaux-Driven Dynamic Planning
* **Paper Grounding (Supp Note 8)**:
  > *"If hypothesis quality has stopped improving THEN EvolveTopHypotheses; Keep tournament running to refine scores."*
* **Current Code Problem**:
  - `SupervisorPlanner` uses fixed heuristic thresholds (`steps_remaining <= 1`) rather than computing Elo stability/convergence (variance of Elo changes over the last batch of matches).
* **Implementation Plan**:
  1. Compute Elo rating change delta $\Delta = \sum |\Delta \text{Elo}|$.
  2. When $\Delta < \epsilon$ (ratings have converged), automatically trigger `EVOLUTION` or `META_REVIEW` rather than redundant ranking matches.

---

### 🟡 SV-03: Finalization Quality Gate
* **Current Code Problem**:
  - If step count reaches limit, `FINALIZE` is returned even if top candidates haven't been reviewed or ranked.
* **Implementation Plan**:
  - Enforce prerequisite checks before finishing: At least $N$ hypotheses must be in `ACCEPTED` state with confirmed citations.
