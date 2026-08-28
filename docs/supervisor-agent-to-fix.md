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
| **SV-02** | Rating Convergence & Plateaux-Driven Dynamic Planning | 🔴 High | Supp Note 8 (`DecideNextSteps`) | ✅ fixed |
| **SV-03** | Finalization Quality Verification Gate | 🟡 Medium | Code audit & Reliability | ✅ fixed |
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
* **Implemented**:
  1. Ranking batches record bounded Elo snapshots in `ContextMemory.supervisor_state`.
  2. The planner compares like-for-like candidate sets and detects a configurable top-Elo plateau.
  3. It runs a bounded additional tournament batch while ratings are moving, then routes plateaux to `EVOLUTION`.

---

### 🟡 SV-03: Finalization Quality Gate
* **Implemented**:
  - `FINALIZE` is gated on a configurable minimum number of accepted hypotheses, completed tournament matches, finalist participation, and verified evidence citations.
  - Premature finalization is routed back to generation, ranking, or grounded evolution. Budget exhaustion is reported as incomplete instead of successful.
