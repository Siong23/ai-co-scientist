# Ranking Agent Gaps & Improvements (Based on Nature Paper)

> Audit Date: 2026-08-18  
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist", Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)  
> Supplementary Notes: [Supplementary Material (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf) (Notes 8, 9.3, 10.9)

---

## 📌 Status Summary

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

| ID | Improvement Area | Priority | Paper Reference | Status |
|---|---|---|---|---|
| **RK-01** | A/B Presentation Position Bias Neutralization | 🔴 High | Main text p. 126087 & Supp Note 9.3 | ⬜ open |
| **RK-02** | Proximity & Novelty Guided Tournament Matchmaking | 🔴 High | Main text §Ranking Agent, p. 126087 & Supp Note 8 | ⬜ open |
| **RK-03** | Tiered Matches: Multi-turn Debate for Top-Ranked vs. Fast Single-turn for Low-Ranked | 🟡 Medium | Main text §Ranking Agent, p. 126087 & Supp Note 9.3 / 10.9 | ⬜ open |
| **RK-04** | Prompt Alignment: Mechanism/Evidence Over Cross-Review Numeric Scores | 🟡 Medium | Supp Note 9.3 Prompt Spec | ⬜ open |
| **RK-05** | Dynamic K-Factor for Early Match Convergence | 🟢 Low | Supp Note 8 Elo specification | ⬜ open |

---

## 🔍 Detailed Gap Analysis & Technical Specs

### 🔴 RK-01: A/B Presentation Position Bias Neutralization
* **Problem**: In `judge_hypotheses` (`app/agents_modules/ranking_helpers.py:504-555`), Hypothesis A is always presented first, and Hypothesis B is always presented second. LLM evaluators exhibit known position bias (tendency to favor the first or last candidate), leading to systematic rating drift.
* **Paper Grounding**: The paper explicitly addresses ordering bias via multi-turn deliberation and symmetric judging (`p. 126087`).
* **Implementation Plan**:
  1. In `run_pairwise_debate`, randomly flip presentation order ($A \leftrightarrow B$) with 50% probability before invoking LLM judge.
  2. Map the winner back to the true canonical hypothesis IDs (`"A"` or `"B"`) before recording and updating Elo.
  3. Support dual-round evaluation mode for high-stakes matches (evaluate $A$ vs $B$ and $B$ vs $A$, averaging scores/checking consistency).
* **Target Files**:
  - `app/agents_modules/ranking_helpers.py` (`run_pairwise_debate`, `judge_hypotheses`)
  - `tests/test_ranking_performance.py`

---

### 🔴 RK-02: Proximity & Novelty Guided Matchmaking
* **Problem**: Currently `RankingAgent.run_tournament` (`app/agents_modules/ranking.py:46-55`) generates an exhaustive $O(N^2)$ Cartesian product of all pairs, filtering only by `new_hypothesis_ids`. It does not prioritize close competitors.
* **Paper Grounding** (Main text p. 126087):
  > *"The Ranking agent prioritizes tournament matches as follows: (1) hypotheses are more likely to be compared with similar ones (on the basis of the Proximity agent’s graph); (2) newer and top-ranking hypotheses are prioritized for participation in tournament matches."*
* **Implementation Plan**:
  1. Leverage `context` proximity adjacency / cosine similarity matrix when available.
  2. Sort / sample candidate pairs by:
     - **Cluster / Mechanism Rivals**: Pairs with similarity $> 0.5$ (resolving ambiguity within same sub-field).
     - **New vs. Top-Ranked**: Pairing newly introduced candidates against existing high-Elo hypotheses to quickly calibrate ratings.
  3. Cap total tournament rounds per cycle to $O(N \log N)$ or configurable `max_tournament_matches` rather than $O(N^2)$.
* **Target Files**:
  - `app/agents_modules/ranking.py` (`run_tournament`, matchmaking pair generation)
  - `config.yaml` (`ranking.max_matches_per_cycle`, `ranking.proximity_guided_matching: true`)

---

### 🟡 RK-03: Tiered Match Hierarchy (Multi-Turn Scientific Debate for Top Tier)
* **Problem**: Every comparison currently uses the identical single-turn prompt (`judge_hypotheses`), regardless of whether the hypotheses are top contenders or bottom outliers.
* **Paper Grounding** (Main text p. 126087 & Supp Note 9.3/10.9):
  > *"Top-ranked hypotheses are compared pairwise in tournament matches through multi-turn scientific debates (typically 3-5 turns, up to 10). Lower-ranked hypotheses undergo single-turn comparisons."*
* **Implementation Plan**:
  1. Distinguish match tiers:
     - **Top-Tier Matches** (both hypotheses in Top-$K$ or Elo $\ge 1250$): Run simulated multi-turn expert debate (Expert 1 defends A, Expert 2 defends B, Panel Moderator probes weaknesses and declares `"better idea: 1 or 2"`).
     - **Standard Matches**: Run single-turn comparative judge prompt.
  2. Implement `run_multi_turn_debate` following Supplementary Note 9.3 / 10.9 prompt structure.
* **Target Files**:
  - `app/agents_modules/ranking_helpers.py` (`run_multi_turn_debate`, `judge_debate`)
  - `app/agents_modules/ranking.py`

---

### 🟡 RK-04: Prompt Alignment: Disregard Cross-Review Absolute Scores
* **Problem**: In `ranking_helpers.py:41-50, 737-746`, explicit numerical scores from independent reflection reports are computed and injected. Independent reviews have score drift, and LLMs may lazily compare numbers rather than scrutinizing scientific mechanisms.
* **Paper Grounding** (Supp Note 9.3 Prompt Spec):
  > *"Each hypothesis includes an independent review. These reviews may contain numerical scores. Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews."*
* **Implementation Plan**:
  1. Update judge system instructions to explicitly require evaluating underlying mechanisms, assumptions, and evidence citations rather than comparing numeric rating values.
  2. Emphasize causal analysis and testability.
* **Target Files**:
  - `app/agents_modules/ranking_helpers.py` (`judge_hypotheses`)

---

### 🟢 RK-05: Dynamic K-Factor for Fast Early Convergence
* **Problem**: Fixed $K=32$ for all hypotheses means new candidates need many matches to reach their true Elo rating.
* **Paper Grounding**: Initial rating is 1,200.
* **Implementation Plan**:
  1. Add a `match_count` tracker to `Hypothesis`.
  2. For hypotheses with $\le 3$ matches, use $K=48$ or $K=64$ (accelerated rating discovery); once $\ge 5$ matches, decay to stable $K=32$.
* **Target Files**:
  - `app/models.py` (`Hypothesis.match_count`)
  - `app/agents_modules/ranking_helpers.py` (`update_elo`)

---

## 📜 Paper Prompt References (Supplementary Note 9.3)

### Single-Turn Tournament Comparison Prompt
```text
You are an expert evaluator tasked with comparing two hypotheses. Evaluate the two provided hypotheses (hypothesis 1 and hypothesis 2) and determine which one is superior based on the specified {idea_attributes}. Provide a concise rationale for your selection, concluding with the phrase "better idea: <1 or 2>".

Goal: {goal}
Evaluation criteria: {preferences}
Considerations: {notes}

Each hypothesis includes an independent review. These reviews may contain numerical scores. Disregard these scores in your comparative analysis, as they may not be directly comparable across reviews.

Hypothesis 1: {hypothesis 1}
Hypothesis 2: {hypothesis 2}
Review of hypothesis 1: {review 1}
Review of hypothesis 2: {review 2}

Reasoning and conclusion (end with "better hypothesis: <1 or 2>"):
```

### Multi-Turn Scientific Debate Prompt
```text
You are an expert in comparative analysis, simulating a panel of domain experts engaged in a structured discussion to evaluate two competing hypotheses. The objective is to rigorously determine which hypothesis is superior based on a predefined set of attributes and criteria. The experts possess no pre-existing biases toward either hypothesis and are solely focused on identifying the optimal choice, given that only one can be implemented.

Goal: {goal}
Criteria for hypothesis superiority: {preferences}
Hypothesis 1: {hypothesis 1}
Hypothesis 2: {hypothesis 2}
Initial review of hypothesis 1: {review 1}
Initial review of hypothesis 2: {review 2}

Debate procedure:
The discussion will unfold in a series of turns, typically ranging from 3 to 5, with a maximum of 10.
Turn 1: begin with a concise summary of both hypotheses and their respective initial reviews.
Subsequent turns:
* Pose clarifying questions to address any ambiguities or uncertainties.
* Critically evaluate each hypothesis in relation to the stated Goal and Criteria (correctness, utility, specificity, novelty, desirability).
* Identify and articulate any weaknesses, limitations, or potential flaws in either hypothesis.

Termination and judgment:
Once the discussion has reached a point of sufficient depth (typically 3-5 turns, up to 10 turns) and all relevant questions and concerns have been thoroughly addressed, provide a conclusive judgment. This judgment should succinctly state the rationale for the selection. Then, indicate the superior hypothesis by writing the phrase "better idea: ", followed by "1" (for hypothesis 1) or "2" (for hypothesis 2).
```
