# Proximity Agent Gaps & Improvements (Based on Nature Paper)

> Audit Date: 2026-08-18  
> Reference: [Gottweis et al. "Accelerating scientific discovery with Co-Scientist", Nature 655, 487–496 (2026)](https://doi.org/10.1038/s41586-026-10644-y)  
> Supplementary Notes: [Supplementary Material (PDF)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10644-y/MediaObjects/41586_2026_10644_MOESM1_ESM.pdf) (Notes 8, 10.8)

---

## 📌 Status Summary

Status legend: ⬜ open · 🟡 in progress · ✅ fixed · ❌ won't fix

| ID | Improvement Area | Priority | Paper Reference | Status |
|---|---|---|---|---|
| **PX-01** | Goal-Aware Semantic Similarity Encoding | 🔴 High | Main text §Proximity Agent, p. 126087 | ⬜ open |
| **PX-02** | Automated Deactivation of Sub-par Near-Duplicates | 🔴 High | Code audit & Meta-review consistency | ⬜ open |
| **PX-03** | Cluster Exemplar Feeding to Evolution & Ranking | 🟡 Medium | Main text p. 126087 & Supp Note 8 | ⬜ open |
| **PX-04** | Dynamic Similarity Thresholding for Sparsification | 🟢 Low | Graph scaling & UI clarity | ⬜ open |

---

## 🔍 Detailed Gap Analysis & Technical Specs

### 🔴 PX-01: Goal-Aware Semantic Similarity Encoding
* **Paper Specification (Main text §Proximity Agent, p. 126087)**:
  > *"The Proximity agent calculates the similarity between research hypotheses and proposals, and builds a proximity graph, taking into account the specific research goal."*
* **Current Code Problem**:
  - `SimilarityScorer.calculate_similarity` (`proximity_helpers.py:100-160`) embeds raw `hypothesis.text` without prefixing or conditioning on `research_goal.description` or target attributes.
* **Implementation Plan**:
  - Format embedding text as `f"Research Goal: {goal.description} | Hypothesis: {h.text}"` when calculating semantic proximity.

---

### 🔴 PX-02: Automated Deactivation of Near-Duplicates
* **Current Code Problem**:
  - `proximity.py` finds pairs with similarity $> 0.90$ and records them in `near_duplicates`, and `meta_review.py:157-159` tells the user that *"the lower-Elo duplicate was automatically deactivated"*, but **no code actually set `is_active = False`**.
* **Implementation Plan**:
  - In `ProximityAgent.get_proximity_analysis`, for each pair $(H_1, H_2)$ with similarity $\ge \text{near\_duplicate\_threshold}$:
    - Keep the one with higher Elo (or higher reflection score if unranked).
    - Set the lower one to `is_active = False` with `deactivation_reason = "near_duplicate_of_{higher_id}"`.

---

### 🟡 PX-03: Cluster Exemplar Feeding
* **Paper Grounding**: Supp Note 8 mentions identifying top exemplars from each distinct cluster to ensure diversity in evolution and meta-review.
* **Implementation Plan**:
  - Export `exemplar_ids` directly into `context` so `EvolutionAgent._strategies_for_cycle` can select diverse parents across clusters rather than drawing all parents from the single largest cluster.
