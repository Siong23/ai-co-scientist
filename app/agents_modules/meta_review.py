"""Hypothesis meta-review agent.

Synthesizes insights from all reviews and proximity topology to identify
recurring patterns, generate high-level critiques, and produce actionable
next-step recommendations for the research session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import ContextMemory
from ..utils import logger


class MetaReviewAgent:
    def summarize_and_feedback(
        self,
        context: ContextMemory,
        adjacency: Dict,
        *,
        proximity_data: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Summarizes research state and provides feedback.

        Parameters
        ----------
        context:
            Shared context memory.
        adjacency:
            Hypothesis adjacency graph from the Proximity agent.
        proximity_data:
            Full proximity result dict (includes ``clusters``, ``cluster_labels``,
            ``outliers``, ``exemplars``, ``near_duplicates``, ``diversity_score``).
            When provided, richer topology-aware critiques are generated.
        """
        active_hypotheses = context.get_active_hypotheses()
        if not active_hypotheses:
            return {
                "meta_review_critique": ["No active hypotheses."],
                "research_overview": {"top_ranked_hypotheses": [], "suggested_next_steps": []},
            }

        # ----------------------------------------------------------------
        # Quality-level critiques from individual hypothesis reviews
        # ----------------------------------------------------------------
        comment_summary: List[str] = []
        low_novelty_count = 0
        low_feasibility_count = 0
        for h in active_hypotheses:
            if h.novelty_review == "LOW":
                low_novelty_count += 1
            if h.feasibility_review == "LOW":
                low_feasibility_count += 1
        if low_novelty_count:
            comment_summary.append(
                f"{low_novelty_count} hypothesis(es) scored LOW on novelty — consider "
                "exploring less-studied mechanisms or cross-disciplinary connections."
            )
        if low_feasibility_count:
            comment_summary.append(
                f"{low_feasibility_count} hypothesis(es) scored LOW on feasibility — consider "
                "simplifying experimental designs or leveraging available model systems."
            )

        # ----------------------------------------------------------------
        # Topology critiques from proximity data
        # ----------------------------------------------------------------
        next_steps: List[str] = ["Refine top hypotheses based on review comments."]

        if proximity_data:
            clusters: Dict[int, List[str]] = proximity_data.get("clusters", {})
            cluster_labels: Dict[int, Dict[str, str]] = proximity_data.get("cluster_labels", {})
            outliers: List[str] = proximity_data.get("outliers", [])
            near_duplicates: List[Dict[str, Any]] = proximity_data.get("near_duplicates", [])
            diversity_score = proximity_data.get("diversity_score")
            connectivity = proximity_data.get("connectivity", {})
            highly_connected = proximity_data.get("highly_connected", [])
            isolated = proximity_data.get("isolated", outliers)

            # Diversity critique
            if isinstance(diversity_score, (int, float)) and diversity_score < 0.35:
                comment_summary.append(
                    f"Hypothesis diversity is LOW (score {diversity_score:.2f}). "
                    "Most ideas are conceptually similar — consider generating hypotheses "
                    "that explore orthogonal mechanisms or alternative experimental approaches."
                )
                next_steps.append(
                    "Instruct the Generation agent to explore underrepresented sub-fields "
                    "or use the 'out_of_box' Evolution strategy to break from the current cluster."
                )
            elif isinstance(diversity_score, (int, float)) and diversity_score > 0.75:
                comment_summary.append(
                    f"Hypothesis diversity is HIGH (score {diversity_score:.2f}). "
                    "The landscape is broad — ranking may benefit from additional iterations "
                    "to establish reliable relative scores between distant clusters."
                )

            # Cluster-level critique
            cluster_members = proximity_data.get("cluster_members")
            if isinstance(cluster_members, dict) and cluster_members:
                normalized_clusters = {
                    cluster_id: [hypothesis_id for hypothesis_id in members if hypothesis_id in context.hypotheses]
                    for cluster_id, members in cluster_members.items()
                    if isinstance(members, (list, tuple, set))
                }
            elif isinstance(clusters, dict) and clusters:
                if all(isinstance(v, (int, str)) for v in clusters.values()):
                    normalized_clusters = {}
                    for hypothesis_id, cluster_id in clusters.items():
                        if hypothesis_id in context.hypotheses:
                            normalized_clusters.setdefault(cluster_id, []).append(hypothesis_id)
                else:
                    normalized_clusters = {
                        cluster_id: [hypothesis_id for hypothesis_id in members if hypothesis_id in context.hypotheses]
                        for cluster_id, members in clusters.items()
                        if isinstance(members, (list, tuple, set))
                    }
            else:
                normalized_clusters = {}

            normalized_clusters = {
                cluster_id: members for cluster_id, members in normalized_clusters.items() if members
            }
            n_clusters = len(normalized_clusters)
            cluster_keys = list(normalized_clusters)

            # Recommendations should name the strongest connected candidate in each
            # direction, rather than asking Evolution to operate on an arbitrary node.
            representative_ids = []
            for members in sorted(normalized_clusters.values(), key=len, reverse=True):
                connected_members = [
                    hypothesis_id
                    for hypothesis_id in members
                    if hypothesis_id in highly_connected
                ]
                candidates = connected_members or members
                representative = max(
                    candidates,
                    key=lambda hypothesis_id: (
                        connectivity.get(hypothesis_id, 0),
                        context.hypotheses[hypothesis_id].elo_score,
                    ),
                )
                representative_ids.append(representative)

            if n_clusters == 1 and len(active_hypotheses) > 2:
                comment_summary.append(
                    "All active hypotheses form a single cluster — the search space may be "
                    "too narrow. Encourage exploration of adjacent research directions."
                )
            elif n_clusters > 1:
                cluster_names = []
                for idx, c_id in enumerate(cluster_keys[:4], start=1):
                    label = (
                        cluster_labels.get(c_id, {}).get("label")
                        if isinstance(cluster_labels, dict) and isinstance(cluster_labels.get(c_id), dict)
                        else None
                    )
                    if not label:
                        label = f"Cluster {idx}"
                    cluster_names.append(label)
                comment_summary.append(
                    f"Identified {n_clusters} hypothesis cluster(s): "
                    + ", ".join(f'"{n}"' for n in cluster_names)
                    + ". Each cluster represents a distinct research direction."
                )
                next_steps.append(
                    "Run Evolution from the strongest connected hypothesis in each "
                    f"cluster ({', '.join(representative_ids[:4])}) to deepen distinct directions."
                )

                largest_cluster = max(len(members) for members in normalized_clusters.values())
                if largest_cluster / len(active_hypotheses) >= 0.75 and len(active_hypotheses) > 2:
                    next_steps.append(
                        "Prioritize hypotheses outside the dominant cluster in the next "
                        "generation cycle to reduce concentration in the search space."
                    )

            if n_clusters == 0 and isolated:
                next_steps.append(
                    "Validate isolated hypotheses against the research goal and evidence "
                    "before expanding them; their lack of semantic neighbors is not by itself "
                    "evidence that they should be deactivated."
                )
            elif n_clusters == 1 and len(active_hypotheses) > 2:
                next_steps.append(
                    "Generate hypotheses using mechanisms or experimental approaches that "
                    "are orthogonal to the current cluster before further ranking."
                )

            # Outlier critique
            if outliers:
                comment_summary.append(
                    f"{len(outliers)} hypothesis(es) are isolated outliers with low "
                    "similarity to all others. These may represent creative long-shots "
                    "worth investigating or off-topic noise — review them manually: " + ", ".join(outliers[:5]) + "."
                )
                next_steps.append(
                    "Review outlier hypotheses for potential breakthrough ideas or "
                    "off-topic artifacts that should be deactivated."
                )

            # Near-duplicate note
            if near_duplicates:
                comment_summary.append(
                    f"{len(near_duplicates)} near-duplicate hypothesis pair(s) were detected "
                    "and should be reviewed for redundant ranking and evidence. Future "
                    "generation cycles should avoid re-proposing these ideas."
                )
                next_steps.append(
                    "Merge or deactivate confirmed near-duplicates after reviewing their "
                    "evidence and Elo scores."
                )

            if isolated and n_clusters > 0:
                next_steps.append(
                    "Manually validate isolated hypotheses for breakthrough potential or "
                    "off-topic content before using them as evolution parents."
                )

        if not comment_summary:
            comment_summary.append("Overall hypothesis quality seems reasonable based on automated review.")

        # ----------------------------------------------------------------
        # Top-ranked hypotheses
        # ----------------------------------------------------------------
        best_hypotheses = sorted(active_hypotheses, key=lambda h: h.elo_score, reverse=True)[:3]
        logger.info(
            "Top hypotheses for meta-review: %s",
            [h.hypothesis_id for h in best_hypotheses],
        )

        # ----------------------------------------------------------------
        # Previous meta-review feedback injection
        # ----------------------------------------------------------------
        if context.meta_review_feedback:
            prev = context.meta_review_feedback[-1]
            prev_steps = (prev.get("research_overview") or {}).get("suggested_next_steps") or []
            if prev_steps:
                next_steps.append(f"[Continuing from prior cycle] {prev_steps[0]}")

        overview = {
            "meta_review_critique": comment_summary,
            "research_overview": {
                "top_ranked_hypotheses": [h.to_dict() for h in best_hypotheses],
                "suggested_next_steps": next_steps,
            },
        }
        context.meta_review_feedback.append(overview)
        logger.info("Meta-review complete with %d critique(s).", len(comment_summary))
        return overview
