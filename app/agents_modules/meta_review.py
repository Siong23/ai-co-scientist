"""Hypothesis meta-review agent."""

from __future__ import annotations

from typing import Dict

from ..models import ContextMemory
from ._compat import _legacy


class MetaReviewAgent:
    def summarize_and_feedback(self, context: ContextMemory, adjacency: Dict) -> Dict:
        """Summarizes research state and provides feedback."""
        active_hypotheses = context.get_active_hypotheses()
        if not active_hypotheses:
            return {
                "meta_review_critique": ["No active hypotheses."],
                "research_overview": {"top_ranked_hypotheses": [], "suggested_next_steps": []},
            }

        comment_summary = set()
        for h in active_hypotheses:
            # Example critique based on reviews
            if h.novelty_review == "LOW":
                comment_summary.add("Some ideas lack novelty.")
            if h.feasibility_review == "LOW":
                comment_summary.add("Some ideas may have low feasibility.")
            # Could add critiques based on adjacency graph (e.g., clusters, outliers)

        best_hypotheses = sorted(active_hypotheses, key=lambda h: h.elo_score, reverse=True)[:3]
        _legacy.logger.info("Top hypotheses for meta-review: %s", [h.hypothesis_id for h in best_hypotheses])

        # Example suggested next steps
        next_steps = [
            "Refine top hypotheses based on review comments.",
            "Consider exploring areas with fewer, less connected hypotheses (if any).",
            "Seek external expert feedback on top candidates.",
        ]
        if not comment_summary:
            comment_summary.add("Overall hypothesis quality seems reasonable based on automated review.")

        overview = {
            "meta_review_critique": list(comment_summary),
            "research_overview": {
                "top_ranked_hypotheses": [h.to_dict() for h in best_hypotheses],  # Use to_dict for serialization
                "suggested_next_steps": next_steps,
            },
        }
        context.meta_review_feedback.append(overview)  # Store feedback in context
        _legacy.logger.info("Meta-review complete: %s", overview)
        return overview
