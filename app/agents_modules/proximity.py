"""Hypothesis proximity analysis agent."""

from __future__ import annotations

from typing import Dict

from ..models import ContextMemory
from ._compat import _legacy


class ProximityAgent:
    def build_proximity_graph(self, context: ContextMemory) -> Dict:
        """Builds proximity graph data based on hypothesis similarity."""
        active_hypotheses = context.get_active_hypotheses()
        adjacency = {}
        if not active_hypotheses:
            _legacy.logger.info("No active hypotheses to build proximity graph.")
            return {"adjacency_graph": {}, "nodes": [], "edges": []}

        for i in range(len(active_hypotheses)):
            hypo_i = active_hypotheses[i]
            adjacency[hypo_i.hypothesis_id] = []
            for j in range(len(active_hypotheses)):
                if i == j:
                    continue
                hypo_j = active_hypotheses[j]
                if hypo_i.text and hypo_j.text:
                    sim = _legacy.similarity_score(hypo_i.text, hypo_j.text)
                    adjacency[hypo_i.hypothesis_id].append({"other_id": hypo_j.hypothesis_id, "similarity": sim})
                else:
                    _legacy.logger.warning(
                        f"Skipping similarity for {hypo_i.hypothesis_id} or {hypo_j.hypothesis_id} due to empty text."
                    )

        visjs_data = _legacy.generate_visjs_data(adjacency)  # Use utility function
        _legacy.logger.info("Built proximity graph adjacency with %d nodes.", len(active_hypotheses))
        return {"adjacency_graph": adjacency, "nodes": visjs_data["nodes"], "edges": visjs_data["edges"]}
