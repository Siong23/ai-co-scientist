"""Hypothesis proximity analysis agent."""

from __future__ import annotations

from typing import Dict, Optional

from ..models import ContextMemory, ResearchGoal
from ..utils import logger
from .proximity_helpers import (
    BatchSimilarityCalculator,
    GraphOptimizer,
    SimilarityConfig,
    SimilarityScorer,
)


class ProximityAgent:
    """Agent for analyzing hypothesis proximity and similarity relationships."""

    def __init__(
        self,
        similarity_config: Optional[SimilarityConfig] = None,
    ):
        self.similarity_config = (
            similarity_config or SimilarityConfig()
        )

        self.scorer = SimilarityScorer(
            self.similarity_config
        )

        self.optimizer = GraphOptimizer()

    def _resolve_threshold(
        self,
        similarity_threshold: Optional[float],
        hypothesis_count: Optional[int] = None,
    ) -> float:
        """
        Resolve the similarity threshold.

        Explicitly checks for None so that 0.0 remains a valid threshold.
        """
        if similarity_threshold is None:
            threshold = self.similarity_config.similarity_threshold
            if self.similarity_config.dynamic_thresholding and hypothesis_count and hypothesis_count > 3:
                threshold += min(0.2, 0.02 * (hypothesis_count - 3))
            return min(1.0, threshold)

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0."
            )

        return similarity_threshold

    def _resolve_top_k(
        self,
        top_k: Optional[int],
    ) -> Optional[int]:
        """Validate the maximum number of edges per node."""

        if top_k is None:
            return None

        if top_k < 0:
            raise ValueError("top_k must be non-negative.")

        return top_k

    def build_proximity_graph(
        self,
        context: ContextMemory,
        method: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        research_goal: Optional[ResearchGoal] = None,
    ):
        """
        Build a proximity graph using pairwise similarity scoring.
        """

        # ---------------------------------------------------------
        # Resolve default method
        # ---------------------------------------------------------
        if method is None:
            method = self.similarity_config.default_method

        # ---------------------------------------------------------
        # Validate parameters
        # ---------------------------------------------------------
        if top_k is not None and top_k < 0:
            raise ValueError(
                "top_k must be non-negative"
            )

        # ---------------------------------------------------------
        # Get active hypotheses
        # ---------------------------------------------------------
        hypotheses = context.get_active_hypotheses()
        similarity_threshold = self._resolve_threshold(similarity_threshold, len(hypotheses))

        if not hypotheses:
            return {
                "adjacency_graph": {},
                "nodes": [],
                "edges": [],
            }

        node_ids = [
            hypothesis.hypothesis_id
            for hypothesis in hypotheses
        ]

        adjacency_graph = {
            node_id: []
            for node_id in node_ids
        }

        # ---------------------------------------------------------
        # Compute pairwise similarities
        # ---------------------------------------------------------
        for i in range(len(hypotheses)):
            for j in range(i + 1, len(hypotheses)):

                hypothesis_a = hypotheses[i]
                hypothesis_b = hypotheses[j]

                similarity = self.scorer.score(
                    self._similarity_text(hypothesis_a.text, research_goal, method),
                    self._similarity_text(hypothesis_b.text, research_goal, method),
                    method=method,
                )

                if similarity >= similarity_threshold:

                    # A -> B
                    adjacency_graph[
                        hypothesis_a.hypothesis_id
                    ].append(
                        {
                            "other_id": hypothesis_b.hypothesis_id,
                            "similarity": similarity,
                        }
                    )

                    # B -> A
                    adjacency_graph[
                        hypothesis_b.hypothesis_id
                    ].append(
                        {
                            "other_id": hypothesis_a.hypothesis_id,
                            "similarity": similarity,
                        }
                    )

        # ---------------------------------------------------------
        # Apply top-k
        # ---------------------------------------------------------
        if top_k is not None:
            if top_k == 0:
                adjacency_graph = {
                    node_id: []
                    for node_id in node_ids
                }
            else:
                adjacency_graph = self.optimizer.get_top_k_edges(
                    adjacency_graph,
                    k=top_k,
                )

        # ---------------------------------------------------------
        # Build edge list
        # ---------------------------------------------------------
        edges = []

        for node_id, node_edges in adjacency_graph.items():
            for edge in node_edges:
                edges.append(
                    {
                        "source": node_id,
                        "target": edge["other_id"],
                        "similarity": edge["similarity"],
                    }
                )

        return {
            "adjacency_graph": adjacency_graph,
            "nodes": node_ids,
            "edges": edges,
        }

    @staticmethod
    def _similarity_text(text: str, research_goal: Optional[ResearchGoal], method: str) -> str:
        """Condition semantic proximity on the research goal when available."""
        if research_goal is None or method.lower() in {"jaccard", "sequence"}:
            return text
        return (
            f"Research goal: {research_goal.description}\n"
            f"Preferences: {research_goal.preferences}\n"
            f"Idea attributes: {research_goal.idea_attributes}\n"
            f"Constraints: {research_goal.constraints or {}}\n"
            f"Hypothesis: {text}"
        )

    def build_proximity_graph_optimized(
        self,
        context: ContextMemory,
        method: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        research_goal: Optional[ResearchGoal] = None,
    ):
        """
        Build a proximity graph using batch similarity calculation.

        This optimized implementation computes the pairwise similarity
        matrix once and constructs the adjacency graph from the matrix.

        Compared with the standard implementation, this avoids repeatedly
        calling the similarity scorer for the same hypothesis pairs.
        """

        # ---------------------------------------------------------
        # Resolve default similarity method
        # ---------------------------------------------------------
        if method is None:
            method = self.similarity_config.default_method

        # ---------------------------------------------------------
        # Validate parameters
        # ---------------------------------------------------------
        if top_k is not None and top_k < 0:
            raise ValueError(
                "top_k must be non-negative"
            )

        # ---------------------------------------------------------
        # Get active hypotheses only
        # ---------------------------------------------------------
        hypotheses = context.get_active_hypotheses()
        similarity_threshold = self._resolve_threshold(similarity_threshold, len(hypotheses))

        if not hypotheses:
            return {
                "adjacency_graph": {},
                "nodes": [],
                "edges": [],
            }

        # ---------------------------------------------------------
        # Extract IDs and text
        # ---------------------------------------------------------
        node_ids = [
            hypothesis.hypothesis_id
            for hypothesis in hypotheses
        ]

        texts = [
            self._similarity_text(hypothesis.text, research_goal, method)
            for hypothesis in hypotheses
        ]

        # ---------------------------------------------------------
        # Compute similarity matrix in batch
        # ---------------------------------------------------------
        calculator = BatchSimilarityCalculator(
            self.scorer
        )

        similarity_matrix = calculator.compute_similarity_matrix(
            texts,
            method=method,
        )

        # ---------------------------------------------------------
        # Convert similarity matrix into adjacency graph
        # ---------------------------------------------------------
        adjacency_graph = calculator.build_adjacency_from_matrix(
            node_ids,
            similarity_matrix,
            threshold=similarity_threshold,
        )

        # ---------------------------------------------------------
        # Apply top-k filtering if requested
        # ---------------------------------------------------------
        if top_k is not None:
            if top_k == 0:
                adjacency_graph = {
                    node_id: []
                    for node_id in node_ids
                }
            else:
                adjacency_graph = self.optimizer.get_top_k_edges(
                    adjacency_graph,
                    k=top_k,
                )

        # ---------------------------------------------------------
        # Build edge list
        # ---------------------------------------------------------
        edges = []

        for node_id, node_edges in adjacency_graph.items():
            for edge in node_edges:
                edges.append(
                    {
                        "source": node_id,
                        "target": edge["other_id"],
                        "similarity": edge["similarity"],
                    }
                )

        return {
            "adjacency_graph": adjacency_graph,
            "nodes": node_ids,
            "edges": edges,
        }

    def get_hypothesis_clusters(
        self,
        context: ContextMemory,
        similarity_threshold: Optional[float] = None,
        research_goal: Optional[ResearchGoal] = None,
    ) -> Dict[str, int]:
        """Identify clusters of similar hypotheses."""

        threshold = self._resolve_threshold(
            similarity_threshold,
            len(context.get_active_hypotheses()),
        )

        graph = self.build_proximity_graph(
            context,
            similarity_threshold=threshold,
            research_goal=research_goal,
        )

        clusters = self.optimizer.identify_clusters(
            graph["adjacency_graph"],
            threshold,
        )

        logger.info(
            "Identified %d clusters from %d hypotheses.",
            len(set(clusters.values())),
            len(clusters),
        )

        return clusters

    def get_node_connectivity(
        self,
        context: ContextMemory,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, int]:
        """Compute the number of retained similarity edges per node."""

        graph = self.build_proximity_graph(
            context,
            similarity_threshold=similarity_threshold,
        )

        return self.optimizer.compute_node_degree(
            graph["adjacency_graph"]
        )

    def get_proximity_analysis(
        self,
        context: ContextMemory,
        similarity_threshold: Optional[float] = None,
        research_goal: Optional[ResearchGoal] = None,
        near_duplicate_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Produce higher-level proximity information for Meta-review.

        Returns:
            clusters
            connectivity
            largest_clusters
            highly_connected
            isolated
        """
        threshold = self._resolve_threshold(
            similarity_threshold,
            len(context.get_active_hypotheses()),
        )

        graph = self.build_proximity_graph(
            context,
            similarity_threshold=threshold,
            research_goal=research_goal,
        )

        adjacency = graph["adjacency_graph"]

        clusters = self.optimizer.identify_clusters(
            adjacency,
            threshold,
        )

        connectivity = self.optimizer.compute_node_degree(
            adjacency
        )

        cluster_members = {}

        for hypothesis_id, cluster_id in clusters.items():
            cluster_members.setdefault(
                cluster_id,
                [],
            ).append(hypothesis_id)

        largest_clusters = sorted(
            cluster_members.values(),
            key=len,
            reverse=True,
        )

        max_degree = (
            max(connectivity.values())
            if connectivity
            else 0
        )

        highly_connected = [
            hypothesis_id
            for hypothesis_id, degree in connectivity.items()
            if max_degree > 0
            and degree == max_degree
        ]

        isolated = [
            hypothesis_id
            for hypothesis_id, degree in connectivity.items()
            if degree == 0
        ]

        duplicate_threshold = near_duplicate_threshold
        if duplicate_threshold is None:
            duplicate_threshold = max(0.9, threshold)
        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be between 0.0 and 1.0")

        near_duplicates = []
        for source_id, source_edges in adjacency.items():
            for edge in source_edges:
                target_id = edge["other_id"]
                if source_id >= target_id or edge["similarity"] < duplicate_threshold:
                    continue
                source = context.hypotheses[source_id]
                target = context.hypotheses[target_id]
                if not source.is_active or not target.is_active:
                    continue
                survivor, duplicate = sorted(
                    (source, target),
                    key=lambda hypothesis: (
                        getattr(hypothesis, "elo_score", 0.0),
                        hypothesis.hypothesis_id,
                    ),
                    reverse=True,
                )
                duplicate.is_active = False
                duplicate.deactivation_reason = f"near_duplicate_of_{survivor.hypothesis_id}"
                near_duplicates.append(
                    {
                        "duplicate_id": duplicate.hypothesis_id,
                        "canonical_id": survivor.hypothesis_id,
                        "similarity": edge["similarity"],
                        "reason": duplicate.deactivation_reason,
                    }
                )

        cluster_exemplars = {}
        exemplar_ids = []
        for cluster_id, members in cluster_members.items():
            active_members = [member for member in members if context.hypotheses[member].is_active]
            if not active_members:
                continue
            exemplar = max(
                active_members,
                key=lambda hypothesis_id: (
                    getattr(context.hypotheses[hypothesis_id], "elo_score", 0.0),
                    connectivity.get(hypothesis_id, 0),
                    hypothesis_id,
                ),
            )
            cluster_exemplars[cluster_id] = exemplar
            exemplar_ids.append(exemplar)

        pair_scores = [
            edge["similarity"]
            for source_id, source_edges in adjacency.items()
            for edge in source_edges
            if source_id < edge["other_id"]
        ]
        diversity_score = 1.0 - (sum(pair_scores) / len(pair_scores)) if pair_scores else 1.0

        result = {
            "graph": graph,
            "clusters": clusters,
            "cluster_members": cluster_members,
            "largest_clusters": largest_clusters,
            "connectivity": connectivity,
            "highly_connected": highly_connected,
            "isolated": isolated,
            "outliers": isolated,
            "near_duplicates": near_duplicates,
            "exemplar_ids": exemplar_ids,
            "cluster_exemplars": cluster_exemplars,
            "diversity_score": diversity_score,
            "cluster_labels": {
                cluster_id: {"label": f"Cluster {index}"}
                for index, cluster_id in enumerate(cluster_members, start=1)
            },
        }
        context.proximity_analysis = result
        return result

    def clear_similarity_cache(self):
        """Clear similarity and embedding caches."""

        self.scorer.clear_cache()

        logger.debug(
            "Cleared similarity and embedding caches."
        )
