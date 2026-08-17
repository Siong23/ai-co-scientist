"""Hypothesis proximity analysis agent."""

from __future__ import annotations

from typing import Dict, Optional

from ..models import ContextMemory
from ._compat import _legacy
from .proximity_helpers import (
    SimilarityScorer,
    SimilarityConfig,
    GraphOptimizer,
    BatchSimilarityCalculator,
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
    ) -> float:
        """
        Resolve the similarity threshold.

        Explicitly checks for None so that 0.0 remains a valid threshold.
        """
        if similarity_threshold is None:
            return self.similarity_config.similarity_threshold

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
        similarity_threshold: float = 0.3,
        top_k: Optional[int] = None,
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
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0"
            )

        if top_k is not None and top_k < 0:
            raise ValueError(
                "top_k must be non-negative"
            )

        # ---------------------------------------------------------
        # Get active hypotheses
        # ---------------------------------------------------------
        hypotheses = context.get_active_hypotheses()

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
                    hypothesis_a.text,
                    hypothesis_b.text,
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

    def build_proximity_graph_optimized(
        self,
        context: ContextMemory,
        method: Optional[str] = None,
        similarity_threshold: float = 0.3,
        top_k: Optional[int] = None,
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
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0"
            )

        if top_k is not None and top_k < 0:
            raise ValueError(
                "top_k must be non-negative"
            )

        # ---------------------------------------------------------
        # Get active hypotheses only
        # ---------------------------------------------------------
        hypotheses = context.get_active_hypotheses()

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
            hypothesis.text
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
    ) -> Dict[str, int]:
        """Identify clusters of similar hypotheses."""

        threshold = self._resolve_threshold(
            similarity_threshold
        )

        graph = self.build_proximity_graph(
            context,
            similarity_threshold=threshold,
        )

        clusters = self.optimizer.identify_clusters(
            graph["adjacency_graph"],
            threshold,
        )

        _legacy.logger.info(
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
            similarity_threshold
        )

        graph = self.build_proximity_graph(
            context,
            similarity_threshold=threshold,
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

        return {
            "graph": graph,
            "clusters": clusters,
            "cluster_members": cluster_members,
            "largest_clusters": largest_clusters,
            "connectivity": connectivity,
            "highly_connected": highly_connected,
            "isolated": isolated,
        }

    def clear_similarity_cache(self):
        """Clear similarity and embedding caches."""

        self.scorer.clear_cache()

        _legacy.logger.debug(
            "Cleared similarity and embedding caches."
        )