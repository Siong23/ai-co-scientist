"""Proximity analysis helpers: similarity scoring, graph optimization, embeddings, and testing utilities."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SimilarityConfig:
    """Configuration for similarity calculations."""

    use_caching: bool = True
    cache_size: int = 256

    # Minimum similarity required for an edge to remain in the graph.
    similarity_threshold: float = 0.3
    dynamic_thresholding: bool = True

    # Similarity methods:
    # jaccard, sequence, semantic, combined
    default_method: str = "combined"

    # Weights used by combined similarity.
    # Semantic similarity receives the highest weight because proximity
    # should primarily capture conceptual similarity rather than word overlap.
    jaccard_weight: float = 0.2
    sequence_weight: float = 0.2
    semantic_weight: float = 0.6

    # None reuses the application's configured embedding provider and model.
    # Set a name explicitly only when a separate local model is intended.
    embedding_model_name: str | None = None

    # Whether semantic similarity should fall back to lexical similarity
    # if sentence-transformers is unavailable.
    allow_embedding_fallback: bool = True


class SimilarityScorer:
    """
    Similarity scorer supporting lexical, sequence-based, semantic,
    and combined similarity.

    Semantic similarity uses sentence embeddings and cosine similarity.
    """

    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()

        self._cache: Dict[str, float] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}

        self._embedding_model = None
        self._embedding_model_loaded = False

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        text1: str,
        text2: str,
        method: str,
    ) -> str:
        """
        Generate an order-independent cache key.

        The similarity method MUST be included because:
            Jaccard(A, B)
            Sequence(A, B)
            Semantic(A, B)

        are different calculations.
        """
        pair = tuple(sorted([text1, text2]))

        raw_key = (
            f"{method}|"
            f"{self.config.embedding_model_name}|"
            f"{self.config.jaccard_weight}|"
            f"{self.config.sequence_weight}|"
            f"{self.config.semantic_weight}|"
            f"{pair[0]}|"
            f"{pair[1]}"
        )

        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def _get_cached_score(
        self,
        text1: str,
        text2: str,
        method: str,
    ) -> Optional[float]:
        if not self.config.use_caching:
            return None

        key = self._cache_key(text1, text2, method)
        return self._cache.get(key)

    def _store_cached_score(
        self,
        text1: str,
        text2: str,
        method: str,
        score: float,
    ) -> None:
        if not self.config.use_caching:
            return

        key = self._cache_key(text1, text2, method)

        if len(self._cache) >= self.config.cache_size:
            # Remove approximately half of the entries.
            half = max(1, len(self._cache) // 2)
            self._cache = dict(list(self._cache.items())[half:])

        self._cache[key] = float(score)

    # ------------------------------------------------------------------
    # Lexical similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Normalize text into simple word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def jaccard_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Calculate token-level Jaccard similarity."""
        if not text1 or not text2:
            return 0.0

        set1 = set(self._tokenize(text1))
        set2 = set(self._tokenize(text2))

        union = set1 | set2

        if not union:
            return 0.0

        intersection = set1 & set2

        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Sequence similarity
    # ------------------------------------------------------------------

    def sequence_matcher_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Calculate textual similarity using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0

        return SequenceMatcher(
            None,
            text1.lower(),
            text2.lower(),
        ).ratio()

    # ------------------------------------------------------------------
    # Semantic similarity
    # ------------------------------------------------------------------

    def _load_embedding_model(self):
        """Load the embedding model lazily."""
        if self._embedding_model_loaded:
            return self._embedding_model

        self._embedding_model_loaded = True

        try:
            if self.config.embedding_model_name is None:
                from ..utils import get_sentence_transformer_model

                self._embedding_model = get_sentence_transformer_model()
            else:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.config.embedding_model_name)

        except Exception as exc:
            self._embedding_model = None

            if not self.config.allow_embedding_fallback:
                raise RuntimeError(
                    f"Unable to load sentence-transformers embedding model '{self.config.embedding_model_name}'."
                ) from exc

        return self._embedding_model

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get and cache an embedding for a single text."""
        if not text:
            return None

        if self.config.use_caching and text in self._embedding_cache:
            return self._embedding_cache[text]

        model = self._load_embedding_model()

        if model is None:
            return None

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        if self.config.use_caching:
            self._embedding_cache[text] = embedding

        return embedding

    def encode_texts(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Encode multiple texts in one batch.

        This is more efficient than calling encode() separately for
        every pair of hypotheses.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._load_embedding_model()

        if model is None:
            raise RuntimeError("Embedding model is unavailable. Install sentence-transformers or enable fallback.")

        missing_texts = [text for text in texts if text and text not in self._embedding_cache]

        if missing_texts:
            embeddings = model.encode(
                missing_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for text, embedding in zip(missing_texts, embeddings):
                self._embedding_cache[text] = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

        result = []

        for text in texts:
            if text in self._embedding_cache:
                result.append(self._embedding_cache[text])
            else:
                # Empty text.
                result.append(np.zeros(384, dtype=np.float32))

        return np.vstack(result)

    def semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings
        and cosine similarity.

        Embeddings are normalized, so their dot product is equivalent
        to cosine similarity.
        """
        if not text1 or not text2:
            return 0.0

        embedding1 = self._get_embedding(text1)
        embedding2 = self._get_embedding(text2)

        if embedding1 is None or embedding2 is None:
            if self.config.allow_embedding_fallback:
                return self.sequence_matcher_similarity(text1, text2)

            return 0.0

        score = float(np.dot(embedding1, embedding2))

        # Numerical safety.
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Combined similarity
    # ------------------------------------------------------------------

    def combined_similarity(
        self,
        text1: str,
        text2: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate weighted similarity from lexical, sequence,
        and semantic signals.
        """
        if not text1 or not text2:
            return 0.0

        if weights is None:
            weights = {
                "jaccard": self.config.jaccard_weight,
                "sequence": self.config.sequence_weight,
                "semantic": self.config.semantic_weight,
            }

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError("Similarity weights must sum to a positive value.")

        normalized_weights = {key: value / total_weight for key, value in weights.items()}

        scores = {
            "jaccard": self.jaccard_similarity(text1, text2),
            "sequence": self.sequence_matcher_similarity(text1, text2),
            "semantic": self.semantic_similarity(text1, text2),
        }

        return float(sum(scores[key] * normalized_weights[key] for key in scores))

    # ------------------------------------------------------------------
    # Public scoring interface
    # ------------------------------------------------------------------

    def score(
        self,
        text1: str,
        text2: str,
        method: str = "combined",
    ) -> float:
        """
        Compute similarity score in [0, 1].

        Supported methods:
            - jaccard
            - sequence
            - semantic
            - combined
        """
        method = method.lower()

        cached = self._get_cached_score(
            text1,
            text2,
            method,
        )

        if cached is not None:
            return cached

        if method == "jaccard":
            result = self.jaccard_similarity(text1, text2)

        elif method == "sequence":
            result = self.sequence_matcher_similarity(text1, text2)

        elif method == "semantic":
            result = self.semantic_similarity(text1, text2)

        elif method == "combined":
            result = self.combined_similarity(text1, text2)

        else:
            raise ValueError(f"Unknown similarity method: {method}. Expected jaccard, sequence, semantic, or combined.")

        self._store_cached_score(
            text1,
            text2,
            method,
            result,
        )

        return float(result)

    def clear_cache(self):
        """Clear all similarity and embedding caches."""
        self._cache.clear()
        self._embedding_cache.clear()


class GraphOptimizer:
    """Graph filtering and post-processing utilities."""

    @staticmethod
    def filter_edges_by_threshold(
        adjacency: Dict[str, List[Dict]],
        threshold: float = 0.3,
    ) -> Dict[str, List[Dict]]:
        """Remove edges below the similarity threshold."""
        filtered = {}

        for node_id, edges in adjacency.items():
            filtered[node_id] = [edge for edge in edges if edge.get("similarity", 0.0) >= threshold]

        return filtered

    @staticmethod
    def get_top_k_edges(
        adjacency: Dict[str, List[Dict]],
        k: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Keep the strongest k edges for each node."""
        if k <= 0:
            return {node_id: [] for node_id in adjacency}

        trimmed = {}

        for node_id, edges in adjacency.items():
            trimmed[node_id] = sorted(
                edges,
                key=lambda x: x.get("similarity", 0.0),
                reverse=True,
            )[:k]

        return trimmed

    @staticmethod
    def compute_node_degree(
        adjacency: Dict[str, List[Dict]],
    ) -> Dict[str, int]:
        """Compute graph degree for each node."""
        return {node_id: len(edges) for node_id, edges in adjacency.items()}

    @staticmethod
    def identify_clusters(
        adjacency: Dict[str, List[Dict]],
        threshold: float = 0.5,
    ) -> Dict[str, int]:
        """
        Identify connected components in the similarity graph.
        """
        clusters: Dict[str, int] = {}
        visited = set()
        cluster_id = 0

        def dfs(node_id: str, cid: int):
            if node_id in visited:
                return

            visited.add(node_id)
            clusters[node_id] = cid

            for edge in adjacency.get(node_id, []):
                if edge.get("similarity", 0.0) >= threshold:
                    neighbor_id = edge["other_id"]

                    if neighbor_id not in visited:
                        dfs(neighbor_id, cid)

        for node_id in adjacency:
            if node_id not in visited:
                dfs(node_id, cluster_id)
                cluster_id += 1

        return clusters


class BatchSimilarityCalculator:
    """
    Batch similarity calculations.

    Semantic embeddings are generated once per hypothesis and then
    compared using matrix multiplication.
    """

    def __init__(
        self,
        scorer: Optional[SimilarityScorer] = None,
    ):
        self.scorer = scorer or SimilarityScorer()

    def compute_similarity_matrix(
        self,
        texts: List[str],
        method: str = "combined",
    ) -> np.ndarray:
        """
        Compute an NxN similarity matrix.

        Semantic similarity is vectorized using normalized embeddings.
        Other methods retain pairwise calculation.
        """
        n = len(texts)

        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        matrix = np.eye(n, dtype=np.float32)

        method = method.lower()

        if method == "semantic":
            embeddings = self.scorer.encode_texts(texts)

            semantic_matrix = np.matmul(
                embeddings,
                embeddings.T,
            )

            matrix = np.clip(
                semantic_matrix,
                0.0,
                1.0,
            ).astype(np.float32)

            np.fill_diagonal(matrix, 1.0)

            return matrix

        if method == "combined":
            # Calculate semantic embeddings once.
            embeddings = self.scorer.encode_texts(texts)

            semantic_matrix = np.clip(
                np.matmul(embeddings, embeddings.T),
                0.0,
                1.0,
            )

            jw = self.scorer.config.jaccard_weight
            sw = self.scorer.config.sequence_weight
            semw = self.scorer.config.semantic_weight

            total_weight = jw + sw + semw

            if total_weight <= 0:
                raise ValueError("Combined similarity weights must sum to a positive value.")

            jw /= total_weight
            sw /= total_weight
            semw /= total_weight

            for i in range(n):
                for j in range(i + 1, n):
                    jaccard = self.scorer.jaccard_similarity(
                        texts[i],
                        texts[j],
                    )

                    sequence = self.scorer.sequence_matcher_similarity(
                        texts[i],
                        texts[j],
                    )

                    combined = jw * jaccard + sw * sequence + semw * float(semantic_matrix[i, j])

                    matrix[i, j] = combined
                    matrix[j, i] = combined

            return matrix

        # Jaccard or SequenceMatcher.
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.scorer.score(
                    texts[i],
                    texts[j],
                    method=method,
                )

                matrix[i, j] = similarity
                matrix[j, i] = similarity

        return matrix

    def build_adjacency_from_matrix(
        self,
        node_ids: List[str],
        similarity_matrix: np.ndarray,
        threshold: float = 0.3,
    ) -> Dict[str, List[Dict]]:
        """Convert a similarity matrix into adjacency data."""
        adjacency = {}

        n = len(node_ids)

        for i in range(n):
            adjacency[node_ids[i]] = []

            for j in range(n):
                if i == j:
                    continue

                similarity = float(similarity_matrix[i, j])

                if similarity >= threshold:
                    adjacency[node_ids[i]].append(
                        {
                            "other_id": node_ids[j],
                            "similarity": similarity,
                        }
                    )

        return adjacency


# ----------------------------------------------------------------------
# Testing utilities
# ----------------------------------------------------------------------


class ProximityTestFactory:
    """Factory for generating deterministic test hypotheses."""

    @staticmethod
    def create_mock_hypothesis(
        hypothesis_id: str,
        text: str,
    ):
        """Create a lightweight mock hypothesis."""

        class MockHypothesis:
            def __init__(self, hid, hypothesis_text):
                self.hypothesis_id = hid
                self.text = hypothesis_text
                self.is_active = True

        return MockHypothesis(
            hypothesis_id,
            text,
        )

    @staticmethod
    def create_test_hypotheses(
        count: int,
    ) -> List:
        """Generate diverse deterministic test hypotheses."""

        templates = [
            "Protein {} interacts with enzyme {} in pathway {}",
            "Gene {} regulates expression of {} through mechanism {}",
            "Mutation in {} causes phenotype {} via interaction with {}",
            "Metabolite {} promotes degradation of {} in system {}",
            "Complex {} binds substrate {} with affinity {}",
        ]

        genes = [
            "BRCA1",
            "TP53",
            "EGFR",
            "MAPK1",
            "STAT3",
            "TLR4",
            "IL6",
            "TNF",
        ]

        processes = [
            "apoptosis",
            "proliferation",
            "differentiation",
            "migration",
            "angiogenesis",
        ]

        hypotheses = []

        for i in range(count):
            template = templates[i % len(templates)]

            text = template.format(
                genes[i % len(genes)],
                genes[(i + 1) % len(genes)],
                processes[i % len(processes)],
            )

            hypotheses.append(
                ProximityTestFactory.create_mock_hypothesis(
                    f"hypo_{i}",
                    text,
                )
            )

        return hypotheses
