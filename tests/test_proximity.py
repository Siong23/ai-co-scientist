"""
Run with:
    pytest tests/test_proximity.py -v -s
"""
import numpy as np
import pytest

from app.agents_modules.proximity import ProximityAgent
from app.agents_modules.proximity_helpers import (
    BatchSimilarityCalculator,
    GraphOptimizer,
    ProximityTestFactory,
    SimilarityConfig,
    SimilarityScorer,
)


@pytest.fixture
def context():
    class MockContext:

        def __init__(self, hypotheses):
            self.hypotheses = {
                h.hypothesis_id: h
                for h in hypotheses
            }

        def get_active_hypotheses(self):
            return [
                h
                for h in self.hypotheses.values()
                if h.is_active
            ]

    hypotheses = [
        ProximityTestFactory.create_mock_hypothesis(
            "H1",
            "Gene A regulates protein B expression",
        ),
        ProximityTestFactory.create_mock_hypothesis(
            "H2",
            "Gene A controls protein B expression",
        ),
        ProximityTestFactory.create_mock_hypothesis(
            "H3",
            "Weather prediction using machine learning",
        ),
    ]

    return MockContext(hypotheses)


# ---------------------------------------------------------------------------
# SimilarityScorer tests
# ---------------------------------------------------------------------------

class TestSimilarityScorer:

    def test_jaccard_similarity_identical_text(self):
        scorer = SimilarityScorer()

        text = "gene regulates protein expression"

        score = scorer.jaccard_similarity(text, text)

        assert score == pytest.approx(1.0)

    def test_jaccard_similarity_different_text(self):
        scorer = SimilarityScorer()

        score = scorer.jaccard_similarity(
            "apple banana orange",
            "computer science mathematics",
        )

        assert score == pytest.approx(0.0)

    def test_sequence_similarity_identical_text(self):
        scorer = SimilarityScorer()

        text = "gene expression regulation"

        score = scorer.sequence_matcher_similarity(text, text)

        assert score == pytest.approx(1.0)

    def test_similarity_empty_text(self):
        scorer = SimilarityScorer()

        assert scorer.jaccard_similarity("", "test") == 0.0
        assert scorer.sequence_matcher_similarity("", "test") == 0.0
        assert scorer.semantic_similarity("", "test") == 0.0

    def test_cache_key_contains_method(self):
        scorer = SimilarityScorer()

        key_jaccard = scorer._cache_key(
            "text A",
            "text B",
            "jaccard",
        )

        key_sequence = scorer._cache_key(
            "text A",
            "text B",
            "sequence",
        )

        assert key_jaccard != key_sequence

    def test_cache_is_order_independent(self):
        scorer = SimilarityScorer()

        key_ab = scorer._cache_key(
            "text A",
            "text B",
            "semantic",
        )

        key_ba = scorer._cache_key(
            "text B",
            "text A",
            "semantic",
        )

        assert key_ab == key_ba

    def test_score_invalid_method(self):
        scorer = SimilarityScorer()

        with pytest.raises(ValueError):
            scorer.score(
                "text A",
                "text B",
                method="invalid",
            )

    def test_combined_similarity_range(self):
        scorer = SimilarityScorer(
            SimilarityConfig(
                allow_embedding_fallback=True
            )
        )

        score = scorer.combined_similarity(
            "gene regulates protein expression",
            "gene controls protein expression",
        )

        assert 0.0 <= score <= 1.0

    def test_clear_cache(self):
        scorer = SimilarityScorer()

        scorer._cache["test"] = 0.5
        scorer._embedding_cache["text"] = np.array([1.0, 0.0])

        scorer.clear_cache()

        assert scorer._cache == {}
        assert scorer._embedding_cache == {}


# ---------------------------------------------------------------------------
# GraphOptimizer tests
# ---------------------------------------------------------------------------

class TestGraphOptimizer:

    @pytest.fixture
    def sample_graph(self):
        return {
            "H1": [
                {"other_id": "H2", "similarity": 0.9},
                {"other_id": "H3", "similarity": 0.4},
                {"other_id": "H4", "similarity": 0.1},
            ],
            "H2": [
                {"other_id": "H1", "similarity": 0.9},
            ],
            "H3": [
                {"other_id": "H1", "similarity": 0.4},
            ],
            "H4": [],
        }

    def test_threshold_filter(self, sample_graph):
        optimizer = GraphOptimizer()

        filtered = optimizer.filter_edges_by_threshold(
            sample_graph,
            threshold=0.3,
        )

        assert len(filtered["H1"]) == 2

        similarities = [
            edge["similarity"]
            for edge in filtered["H1"]
        ]

        assert 0.9 in similarities
        assert 0.4 in similarities
        assert 0.1 not in similarities

    def test_top_k_edges(self, sample_graph):
        optimizer = GraphOptimizer()

        result = optimizer.get_top_k_edges(
            sample_graph,
            k=1,
        )

        assert len(result["H1"]) == 1
        assert result["H1"][0]["similarity"] == 0.9

    def test_top_k_zero(self, sample_graph):
        optimizer = GraphOptimizer()

        result = optimizer.get_top_k_edges(
            sample_graph,
            k=0,
        )

        for edges in result.values():
            assert edges == []

    def test_node_degree(self, sample_graph):
        optimizer = GraphOptimizer()

        degree = optimizer.compute_node_degree(
            sample_graph
        )

        assert degree["H1"] == 3
        assert degree["H2"] == 1
        assert degree["H4"] == 0

    def test_identify_clusters(self):
        optimizer = GraphOptimizer()

        graph = {
            "H1": [
                {"other_id": "H2", "similarity": 0.9}
            ],
            "H2": [
                {"other_id": "H1", "similarity": 0.9}
            ],
            "H3": [],
            "H4": [],
        }

        clusters = optimizer.identify_clusters(
            graph,
            threshold=0.5,
        )

        assert clusters["H1"] == clusters["H2"]
        assert clusters["H1"] != clusters["H3"]
        assert clusters["H3"] != clusters["H4"]

    def test_build_proximity_graph_optimized(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph_optimized(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        assert "adjacency_graph" in result
        assert "nodes" in result
        assert "edges" in result

        assert len(result["adjacency_graph"]) == 3

    def test_default_method_from_config(self, context):
        config = SimilarityConfig(
            default_method="jaccard"
        )

        agent = ProximityAgent(config)

        result = agent.build_proximity_graph(
            context,
            similarity_threshold=0.0,
        )

        assert "adjacency_graph" in result

    def test_optimized_graph(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph_optimized(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        assert "adjacency_graph" in result
        assert "nodes" in result
        assert "edges" in result

        assert len(result["adjacency_graph"]) == 3

    def test_optimized_graph_is_symmetric(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph_optimized(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        adjacency = result["adjacency_graph"]

        for node_id, edges in adjacency.items():
            for edge in edges:
                other_id = edge["other_id"]

                reverse_edges = [
                    reverse_edge
                    for reverse_edge in adjacency[other_id]
                    if reverse_edge["other_id"] == node_id
                ]

                assert len(reverse_edges) == 1

                assert reverse_edges[0]["similarity"] == pytest.approx(
                    edge["similarity"]
                )

    def test_optimized_graph_top_k(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph_optimized(
            context,
            method="jaccard",
            similarity_threshold=0.0,
            top_k=1,
        )

        for edges in result["adjacency_graph"].values():
            assert len(edges) <= 1

    def test_optimized_graph_matches_standard_graph(self, context):
        agent = ProximityAgent()

        standard = agent.build_proximity_graph(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        optimized = agent.build_proximity_graph_optimized(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        standard_graph = standard["adjacency_graph"]
        optimized_graph = optimized["adjacency_graph"]

        assert set(standard_graph.keys()) == set(
            optimized_graph.keys()
        )

        for node_id in standard_graph:
            standard_edges = {
                edge["other_id"]: edge["similarity"]
                for edge in standard_graph[node_id]
            }

            optimized_edges = {
                edge["other_id"]: edge["similarity"]
                for edge in optimized_graph[node_id]
            }

            assert standard_edges.keys() == optimized_edges.keys()

            for other_id in standard_edges:
                assert optimized_edges[other_id] == pytest.approx(
                    standard_edges[other_id]
                )


# ---------------------------------------------------------------------------
# BatchSimilarityCalculator tests
# ---------------------------------------------------------------------------

class TestBatchSimilarityCalculator:

    def test_empty_similarity_matrix(self):
        scorer = SimilarityScorer()

        calculator = BatchSimilarityCalculator(scorer)

        matrix = calculator.compute_similarity_matrix(
            [],
            method="jaccard",
        )

        assert matrix.shape == (0, 0)

    def test_similarity_matrix_shape(self):
        scorer = SimilarityScorer()

        calculator = BatchSimilarityCalculator(scorer)

        texts = [
            "gene regulates protein",
            "protein controls gene",
            "weather prediction model",
        ]

        matrix = calculator.compute_similarity_matrix(
            texts,
            method="jaccard",
        )

        assert matrix.shape == (3, 3)

    def test_similarity_matrix_is_symmetric(self):
        scorer = SimilarityScorer()

        calculator = BatchSimilarityCalculator(scorer)

        texts = [
            "gene regulates protein",
            "protein controls gene",
            "weather prediction model",
        ]

        matrix = calculator.compute_similarity_matrix(
            texts,
            method="jaccard",
        )

        assert np.allclose(
            matrix,
            matrix.T,
        )

    def test_similarity_matrix_diagonal(self):
        scorer = SimilarityScorer()

        calculator = BatchSimilarityCalculator(scorer)

        texts = [
            "gene regulates protein",
            "protein controls gene",
        ]

        matrix = calculator.compute_similarity_matrix(
            texts,
            method="jaccard",
        )

        assert np.allclose(
            np.diag(matrix),
            1.0,
        )

    def test_build_adjacency_from_matrix(self):
        scorer = SimilarityScorer()

        calculator = BatchSimilarityCalculator(scorer)

        node_ids = ["H1", "H2", "H3"]

        matrix = np.array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        )

        adjacency = calculator.build_adjacency_from_matrix(
            node_ids,
            matrix,
            threshold=0.5,
        )

        assert len(adjacency["H1"]) == 1
        assert adjacency["H1"][0]["other_id"] == "H2"

        assert len(adjacency["H2"]) == 1
        assert adjacency["H2"][0]["other_id"] == "H1"

        assert adjacency["H3"] == []

    def test_invalid_top_k(self, context):
        agent = ProximityAgent()

        with pytest.raises(ValueError):
            agent.build_proximity_graph(
                context,
                method="jaccard",
                similarity_threshold=0.0,
                top_k=-1,
            )


# ---------------------------------------------------------------------------
# ProximityTestFactory tests
# ---------------------------------------------------------------------------

class TestProximityTestFactory:

    def test_create_mock_hypothesis(self):
        hypothesis = (
            ProximityTestFactory.create_mock_hypothesis(
                "H1",
                "Test hypothesis",
            )
        )

        assert hypothesis.hypothesis_id == "H1"
        assert hypothesis.text == "Test hypothesis"
        assert hypothesis.is_active is True

    def test_create_test_hypotheses_count(self):
        hypotheses = (
            ProximityTestFactory.create_test_hypotheses(10)
        )

        assert len(hypotheses) == 10

    def test_create_test_hypotheses_unique_ids(self):
        hypotheses = (
            ProximityTestFactory.create_test_hypotheses(10)
        )

        ids = [
            h.hypothesis_id
            for h in hypotheses
        ]

        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# ProximityAgent tests
# ---------------------------------------------------------------------------

class TestProximityAgent:

    # @pytest.fixture
    # def context(self):
    #     class MockContext:

    #         def __init__(self, hypotheses):
    #             self.hypotheses = {
    #                 h.hypothesis_id: h
    #                 for h in hypotheses
    #             }

    #         def get_active_hypotheses(self):
    #             return [
    #                 h
    #                 for h in self.hypotheses.values()
    #                 if h.is_active
    #             ]

    #     hypotheses = [
    #         ProximityTestFactory.create_mock_hypothesis(
    #             "H1",
    #             "Gene A regulates protein B expression",
    #         ),
    #         ProximityTestFactory.create_mock_hypothesis(
    #             "H2",
    #             "Gene A controls protein B expression",
    #         ),
    #         ProximityTestFactory.create_mock_hypothesis(
    #             "H3",
    #             "Weather prediction using machine learning",
    #         ),
    #     ]

    #     return MockContext(hypotheses)

    def test_empty_context(self):
        class EmptyContext:

            def get_active_hypotheses(self):
                return []

        agent = ProximityAgent()

        result = agent.build_proximity_graph(
            EmptyContext()
        )

        assert result["adjacency_graph"] == {}
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_build_proximity_graph(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        assert "adjacency_graph" in result
        assert "nodes" in result
        assert "edges" in result

        assert len(result["adjacency_graph"]) == 3

    def test_graph_is_symmetric(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        adjacency = result["adjacency_graph"]

        for node_id, edges in adjacency.items():

            for edge in edges:

                other_id = edge["other_id"]

                reverse_edges = [
                    reverse_edge
                    for reverse_edge in adjacency[other_id]
                    if reverse_edge["other_id"] == node_id
                ]

                assert len(reverse_edges) == 1

                assert reverse_edges[0]["similarity"] == pytest.approx(
                    edge["similarity"]
                )

    def test_threshold_removes_weak_edges(self, context):
        agent = ProximityAgent()

        low_threshold = agent.build_proximity_graph(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        high_threshold = agent.build_proximity_graph(
            context,
            method="jaccard",
            similarity_threshold=0.8,
        )

        low_edges = sum(
            len(edges)
            for edges in low_threshold["adjacency_graph"].values()
        )

        high_edges = sum(
            len(edges)
            for edges in high_threshold["adjacency_graph"].values()
        )

        assert high_edges <= low_edges

    def test_zero_threshold_is_respected(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph(
            context,
            method="jaccard",
            similarity_threshold=0.0,
        )

        assert len(result["adjacency_graph"]) == 3

    def test_invalid_threshold(self, context):
        agent = ProximityAgent()

        with pytest.raises(ValueError):

            agent.build_proximity_graph(
                context,
                similarity_threshold=1.5,
            )

    def test_top_k(self, context):
        agent = ProximityAgent()

        result = agent.build_proximity_graph(
            context,
            method="jaccard",
            similarity_threshold=0.0,
            top_k=1,
        )

        for edges in result["adjacency_graph"].values():
            assert len(edges) <= 1

    def test_node_connectivity(self, context):
        agent = ProximityAgent()

        connectivity = agent.get_node_connectivity(
            context,
            similarity_threshold=0.0,
        )

        assert set(connectivity.keys()) == {
            "H1",
            "H2",
            "H3",
        }

        assert all(
            isinstance(value, int)
            for value in connectivity.values()
        )

    def test_get_hypothesis_clusters(self, context):
        agent = ProximityAgent()

        clusters = agent.get_hypothesis_clusters(
            context,
            similarity_threshold=0.0,
        )

        assert set(clusters.keys()) == {
            "H1",
            "H2",
            "H3",
        }

    def test_proximity_analysis(self, context):
        agent = ProximityAgent()

        result = agent.get_proximity_analysis(
            context,
            similarity_threshold=0.0,
        )

        assert "graph" in result
        assert "clusters" in result
        assert "cluster_members" in result
        assert "largest_clusters" in result
        assert "connectivity" in result
        assert "highly_connected" in result
        assert "isolated" in result

    def test_proximity_analysis_consistency(self, context):
        agent = ProximityAgent()

        result = agent.get_proximity_analysis(
            context,
            similarity_threshold=0.0,
        )

        clusters = result["clusters"]
        cluster_members = result["cluster_members"]

        for hypothesis_id, cluster_id in clusters.items():

            assert hypothesis_id in cluster_members[cluster_id]

    def test_clear_similarity_cache(self):
        agent = ProximityAgent()

        agent.scorer._cache["test"] = 0.5
        agent.scorer._embedding_cache["test"] = np.array(
            [1.0, 0.0]
        )

        agent.clear_similarity_cache()

        assert agent.scorer._cache == {}
        assert agent.scorer._embedding_cache == {}

    def test_semantic_similarity_uses_embeddings(self, monkeypatch):
        scorer = SimilarityScorer()

        embedding_a = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

        embedding_b = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

        def fake_get_embedding(text):
            if text == "A":
                return embedding_a
            return embedding_b

        monkeypatch.setattr(
            scorer,
            "_get_embedding",
            fake_get_embedding,
        )

        score = scorer.semantic_similarity(
            "A",
            "B",
        )

        assert score == pytest.approx(1.0)

    def test_cache_does_not_mix_similarity_methods(self):
        scorer = SimilarityScorer()

        text1 = "gene regulates protein"
        text2 = "gene controls protein"

        jaccard_score = scorer.score(
            text1,
            text2,
            method="jaccard",
        )

        sequence_score = scorer.score(
            text1,
            text2,
            method="sequence",
        )

        assert jaccard_score != sequence_score

    def test_cache_returns_correct_method_specific_result(self):
        scorer = SimilarityScorer()

        text1 = "gene regulates protein"
        text2 = "gene controls protein"

        jaccard_first = scorer.score(
            text1,
            text2,
            method="jaccard",
        )

        sequence_first = scorer.score(
            text1,
            text2,
            method="sequence",
        )

        jaccard_second = scorer.score(
            text1,
            text2,
            method="jaccard",
        )

        sequence_second = scorer.score(
            text1,
            text2,
            method="sequence",
        )

        assert jaccard_first == jaccard_second
        assert sequence_first == sequence_second

    def test_inactive_hypotheses_are_excluded(self):
        active = ProximityTestFactory.create_mock_hypothesis(
            "H1",
            "Gene regulates protein expression",
        )

        inactive = ProximityTestFactory.create_mock_hypothesis(
            "H2",
            "Gene controls protein expression",
        )

        inactive.is_active = False

        class MockContext:
            def get_active_hypotheses(self):
                return [active]

        agent = ProximityAgent()

        result = agent.build_proximity_graph(
            MockContext(),
            method="jaccard",
            similarity_threshold=0.0,
        )

        assert "H1" in result["adjacency_graph"]
        assert "H2" not in result["adjacency_graph"]