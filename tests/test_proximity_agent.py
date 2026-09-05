"""Tests for ProximityAgent hypothesis proximity and topology analysis."""

from unittest.mock import MagicMock, patch

import pytest

from app.agents_modules.semantic_topology import (
    ProximityAgent,
    call_llm_for_cluster_label,
    call_llm_for_near_duplicate_check,
)
from app.models import ContextMemory, Hypothesis


def _make_context(hypotheses_data):
    context = ContextMemory()
    for h_id, title, text, elo in hypotheses_data:
        h = Hypothesis(hypothesis_id=h_id, title=title, text=text, elo_score=elo)
        context.add_hypothesis(h)
    return context


# ---------------------------------------------------------------------------
# Baseline / smoke tests
# ---------------------------------------------------------------------------


def test_proximity_agent_empty_context():
    agent = ProximityAgent(label_clusters=False)
    context = ContextMemory()
    result = agent.build_proximity_graph(context, run_near_duplicate_check=False)

    assert result["adjacency_graph"] == {}
    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["clusters"] == {}
    assert result["cluster_labels"] == {}
    assert result["outliers"] == []
    assert result["exemplars"] == []
    assert result["near_duplicates"] == []
    assert result["diversity_score"] == 0.0
    assert result["mean_similarities"] == {}


def test_proximity_agent_single_hypothesis():
    agent = ProximityAgent(label_clusters=False)
    context = _make_context([("H1", "Quantum Entanglement", "Study on entanglement in 5G.", 1300.0)])
    result = agent.build_proximity_graph(context, run_near_duplicate_check=False)

    assert "H1" in result["adjacency_graph"]
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["id"] == "H1"
    assert result["edges"] == []
    assert result["clusters"] == {0: ["H1"]}
    assert result["exemplars"] == ["H1"]
    assert result["diversity_score"] == 1.0
    assert result["mean_similarities"] == {"H1": 0.0}
    assert result["near_duplicates"] == []


def test_proximity_agent_multi_hypotheses_and_clustering():
    agent = ProximityAgent(
        similarity_threshold=0.2,
        cluster_threshold=0.5,
        outlier_threshold=0.3,
        label_clusters=False,
    )

    context = _make_context(
        [
            ("H1", "Quantum 5G Encryption", "Quantum key distribution for 5G network slices.", 1400.0),
            ("H2", "Quantum 5G Routing", "Quantum cryptographic routing in 5G architectures.", 1350.0),
            ("H3", "Soil Agriculture Crop", "Deep learning for crop soil fertility optimization.", 1200.0),
        ]
    )

    def mock_sim(t1, t2):
        if "5G" in t1 and "5G" in t2:
            return 0.85
        return 0.10

    with patch("app.agents_modules.semantic_topology.similarity_score", side_effect=mock_sim):
        result = agent.build_proximity_graph(context, run_near_duplicate_check=False)

        assert len(result["nodes"]) == 3
        assert set(result["adjacency_graph"].keys()) == {"H1", "H2", "H3"}

        # H1 and H2 should cluster together, H3 in its own cluster
        clusters = result["clusters"]
        assert len(clusters) == 2

        cluster_0 = clusters[0]
        cluster_1 = clusters[1]
        if "H1" in cluster_0:
            assert "H2" in cluster_0
            assert "H3" in cluster_1
        else:
            assert "H3" in cluster_0
            assert "H1" in cluster_1 and "H2" in cluster_1

        # Exemplars: H1 (1400 > 1350) and H3 (1200)
        assert "H1" in result["exemplars"]
        assert "H3" in result["exemplars"]

        # H3 should be an outlier (mean similarity 0.10 < 0.30)
        assert "H3" in result["outliers"]

        # Node enrichment
        node_map = {n["id"]: n for n in result["nodes"]}
        assert node_map["H1"]["value"] > 0
        assert "Elo: 1400.0" in node_map["H1"]["title"]


def test_proximity_agent_caching():
    agent = ProximityAgent(label_clusters=False)
    context = _make_context(
        [
            ("H1", "T1", "Text 1", 1200.0),
            ("H2", "T2", "Text 2", 1200.0),
        ]
    )

    sim_mock = MagicMock(return_value=0.75)
    with patch("app.agents_modules.semantic_topology.similarity_score", sim_mock):
        res1 = agent.build_proximity_graph(context, run_near_duplicate_check=False)
        res2 = agent.build_proximity_graph(context, run_near_duplicate_check=False)

        # sim_mock should only have been called once for pair (H1, H2)
        assert sim_mock.call_count == 1
        assert res1["diversity_score"] == res2["diversity_score"]


def test_proximity_agent_empty_text_handling():
    agent = ProximityAgent(label_clusters=False)
    context = _make_context(
        [
            ("H1", "Title 1", "", 1200.0),
            ("H2", "Title 2", "Valid text content", 1200.0),
        ]
    )

    with patch("app.agents_modules.semantic_topology.similarity_score", return_value=0.5):
        result = agent.build_proximity_graph(context, run_near_duplicate_check=False)
        assert len(result["nodes"]) == 2


# ---------------------------------------------------------------------------
# Near-duplicate detection tests
# ---------------------------------------------------------------------------


def test_near_duplicate_detected():
    """Near-duplicates above the sim threshold get passed to LLM and flagged."""
    agent = ProximityAgent(
        near_dup_sim_threshold=0.70,
        near_dup_llm_confidence=0.75,
        label_clusters=False,
    )
    context = _make_context(
        [
            ("H1", "Drug A in Cancer", "Drug A inhibits kinase X in pancreatic cancer.", 1300.0),
            ("H2", "Drug A in Cancer (v2)", "Drug A inhibits kinase X in pancreatic tumors.", 1250.0),
            ("H3", "Soil Microbiome", "Bacterial diversity improves crop yield.", 1200.0),
        ]
    )

    def mock_sim(t1, t2):
        if "Drug A" in t1 and "Drug A" in t2:
            return 0.92
        return 0.05

    llm_response = '{"near_duplicate": true, "confidence": 0.95, "reasoning": "Same mechanism, different phrasing."}'

    with (
        patch("app.agents_modules.semantic_topology.similarity_score", side_effect=mock_sim),
        patch("app.agents_modules.semantic_topology.call_llm", return_value=llm_response),
    ):
        result = agent.build_proximity_graph(context, run_near_duplicate_check=True)

    assert len(result["near_duplicates"]) == 1
    dup = result["near_duplicates"][0]
    assert {dup["id_a"], dup["id_b"]} == {"H1", "H2"}
    assert dup["confidence"] == pytest.approx(0.95)
    # Near-duplicate warning should appear in node tooltip
    node_map = {n["id"]: n for n in result["nodes"]}
    assert "near-duplicate" in node_map["H1"]["title"] or "near-duplicate" in node_map["H2"]["title"]


def test_near_duplicate_rejected_by_low_confidence():
    """LLM low-confidence result is NOT classified as near-duplicate."""
    agent = ProximityAgent(
        near_dup_sim_threshold=0.70,
        near_dup_llm_confidence=0.75,
        label_clusters=False,
    )
    context = _make_context(
        [
            ("H1", "Enzyme X", "Enzyme X drives pathway Y.", 1300.0),
            ("H2", "Enzyme X (alt)", "Enzyme X modulates pathway Y via Z.", 1250.0),
        ]
    )

    llm_response = '{"near_duplicate": true, "confidence": 0.60, "reasoning": "Partially similar."}'

    with (
        patch("app.agents_modules.semantic_topology.similarity_score", return_value=0.80),
        patch("app.agents_modules.semantic_topology.call_llm", return_value=llm_response),
    ):
        result = agent.build_proximity_graph(context, run_near_duplicate_check=True)

    assert result["near_duplicates"] == []


def test_near_duplicate_check_skipped_when_disabled():
    """When run_near_duplicate_check=False, no LLM calls for dedup are made."""
    agent = ProximityAgent(near_dup_sim_threshold=0.70, label_clusters=False)
    context = _make_context(
        [
            ("H1", "Idea A", "Very similar idea A.", 1300.0),
            ("H2", "Idea A copy", "Very similar idea A.", 1250.0),
        ]
    )
    llm_mock = MagicMock(return_value='{"near_duplicate": true, "confidence": 0.99}')
    with (
        patch("app.agents_modules.semantic_topology.similarity_score", return_value=0.99),
        patch("app.agents_modules.semantic_topology.call_llm", llm_mock),
    ):
        result = agent.build_proximity_graph(context, run_near_duplicate_check=False)

    assert result["near_duplicates"] == []
    llm_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Cluster label tests
# ---------------------------------------------------------------------------


def test_cluster_label_generated():
    """LLM cluster labels are generated for multi-member clusters."""
    agent = ProximityAgent(
        cluster_threshold=0.5,
        label_clusters=True,
        near_dup_sim_threshold=0.99,  # effectively disable near-dup for this test
    )
    context = _make_context(
        [
            ("H1", "Quantum Key", "Quantum cryptography for secure communications.", 1300.0),
            ("H2", "Quantum Comm", "Quantum entanglement for secure key distribution.", 1250.0),
        ]
    )

    label_response = '{"label": "Quantum Cryptography", "explanation": "Both address quantum-secure comms."}'

    with (
        patch("app.agents_modules.semantic_topology.similarity_score", return_value=0.85),
        patch("app.agents_modules.semantic_topology.call_llm", return_value=label_response),
    ):
        result = agent.build_proximity_graph(context, run_near_duplicate_check=False)

    assert 0 in result["cluster_labels"]
    assert result["cluster_labels"][0]["label"] == "Quantum Cryptography"


def test_cluster_label_fallback_on_llm_error():
    """Malformed LLM response triggers generic fallback label."""
    agent = ProximityAgent(
        cluster_threshold=0.5,
        label_clusters=True,
        near_dup_sim_threshold=0.99,
    )
    context = _make_context(
        [
            ("H1", "Topic A", "Research topic A details.", 1300.0),
            ("H2", "Topic A alt", "Research topic A alternative details.", 1250.0),
        ]
    )

    with (
        patch("app.agents_modules.semantic_topology.similarity_score", return_value=0.85),
        patch("app.agents_modules.semantic_topology.call_llm", return_value="not valid json"),
    ):
        result = agent.build_proximity_graph(context, run_near_duplicate_check=False)

    # Fallback label should be returned without crashing
    assert 0 in result["cluster_labels"]
    assert isinstance(result["cluster_labels"][0]["label"], str)


# ---------------------------------------------------------------------------
# call_llm_for_cluster_label unit tests
# ---------------------------------------------------------------------------


def test_call_llm_for_cluster_label_ok():
    response = '{"label": "Gene Therapy", "explanation": "Both relate to gene editing."}'
    with patch("app.agents_modules.semantic_topology.call_llm", return_value=response):
        result = call_llm_for_cluster_label(["Hypo A: CRISPR editing.", "Hypo B: Gene knockdown."])
    assert result["label"] == "Gene Therapy"
    assert "gene editing" in result["explanation"]


def test_call_llm_for_cluster_label_fallback():
    with patch("app.agents_modules.semantic_topology.call_llm", return_value="garbage"):
        result = call_llm_for_cluster_label(["Hypo A: Something.", "Hypo B: Something else."])
    assert "label" in result
    assert isinstance(result["label"], str)


# ---------------------------------------------------------------------------
# call_llm_for_near_duplicate_check unit tests
# ---------------------------------------------------------------------------


def test_call_llm_for_near_duplicate_check_true():
    h_a = Hypothesis(hypothesis_id="A", title="T1", text="Idea about kinase inhibition.")
    h_b = Hypothesis(hypothesis_id="B", title="T2", text="Idea about kinase blocking.")
    response = '{"near_duplicate": true, "confidence": 0.88, "reasoning": "Same mechanism."}'
    with patch("app.agents_modules.semantic_topology.call_llm", return_value=response):
        result = call_llm_for_near_duplicate_check(h_a, h_b)
    assert result["near_duplicate"] is True
    assert result["confidence"] == pytest.approx(0.88)
    assert result["reasoning"] == "Same mechanism."


def test_call_llm_for_near_duplicate_check_false():
    h_a = Hypothesis(hypothesis_id="A", title="T1", text="Kinase inhibition in cancer.")
    h_b = Hypothesis(hypothesis_id="B", title="T2", text="Soil microbiome diversity.")
    response = '{"near_duplicate": false, "confidence": 0.95, "reasoning": "Completely different fields."}'
    with patch("app.agents_modules.semantic_topology.call_llm", return_value=response):
        result = call_llm_for_near_duplicate_check(h_a, h_b)
    assert result["near_duplicate"] is False


def test_call_llm_for_near_duplicate_check_fallback_on_bad_json():
    h_a = Hypothesis(hypothesis_id="A", title="T1", text="Some text.")
    h_b = Hypothesis(hypothesis_id="B", title="T2", text="Other text.")
    with patch("app.agents_modules.semantic_topology.call_llm", return_value="not json at all"):
        result = call_llm_for_near_duplicate_check(h_a, h_b)
    assert result["near_duplicate"] is False
    assert result["confidence"] == pytest.approx(0.0)


def test_call_llm_for_near_duplicate_check_fenced_json():
    h_a = Hypothesis(hypothesis_id="A", title="T1", text="Drug A inhibits kinase X.")
    h_b = Hypothesis(hypothesis_id="B", title="T2", text="Drug A blocks kinase X activity.")
    response = '```json\n{"near_duplicate": true, "confidence": 0.91, "reasoning": "Same drug, same target."}\n```'
    with patch("app.agents_modules.semantic_topology.call_llm", return_value=response):
        result = call_llm_for_near_duplicate_check(h_a, h_b)
    assert result["near_duplicate"] is True
    assert result["confidence"] == pytest.approx(0.91)
