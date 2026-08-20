"""
Evaluation of the Improved Proximity Agent.

This script evaluates:
1. Similarity separation
2. Threshold correctness
3. Diversity / cluster detection
4. Consistency across repeated runs
5. Standard vs optimized performance
6. Pair-count correctness

Run with:

    python -m tests.test_proximity_evaluation
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.agents_modules.proximity import ProximityAgent
from app.agents_modules.proximity_helpers import SimilarityConfig

# ============================================================
# Lightweight evaluation hypothesis/context
# ============================================================

@dataclass
class EvaluationHypothesis:
    hypothesis_id: str
    text: str
    is_active: bool = True


class EvaluationContext:
    """
    Minimal ContextMemory-compatible object used only for evaluation.

    ProximityAgent requires get_active_hypotheses().
    """

    def __init__(self, hypotheses: List[EvaluationHypothesis]):
        self.hypotheses = hypotheses

    def get_active_hypotheses(self):
        return [
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.is_active
        ]


# ============================================================
# Ground-truth evaluation dataset
# ============================================================

def create_evaluation_hypotheses() -> List[EvaluationHypothesis]:
    """
    Create hypotheses with known similarity relationships.

    Group A:
        Cancer / TP53-related hypotheses

    Group B:
        Network / intrusion detection hypotheses

    Group C:
        Agriculture / crop-related hypothesis

    The expected structure is:

        Group A: A1, A2, A3
        Group B: B1, B2
        Group C: C1

    A7 is intentionally unrelated to the others.
    """

    return [
        # ----------------------------------------------------
        # Group A: Cancer / TP53
        # ----------------------------------------------------
        EvaluationHypothesis(
            "A1",
            "TP53 regulates apoptosis in cancer cells "
            "by controlling programmed cell death."
        ),

        EvaluationHypothesis(
            "A2",
            "The TP53 gene controls programmed cell death "
            "and regulates apoptosis in cancer cells."
        ),

        EvaluationHypothesis(
            "A3",
            "Activation of p53 promotes apoptosis "
            "and suppresses tumor cell survival."
        ),

        # ----------------------------------------------------
        # Group B: Network intrusion detection
        # ----------------------------------------------------
        EvaluationHypothesis(
            "B1",
            "Machine learning can detect network intrusions "
            "by analyzing abnormal traffic patterns."
        ),

        EvaluationHypothesis(
            "B2",
            "Network intrusion detection can use machine learning "
            "to identify unusual network traffic behavior."
        ),

        # ----------------------------------------------------
        # Group C: Agriculture
        # ----------------------------------------------------
        EvaluationHypothesis(
            "C1",
            "Crop yield can be improved by optimizing irrigation "
            "and soil nutrient management."
        ),

        # ----------------------------------------------------
        # Intentionally unrelated
        # ----------------------------------------------------
        EvaluationHypothesis(
            "C2",
            "Deep ocean microorganisms can survive under extreme "
            "pressure and low temperature conditions."
        ),
    ]


# ============================================================
# Expected similarity relationships
# ============================================================

SIMILAR_PAIRS = {
    ("A1", "A2"),
    ("A1", "A3"),
    ("A2", "A3"),
    ("B1", "B2"),
}

DIFFERENT_PAIRS = {
    ("A1", "B1"),
    ("A1", "C1"),
    ("A1", "C2"),
    ("B1", "C1"),
    ("B1", "C2"),
    ("C1", "C2"),
}


def normalize_pair(a: str, b: str) -> Tuple[str, str]:
    """Create an order-independent pair."""
    return tuple(sorted((a, b)))


# ============================================================
# Helper functions
# ============================================================

def build_score_dictionary(
    agent: ProximityAgent,
    hypotheses: List[EvaluationHypothesis],
    method: str = "combined",
) -> Dict[Tuple[str, str], float]:
    """
    Calculate similarity for every unique hypothesis pair.
    """

    scores = {}

    for i in range(len(hypotheses)):
        for j in range(i + 1, len(hypotheses)):

            h1 = hypotheses[i]
            h2 = hypotheses[j]

            score = agent.scorer.score(
                h1.text,
                h2.text,
                method=method,
            )

            scores[
                normalize_pair(
                    h1.hypothesis_id,
                    h2.hypothesis_id,
                )
            ] = score

    return scores


# ============================================================
# 1. Similarity separation
# ============================================================

def evaluate_similarity_separation(
    scores: Dict[Tuple[str, str], float],
):
    """
    Determine whether known-similar hypotheses receive higher
    scores than known-different hypotheses.
    """

    similar_scores = [
        scores[pair]
        for pair in SIMILAR_PAIRS
        if pair in scores
    ]

    different_scores = [
        scores[pair]
        for pair in DIFFERENT_PAIRS
        if pair in scores
    ]

    similar_average = (
        statistics.mean(similar_scores)
        if similar_scores
        else 0.0
    )

    different_average = (
        statistics.mean(different_scores)
        if different_scores
        else 0.0
    )

    separation = similar_average - different_average

    print("\n1. SIMILARITY SEPARATION")
    print("-" * 50)

    print(
        f"Similar pairs evaluated:       {len(similar_scores)}"
    )

    print(
        f"Different pairs evaluated:     {len(different_scores)}"
    )

    print(
        f"Average similar-pair score:    {similar_average:.3f}"
    )

    print(
        f"Average different-pair score:  {different_average:.3f}"
    )

    print(
        f"Similarity separation:         {separation:.3f}"
    )

    if separation > 0:
        print("Result: PASS")
    else:
        print("Result: FAIL")

    return {
        "similar_average": similar_average,
        "different_average": different_average,
        "separation": separation,
        "pass": separation > 0,
    }


# ============================================================
# 2. Threshold correctness
# ============================================================

def evaluate_threshold_correctness(
    scores: Dict[Tuple[str, str], float],
    threshold: float,
):
    """
    Evaluate whether the selected threshold separates known
    similar and different hypothesis pairs.
    """

    similar_correct = 0
    similar_total = 0

    different_correct = 0
    different_total = 0

    for pair in SIMILAR_PAIRS:

        if pair not in scores:
            continue

        similar_total += 1

        if scores[pair] >= threshold:
            similar_correct += 1

    for pair in DIFFERENT_PAIRS:

        if pair not in scores:
            continue

        different_total += 1

        if scores[pair] < threshold:
            different_correct += 1

    similar_accuracy = (
        similar_correct / similar_total
        if similar_total
        else 0.0
    )

    different_accuracy = (
        different_correct / different_total
        if different_total
        else 0.0
    )

    overall_total = (
        similar_total + different_total
    )

    overall_correct = (
        similar_correct + different_correct
    )

    overall_accuracy = (
        overall_correct / overall_total
        if overall_total
        else 0.0
    )

    print("\n2. THRESHOLD CORRECTNESS")
    print("-" * 50)

    print(f"Threshold:                    {threshold:.2f}")

    print(
        f"Similar pairs correctly connected: "
        f"{similar_correct}/{similar_total}"
    )

    print(
        f"Different pairs correctly separated: "
        f"{different_correct}/{different_total}"
    )

    print(
        f"Similar-pair accuracy:         "
        f"{similar_accuracy:.3f}"
    )

    print(
        f"Different-pair accuracy:       "
        f"{different_accuracy:.3f}"
    )

    print(
        f"Overall threshold accuracy:    "
        f"{overall_accuracy:.3f}"
    )

    if overall_accuracy >= 0.80:
        print("Result: PASS")
    else:
        print("Result: REVIEW")

    return {
        "accuracy": overall_accuracy,
        "similar_accuracy": similar_accuracy,
        "different_accuracy": different_accuracy,
    }


# ============================================================
# 3. Diversity / cluster detection
# ============================================================

def evaluate_diversity_detection(
    agent: ProximityAgent,
    context: EvaluationContext,
    threshold: float,
):
    """
    Evaluate whether the Proximity Agent separates substantially
    different hypothesis groups.
    """

    clusters = agent.get_hypothesis_clusters(
        context,
        similarity_threshold=threshold,
    )

    cluster_groups = {}

    for hypothesis_id, cluster_id in clusters.items():

        cluster_groups.setdefault(
            cluster_id,
            [],
        ).append(hypothesis_id)

    print("\n3. DIVERSITY / CLUSTER DETECTION")
    print("-" * 50)

    print(
        f"Number of detected clusters: {len(cluster_groups)}"
    )

    for cluster_id, members in sorted(
        cluster_groups.items()
    ):
        print(
            f"Cluster {cluster_id}: "
            f"{', '.join(sorted(members))}"
        )

    # Basic diversity criterion:
    # There should be more than one cluster.
    passed = len(cluster_groups) > 1

    if passed:
        print("Result: PASS")
    else:
        print("Result: REVIEW")

    return {
        "number_of_clusters": len(cluster_groups),
        "clusters": cluster_groups,
        "pass": passed,
    }


# ============================================================
# 4. Consistency
# ============================================================

def evaluate_consistency(
    agent: ProximityAgent,
    hypotheses: List[EvaluationHypothesis],
    runs: int = 5,
):
    """
    Run the same similarity calculations repeatedly and determine
    whether the results remain stable.
    """

    pair = ("A1", "A2")

    h1 = next(
        h for h in hypotheses
        if h.hypothesis_id == pair[0]
    )

    h2 = next(
        h for h in hypotheses
        if h.hypothesis_id == pair[1]
    )

    scores = []

    for _ in range(runs):

        # Clear cache to ensure the calculation is repeated.
        agent.clear_similarity_cache()

        score = agent.scorer.score(
            h1.text,
            h2.text,
            method="combined",
        )

        scores.append(score)

    max_difference = max(scores) - min(scores)

    standard_deviation = (
        statistics.stdev(scores)
        if len(scores) > 1
        else 0.0
    )

    print("\n4. CONSISTENCY")
    print("-" * 50)

    print(f"Runs:                         {runs}")

    print(
        "Scores:                       "
        + ", ".join(
            f"{score:.6f}"
            for score in scores
        )
    )

    print(
        f"Maximum score difference:     "
        f"{max_difference:.8f}"
    )

    print(
        f"Standard deviation:           "
        f"{standard_deviation:.8f}"
    )

    # Small numerical differences are acceptable.
    passed = max_difference <= 0.0001

    if passed:
        print("Result: PASS")
    else:
        print("Result: REVIEW")

    return {
        "scores": scores,
        "max_difference": max_difference,
        "standard_deviation": standard_deviation,
        "pass": passed,
    }


# ============================================================
# 5. Performance evaluation
# ============================================================

def benchmark(
    function,
    repetitions: int = 3,
):
    """
    Measure average execution time.
    """

    times = []

    for _ in range(repetitions):

        start = time.perf_counter()

        function()

        end = time.perf_counter()

        times.append(end - start)

    return statistics.mean(times)


def evaluate_performance(
    agent: ProximityAgent,
    context: EvaluationContext,
    repetitions: int = 3,
):
    """
    Compare standard and optimized proximity graph construction.
    """

    # --------------------------------------------------------
    # Warm-up.
    #
    # This prevents model-loading time from dominating the
    # comparison.
    # --------------------------------------------------------

    agent.clear_similarity_cache()

    agent.build_proximity_graph(
        context,
        method="combined",
        similarity_threshold=0.3,
    )

    agent.clear_similarity_cache()

    agent.build_proximity_graph_optimized(
        context,
        method="combined",
        similarity_threshold=0.3,
    )

    # --------------------------------------------------------
    # Standard implementation
    # --------------------------------------------------------

    def run_standard():

        agent.clear_similarity_cache()

        agent.build_proximity_graph(
            context,
            method="combined",
            similarity_threshold=0.3,
        )

    # --------------------------------------------------------
    # Optimized implementation
    # --------------------------------------------------------

    def run_optimized():

        agent.clear_similarity_cache()

        agent.build_proximity_graph_optimized(
            context,
            method="combined",
            similarity_threshold=0.3,
        )

    standard_time = benchmark(
        run_standard,
        repetitions,
    )

    optimized_time = benchmark(
        run_optimized,
        repetitions,
    )

    if optimized_time > 0:
        speedup = standard_time / optimized_time
    else:
        speedup = 0.0

    improvement = (
        (standard_time - optimized_time)
        / standard_time
        * 100
        if standard_time > 0
        else 0.0
    )

    print("\n5. PERFORMANCE")
    print("-" * 50)

    print(
        f"Standard implementation:      "
        f"{standard_time:.4f} seconds"
    )

    print(
        f"Optimized implementation:     "
        f"{optimized_time:.4f} seconds"
    )

    print(
        f"Speedup:                       "
        f"{speedup:.2f}x"
    )

    print(
        f"Execution-time improvement:    "
        f"{improvement:.2f}%"
    )

    if optimized_time < standard_time:
        print("Result: PASS")
    else:
        print("Result: REVIEW")

    return {
        "standard_time": standard_time,
        "optimized_time": optimized_time,
        "speedup": speedup,
        "improvement_percent": improvement,
    }


# ============================================================
# 6. Pair-count correctness
# ============================================================

def evaluate_pair_count(
    agent: ProximityAgent,
    context: EvaluationContext,
):
    """
    Verify unique pair count and directed graph connection count.
    """

    hypotheses = context.get_active_hypotheses()

    n = len(hypotheses)

    expected_unique_pairs = (
        n * (n - 1) // 2
    )

    graph = agent.build_proximity_graph(
        context,
        method="combined",
        similarity_threshold=0.0,
    )

    directed_connections = len(
        graph["edges"]
    )

    expected_directed_connections = (
        expected_unique_pairs * 2
    )

    print("\n6. PAIR-COUNT CORRECTNESS")
    print("-" * 50)

    print(
        f"Number of hypotheses:          {n}"
    )

    print(
        f"Expected unique pairs:         "
        f"{expected_unique_pairs}"
    )

    print(
        f"Directed graph connections:    "
        f"{directed_connections}"
    )

    print(
        f"Expected directed connections: "
        f"{expected_directed_connections}"
    )

    passed = (
        directed_connections
        == expected_directed_connections
    )

    if passed:
        print("Result: PASS")
    else:
        print("Result: REVIEW")

    return {
        "hypotheses": n,
        "unique_pairs": expected_unique_pairs,
        "directed_connections": directed_connections,
        "expected_directed_connections":
            expected_directed_connections,
        "pass": passed,
    }


# ============================================================
# Detailed similarity table
# ============================================================

def print_similarity_scores(
    scores: Dict[Tuple[str, str], float],
):
    """Print all evaluated pair scores."""

    print("\nPAIRWISE SIMILARITY SCORES")
    print("=" * 60)

    for pair, score in sorted(scores.items()):

        if pair in SIMILAR_PAIRS:
            category = "SIMILAR"

        elif pair in DIFFERENT_PAIRS:
            category = "DIFFERENT"

        else:
            category = "OTHER"

        print(
            f"{pair[0]:>3} ↔ {pair[1]:<3} | "
            f"{score:.3f} | {category}"
        )


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 60)
    print("IMPROVED PROXIMITY AGENT EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = SimilarityConfig(
        use_caching=True,
        cache_size=256,
        similarity_threshold=0.3,
        default_method="combined",
        jaccard_weight=0.2,
        sequence_weight=0.2,
        semantic_weight=0.6,
        embedding_model_name=(
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        allow_embedding_fallback=True,
    )

    agent = ProximityAgent(
        similarity_config=config
    )

    hypotheses = create_evaluation_hypotheses()

    context = EvaluationContext(
        hypotheses
    )

    threshold = 0.3

    # --------------------------------------------------------
    # Calculate pairwise scores
    # --------------------------------------------------------

    scores = build_score_dictionary(
        agent,
        hypotheses,
        method="combined",
    )

    print_similarity_scores(scores)

    # --------------------------------------------------------
    # Run evaluations
    # --------------------------------------------------------

    similarity_result = (
        evaluate_similarity_separation(
            scores
        )
    )

    threshold_result = (
        evaluate_threshold_correctness(
            scores,
            threshold,
        )
    )

    diversity_result = (
        evaluate_diversity_detection(
            agent,
            context,
            threshold,
        )
    )

    consistency_result = (
        evaluate_consistency(
            agent,
            hypotheses,
            runs=5,
        )
    )

    performance_result = (
        evaluate_performance(
            agent,
            context,
            repetitions=3,
        )
    )

    pair_result = (
        evaluate_pair_count(
            agent,
            context,
        )
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Similarity separation: "
        f"{'PASS' if similarity_result['pass'] else 'FAIL'}"
    )

    print(
        f"Threshold correctness: "
        f"{'PASS' if threshold_result['accuracy'] >= 0.80 else 'REVIEW'}"
    )

    print(
        f"Diversity detection: "
        f"{'PASS' if diversity_result['pass'] else 'REVIEW'}"
    )

    print(
        f"Consistency: "
        f"{'PASS' if consistency_result['pass'] else 'REVIEW'}"
    )

    print(
        f"Performance: "
        f"{'PASS' if performance_result['optimized_time'] < performance_result['standard_time'] else 'REVIEW'}"
    )

    print(
        f"Pair count: "
        f"{'PASS' if pair_result['pass'] else 'REVIEW'}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()