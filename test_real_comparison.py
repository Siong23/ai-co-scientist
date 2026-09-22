"""
A Real Test for the Automated Experiment Runner.

This test is meant to be run manually, as it requires a real experiment to be run.
It is not meant to be run as part of the automated test suite.

Run with:
    python test_real_comparison.py
    
"""

import json
from pathlib import Path

from app.experiments.experiment_orchestrator import ExperimentOrchestrator


# ============================================================
# Load existing Rank #1 hypothesis
# ============================================================

config_path = Path(
    "app/experiments/results/runs/"
    "G7417_20260917_130843_533045/" # can change this according to your runs
    "experiment_config.json"
)

with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)

specification = config["specification"]
hypothesis = specification["selected_hypothesis"]
research_goal = specification.get("research_goal")


print("=" * 70)
print("REAL RANK #1 EXPERIMENT TEST")
print("=" * 70)

print("\nRank #1 Hypothesis:")
print(hypothesis.get("title"))

print("\nResearch Goal:")
print(research_goal)

print("\nEvidence Sources:")

for source in hypothesis.get("evidence_sources", []):
    print(
        "-",
        source.get("title"),
        "|",
        source.get("canonical_url")
        or source.get("url")
        or source.get("arxiv_url"),
    )


# ============================================================
# Create orchestrator
# ============================================================

print("\n" + "=" * 70)
print("CREATING EXPERIMENT ORCHESTRATOR")
print("=" * 70)

orchestrator = ExperimentOrchestrator(
    dataset_name="5G-NIDD",
    dataset_path="data/5g_nidd/5g_nidd.csv",
    device="cpu",
)

print("ExperimentOrchestrator created successfully.")
print("Shared PaperLibrary:", type(orchestrator.paper_library).__name__)
print(
    "PaperReader:",
    type(orchestrator.paper_reader).__name__,
)
print(
    "ExperimentComparator:",
    type(orchestrator.experiment_comparator).__name__,
)


# ============================================================
# Run complete experiment pipeline
# ============================================================

print("\n" + "=" * 70)
print("RUNNING COMPLETE EXPERIMENT PIPELINE")
print("=" * 70)

print(
    "\nFlow:"
    "\nRank #1 Hypothesis"
    "\n    -> Evidence Sources"
    "\n    -> PaperReader / PaperLibrary"
    "\n    -> Reference Experiment"
    "\n    -> CodeGenerationAgent"
    "\n    -> ExperimentRunner"
    "\n    -> ExperimentComparator"
    "\n    -> Paper vs Experiment"
)

result = orchestrator.run_experiment(
    context=specification,
    research_goal=research_goal,
    hypothesis=hypothesis,
    execute_generated_code=True,
)


# ============================================================
# Display pipeline result
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE RESULT")
print("=" * 70)

print("Success:", result.get("success"))
print("Status:", result.get("status"))

print("\nErrors:")

errors = result.get("errors", [])

if errors:
    for error in errors:
        print("-", error)
else:
    print("None")


# ============================================================
# Reference experiment extracted from paper
# ============================================================

preparation = result.get("experiment_preparation", {})

reference_experiment = preparation.get(
    "reference_experiment"
)

print("\n" + "=" * 70)
print("REFERENCE EXPERIMENT FROM PAPER")
print("=" * 70)

if reference_experiment:
    print(
        json.dumps(
            reference_experiment,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nReference source count:")
    print(
        reference_experiment.get(
            "source_count"
        )
    )

    for index, source in enumerate(
        reference_experiment.get("sources", []),
        start=1,
    ):
        print(
            f"\nReference Source #{index}:"
        )

        print(
            "Title:",
            source.get("title")
            or source.get("source_title"),
        )

        print(
            "URL:",
            source.get("source_url"),
        )

        print(
            "Source ID:",
            source.get("source_id"),
        )

        print(
            "Indexed:",
            source.get("indexed"),
        )

        experiment_details = source.get(
            "experiment_details"
        )

        if experiment_details:
            print(
                "Experiment details extracted: YES"
            )
        else:
            print(
                "Experiment details extracted: NO"
            )

else:
    print("No reference experiment was extracted.")


# ============================================================
# Automated experiment
# ============================================================

execution = result.get("execution")

print("\n" + "=" * 70)
print("AUTOMATED EXPERIMENT")
print("=" * 70)

if execution:
    print(
        json.dumps(
            execution,
            indent=2,
            ensure_ascii=False,
        )
    )
else:
    print("No execution result.")


# ============================================================
# Paper vs experiment comparison
# ============================================================

comparison = result.get("comparison")

print("\n" + "=" * 70)
print("PAPER VS AUTOMATED EXPERIMENT")
print("=" * 70)

if comparison:
    print(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nComparison status:")
    print(
        comparison.get("status")
    )

    print(
        "Comparable:",
        comparison.get("comparable"),
    )

    print(
        "Comparison success:",
        comparison.get("success"),
    )

    print(
        "Metrics:",
        json.dumps(
            comparison.get("metrics", {}),
            indent=2,
            ensure_ascii=False,
        ),
    )

    if comparison.get("errors"):
        print("\nComparison errors:")

        for error in comparison["errors"]:
            print("-", error)

else:
    print("No comparison result.")


# ============================================================
# Final test summary
# ============================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

print(
    "Pipeline success:",
    result.get("success"),
)

print(
    "Reference experiment available:",
    bool(reference_experiment),
)

print(
    "Automated execution available:",
    bool(execution),
)

print(
    "Comparison available:",
    bool(comparison),
)

if reference_experiment:
    indexed_sources = [
        source
        for source in reference_experiment.get(
            "sources",
            [],
        )
        if source.get("indexed") is True
    ]

    print(
        "Indexed paper sources used:",
        len(indexed_sources),
    )

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)



# import json
# from pathlib import Path

# import torch
# import torch.nn as nn


# # ============================================================
# # Load existing Rank #1 hypothesis
# # ============================================================

# config_path = Path(
#     "app/experiments/results/runs/"
#     "G7417_20260917_130843_533045/"
#     "experiment_config.json"
# )

# with config_path.open("r", encoding="utf-8") as f:
#     config = json.load(f)

# specification = config["specification"]

# hypothesis = specification["selected_hypothesis"]
# research_goal = specification.get("research_goal")


# print("=" * 70)
# print("RANK #1 HYPOTHESIS")
# print("=" * 70)

# print("Title:", hypothesis.get("title"))
# print("Research goal:", research_goal)

# print("\nEvidence Sources:")
# for source in hypothesis.get("evidence_sources", []):
#     print(
#         "-",
#         source.get("title"),
#         "|",
#         source.get("canonical_url")
#         or source.get("url")
#         or source.get("arxiv_url"),
#     )


# # ============================================================
# # PyTorch experiment configuration
# # ============================================================

# DEVICE = torch.device("cpu")

# NUM_TIME_STEPS = 1000
# NUM_SLICES = 3

# BASE_LATENCY_MS = 5.0
# MLKEM_ROTATION_OVERHEAD_MS = 3.5
# LATENCY_THRESHOLD_MS = 10.0

# KEY_ROTATION_INTERVAL = 50

# torch.manual_seed(42)


# # ============================================================
# # Simulate 5G network load
# # ============================================================

# def generate_network_load(
#     num_time_steps: int,
#     num_slices: int,
# ) -> torch.Tensor:
#     """
#     Generate synthetic 5G slice load values.

#     Values are normalized between 0 and 1:
#         0.0 = low load
#         1.0 = high load
#     """

#     load = torch.rand(
#         num_time_steps,
#         num_slices,
#         device=DEVICE,
#     )

#     # Add a changing load pattern over time.
#     time_pattern = torch.sin(
#         torch.linspace(
#             0,
#             8 * torch.pi,
#             num_time_steps,
#             device=DEVICE,
#         )
#     )

#     time_pattern = (
#         time_pattern.unsqueeze(1) + 1.0
#     ) / 2.0

#     load = 0.5 * load + 0.5 * time_pattern

#     return load.clamp(0.0, 1.0)


# # ============================================================
# # Latency model
# # ============================================================

# def calculate_latency(
#     load: torch.Tensor,
#     key_rotation: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Calculate latency for each time step and network slice.

#     Latency consists of:
#         base latency
#         traffic/load latency
#         ML-KEM key-rotation overhead
#     """

#     traffic_latency = load * 8.0

#     rotation_overhead = (
#         key_rotation
#         * MLKEM_ROTATION_OVERHEAD_MS
#     )

#     latency = (
#         BASE_LATENCY_MS
#         + traffic_latency
#         + rotation_overhead
#     )

#     return latency


# # ============================================================
# # Baseline policy
# # ============================================================

# def fixed_period_key_rotation(
#     num_time_steps: int,
#     num_slices: int,
#     interval: int,
# ) -> torch.Tensor:
#     """
#     Baseline policy.

#     Rotate ML-KEM keys at a fixed interval,
#     regardless of current network load.
#     """

#     key_rotation = torch.zeros(
#         num_time_steps,
#         num_slices,
#         device=DEVICE,
#     )

#     for step in range(0, num_time_steps, interval):
#         key_rotation[step, :] = 1.0

#     return key_rotation


# # ============================================================
# # CMDP-inspired policy
# # ============================================================

# def cmdp_scheduled_key_rotation(
#     load: torch.Tensor,
#     load_threshold: float = 0.45,
#     minimum_rotation_interval: int = 50,
# ) -> torch.Tensor:
#     """
#     CMDP-inspired load-aware key-rotation policy.

#     Keys are rotated only when:

#     1. The average network load is below the threshold.
#     2. The minimum interval since the previous rotation
#        has been reached.

#     This is a simplified policy approximation. It is not
#     a complete trained CMDP solver.
#     """

#     num_time_steps, num_slices = load.shape

#     key_rotation = torch.zeros(
#         num_time_steps,
#         num_slices,
#         device=DEVICE,
#     )

#     last_rotation_step = -minimum_rotation_interval

#     for step in range(num_time_steps):
#         average_load = load[step].mean().item()

#         enough_time_passed = (
#             step - last_rotation_step
#             >= minimum_rotation_interval
#         )

#         low_load_window = (
#             average_load <= load_threshold
#         )

#         if enough_time_passed and low_load_window:
#             key_rotation[step, :] = 1.0
#             last_rotation_step = step

#     return key_rotation


# # ============================================================
# # Evaluation metrics
# # ============================================================

# def evaluate_policy(
#     name: str,
#     latency: torch.Tensor,
#     key_rotation: torch.Tensor,
# ) -> dict:
#     """
#     Evaluate latency, QoS violations, and key rotations.
#     """

#     average_latency = latency.mean().item()

#     maximum_latency = latency.max().item()

#     qos_violations = (
#         latency > LATENCY_THRESHOLD_MS
#     ).float().mean().item()

#     total_key_rotations = (
#         key_rotation.sum().item()
#     )

#     return {
#         "policy": name,
#         "average_latency_ms": round(
#             average_latency,
#             4,
#         ),
#         "maximum_latency_ms": round(
#             maximum_latency,
#             4,
#         ),
#         "qos_violation_rate": round(
#             qos_violations,
#             4,
#         ),
#         "total_key_rotations": int(
#             total_key_rotations
#         ),
#     }


# # ============================================================
# # Optional PyTorch model
# # ============================================================

# class LoadAwareRotationModel(nn.Module):
#     """
#     Small PyTorch model representing a load-aware
#     key-rotation decision function.

#     The main experiment uses the explicit CMDP-inspired
#     policy above. This model is included as a PyTorch
#     implementation component for future training.
#     """

#     def __init__(self):
#         super().__init__()

#         self.network = nn.Sequential(
#             nn.Linear(1, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, load):
#         return self.network(load)


# # ============================================================
# # Run experiment
# # ============================================================

# def main():
#     print("\n" + "=" * 70)
#     print("PYTORCH EXPERIMENT")
#     print("=" * 70)

#     print("Device:", DEVICE)
#     print("Time steps:", NUM_TIME_STEPS)
#     print("Network slices:", NUM_SLICES)
#     print(
#         "ML-KEM overhead:",
#         MLKEM_ROTATION_OVERHEAD_MS,
#         "ms",
#     )

#     load = generate_network_load(
#         num_time_steps=NUM_TIME_STEPS,
#         num_slices=NUM_SLICES,
#     )

#     # --------------------------------------------------------
#     # Baseline: fixed-period rotation
#     # --------------------------------------------------------

#     baseline_rotation = fixed_period_key_rotation(
#         num_time_steps=NUM_TIME_STEPS,
#         num_slices=NUM_SLICES,
#         interval=KEY_ROTATION_INTERVAL,
#     )

#     baseline_latency = calculate_latency(
#         load=load,
#         key_rotation=baseline_rotation,
#     )

#     baseline_result = evaluate_policy(
#         name="Fixed-period key rotation",
#         latency=baseline_latency,
#         key_rotation=baseline_rotation,
#     )

#     # --------------------------------------------------------
#     # Proposed: CMDP-inspired low-load scheduling
#     # --------------------------------------------------------

#     proposed_rotation = cmdp_scheduled_key_rotation(
#         load=load,
#         load_threshold=0.45,
#     )

#     proposed_latency = calculate_latency(
#         load=load,
#         key_rotation=proposed_rotation,
#     )

#     proposed_result = evaluate_policy(
#         name="CMDP-inspired low-load scheduling",
#         latency=proposed_latency,
#         key_rotation=proposed_rotation,
#     )

#     # --------------------------------------------------------
#     # Compare results
#     # --------------------------------------------------------

#     average_latency_difference = (
#         proposed_result["average_latency_ms"]
#         - baseline_result["average_latency_ms"]
#     )

#     qos_violation_difference = (
#         proposed_result["qos_violation_rate"]
#         - baseline_result["qos_violation_rate"]
#     )

#     key_rotation_difference = (
#         proposed_result["total_key_rotations"]
#         - baseline_result["total_key_rotations"]
#     )

#     comparison = {
#         "hypothesis": hypothesis.get("title"),
#         "research_goal": research_goal,
#         "experiment_type": (
#             "Synthetic PyTorch simulation of "
#             "load-aware ML-KEM key rotation"
#         ),
#         "baseline": baseline_result,
#         "proposed_method": proposed_result,
#         "comparison": {
#             "average_latency_difference_ms": round(
#                 average_latency_difference,
#                 4,
#             ),
#             "qos_violation_rate_difference": round(
#                 qos_violation_difference,
#                 4,
#             ),
#             "key_rotation_difference": int(
#                 key_rotation_difference
#             ),
#             "average_latency_result": (
#                 "improved"
#                 if average_latency_difference < 0
#                 else "worse"
#                 if average_latency_difference > 0
#                 else "unchanged"
#             ),
#             "qos_result": (
#                 "improved"
#                 if qos_violation_difference < 0
#                 else "worse"
#                 if qos_violation_difference > 0
#                 else "unchanged"
#             ),
#             "key_rotation_result": (
#                 "reduced"
#                 if key_rotation_difference < 0
#                 else "increased"
#                 if key_rotation_difference > 0
#                 else "unchanged"
#             ),
#         },
#         "limitations": [
#             "Synthetic network-load data was used.",
#             "The CMDP scheduler is represented by a "
#             "low-load threshold policy.",
#             "The experiment does not reproduce the "
#             "paper's complete protocol or hardware setup.",
#             "The result is an adaptation of the hypothesis "
#             "rather than a direct paper reproduction.",
#         ],
#     }

#     print("\n" + "=" * 70)
#     print("BASELINE RESULT")
#     print("=" * 70)

#     print(json.dumps(
#         baseline_result,
#         indent=2,
#     ))

#     print("\n" + "=" * 70)
#     print("PROPOSED METHOD RESULT")
#     print("=" * 70)

#     print(json.dumps(
#         proposed_result,
#         indent=2,
#     ))

#     print("\n" + "=" * 70)
#     print("COMPARISON RESULT")
#     print("=" * 70)

#     print(json.dumps(
#         comparison,
#         indent=2,
#     ))

#     # --------------------------------------------------------
#     # Save experiment result
#     # --------------------------------------------------------

#     output_path = Path(
#         "app/experiments/results/"
#         "manual_pytorch_comparison.json"
#     )

#     output_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     with output_path.open(
#         "w",
#         encoding="utf-8",
#     ) as f:
#         json.dump(
#             comparison,
#             f,
#             indent=2,
#             ensure_ascii=False,
#         )

#     print("\nSaved result to:")
#     print(output_path)

#     print("\n" + "=" * 70)
#     print("TEST COMPLETE")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()