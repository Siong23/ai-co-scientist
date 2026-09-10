"""Metric suite names.

Kept in its own module so the CLI can validate ``--metric-suite`` without
importing DeepEval: deterministic parsing must stay independent of the judge.
"""

from __future__ import annotations

#: Scores the hypothesis itself against the research goal.
SUITE_HYPOTHESIS = "hypothesis"

#: Scores the hypothesis against the evidence retrieved for it.
SUITE_RAG = "rag"

#: The superseded Evidence support metric, kept out of the default suites.
SUITE_LEGACY = "legacy"

#: Every suite except ``legacy``.
SUITE_ALL = "all"

METRIC_SUITES = (SUITE_HYPOTHESIS, SUITE_RAG, SUITE_LEGACY, SUITE_ALL)

DEFAULT_METRIC_SUITE = SUITE_ALL

__all__ = [
    "DEFAULT_METRIC_SUITE",
    "METRIC_SUITES",
    "SUITE_ALL",
    "SUITE_HYPOTHESIS",
    "SUITE_LEGACY",
    "SUITE_RAG",
]
