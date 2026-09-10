"""Research-mode normalization and hypothesis-pipeline policy."""

from __future__ import annotations

from typing import Literal

ResearchType = Literal[
    "hypothesis_testing",
    "causal",
    "comparative",
    "exploratory",
    "literature_review",
    "due_diligence",
]

RESEARCH_TYPES: tuple[ResearchType, ...] = (
    "hypothesis_testing",
    "causal",
    "comparative",
    "exploratory",
    "literature_review",
    "due_diligence",
)

HYPOTHESIS_REQUIRED_RESEARCH_TYPES: frozenset[str] = frozenset(RESEARCH_TYPES)
HYPOTHESIS_OPTIONAL_RESEARCH_TYPES: frozenset[str] = frozenset()


def normalize_research_type(
    value: object,
    *,
    default: ResearchType = "hypothesis_testing",
) -> ResearchType:
    """Return one supported mode without over-interpreting arbitrary labels."""

    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "hypothesis": "hypothesis_testing",
        "hypothesis_test": "hypothesis_testing",
        "hypothesis_testing": "hypothesis_testing",
        "causal": "causal",
        "causal_analysis": "causal",
        "comparative": "comparative",
        "comparison": "comparative",
        "exploratory": "exploratory",
        "exploration": "exploratory",
        "literature_review": "literature_review",
        "systematic_review": "literature_review",
        "due_diligence": "due_diligence",
        "verification": "due_diligence",
    }
    return aliases.get(normalized, default)  # type: ignore[return-value]


def research_type_requires_hypotheses(research_type: object) -> bool:
    """Return whether the mode intrinsically requires hypothesis scaffolds."""

    return normalize_research_type(research_type) in HYPOTHESIS_REQUIRED_RESEARCH_TYPES


def research_type_allows_hypotheses(research_type: object) -> bool:
    """Return whether hypotheses are a meaningful output for the mode."""

    normalized = normalize_research_type(research_type)
    return normalized in HYPOTHESIS_REQUIRED_RESEARCH_TYPES | HYPOTHESIS_OPTIONAL_RESEARCH_TYPES
