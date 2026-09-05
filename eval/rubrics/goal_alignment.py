"""Deterministic goal matching; semantic/LLM scoring is intentionally absent."""


def normalize_goal(goal: str) -> str:
    """Normalize inconsequential whitespace before exact goal comparison."""
    if not isinstance(goal, str):
        raise TypeError("goal must be a string")
    return " ".join(goal.split())


def goals_match(actual: str, expected: str) -> bool:
    """Return whether two research goals match after whitespace normalization."""
    return normalize_goal(actual) == normalize_goal(expected)

