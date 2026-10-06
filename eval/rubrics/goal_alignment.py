"""Deterministic goal matching; semantic/LLM scoring is intentionally absent."""


def normalize_goal(goal: str) -> str:
    """Normalize whitespace and one optional final period before comparison."""
    if not isinstance(goal, str):
        raise TypeError("goal must be a string")
    return " ".join(goal.split()).removesuffix(".")


def goals_match(actual: str, expected: str) -> bool:
    """Return whether goals match after whitespace and terminal-period normalization."""
    return normalize_goal(actual) == normalize_goal(expected)
