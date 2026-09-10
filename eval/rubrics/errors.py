"""Errors shared by the LLM-backed rubric modules."""

from __future__ import annotations


class LLMEvaluationError(RuntimeError):
    """Raised when an LLM-backed evaluation cannot be completed."""


__all__ = ["LLMEvaluationError"]
