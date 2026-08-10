"""Evolution-agent hypothesis-combination helpers."""

from __future__ import annotations

from ..models import Hypothesis
from ..utils import generate_unique_id, logger


def combine_hypotheses(hypoA: Hypothesis, hypoB: Hypothesis) -> Hypothesis:
    """Combines two hypotheses into a new one."""
    new_id = generate_unique_id("E")  # Use utility function
    combined_title = f"Combined: {hypoA.title} & {hypoB.title}"
    # Keep the combined text plain and structured so downstream code can process it safely.
    combined_text = f"Combination of:<br>1. {hypoA.text}<br>2. {hypoB.text}"

    logger.info("Combining hypotheses %s and %s into %s", hypoA.hypothesis_id, hypoB.hypothesis_id, new_id)
    new_hypothesis = Hypothesis(new_id, combined_title, combined_text)
    new_hypothesis.parent_ids = [hypoA.hypothesis_id, hypoB.hypothesis_id]
    new_hypothesis.evidence_source_ids = list(dict.fromkeys(hypoA.evidence_source_ids + hypoB.evidence_source_ids))
    return new_hypothesis
