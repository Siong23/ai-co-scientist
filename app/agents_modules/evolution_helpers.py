"""Helpers for creating evolved hypotheses without mutating their parents."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Literal

from ..models import Hypothesis, ResearchGoal
from ..utils import generate_unique_id, logger, redact_secrets
from .generation_helpers import _call_llm

EvolutionStrategy = Literal[
    "grounding",
    "feasibility",
    "inspiration",
    "combination",
    "simplification",
    "out_of_box",
]

EVOLUTION_STRATEGIES: tuple[EvolutionStrategy, ...] = (
    "combination",
    "feasibility",
    "simplification",
    "grounding",
    "inspiration",
    "out_of_box",
)

_NEAR_DUPLICATE_THRESHOLD = 0.92

_STRATEGY_INSTRUCTIONS: dict[EvolutionStrategy, str] = {
    "grounding": (
        "Strengthen the hypothesis using only the supplied evidence. Identify its weakest reasoning gap, "
        "repair that gap with traceable supporting details, and explicitly preserve uncertainty where the evidence is "
        "insufficient. Do not invent sources or findings."
    ),
    "feasibility": (
        "Improve coherence, practicality, and feasibility. Repair invalid assumptions, make the mechanism specific, "
        "and describe an implementable validation path while preserving the original novelty."
    ),
    "inspiration": (
        "Create one new hypothesis inspired by useful principles in the parents. Transfer an underlying mechanism or "
        "analogy instead of merely rephrasing or aggregating the parents."
    ),
    "combination": (
        "Combine only complementary strengths from the parents into one coherent hypothesis. Resolve contradictions "
        "between them and avoid returning a list or a simple concatenation."
    ),
    "simplification": (
        "Refine the strongest parent into a simpler, clearer, and more testable hypothesis. Remove unnecessary claims "
        "and expose the smallest decisive experiment without making the hypothesis trivial."
    ),
    "out_of_box": (
        "Generate one divergent, out-of-the-box alternative inspired by the parents. It must explore a meaningfully "
        "different mechanism and must not be a direct combination of existing entities or methods."
    ),
}


def _json_objects(response: str):
    """Yield JSON objects found in fenced or explanatory local-model output."""
    fenced = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    decoder = json.JSONDecoder()
    for candidate_text in [*fenced, response]:
        for index, character in enumerate(candidate_text):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _parse_evolution_response(response: str) -> tuple[dict[str, str] | None, str]:
    if not response or not response.strip():
        return None, "empty_response"
    if response.lstrip().lower().startswith("error:"):
        return None, "llm_error"

    found_object = False
    for payload in _json_objects(response):
        found_object = True
        normalised = {str(key).strip().casefold(): value for key, value in payload.items()}
        title = normalised.get("title")
        text = normalised.get("hypothesis", normalised.get("text"))
        if isinstance(title, str) and title.strip() and isinstance(text, str) and text.strip():
            return {"title": title.strip(), "text": text.strip()}, "accepted"

    reason = "missing_required_fields" if found_object else "no_json_object"
    return None, reason


def parse_evolution_response(response: str) -> dict[str, str] | None:
    """Parse a validated JSON hypothesis from common local-model output shapes."""
    parsed, _ = _parse_evolution_response(response)
    return parsed


def _normalise_hypothesis_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def validate_evolution_candidate(
    candidate: Mapping[str, str],
    parents: Sequence[Hypothesis],
    strategy: EvolutionStrategy,
) -> str | None:
    """Return a deterministic rejection reason for non-evolutionary output."""
    text = str(candidate.get("text") or candidate.get("hypothesis") or "").strip()
    normalised = _normalise_hypothesis_text(text)
    if not normalised:
        return "empty_hypothesis"

    if len(re.findall(r"\bhypothesis\s*:", text, flags=re.IGNORECASE)) > 1:
        return "multiple_hypotheses"

    lowered = text.casefold().lstrip()
    if strategy == "combination" and (
        lowered.startswith("combination of") or "<br>1." in lowered or "<br>2." in lowered
    ):
        return "stitched_combination"

    for parent in parents:
        parent_text = _normalise_hypothesis_text(parent.text)
        if not parent_text:
            continue
        similarity = SequenceMatcher(None, parent_text, normalised).ratio()
        if similarity >= _NEAR_DUPLICATE_THRESHOLD:
            return f"near_duplicate_parent:{parent.hypothesis_id}"
    return None


def _build_quality_repair_prompt(
    original_prompt: str,
    candidate: Mapping[str, str],
    rejection_reason: str,
) -> str:
    guidance = {
        "multiple_hypotheses": "Return one unified causal claim with one coherent validation plan.",
        "stitched_combination": "Synthesize the parents into one mechanism; do not list or concatenate them.",
    }.get(
        rejection_reason.split(":", 1)[0],
        "Make a substantive change to the mechanism, prediction, or decisive experiment; do not merely rephrase the parent.",
    )
    prior_candidate = json.dumps(dict(candidate), ensure_ascii=False, default=str)
    return f"""
{original_prompt}

Your previous candidate was rejected by the Evolution quality gate.
Rejection reason: {rejection_reason}
Required correction: {guidance}
Rejected candidate: {prior_candidate[:4000]}

Return only the corrected JSON object using the exact schema above.
""".strip()


def _parent_payload(parent: Hypothesis) -> dict:
    reflection = parent.reflection_report.model_dump() if parent.reflection_report else None
    return {
        "id": parent.hypothesis_id,
        "title": parent.title,
        "hypothesis": parent.text,
        "elo_score": parent.elo_score,
        "novelty_review": parent.novelty_review,
        "feasibility_review": parent.feasibility_review,
        "review_comments": parent.review_comments,
        "reflection_report": reflection,
    }


def resolve_parent_evidence(
    parents: Sequence[Hypothesis],
    retrieved_sources: Sequence[Mapping] = (),
) -> list[Mapping]:
    """Resolve inherited evidence IDs without mutating any parent hypothesis."""
    resolved: list[Mapping] = []
    seen = set()

    def add_source(source: Mapping) -> None:
        marker = str(
            source.get("source_id")
            or source.get("id")
            or source.get("url")
            or json.dumps(dict(source), sort_keys=True, default=str)
        )
        if marker not in seen:
            seen.add(marker)
            resolved.append(source)

    for parent in parents:
        for source in parent.evidence_sources:
            if isinstance(source, Mapping):
                add_source(source)

    source_by_id: dict[str, Mapping] = {}
    for source in retrieved_sources:
        if not isinstance(source, Mapping):
            continue
        source_id = str(source.get("source_id") or "").strip()
        parent_source_id = str(source.get("parent_source_id") or "").strip()
        if source_id:
            source_by_id.setdefault(source_id, source)
        if parent_source_id:
            source_by_id.setdefault(parent_source_id, source)

    for parent in parents:
        for source_id in parent.evidence_source_ids:
            source = source_by_id.get(str(source_id))
            if source is not None:
                add_source(source)
    return resolved


def _evidence_context(evidence_sources: Sequence[Mapping], *, limit: int = 8) -> str:
    evidence = []
    seen = set()
    for source in evidence_sources:
        source_id = str(source.get("source_id") or source.get("id") or source.get("url") or "").strip()
        marker = source_id or json.dumps(dict(source), sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        title = str(source.get("title") or "Untitled source").strip()
        excerpt = str(
            source.get("abstract") or source.get("summary") or source.get("content") or source.get("text") or ""
        ).strip()
        evidence.append({"source_id": source_id, "title": title, "excerpt": excerpt[:1200]})
        if len(evidence) >= limit:
            return json.dumps(evidence, indent=2, ensure_ascii=False)
    return json.dumps(evidence, indent=2, ensure_ascii=False)


def _format_evolution_meta_review(feedback: Sequence[Mapping] | str | None) -> str:
    if not feedback:
        return ""
    if isinstance(feedback, str):
        return f"\nPrior cycle meta-review feedback to address:\n{feedback}\n"
    if isinstance(feedback, (list, tuple)) and feedback:
        latest = feedback[-1]
        if isinstance(latest, dict):
            critiques = latest.get("meta_review_critique", [])
            next_steps = (latest.get("research_overview", {}) or {}).get("suggested_next_steps", [])
            parts = []
            if critiques:
                parts.append("Critiques:\n" + "\n".join(f"- {c}" for c in critiques))
            if next_steps:
                parts.append("Suggested next steps:\n" + "\n".join(f"- {s}" for s in next_steps))
            if parts:
                return "\nPrior cycle meta-review feedback to address:\n" + "\n\n".join(parts) + "\n"
    return ""


def build_evolution_prompt(
    strategy: EvolutionStrategy,
    parents: Sequence[Hypothesis],
    research_goal: ResearchGoal,
    *,
    evidence_sources: Sequence[Mapping] | None = None,
    meta_review_feedback: Sequence[Mapping] | str | None = None,
) -> str:
    """Build a strategy-specific prompt grounded in tournament and review state."""
    parent_payload = [_parent_payload(parent) for parent in parents]
    resolved_evidence = resolve_parent_evidence(parents) if evidence_sources is None else list(evidence_sources)
    feedback_section = _format_evolution_meta_review(meta_review_feedback)
    return f"""
You are the Evolution agent in a scientific hypothesis tournament. Create exactly one NEW hypothesis; never edit,
replace, or claim to overwrite a parent. The new hypothesis will be independently reviewed and must compete in the
tournament before it can displace an existing idea.

Research goal:
{research_goal.description}

Evaluation criteria:
{research_goal.preferences}

Constraints:
{json.dumps(research_goal.constraints, indent=2, ensure_ascii=False, default=str)}
{feedback_section}
Evolution strategy: {strategy}
Strategy instruction: {_STRATEGY_INSTRUCTIONS[strategy]}

Top-ranked parent hypotheses and their existing reviews:
{json.dumps(parent_payload, indent=2, ensure_ascii=False, default=str)}

Evidence already attached to the parents:
{_evidence_context(resolved_evidence)}

Return only valid JSON with this exact schema:
{{"title": "concise title", "hypothesis": "detailed, self-contained and empirically testable hypothesis"}}
""".strip()


def call_llm_for_evolution(
    strategy: EvolutionStrategy,
    parents: Sequence[Hypothesis],
    research_goal: ResearchGoal,
    *,
    max_tokens: int = 2048,
    evidence_sources: Sequence[Mapping] | None = None,
    diagnostics: list[dict] | None = None,
    quality_repair_attempts: int = 1,
    meta_review_feedback: Sequence[Mapping] | str | None = None,
) -> dict[str, str] | None:
    """Create one evolved candidate through the shared, mockable LLM boundary."""
    base_prompt = build_evolution_prompt(
        strategy,
        parents,
        research_goal,
        evidence_sources=evidence_sources,
        meta_review_feedback=meta_review_feedback,
    )
    prompt = base_prompt
    quality_rejections: list[str] = []
    response = ""
    parsed = None
    reason = "no_response"
    for attempt in range(max(0, quality_repair_attempts) + 1):
        response = _call_llm(
            prompt,
            temperature=research_goal.generation_temperature,
            model=research_goal.llm_model,
            max_tokens=max_tokens,
            reasoning="off",
        )
        parsed, reason = _parse_evolution_response(response)
        if parsed is None:
            break

        quality_rejection = validate_evolution_candidate(parsed, parents, strategy)
        if quality_rejection is None:
            reason = "accepted_after_quality_repair" if quality_rejections else "accepted"
            break

        rejected_candidate = parsed
        quality_rejections.append(quality_rejection)
        parsed = None
        reason = quality_rejection
        if attempt >= max(0, quality_repair_attempts):
            break
        logger.warning(
            "Evolution strategy %s failed quality gate (%s); requesting one repair.",
            strategy,
            quality_rejection,
        )
        prompt = _build_quality_repair_prompt(
            base_prompt,
            rejected_candidate,
            quality_rejection,
        )
    attempt = {
        "strategy": strategy,
        "parent_ids": [parent.hypothesis_id for parent in parents],
        "status": "accepted" if parsed is not None else "rejected",
        "reason": reason,
    }
    if quality_rejections:
        attempt["quality_rejections"] = quality_rejections
    if parsed is None:
        excerpt = redact_secrets(" ".join(str(response).split()))[:500]
        attempt["response_excerpt"] = excerpt
        logger.warning(
            "Evolution strategy %s returned no usable hypothesis: %s; response=%s",
            strategy,
            reason,
            excerpt,
        )
    if diagnostics is not None:
        diagnostics.append(attempt)
    return parsed


def create_evolved_hypothesis(
    candidate: Mapping[str, str],
    parents: Sequence[Hypothesis],
    strategy: EvolutionStrategy,
    *,
    evidence_sources: Sequence[Mapping] | None = None,
) -> Hypothesis:
    """Create a tournament-ready child while retaining lineage and evidence."""
    evolved = Hypothesis(generate_unique_id("E"), candidate["title"], candidate["text"])
    evolved.parent_ids = list(dict.fromkeys(parent.hypothesis_id for parent in parents))
    evolved.evolution_strategy = strategy
    evolved.evidence_source_ids = list(
        dict.fromkeys(source_id for parent in parents for source_id in parent.evidence_source_ids)
    )
    evolved.references = [reference for parent in parents for reference in parent.references]

    resolved_evidence = resolve_parent_evidence(parents) if evidence_sources is None else list(evidence_sources)
    evolved.evidence_sources = [dict(source) for source in resolved_evidence]
    return evolved


def combine_hypotheses(hypoA: Hypothesis, hypoB: Hypothesis) -> Hypothesis:
    """Deterministically combine two hypotheses as an availability fallback."""
    new_id = generate_unique_id("E")  # Use utility function
    combined_title = f"Combined: {hypoA.title} & {hypoB.title}"
    # Keep the combined text plain and structured so downstream code can process it safely.
    combined_text = f"Combination of:<br>1. {hypoA.text}<br>2. {hypoB.text}"

    logger.info("Combining hypotheses %s and %s into %s", hypoA.hypothesis_id, hypoB.hypothesis_id, new_id)
    new_hypothesis = Hypothesis(new_id, combined_title, combined_text)
    new_hypothesis.parent_ids = [hypoA.hypothesis_id, hypoB.hypothesis_id]
    new_hypothesis.evolution_strategy = "combination_fallback"
    new_hypothesis.evidence_source_ids = list(dict.fromkeys(hypoA.evidence_source_ids + hypoB.evidence_source_ids))
    new_hypothesis.references = list(hypoA.references) + list(hypoB.references)
    new_hypothesis.evidence_sources = list(hypoA.evidence_sources)
    seen_sources = {
        json.dumps(source, sort_keys=True, default=str) if isinstance(source, Mapping) else repr(source)
        for source in new_hypothesis.evidence_sources
    }
    for source in hypoB.evidence_sources:
        marker = json.dumps(source, sort_keys=True, default=str) if isinstance(source, Mapping) else repr(source)
        if marker not in seen_sources:
            seen_sources.add(marker)
            new_hypothesis.evidence_sources.append(source)
    return new_hypothesis
