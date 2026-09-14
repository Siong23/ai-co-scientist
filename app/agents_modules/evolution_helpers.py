"""Helpers for creating evolved hypotheses without mutating their parents."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any, Literal

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
_MAX_PARENT_TEXT_CHARS = 3500
_MAX_REVIEW_TEXT_CHARS = 500
_MAX_REVIEW_ITEMS = 2
_MAX_REVIEW_CLAIMS = 4
_MAX_CLAIM_TEXT_CHARS = 400
_MAX_CLAIM_SOURCE_IDS = 6
_MAX_EVIDENCE_SOURCES = 6
_MAX_EVIDENCE_EXCERPT_CHARS = 800
_MAX_EVIDENCE_REFS_PER_SOURCE = 3
_MAX_EVIDENCE_REF_TEXT_CHARS = 500
_MAX_META_REVIEW_ITEMS = 3
_MAX_META_REVIEW_TEXT_CHARS = 800

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


def _parse_evolution_response(response: str) -> tuple[dict[str, Any] | None, str]:
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
            parsed: dict[str, Any] = {"title": title.strip(), "text": text.strip()}
            raw_source_ids = normalised.get("evidence_source_ids")
            if isinstance(raw_source_ids, list):
                parsed["evidence_source_ids"] = list(
                    dict.fromkeys(
                        source_id.strip()
                        for source_id in raw_source_ids
                        if isinstance(source_id, str) and source_id.strip()
                    )
                )
            raw_evidence_refs = normalised.get("evidence_refs")
            if isinstance(raw_evidence_refs, list):
                parsed["evidence_refs"] = list(
                    dict.fromkeys(
                        ref_id.strip() for ref_id in raw_evidence_refs if isinstance(ref_id, str) and ref_id.strip()
                    )
                )
            return parsed, "accepted"

    reason = "missing_required_fields" if found_object else "no_json_object"
    return None, reason


def parse_evolution_response(response: str) -> dict[str, Any] | None:
    """Parse a validated JSON hypothesis from common local-model output shapes."""
    parsed, _ = _parse_evolution_response(response)
    return parsed


def _normalise_hypothesis_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def validate_evolution_candidate(
    candidate: Mapping[str, Any],
    parents: Sequence[Hypothesis],
    strategy: EvolutionStrategy,
    *,
    available_evidence_source_ids: Sequence[str] = (),
    evidence_ref_source_ids: Mapping[str, str] | None = None,
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

    available_ids = {str(source_id).strip() for source_id in available_evidence_source_ids if str(source_id).strip()}
    if available_ids:
        selected_ids = candidate.get("evidence_source_ids")
        if not isinstance(selected_ids, list):
            return "missing_evidence_source_ids"
        if not selected_ids:
            return "no_valid_evidence_source_ids"
        unknown_ids = [source_id for source_id in selected_ids if source_id not in available_ids]
        if unknown_ids:
            return "unknown_evidence_source_ids:" + ",".join(unknown_ids)

    available_ref_ids = set(evidence_ref_source_ids or ())
    if available_ref_ids:
        selected_refs = candidate.get("evidence_refs")
        if not isinstance(selected_refs, list):
            return "missing_evidence_refs"
        if not selected_refs:
            return "no_valid_evidence_refs"
        unknown_refs = [ref_id for ref_id in selected_refs if ref_id not in available_ref_ids]
        if unknown_refs:
            return "unknown_evidence_refs:" + ",".join(unknown_refs)
        selected_source_ids = set(candidate.get("evidence_source_ids") or ())
        mismatched_refs = [
            ref_id for ref_id in selected_refs if evidence_ref_source_ids[ref_id] not in selected_source_ids
        ]
        if mismatched_refs:
            return "evidence_ref_source_mismatch:" + ",".join(mismatched_refs)
    return None


def _build_quality_repair_prompt(
    original_prompt: str,
    candidate: Mapping[str, Any],
    rejection_reason: str,
) -> str:
    guidance = {
        "multiple_hypotheses": "Return one unified causal claim with one coherent validation plan.",
        "stitched_combination": "Synthesize the parents into one mechanism; do not list or concatenate them.",
        "missing_evidence_source_ids": (
            "Add evidence_source_ids and select only the supplied sources that directly support the new hypothesis."
        ),
        "no_valid_evidence_source_ids": (
            "Select at least one supplied source that directly supports the new hypothesis."
        ),
        "unknown_evidence_source_ids": (
            "Use only exact source_id values from the supplied evidence and remove invented or unavailable IDs."
        ),
        "missing_evidence_refs": (
            "Add evidence_refs and select only exact chunk_id values whose passages directly support the new hypothesis."
        ),
        "no_valid_evidence_refs": "Select at least one supplied evidence passage that directly supports the new hypothesis.",
        "unknown_evidence_refs": "Use only exact chunk_id values shown in the supplied evidence.",
        "evidence_ref_source_mismatch": (
            "Every selected chunk_id must belong to one of the selected evidence_source_ids."
        ),
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


def _bounded_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _claim_source_ids(evidence: object) -> list[str]:
    if not isinstance(evidence, (list, tuple)):
        return []
    source_ids = []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        source_id = str(item.get("source_id") or item.get("id") or item.get("url") or "").strip()
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
        if len(source_ids) >= _MAX_CLAIM_SOURCE_IDS:
            break
    return source_ids


def _reflection_summary(parent: Hypothesis) -> dict | None:
    """Keep scientific judgments while excluding full retrieved documents."""
    report = parent.reflection_report
    if report is None:
        return None

    claims = []
    for claim in report.claims[:_MAX_REVIEW_CLAIMS]:
        claims.append(
            {
                "claim": _bounded_text(claim.claim, _MAX_CLAIM_TEXT_CHARS),
                "status": claim.status,
                "confidence": claim.confidence,
                "supporting_source_ids": _claim_source_ids(claim.supporting_evidence),
                "contradictory_source_ids": _claim_source_ids(claim.contradictory_evidence),
            }
        )

    return {
        "alignment_score": report.alignment_score,
        "novelty_score": report.novelty_score,
        "feasibility_score": report.feasibility_score,
        "plausibility_score": report.plausibility_score,
        "testability_score": report.testability_score,
        "evidence_quality_score": report.evidence_quality_score,
        "expected_research_value_score": report.expected_research_value_score,
        "strengths": [_bounded_text(item, _MAX_REVIEW_TEXT_CHARS) for item in report.strengths[:_MAX_REVIEW_ITEMS]],
        "weaknesses": [_bounded_text(item, _MAX_REVIEW_TEXT_CHARS) for item in report.weaknesses[:_MAX_REVIEW_ITEMS]],
        "recommendation": report.recommendation,
        "claims": claims,
        "overall_confidence": report.overall_confidence,
    }


def _parent_payload(parent: Hypothesis) -> dict:
    return {
        "id": parent.hypothesis_id,
        "title": _bounded_text(parent.title, _MAX_REVIEW_TEXT_CHARS),
        "hypothesis": _bounded_text(parent.text, _MAX_PARENT_TEXT_CHARS),
        "elo_score": parent.elo_score,
        "novelty_review": parent.novelty_review,
        "feasibility_review": parent.feasibility_review,
        "review_comments": [
            _bounded_text(item, _MAX_REVIEW_TEXT_CHARS) for item in parent.review_comments[:_MAX_REVIEW_ITEMS]
        ],
        "reflection_report": _reflection_summary(parent),
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


def _evidence_source_aliases(source: Mapping) -> tuple[str, ...]:
    """Return stable identifiers that may refer to one evidence record."""

    return tuple(
        dict.fromkeys(
            value
            for field in ("source_id", "parent_source_id", "id", "url")
            if (value := str(source.get(field) or "").strip())
        )
    )


def _bounded_evidence_payload(
    evidence_sources: Sequence[Mapping],
    *,
    limit: int = _MAX_EVIDENCE_SOURCES,
) -> list[dict[str, Any]]:
    evidence = []
    seen = set()
    for source in evidence_sources:
        source_id = str(source.get("source_id") or source.get("id") or source.get("url") or "").strip()
        marker = source_id or json.dumps(dict(source), sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        title = str(source.get("title") or "Untitled source").strip()
        evidence_refs = []
        raw_refs = source.get("evidence_refs", ())
        if isinstance(raw_refs, (list, tuple)):
            for raw_ref in raw_refs:
                if not isinstance(raw_ref, Mapping):
                    continue
                chunk_id = str(raw_ref.get("chunk_id") or "").strip()
                text = str(raw_ref.get("text") or "").strip()
                if not chunk_id or not text:
                    continue
                evidence_refs.append(
                    {
                        "chunk_id": chunk_id,
                        "excerpt": _bounded_text(text, _MAX_EVIDENCE_REF_TEXT_CHARS),
                    }
                )
                if len(evidence_refs) >= _MAX_EVIDENCE_REFS_PER_SOURCE:
                    break
        item: dict[str, Any] = {
            "source_id": source_id,
            "title": _bounded_text(title, _MAX_REVIEW_TEXT_CHARS),
        }
        if evidence_refs:
            item["evidence_refs"] = evidence_refs
        else:
            excerpt = str(
                source.get("abstract") or source.get("summary") or source.get("content") or source.get("text") or ""
            ).strip()
            item["excerpt"] = _bounded_text(excerpt, _MAX_EVIDENCE_EXCERPT_CHARS)
        evidence.append(item)
        if len(evidence) >= limit:
            break
    return evidence


def _evidence_context(
    evidence_sources: Sequence[Mapping],
    *,
    limit: int = _MAX_EVIDENCE_SOURCES,
) -> str:
    return json.dumps(
        _bounded_evidence_payload(evidence_sources, limit=limit),
        indent=2,
        ensure_ascii=False,
    )


def _evidence_ref_source_ids(evidence_sources: Sequence[Mapping]) -> dict[str, str]:
    return {
        str(ref["chunk_id"]): str(source["source_id"])
        for source in _bounded_evidence_payload(evidence_sources)
        for ref in source.get("evidence_refs", ())
    }


def _format_evolution_meta_review(feedback: Sequence[Mapping] | str | None) -> str:
    if not feedback:
        return ""
    if isinstance(feedback, str):
        bounded = _bounded_text(feedback, _MAX_META_REVIEW_TEXT_CHARS * _MAX_META_REVIEW_ITEMS)
        return f"\nPrior cycle meta-review feedback to address:\n{bounded}\n"
    if isinstance(feedback, (list, tuple)) and feedback:
        latest = feedback[-1]
        if isinstance(latest, dict):
            critiques = latest.get("meta_review_critique", [])
            next_steps = (latest.get("research_overview", {}) or {}).get("suggested_next_steps", [])
            parts = []
            if critiques:
                parts.append(
                    "Critiques:\n"
                    + "\n".join(
                        f"- {_bounded_text(item, _MAX_META_REVIEW_TEXT_CHARS)}"
                        for item in critiques[:_MAX_META_REVIEW_ITEMS]
                    )
                )
            if next_steps:
                parts.append(
                    "Suggested next steps:\n"
                    + "\n".join(
                        f"- {_bounded_text(item, _MAX_META_REVIEW_TEXT_CHARS)}"
                        for item in next_steps[:_MAX_META_REVIEW_ITEMS]
                    )
                )
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

Maintain strict alignment with the research goal: do not drift into secondary metrics or assume unrequested paradigms (e.g. generic AI does not imply an LLM).
If multi-agent coordination is requested, specify concrete interaction or communication mechanisms rather than comparing algorithms side-by-side.
If latency guarantees or real-time control are claimed, specify the supporting operational mechanism (e.g. asynchrony, timeouts, or hierarchical decoupling).

Top-ranked parent hypotheses and their existing reviews:
{json.dumps(parent_payload, indent=2, ensure_ascii=False, default=str)}

Evidence already attached to the parents:
{_evidence_context(resolved_evidence)}

Evidence selection rules:
- Reassess evidence against the NEW hypothesis and the research goal.
- Select the smallest subset of supplied sources that directly supports the new hypothesis.
- Do not automatically inherit every parent source. Omit sources that support only discarded parent claims,
  tangential methods, or a different domain or dataset.
- Copy source IDs exactly from the supplied evidence. Do not invent IDs.
- When chunk-level passages are supplied, select the smallest set of exact chunk_id values that directly supports
  the new hypothesis. Do not retain unrelated passages from an otherwise relevant paper.

Return only valid JSON with this exact schema:
{{"title": "concise title", "hypothesis": "detailed, self-contained and empirically testable hypothesis", "evidence_source_ids": ["exact supplied source_id"], "evidence_refs": ["exact supplied chunk_id"]}}
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
    transport_retry_attempts: int = 2,
    meta_review_feedback: Sequence[Mapping] | str | None = None,
) -> dict[str, Any] | None:
    """Create one evolved candidate through the shared, mockable LLM boundary."""
    resolved_evidence = resolve_parent_evidence(parents) if evidence_sources is None else list(evidence_sources)
    base_prompt = build_evolution_prompt(
        strategy,
        parents,
        research_goal,
        evidence_sources=resolved_evidence,
        meta_review_feedback=meta_review_feedback,
    )
    available_evidence_source_ids = list(
        dict.fromkeys(
            alias for source in resolved_evidence[:_MAX_EVIDENCE_SOURCES] for alias in _evidence_source_aliases(source)
        )
    )
    evidence_ref_source_ids = _evidence_ref_source_ids(resolved_evidence)
    prompt = base_prompt
    quality_rejections: list[str] = []
    response = ""
    parsed = None
    reason = "no_response"
    transport_retries = 0
    for attempt in range(max(0, quality_repair_attempts) + 1):
        for transport_attempt in range(max(0, transport_retry_attempts) + 1):
            response = _call_llm(
                prompt,
                temperature=research_goal.generation_temperature,
                model=research_goal.llm_model,
                max_tokens=max_tokens,
                reasoning="off",
            )
            parsed, reason = _parse_evolution_response(response)
            if parsed is not None or reason not in {"llm_error", "empty_response"}:
                break
            if transport_attempt >= max(0, transport_retry_attempts):
                break
            transport_retries += 1
            logger.warning(
                "Evolution strategy %s hit a transient LLM failure; retrying (%d/%d).",
                strategy,
                transport_attempt + 1,
                max(0, transport_retry_attempts),
            )
        if parsed is None:
            break

        quality_rejection = validate_evolution_candidate(
            parsed,
            parents,
            strategy,
            available_evidence_source_ids=available_evidence_source_ids,
            evidence_ref_source_ids=evidence_ref_source_ids,
        )
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
    if transport_retries:
        attempt["transport_retries"] = transport_retries
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
    candidate: Mapping[str, Any],
    parents: Sequence[Hypothesis],
    strategy: EvolutionStrategy,
    *,
    evidence_sources: Sequence[Mapping] | None = None,
) -> Hypothesis:
    """Create a tournament-ready child while retaining lineage and evidence."""
    evolved = Hypothesis(generate_unique_id("E"), candidate["title"], candidate["text"])
    evolved.parent_ids = list(dict.fromkeys(parent.hypothesis_id for parent in parents))
    evolved.evolution_strategy = strategy
    evolved.references = [reference for parent in parents for reference in parent.references]

    resolved_evidence = resolve_parent_evidence(parents) if evidence_sources is None else list(evidence_sources)
    requested_source_ids = candidate.get("evidence_source_ids")
    if isinstance(requested_source_ids, list):
        source_by_alias = {alias: source for source in resolved_evidence for alias in _evidence_source_aliases(source)}
        selected_sources = []
        selected_source_ids = []
        seen_sources = set()
        requested_ref_ids = candidate.get("evidence_refs")
        selected_ref_ids = (
            {str(value).strip() for value in requested_ref_ids if isinstance(value, str) and str(value).strip()}
            if isinstance(requested_ref_ids, list)
            else set()
        )
        for requested_id in requested_source_ids:
            source = source_by_alias.get(str(requested_id).strip())
            if source is None:
                continue
            source_id = str(source.get("source_id") or source.get("id") or source.get("url") or "").strip()
            if not source_id or source_id in seen_sources:
                continue
            selected_source = dict(source)
            if selected_ref_ids:
                raw_refs = source.get("evidence_refs", ())
                selected_refs = (
                    [
                        dict(ref)
                        for ref in raw_refs
                        if isinstance(ref, Mapping) and str(ref.get("chunk_id") or "").strip() in selected_ref_ids
                    ]
                    if isinstance(raw_refs, (list, tuple))
                    else []
                )
                if not selected_refs:
                    continue
                selected_source["evidence_refs"] = selected_refs
                selected_source["selected_chunk_ids"] = [
                    str(ref["chunk_id"])
                    for ref in selected_refs
                    if ref.get("evidence_type") == "full_text" and ref.get("chunk_id")
                ]
                selected_source["context_chunk_ids"] = list(selected_source["selected_chunk_ids"])
            seen_sources.add(source_id)
            selected_source_ids.append(source_id)
            selected_sources.append(selected_source)
        evolved.evidence_source_ids = selected_source_ids
        evolved.evidence_sources = selected_sources
        evolved.evidence_refs = [
            ref_id
            for ref_id in candidate.get("evidence_refs", [])
            if isinstance(ref_id, str)
            and any(
                str(ref.get("chunk_id") or "") == ref_id
                for source in selected_sources
                for ref in source.get("evidence_refs", ())
                if isinstance(ref, Mapping)
            )
        ]
    else:
        # Compatibility for deterministic callers and pre-change cached responses.
        evolved.evidence_source_ids = list(
            dict.fromkeys(source_id for parent in parents for source_id in parent.evidence_source_ids)
        )
        evolved.evidence_sources = [dict(source) for source in resolved_evidence]
    return evolved


def combine_hypotheses(hypoA: Hypothesis, hypoB: Hypothesis) -> Hypothesis:
    """Deterministically combine two hypotheses as an availability fallback."""
    new_id = generate_unique_id("E")  # Use utility function
    combined_title = f"Combined: {hypoA.title} & {hypoB.title}"
    # Keep the combined text plain and structured so downstream code can process it safely.
    combined_text = f"Combination of:<br>1. {hypoA.text}<br>2. {hypoB.text}"

    logger.debug("Combining hypotheses %s and %s into %s", hypoA.hypothesis_id, hypoB.hypothesis_id, new_id)
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
    new_hypothesis.evidence_refs = list(dict.fromkeys(hypoA.evidence_refs + hypoB.evidence_refs))
    return new_hypothesis
