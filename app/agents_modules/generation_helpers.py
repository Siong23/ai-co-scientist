"""Generation and literature-analysis helper functions.

These functions are kept separate from the public agent façade so the
GenerationAgent can own orchestration while this module owns LLM protocols and
structured evidence validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..config import config
from ..models import EvidenceClaim
from ..rag_retriever import EvidenceAspect, SearchQueryPlan
from ..utils import logger


def _call_llm(*args, **kwargs):
    """Call the façade's LLM boundary so existing mocks remain effective."""
    from .. import agents as facade

    return facade.call_llm(*args, **kwargs)


def _parse_generation_response(response: str) -> List[Dict]:
    """Parse the small set of JSON shapes commonly returned by local LLMs."""
    try:
        cleaned = response.strip()
        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_match:
            cleaned = fenced_match.group(1).strip()

        # Some local models add a short explanation before or after the JSON.
        # raw_decode accepts the first complete JSON value without weakening
        # validation of the hypothesis objects themselves.
        starts = [index for index in (cleaned.find("["), cleaned.find("{")) if index >= 0]
        if not starts:
            raise ValueError("No JSON object or array was found.")
        hypotheses_data, _ = json.JSONDecoder().raw_decode(cleaned[min(starts) :])

        if isinstance(hypotheses_data, dict):
            error_text = hypotheses_data.get("error")
            if isinstance(error_text, str) and error_text.strip():
                return [
                    {
                        "title": "Error",
                        "text": error_text.strip(),
                    }
                ]
            for wrapper_key in ("hypotheses", "results", "items"):
                wrapped = hypotheses_data.get(wrapper_key)
                if isinstance(wrapped, list):
                    hypotheses_data = wrapped
                    break

        if isinstance(hypotheses_data, list):
            aliases = {
                "Title": "title",
                "Hypothesis": "hypothesis",
                "Rationale": "rationale",
                "Feasibility": "feasibility",
                "sourceIds": "source_ids",
                "Source IDs": "source_ids",
                "evidence_sources": "source_ids",
            }
            hypotheses_data = [
                {aliases.get(key, key): value for key, value in item.items()} if isinstance(item, dict) else item
                for item in hypotheses_data
            ]

        required_fields = {
            "title",
            "hypothesis",
            "rationale",
            "feasibility",
            "source_ids",
        }
        if not isinstance(hypotheses_data, list) or not all(
            isinstance(h, dict)
            and required_fields.issubset(h)
            and all(isinstance(h[field], str) for field in required_fields - {"source_ids"})
            and isinstance(h["source_ids"], list)
            for h in hypotheses_data
        ):
            error_message = (
                "Invalid JSON format: Expected a hypothesis array with "
                "'title', 'hypothesis', 'rationale', 'feasibility', and "
                "'source_ids' fields, or an object with an 'error' string."
            )
            raise ValueError(error_message)
        logger.info("Parsed generated hypotheses: %s", hypotheses_data)
        return hypotheses_data
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


# Updated signature to accept temperature
def call_llm_for_generation(
    prompt: str, num_hypotheses: int = 3, temperature: float = 0.7, model: str | None = None
) -> List[Dict]:
    """Call the LLM and make one format-repair attempt when JSON is malformed."""
    logger.info(
        "LLM generation called with prompt: %s, num_hypotheses: %d, temperature: %.2f",
        prompt,
        num_hypotheses,
        temperature,
    )
    schema_instruction = (
        "Return only valid JSON with no Markdown or commentary. On success, "
        f"return an array containing at most {num_hypotheses} objects. Returning fewer is valid "
        "when only fewer candidates are scientifically supportable. "
        "Every object must contain 'title', 'hypothesis', 'rationale', "
        "'feasibility', and 'source_ids'. It may also contain an "
        "'evidence_refs' array of exact chunk IDs from the retrieved context. The "
        "first four values must be strings and 'source_ids' must be an array "
        "of exact Source ID strings from the retrieved context. If the "
        "retrieved context is insufficient or not directly relevant, return "
        'exactly {"error": "The retrieved context is insufficient to generate '
        'grounded hypotheses."}.'
    )
    full_prompt = f"{prompt}\n\n{schema_instruction}"

    response = _call_llm(full_prompt, temperature=temperature, model=model)
    logger.info("LLM generation response: %s", response)

    if response.startswith("Error:"):
        logger.error("LLM generation call failed: %s", response)
        return [{"title": "Error", "text": response}]

    try:
        return _parse_generation_response(response)
    except ValueError as first_error:
        logger.warning(
            "Initial generation output was not valid structured JSON; requesting one format-only repair: %s",
            first_error,
        )

    repair_prompt = (
        "Reformat the candidate response below to satisfy the JSON schema. "
        "Preserve its scientific meaning and exact source IDs. Do not add "
        "facts, citations, explanations, or Markdown. If a required field is "
        "absent and cannot be recovered, return the specified error object.\n\n"
        f"{schema_instruction}\n\nCandidate response:\n{response}"
    )
    repaired_response = _call_llm(
        repair_prompt,
        temperature=0.0,
        model=model,
    )
    logger.info("LLM generation format-repair response: %s", repaired_response)
    if repaired_response.startswith("Error:"):
        logger.error("LLM generation format-repair call failed: %s", repaired_response)
        return [{"title": "Error", "text": repaired_response}]

    try:
        return _parse_generation_response(repaired_response)
    except ValueError as exc:
        logger.error(
            "Could not parse repaired LLM generation response as JSON: %s",
            repaired_response,
            exc_info=True,
        )
        return [
            {
                "title": "Error",
                "text": f"Could not parse LLM response after format repair: {exc}",
            }
        ]


def call_llm_for_search_queries(
    research_goal: str,
    model: str | None = None,
    query_count: int = 5,
    research_planner_prompt: str | None = None,
    query_rewriter_prompt: str | None = None,
) -> tuple[SearchQueryPlan | None, str | None]:
    """Plan the research first, then rewrite the plan into search queries."""

    if research_planner_prompt is None or query_rewriter_prompt is None:
        from .generation import (
            QUERY_REWRITER_SYSTEM_PROMPT,
            RESEARCH_PLANNER_SYSTEM_PROMPT,
        )

        research_planner_prompt = research_planner_prompt or RESEARCH_PLANNER_SYSTEM_PROMPT
        query_rewriter_prompt = query_rewriter_prompt or QUERY_REWRITER_SYSTEM_PROMPT

    def parse_json_object(response: str) -> dict:
        cleaned_response = response.strip()
        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_match:
            cleaned_response = fenced_match.group(1).strip()
        payload = json.loads(cleaned_response)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def parse_research_plan(response: str) -> dict:
        payload = parse_json_object(response)
        string_fields = (
            "research_goal",
            "research_type",
            "freshness_requirement",
            "search_strategy",
        )
        list_fields = (
            "key_entities",
            "constraints",
            "sub_questions",
            "evidence_requirements",
            "ambiguities",
        )
        if any(not isinstance(payload.get(field), str) for field in string_fields) or any(
            not isinstance(payload.get(field), list) for field in list_fields
        ):
            raise ValueError("Research Planner returned an incomplete plan schema.")
        return payload

    def parse_response(response: str) -> SearchQueryPlan:
        payload = parse_json_object(response)
        queries = payload.get("queries")
        required_terms = payload.get("required_terms")
        raw_requirements = payload.get("explicit_requirements")
        raw_directions = payload.get("exploration_directions")
        if (
            not isinstance(queries, list)
            or not isinstance(required_terms, list)
            or not isinstance(raw_requirements, list)
            or not isinstance(raw_directions, list)
        ):
            raise ValueError(
                "Expected 'queries', 'required_terms', 'explicit_requirements', and 'exploration_directions' arrays."
            )

        query_texts = []
        for query in queries:
            if isinstance(query, str):
                query_text = query.strip()
            elif isinstance(query, dict):
                query_text = str(query.get("query", "")).strip()
            else:
                query_text = ""
            if query_text:
                query_texts.append(query_text)
        normalized_queries = tuple(dict.fromkeys(query_texts))
        normalized_terms = tuple(
            dict.fromkeys(term.strip() for term in required_terms if isinstance(term, str) and term.strip())
        )
        if len(normalized_queries) != query_count:
            raise ValueError(f"Expected exactly {query_count} unique search queries.")
        explicit_requirements: list[EvidenceAspect] = []
        seen_aspect_ids: set[str] = set()
        seen_evidence_needs: set[str] = set()
        for raw_aspect in raw_requirements:
            if not isinstance(raw_aspect, dict):
                continue
            aspect_id = str(raw_aspect.get("id", "")).strip()
            goal_quote = str(raw_aspect.get("goal_quote", "")).strip()
            evidence_need = str(raw_aspect.get("evidence_need", "")).strip()
            normalized_goal = " ".join(research_goal.casefold().split())
            normalized_quote = " ".join(goal_quote.casefold().split())
            normalized_evidence_need = " ".join(evidence_need.casefold().split())
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]{1,39}", aspect_id)
                or not normalized_quote
                or normalized_quote not in normalized_goal
                or len(goal_quote.split()) > 16
                or (evidence_need and len(evidence_need.split()) > 24)
                or aspect_id in seen_aspect_ids
                or (normalized_evidence_need and normalized_evidence_need in seen_evidence_needs)
            ):
                continue
            seen_aspect_ids.add(aspect_id)
            if normalized_evidence_need:
                seen_evidence_needs.add(normalized_evidence_need)
            explicit_requirements.append(
                EvidenceAspect(
                    aspect_id=aspect_id,
                    # The quote proves goal fidelity. The evidence need gives
                    # the coverage grader a literature-oriented concept rather
                    # than an imperative such as "develop a framework", which
                    # no retrieved paper can literally satisfy.
                    description=evidence_need or goal_quote,
                )
            )
        if not 1 <= len(explicit_requirements) <= 5:
            raise ValueError("Expected 1 to 5 unique explicit requirements with verbatim goal quotes.")
        exploration_directions = tuple(
            dict.fromkeys(
                direction.strip() for direction in raw_directions if isinstance(direction, str) and direction.strip()
            )
        )
        if len(exploration_directions) > 5:
            raise ValueError("Expected no more than 5 exploration directions.")

        return SearchQueryPlan(
            queries=normalized_queries,
            required_terms=normalized_terms,
            explicit_requirements=tuple(explicit_requirements),
            exploration_directions=exploration_directions,
        )

    planner_prompt = f"""
USER RESEARCH GOAL
{research_goal}
""".strip()
    planner_response = _call_llm(
        planner_prompt,
        temperature=0.0,
        model=model,
        system_prompt=research_planner_prompt,
    )
    if planner_response.startswith("Error:"):
        return None, f"Query rewriting failed: {planner_response}"

    # Preserve compatibility with callers that return the historical combined
    # query-plan schema. Normal operation takes the two-stage path below.
    try:
        first_payload = parse_json_object(planner_response)
    except (json.JSONDecodeError, ValueError):
        first_payload = {}
    legacy_query_response = (
        planner_response
        if {"queries", "required_terms", "explicit_requirements", "exploration_directions"}.issubset(first_payload)
        else None
    )

    if legacy_query_response is None:
        try:
            research_plan = parse_research_plan(planner_response)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Could not parse Research Planner response: %s", planner_response, exc_info=True)
            return None, f"Query rewriting failed: Research planning failed: {exc}"

        query_example = ",\n    ".join(
            f'{{"query": "query {index}", "purpose": "...", '
            '"sub_question": "...", "preferred_sources": [], "freshness": "..."}'
            for index in range(1, query_count + 1)
        )
        rewriter_system_prompt = f"""
{query_rewriter_prompt}

Generate exactly {query_count} distinct queries. In addition to the fields in
the Query Rewriter prompt, include the evidence-control fields below so the
retrieval system can validate coverage without turning optional ideas into
hard requirements.

- required_terms: only indispensable named entities, locations, organisms,
  materials, diseases, or technologies (and close synonyms).
- explicit_requirements: 1 to 5 non-overlapping objects with a stable
  snake_case id, a goal_quote copied verbatim from the original request, and
  an evidence_need describing the literature evidence to retrieve. Each quote
  must be at most 16 words and each evidence_need at most 24 words. Write the
  evidence_need as a scientific topic or finding, never as a user action such
  as "develop", "design", "write", or "generate". Do not require literature
  to have already completed the user's proposed research. Atomize comparisons
  into focal method, comparator, domain, and requested outcomes when those are
  explicitly present, and do not emit overlapping or duplicate requirements.
- exploration_directions: 0 to 5 optional search angles that must never become
  evidence gates.

Return only valid JSON with this extended shape:
{{
  "queries": [
    {query_example}
  ],
  "required_terms": ["entity", "synonym"],
  "explicit_requirements": [
    {{
      "id": "short_id",
      "goal_quote": "verbatim words from the original request",
      "evidence_need": "concise literature-oriented concept to substantiate"
    }}
  ],
  "exploration_directions": ["optional search direction"]
}}
""".strip()

        base_prompt = f"""
ORIGINAL USER REQUEST
{research_goal}

STRUCTURED RESEARCH PLAN
{json.dumps(research_plan, ensure_ascii=False, indent=2)}
""".strip()
    else:
        base_prompt = planner_prompt
        rewriter_system_prompt = query_rewriter_prompt

    correction = ""
    for attempt in range(2):
        if attempt == 0 and legacy_query_response is not None:
            response = legacy_query_response
        else:
            response = _call_llm(
                base_prompt + correction,
                temperature=0.0,
                model=model,
                system_prompt=rewriter_system_prompt,
            )
        if response.startswith("Error:"):
            return None, f"Query rewriting failed: {response}"
        try:
            return parse_response(response), None
        except (json.JSONDecodeError, AttributeError, ValueError) as exc:
            logger.warning(
                "Query plan attempt %d was invalid: %s",
                attempt + 1,
                exc,
            )
            if attempt == 1:
                logger.error(
                    "Could not parse query-rewriting response: %s",
                    response,
                    exc_info=True,
                )
                return None, f"Query rewriting failed: {exc}"
            correction = (
                "\n\nYour previous response was invalid because: "
                f"{exc}. Return a corrected JSON object. Atomize long or "
                "composite goal quotes into separate verbatim spans of at "
                "most 16 words; do not add anything absent from the goal."
            )

    return None, "Query rewriting failed."


def _canonical_arxiv_id(source_id: str) -> str | None:
    """Return an unversioned arXiv ID for a commonly emitted source ID."""

    normalized_id = source_id.strip()
    normalized_id = re.sub(
        r"^arxiv:\s*",
        "",
        normalized_id,
        flags=re.IGNORECASE,
    )
    if not re.fullmatch(
        r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})(?:v\d+)?",
        normalized_id,
        flags=re.IGNORECASE,
    ):
        return None
    return re.sub(
        r"v\d+$",
        "",
        normalized_id,
        flags=re.IGNORECASE,
    ).casefold()


def _resolve_retrieved_source_id(
    source_id: str,
    available_source_ids: set[str],
) -> str | None:
    """Resolve a model-emitted ID to one unique retrieved source."""

    normalized_id = source_id.strip()
    exact_matches = [
        available_id for available_id in available_source_ids if available_id.casefold() == normalized_id.casefold()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    requested_canonical_id = _canonical_arxiv_id(source_id)
    if requested_canonical_id is None:
        return None

    matches = [
        available_id
        for available_id in available_source_ids
        if _canonical_arxiv_id(available_id) == requested_canonical_id
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _resolve_retrieved_source_ids(
    source_ids,
    available_source_ids: set[str],
) -> list[str]:
    """Resolve and deduplicate model-emitted IDs in their original order."""

    resolved_ids: list[str] = []
    for source_id in source_ids:
        if not isinstance(source_id, str):
            continue
        resolved_id = _resolve_retrieved_source_id(
            source_id,
            available_source_ids,
        )
        if resolved_id is not None and resolved_id not in resolved_ids:
            resolved_ids.append(resolved_id)
    return resolved_ids


def call_llm_for_relevance_filter(
    research_goal: str,
    retrieved_context: str,
    available_source_ids: set[str],
    model: str | None = None,
    explicit_requirements: tuple[EvidenceAspect, ...] = (),
) -> tuple[list[str] | None, str | None]:
    """Select papers relevant enough to justify bounded PDF acquisition."""

    aspect_text = "\n".join(f"- {aspect.aspect_id}: {aspect.description}" for aspect in explicit_requirements)
    prompt = f"""
You are a high-recall candidate-paper filter for scientific literature retrieval.

Keep a retrieved source when its title and abstract make it plausibly useful
for at least one explicit requirement below. This gate decides whether a PDF
is worth acquiring; it does not certify a detailed factual claim. A source does
not need to cover the entire research goal by itself. Remove clearly irrelevant
lexical collisions, analogies, and papers about a different domain. Preserve
high recall for candidates that may contain useful Methods, Results,
Discussion, or Limitations evidence in their full text.

Explicit requirements copied from the user's goal:
{aspect_text or "- general_goal: the core scientific subject of the goal"}

Return only valid JSON:
{{
  "relevant_source_ids": ["exact Source ID"],
  "reason": "brief explanation"
}}

An empty relevant_source_ids array is valid when none of the sources directly
support the goal. Never invent a Source ID.

Research goal:
{research_goal}

Retrieved sources:
{retrieved_context}
""".strip()
    response = _call_llm(
        prompt,
        temperature=0.0,
        model=model,
    )
    if response.startswith("Error:"):
        return None, f"Evidence relevance grading failed: {response}"

    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        payload = json.loads(cleaned_response.strip())
        source_ids = payload.get("relevant_source_ids")
        if not isinstance(source_ids, list):
            raise ValueError("Expected a 'relevant_source_ids' array.")

        selected_ids = _resolve_retrieved_source_ids(
            source_ids,
            available_source_ids,
        )

        logger.info(
            "Abstract candidate-paper filter selected %d/%d sources: %s",
            len(selected_ids),
            len(available_source_ids),
            selected_ids,
        )
        return selected_ids, None
    except (json.JSONDecodeError, AttributeError, ValueError) as exc:
        logger.error(
            "Could not parse evidence relevance response: %s",
            response,
            exc_info=True,
        )
        return None, f"Evidence relevance grading failed: {exc}"


@dataclass(frozen=True)
class EvidenceCoverage:
    """Validated collective coverage of user-stated requirements."""

    aspect_source_ids: dict[str, tuple[str, ...]]
    missing_aspect_ids: tuple[str, ...]
    gap_queries: tuple[str, ...]
    reason: str
    aspect_evidence_refs: dict[str, tuple[dict, ...]] = field(default_factory=dict)
    stage: str = "discovery"

    @property
    def sufficient(self) -> bool:
        return not self.missing_aspect_ids


AUDIT_SCORE_WEIGHTS = {
    "evidence_validity": 15,
    "claim_evidence_entailment": 20,
    "novelty_against_prior_art": 20,
    "cross_paper_synthesis": 15,
    "mechanistic_plausibility": 10,
    "operational_falsifiability": 10,
    "unsupported_specificity": 10,
}

_QUANTITATIVE_PATTERN = re.compile(
    r"(?<![\w.])(?:[<>]=?|[≤≥~≈]\s*)?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|percent(?:age)?|µs|μs|us|microseconds?|ms|milliseconds?|s|seconds?|"
    r"minutes?|hours?|kbps|mbps|gbps|tbps|hz|khz|mhz|ghz|dbm?|watts?|w|joules?|j|"
    r"ttis?|intervals?|epochs?|iterations?|seeds?|runs?|repetitions?|users?|samples?|"
    r"packets?|frames?|prbs?|resource blocks?|accuracy|f1(?:[- ]score)?|auc|"
    r"fairness index|spectral efficiency|bits?/s/Hz)(?!\w)",
    flags=re.IGNORECASE,
)


def _normalize_quantitative_value(value: str) -> str:
    return re.sub(r"[\s,]+", "", value).casefold()


def _is_experimental_design_parameter(field_name: str, text: str, match: re.Match) -> bool:
    """Allow explicitly proposed experiment sizes, not predicted outcomes."""

    if field_name != "feasibility":
        return False
    window = text[max(0, match.start() - 80) : min(len(text), match.end() + 80)].casefold()
    design_markers = (
        "run ",
        "evaluate ",
        "simulate ",
        "collect ",
        "repeat ",
        "use ",
        "across ",
        "for the experiment",
        "design parameter",
        "non-inferiority margin",
    )
    outcome_markers = (
        "will improve",
        "will reduce",
        "will achieve",
        "remains stable",
        "outperform by",
        "increase by",
        "decrease by",
    )
    return any(marker in window for marker in design_markers) and not any(
        marker in window for marker in outcome_markers
    )


def classify_numeric_specificity(
    final_hypothesis: dict,
    retrieved_context: str,
    user_text: str = "",
) -> list[dict[str, str]]:
    """Classify quantitative specificity before the LLM audit verdict is trusted."""

    normalized_evidence = _normalize_quantitative_value(retrieved_context)
    normalized_user_text = _normalize_quantitative_value(user_text)
    classifications: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field_name in ("hypothesis", "rationale", "feasibility"):
        text = str(final_hypothesis.get(field_name, ""))
        for match in _QUANTITATIVE_PATTERN.finditer(text):
            value = match.group(0).strip()
            key = (field_name, value.casefold())
            if key in seen:
                continue
            seen.add(key)
            if _normalize_quantitative_value(value) in normalized_evidence:
                classification = "evidence_derived"
            elif normalized_user_text and _normalize_quantitative_value(value) in normalized_user_text:
                classification = "user_provided"
            elif _is_experimental_design_parameter(field_name, text, match):
                window = text[max(0, match.start() - 80) : match.end() + 80].casefold()
                classification = (
                    "derived_research_margin"
                    if "margin" in window
                    else "experimental_design_parameter"
                )
            else:
                classification = "unsupported"
            classifications.append(
                {
                    "value": value,
                    "field": field_name,
                    "classification": classification,
                }
            )
    return classifications


def _numeric_specificity_not_in_evidence(
    final_hypothesis: dict,
    retrieved_context: str,
    user_text: str = "",
) -> list[str]:
    """Return unsupported quantitative claims, excluding design parameters."""

    return list(
        dict.fromkeys(
            item["value"]
            for item in classify_numeric_specificity(final_hypothesis, retrieved_context, user_text)
            if item["classification"] == "unsupported"
        )
    )


def call_llm_for_hypothesis_audit(
    research_goal: str,
    hypotheses: list[dict],
    retrieved_context: str,
    available_source_ids: set[str],
    model: str | None = None,
    system_prompt: str | None = None,
    available_evidence_ref_ids: set[str] | None = None,
    available_evidence_refs: dict[str, dict] | None = None,
) -> tuple[list[dict] | None, str | None]:
    """Audit, revise, and gate generated hypotheses against retrieved evidence."""

    prompt = f"""
RESEARCH GOAL
{research_goal}

CANDIDATE HYPOTHESES
{json.dumps(hypotheses, ensure_ascii=False, indent=2)}

VERIFIED RETRIEVED SOURCES
{retrieved_context}

Audit every candidate and return one result for every zero-based candidate
index. Scores must evaluate the final_hypothesis after any revision, not the
original draft. A rejected item may use null for final_hypothesis.
""".strip()
    response = _call_llm(
        prompt,
        temperature=0.0,
        model=model,
        system_prompt=system_prompt,
    )
    if response.startswith("Error:"):
        return None, f"Hypothesis audit failed: {response}"

    def parse_payload(candidate_response: str) -> dict:
        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            candidate_response.strip(),
            flags=re.DOTALL | re.IGNORECASE,
        )
        payload = json.loads(fenced_match.group(1) if fenced_match else candidate_response.strip())
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    try:
        payload = parse_payload(response)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as first_error:
        logger.warning(
            "Hypothesis audit output was malformed; requesting one format-only repair: %s",
            first_error,
        )
        repair_prompt = (
            "Reformat the audit response below as valid JSON matching the original requested schema. "
            "Preserve all scientific content, verdicts, scores, source IDs, unsupported claims, and "
            "unsupported numbers exactly. Do not add or remove scientific conclusions. Return JSON only.\n\n"
            f"Candidate audit response:\n{response}"
        )
        repaired_response = _call_llm(
            repair_prompt,
            temperature=0.0,
            model=model,
            system_prompt=system_prompt,
        )
        if repaired_response.startswith("Error:"):
            return None, f"Hypothesis audit failed: {repaired_response}"
        try:
            payload = parse_payload(repaired_response)
            response = repaired_response
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            logger.error("Could not parse repaired hypothesis audit response: %s", repaired_response)
            return None, f"Hypothesis audit failed after format repair: {exc}"

    try:
        raw_audits = payload.get("audited_hypotheses")
        if not isinstance(raw_audits, list) or len(raw_audits) != len(hypotheses):
            raise ValueError("Expected one audited_hypotheses item per candidate.")

        audits_by_index: dict[int, dict] = {}
        for raw_audit in raw_audits:
            if not isinstance(raw_audit, dict):
                raise ValueError("Every audit item must be an object.")
            index = raw_audit.get("candidate_index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(hypotheses):
                raise ValueError("Invalid candidate_index in hypothesis audit.")
            if index in audits_by_index:
                raise ValueError("Duplicate candidate_index in hypothesis audit.")

            raw_scores = raw_audit.get("scores")
            if not isinstance(raw_scores, dict):
                raise ValueError("Hypothesis audit scores must be an object.")

            scores: dict[str, float] = {}
            for score_name in AUDIT_SCORE_WEIGHTS:
                score = raw_scores.get(score_name)
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 10:
                    raise ValueError(f"Invalid audit score: {score_name}.")
                scores[score_name] = float(score)
            if scores and max(scores.values()) <= 1:
                scores = {score_name: score * 10 for score_name, score in scores.items()}

            known_evidence_ref_ids = (
                set(available_evidence_refs)
                if available_evidence_refs is not None
                else available_evidence_ref_ids
            )
            claim_assessments: list[dict] = []
            raw_claim_assessments = raw_audit.get("claim_assessments", [])
            if isinstance(raw_claim_assessments, list):
                for claim_index, raw_claim in enumerate(raw_claim_assessments):
                    if not isinstance(raw_claim, dict):
                        continue
                    claim = str(raw_claim.get("claim", "")).strip()
                    status = str(raw_claim.get("support_status", "")).strip().casefold()
                    if not claim or status not in {
                        "entailed",
                        "partially_supported",
                        "unsupported",
                        "contradicted",
                    }:
                        continue
                    raw_chunk_ids = raw_claim.get("chunk_ids", [])
                    chunk_ids = [
                        str(chunk_id).strip()
                        for chunk_id in raw_chunk_ids
                        if isinstance(chunk_id, str)
                        and str(chunk_id).strip()
                        and (
                            known_evidence_ref_ids is None
                            or str(chunk_id).strip() in known_evidence_ref_ids
                        )
                    ] if isinstance(raw_chunk_ids, list) else []
                    chunk_ids = list(dict.fromkeys(chunk_ids))
                    raw_spans = raw_claim.get("evidence_spans", [])
                    evidence_spans = [
                        span.strip()
                        for span in raw_spans
                        if isinstance(span, str)
                        and span.strip()
                        and span.strip().casefold() in retrieved_context.casefold()
                    ] if isinstance(raw_spans, list) else []
                    source_id = _resolve_retrieved_source_id(
                        str(raw_claim.get("source_id", "")),
                        available_source_ids,
                    )
                    provenance = (
                        available_evidence_refs.get(chunk_ids[0], {})
                        if available_evidence_refs and chunk_ids
                        else {}
                    )
                    if source_id is None and provenance:
                        source_id = str(provenance.get("source_id", "")) or None
                    if status in {"entailed", "partially_supported"} and not chunk_ids:
                        status = "unsupported"
                    assessment = EvidenceClaim(
                        claim_id=str(raw_claim.get("claim_id") or f"claim_{claim_index + 1}"),
                        claim=claim,
                        support_status=status,
                        source_id=source_id,
                        chunk_ids=chunk_ids,
                        evidence_spans=evidence_spans,
                        section=str(provenance.get("section")) if provenance.get("section") else None,
                        page=int(provenance["page"]) if provenance.get("page") is not None else None,
                        evidence_type=(
                            str(provenance.get("evidence_type"))
                            if provenance.get("evidence_type") in {"full_text", "abstract_only"}
                            else None
                        ),
                        reason=str(raw_claim.get("reason", "")).strip(),
                    )
                    claim_assessments.append(assessment.model_dump())

            final_hypothesis = raw_audit.get("final_hypothesis")
            valid_final = None
            valid_source_ids: list[str] = []
            if isinstance(final_hypothesis, dict):
                required_fields = ("title", "hypothesis", "rationale", "feasibility")
                if all(
                    isinstance(final_hypothesis.get(field), str) and final_hypothesis[field].strip()
                    for field in required_fields
                ) and isinstance(final_hypothesis.get("source_ids"), list):
                    valid_source_ids = _resolve_retrieved_source_ids(
                        final_hypothesis["source_ids"],
                        available_source_ids,
                    )
                    if valid_source_ids:
                        valid_final = {field: final_hypothesis[field].strip() for field in required_fields}
                        valid_final["source_ids"] = valid_source_ids
                        raw_evidence_refs = final_hypothesis.get("evidence_refs", [])
                        valid_final["evidence_refs"] = [
                            str(value).strip()
                            for value in raw_evidence_refs
                            if isinstance(value, str)
                            and str(value).strip()
                            and (
                                known_evidence_ref_ids is None
                                or str(value).strip() in known_evidence_ref_ids
                            )
                        ] if isinstance(raw_evidence_refs, list) else []

            draft_unsupported_claims = tuple(
                dict.fromkeys(
                    value.strip()
                    for value in raw_audit.get(
                        "draft_unsupported_claims",
                        raw_audit.get("unsupported_claims", []),
                    )
                    if isinstance(value, str) and value.strip()
                )
            )
            draft_unsupported_numbers = list(
                dict.fromkeys(
                    value.strip()
                    for value in raw_audit.get(
                        "draft_unsupported_numbers",
                        raw_audit.get("unsupported_numbers", []),
                    )
                    if isinstance(value, str) and value.strip()
                )
            )
            unsupported_claims = tuple(
                dict.fromkeys(
                    value.strip()
                    for value in raw_audit.get("remaining_unsupported_claims", [])
                    if isinstance(value, str) and value.strip()
                )
            )
            unsupported_numbers = list(
                dict.fromkeys(
                    value.strip()
                    for value in raw_audit.get("remaining_unsupported_numbers", [])
                    if isinstance(value, str) and value.strip()
                )
            )
            numeric_grounding_enabled = bool(
                config.get("validation", {}).get("numeric_grounding_enabled", True)
            )
            if valid_final is not None and numeric_grounding_enabled:
                unsupported_numbers = list(
                    dict.fromkeys(
                        (
                            *unsupported_numbers,
                            *_numeric_specificity_not_in_evidence(
                                valid_final,
                                retrieved_context,
                                research_goal,
                            ),
                        )
                    )
                )

            closest_prior_art = []
            for item in raw_audit.get("closest_prior_art", []):
                if not isinstance(item, dict):
                    continue
                resolved_id = _resolve_retrieved_source_id(
                    str(item.get("source_id", "")),
                    available_source_ids,
                )
                if resolved_id is None:
                    continue
                closest_prior_art.append(
                    {
                        "source_id": resolved_id,
                        "overlap": str(item.get("overlap", "")).strip(),
                        "remaining_novelty": str(item.get("remaining_novelty", "")).strip(),
                    }
                )

            audit_warnings = []
            numeric_classifications = (
                classify_numeric_specificity(valid_final, retrieved_context, research_goal)
                if valid_final is not None and numeric_grounding_enabled
                else []
            )
            if unsupported_numbers:
                scores["unsupported_specificity"] = min(
                    scores["unsupported_specificity"],
                    5.0,
                )
                audit_warnings.append(
                    "The final hypothesis contains numeric specificity not found verbatim in the evidence."
                )

            weighted_score = round(
                sum(scores[name] * weight for name, weight in AUDIT_SCORE_WEIGHTS.items()) / 10,
                1,
            )
            hard_failures = []
            if valid_final is None:
                hard_failures.append("No valid final hypothesis with retrieved citations.")
            elif known_evidence_ref_ids is not None and not valid_final.get("evidence_refs"):
                hard_failures.append("No valid chunk-level evidence references.")
            if known_evidence_ref_ids is not None and not claim_assessments:
                hard_failures.append("No atomic claim assessments were provided.")
            non_entailed_claims = [
                assessment
                for assessment in claim_assessments
                if assessment["support_status"] != "entailed"
            ]
            if non_entailed_claims:
                hard_failures.append("One or more final factual claims are not entailed by their evidence chunks.")
            if scores["evidence_validity"] < 5:
                hard_failures.append("Evidence validity score is below 5/10.")
            if scores["claim_evidence_entailment"] < 5:
                hard_failures.append("Claim-evidence entailment score is below 5/10.")
            if scores["novelty_against_prior_art"] < 5:
                hard_failures.append("Novelty score is below 5/10.")
            if unsupported_claims:
                hard_failures.append("The final hypothesis contains unsupported claims.")
            if unsupported_numbers:
                hard_failures.append("The final hypothesis contains unsupported quantitative claims.")
            if weighted_score < 70:
                hard_failures.append("Weighted audit score is below 70/100.")
            if str(raw_audit.get("verdict", "")).strip().casefold() == "reject":
                hard_failures.append("The novelty auditor rejected the candidate.")

            audit_report = {
                "scores": scores,
                "weighted_score": weighted_score,
                "closest_prior_art": closest_prior_art,
                "draft_unsupported_claims": list(draft_unsupported_claims),
                "draft_unsupported_numbers": draft_unsupported_numbers,
                "unsupported_claims": list(unsupported_claims),
                "unsupported_numbers": unsupported_numbers,
                "numeric_classifications": numeric_classifications,
                "claim_assessments": claim_assessments,
                "warnings": audit_warnings,
                "revision_instruction": str(raw_audit.get("revision_instruction", "")).strip(),
                "verdict": "REJECT" if hard_failures else "PASS",
                "hard_failures": hard_failures,
            }
            audits_by_index[index] = {
                "candidate_index": index,
                "passed": not hard_failures,
                "final_hypothesis": valid_final,
                "audit_report": audit_report,
            }

        return [audits_by_index[index] for index in range(len(hypotheses))], None
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        logger.error("Could not parse hypothesis audit response: %s", response, exc_info=True)
        return None, f"Hypothesis audit failed: {exc}"


def call_llm_for_evidence_coverage(
    research_goal: str,
    explicit_requirements: tuple[EvidenceAspect, ...],
    retrieved_context: str,
    available_source_ids: set[str],
    model: str | None = None,
    max_gap_queries: int = 5,
) -> tuple[EvidenceCoverage | None, str | None]:
    """Check abstract/metadata discovery coverage before any PDF acquisition."""

    aspect_text = "\n".join(f"- {aspect.aspect_id}: {aspect.description}" for aspect in explicit_requirements)
    prompt = f"""
You are the discovery-coverage auditor for scientific literature retrieval.

This stage sees title, abstract, and metadata only. It decides whether the
candidate paper pool broadly covers the research goal and whether search needs
expansion. It does not certify detailed factual claims, methods, numerical
results, limitations, or claim entailment.

For each explicit requirement, identify exact retrieved Source IDs whose title
and abstract substantively support that requirement. Do not infer support from
the research goal itself. Do not add stricter subrequirements, metrics,
datasets, failure modes, or mechanisms that the user did not request. Mere
keyword mention is not support. A source may cover multiple requirements, and
multiple sources may collectively cover one requirement.

The research goal may ask whether one method improves on another or propose a
new causal relationship. Do not require retrieved literature to have already
performed that exact comparison, established the improvement, or proven the
new relationship. Those are legitimate knowledge gaps. Treat the evidence as
sufficient when the retrieved sources collectively ground the named methods,
domain, and existing findings needed to formulate the requested testable
hypotheses.

If any requirement is unsupported, provide 1 to {max_gap_queries} concise
arXiv search queries targeted specifically at the missing requirement. Include critical
domain, method, comparator, or outcome terms needed to avoid broad lexical
matches. Do not generate a hypothesis.

Return only valid JSON:
{{
  "aspect_coverage": [
    {{"aspect_id": "exact aspect ID", "source_ids": ["exact Source ID"]}}
  ],
  "gap_queries": ["targeted query for missing evidence"],
  "reason": "brief coverage assessment"
}}

Research goal:
{research_goal}

Explicit requirements:
{aspect_text}

Retrieved sources:
{retrieved_context}
""".strip()
    response = _call_llm(
        prompt,
        temperature=0.0,
        model=model,
    )
    if response.startswith("Error:"):
        return None, f"Evidence coverage grading failed: {response}"

    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        payload = json.loads(cleaned_response.strip())
        raw_coverage = payload.get("aspect_coverage")
        raw_gap_queries = payload.get("gap_queries")
        if not isinstance(raw_coverage, list) or not isinstance(
            raw_gap_queries,
            list,
        ):
            raise ValueError("Expected 'aspect_coverage' and 'gap_queries' arrays.")

        known_aspect_ids = {aspect.aspect_id for aspect in explicit_requirements}
        aspect_source_ids: dict[str, tuple[str, ...]] = {aspect_id: () for aspect_id in known_aspect_ids}
        for item in raw_coverage:
            if not isinstance(item, dict):
                continue
            aspect_id = str(item.get("aspect_id", "")).strip()
            raw_source_ids = item.get("source_ids")
            if aspect_id not in known_aspect_ids or not isinstance(raw_source_ids, list):
                continue
            valid_ids = _resolve_retrieved_source_ids(
                raw_source_ids,
                available_source_ids,
            )
            aspect_source_ids[aspect_id] = tuple(dict.fromkeys((*aspect_source_ids[aspect_id], *valid_ids)))

        missing_aspect_ids = tuple(
            aspect.aspect_id for aspect in explicit_requirements if not aspect_source_ids[aspect.aspect_id]
        )
        gap_queries = tuple(
            dict.fromkeys(query.strip() for query in raw_gap_queries if isinstance(query, str) and query.strip())
        )[:max_gap_queries]
        reason = str(payload.get("reason", "")).strip()

        coverage = EvidenceCoverage(
            aspect_source_ids=aspect_source_ids,
            missing_aspect_ids=missing_aspect_ids,
            gap_queries=gap_queries,
            reason=reason,
        )
        logger.info(
            "Discovery coverage sufficient=%s missing=%s map=%s reason=%s",
            coverage.sufficient,
            coverage.missing_aspect_ids,
            coverage.aspect_source_ids,
            coverage.reason,
        )
        return coverage, None
    except (json.JSONDecodeError, AttributeError, ValueError) as exc:
        logger.error(
            "Could not parse evidence coverage response: %s",
            response,
            exc_info=True,
        )
        return None, f"Evidence coverage grading failed: {exc}"


def build_evidence_queries(
    research_goal: str,
    explicit_requirements: tuple[EvidenceAspect, ...],
    *,
    max_queries: int | None = None,
) -> tuple[str, ...]:
    """Build focused full-text queries without another large planning call."""

    if max_queries is None:
        max_queries = max(
            1,
            int(config.get("evidence_retrieval", {}).get("max_queries", 8)),
        )
    queries = [research_goal.strip()]
    suffixes = (
        "method mechanism implementation",
        "experimental results metrics baseline comparison",
        "limitations failure cases assumptions",
    )
    for requirement in explicit_requirements:
        for suffix in suffixes:
            queries.append(f"{requirement.description} {suffix}".strip())
    return tuple(dict.fromkeys(query for query in queries if query))[:max_queries]


def call_llm_for_full_text_evidence_coverage(
    research_goal: str,
    explicit_requirements: tuple[EvidenceAspect, ...],
    retrieved_context: str,
    available_evidence_refs: dict[str, dict],
    model: str | None = None,
    max_gap_queries: int = 5,
) -> tuple[EvidenceCoverage | None, str | None]:
    """Validate requirements against exact retrieved passages after indexing."""

    aspect_text = "\n".join(
        f"- {aspect.aspect_id}: {aspect.description}" for aspect in explicit_requirements
    )
    prompt = f"""
You are the full-text evidence-coverage gate for scientific hypothesis generation.

This stage runs after candidate papers were filtered by title/abstract and after
accessible PDFs were parsed and searched. For every explicit requirement,
identify the exact evidence chunks that directly establish useful methods,
results, comparisons, assumptions, or limitations. Paper relevance alone is
not evidence. Do not treat shared vocabulary as entailment.

Abstract evidence has evidence_type=abstract_only. It may support only what its
text explicitly states and must not be used to infer absent methods,
limitations, experimental settings, or detailed results. Prefer full_text
chunks when both evidence modes are present.

Return only valid JSON:
{{
  "aspect_coverage": [
    {{
      "aspect_id": "exact aspect ID",
      "evidence_refs": [
        {{"source_id": "exact Source ID", "chunk_id": "exact chunk ID"}}
      ]
    }}
  ],
  "gap_queries": ["focused missing-evidence query"],
  "reason": "brief passage-level coverage assessment"
}}

Research goal:
{research_goal}

Explicit requirements:
{aspect_text}

Retrieved evidence:
{retrieved_context}
""".strip()
    response = _call_llm(prompt, temperature=0.0, model=model)
    if response.startswith("Error:"):
        return None, f"Full-text evidence coverage failed: {response}"

    try:
        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            response.strip(),
            flags=re.DOTALL | re.IGNORECASE,
        )
        payload = json.loads(fenced_match.group(1) if fenced_match else response.strip())
        raw_coverage = payload.get("aspect_coverage")
        raw_gap_queries = payload.get("gap_queries")
        if not isinstance(raw_coverage, list) or not isinstance(raw_gap_queries, list):
            raise ValueError("Expected 'aspect_coverage' and 'gap_queries' arrays.")

        known_aspect_ids = {aspect.aspect_id for aspect in explicit_requirements}
        aspect_evidence_refs: dict[str, tuple[dict, ...]] = {
            aspect_id: () for aspect_id in known_aspect_ids
        }
        aspect_source_ids: dict[str, tuple[str, ...]] = {
            aspect_id: () for aspect_id in known_aspect_ids
        }
        for item in raw_coverage:
            if not isinstance(item, dict):
                continue
            aspect_id = str(item.get("aspect_id", "")).strip()
            raw_refs = item.get("evidence_refs")
            if aspect_id not in known_aspect_ids or not isinstance(raw_refs, list):
                continue
            valid_refs: list[dict] = []
            for raw_ref in raw_refs:
                if not isinstance(raw_ref, dict):
                    continue
                chunk_id = str(raw_ref.get("chunk_id", "")).strip()
                source_id = str(raw_ref.get("source_id", "")).strip()
                known_ref = available_evidence_refs.get(chunk_id)
                if known_ref is None or str(known_ref.get("source_id", "")) != source_id:
                    continue
                valid_refs.append(dict(known_ref))
            deduplicated = {
                str(ref["chunk_id"]): ref for ref in valid_refs if ref.get("chunk_id")
            }
            ordered_refs = tuple(deduplicated.values())
            aspect_evidence_refs[aspect_id] = ordered_refs
            aspect_source_ids[aspect_id] = tuple(
                dict.fromkeys(str(ref["source_id"]) for ref in ordered_refs)
            )

        missing_aspect_ids = tuple(
            aspect.aspect_id
            for aspect in explicit_requirements
            if not aspect_evidence_refs[aspect.aspect_id]
        )
        gap_queries = tuple(
            dict.fromkeys(
                query.strip()
                for query in raw_gap_queries
                if isinstance(query, str) and query.strip()
            )
        )[:max_gap_queries]
        coverage = EvidenceCoverage(
            aspect_source_ids=aspect_source_ids,
            missing_aspect_ids=missing_aspect_ids,
            gap_queries=gap_queries,
            reason=str(payload.get("reason", "")).strip(),
            aspect_evidence_refs=aspect_evidence_refs,
            stage="full_text",
        )
        logger.info(
            "Full-text evidence coverage sufficient=%s missing=%s refs=%s",
            coverage.sufficient,
            coverage.missing_aspect_ids,
            {
                aspect_id: [ref.get("chunk_id") for ref in refs]
                for aspect_id, refs in coverage.aspect_evidence_refs.items()
            },
        )
        return coverage, None
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        logger.error("Could not parse full-text evidence coverage response: %s", response, exc_info=True)
        return None, f"Full-text evidence coverage failed: {exc}"


@dataclass(frozen=True)
class LiteratureFinding:
    """One source-grounded claim in the literature synthesis."""

    claim: str
    source_ids: tuple[str, ...]
    evidence_refs: tuple[dict, ...] = ()


@dataclass(frozen=True)
class LiteratureSynthesis:
    """Evidence summary and reasoning supplied to the Generation agent."""

    established_findings: tuple[LiteratureFinding, ...]
    contradictions: tuple[LiteratureFinding, ...]
    knowledge_gaps: tuple[str, ...]
    analytical_rationale: str


def call_llm_for_literature_synthesis(
    research_goal: str,
    explicit_requirements: tuple[EvidenceAspect, ...],
    exploration_directions: tuple[str, ...],
    retrieved_context: str,
    available_source_ids: set[str],
    model: str | None = None,
    available_evidence_refs: dict[str, dict] | None = None,
) -> tuple[LiteratureSynthesis | None, str | None]:
    """Build a citation-validated literature review before generation."""

    requirement_text = "\n".join(
        f"- {requirement.aspect_id}: {requirement.description}" for requirement in explicit_requirements
    )
    direction_text = "\n".join(f"- {direction}" for direction in exploration_directions)
    prompt = f"""
You are preparing the literature review and analytical rationale for a
scientific Generation agent.

Use only the retrieved sources below. Extract what the sources actually
establish, identify source-supported contradictions, and identify genuine
knowledge gaps. Do not treat the research goal or optional exploration
directions as evidence. Do not introduce facts, datasets, metrics, mechanisms,
or results absent from the retrieved text.

When full-text Results, Discussion, or Limitations evidence qualifies a claim
made in an abstract, preserve that qualification. Detailed full-text evidence
takes precedence over a stronger abstract-only interpretation. Never discard a
limitation merely because the abstract states a positive headline result.

The analytical rationale may connect established findings into promising
research directions, but it must clearly distinguish established evidence from
new inference. Optional exploration directions may guide analysis but are not
requirements and need not be present in the literature.

Return only valid JSON:
{{
  "established_findings": [
    {{
      "claim": "source-supported finding",
      "source_ids": ["exact Source ID"],
      "evidence_refs": [
        {{"source_id": "exact Source ID", "chunk_id": "exact chunk ID"}}
      ]
    }}
  ],
  "contradictions": [
    {{
      "claim": "source-supported contradiction",
      "source_ids": ["exact Source ID"],
      "evidence_refs": [
        {{"source_id": "exact Source ID", "chunk_id": "exact chunk ID"}}
      ]
    }}
  ],
  "knowledge_gaps": ["unresolved question supported by the review"],
  "analytical_rationale": "how the established findings motivate new, testable directions"
}}

Research goal:
{research_goal}

Explicit requirements:
{requirement_text}

Optional exploration directions:
{direction_text or "- None"}

Retrieved sources:
{retrieved_context}
""".strip()
    response = _call_llm(
        prompt,
        temperature=0.0,
        model=model,
    )
    if response.startswith("Error:"):
        return None, f"Literature synthesis failed: {response}"

    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        payload = json.loads(cleaned_response.strip())

        def validated_findings(field_name: str) -> tuple[LiteratureFinding, ...]:
            raw_findings = payload.get(field_name)
            if not isinstance(raw_findings, list):
                raise ValueError(f"Expected a '{field_name}' array.")
            findings: list[LiteratureFinding] = []
            for raw_finding in raw_findings:
                if not isinstance(raw_finding, dict):
                    continue
                claim = str(raw_finding.get("claim", "")).strip()
                raw_source_ids = raw_finding.get("source_ids")
                if not claim or not isinstance(raw_source_ids, list):
                    continue
                valid_source_ids = _resolve_retrieved_source_ids(
                    raw_source_ids,
                    available_source_ids,
                )
                evidence_refs: list[dict] = []
                raw_evidence_refs = raw_finding.get("evidence_refs", [])
                if isinstance(raw_evidence_refs, list) and available_evidence_refs:
                    for raw_ref in raw_evidence_refs:
                        if not isinstance(raw_ref, dict):
                            continue
                        chunk_id = str(raw_ref.get("chunk_id", "")).strip()
                        source_id = str(raw_ref.get("source_id", "")).strip()
                        known_ref = available_evidence_refs.get(chunk_id)
                        if known_ref is None or str(known_ref.get("source_id", "")) != source_id:
                            continue
                        evidence_refs.append(dict(known_ref))
                if not evidence_refs and available_evidence_refs:
                    for source_id in valid_source_ids:
                        abstract_ref = available_evidence_refs.get(f"abstract:{source_id}")
                        if abstract_ref is not None:
                            evidence_refs.append(dict(abstract_ref))
                evidence_refs = list(
                    {str(ref["chunk_id"]): ref for ref in evidence_refs}.values()
                )
                if evidence_refs:
                    valid_source_ids = list(
                        dict.fromkeys(str(ref["source_id"]) for ref in evidence_refs)
                    )
                # Once chunk provenance is available, paper-level citations by
                # themselves are not sufficient for an established finding.
                provenance_valid = not available_evidence_refs or bool(evidence_refs)
                if valid_source_ids and provenance_valid:
                    findings.append(
                        LiteratureFinding(
                            claim=claim,
                            source_ids=tuple(valid_source_ids),
                            evidence_refs=tuple(evidence_refs),
                        )
                    )
            return tuple(findings)

        established_findings = validated_findings("established_findings")
        contradictions = validated_findings("contradictions")
        raw_gaps = payload.get("knowledge_gaps")
        analytical_rationale = str(payload.get("analytical_rationale", "")).strip()
        if not isinstance(raw_gaps, list):
            raise ValueError("Expected a 'knowledge_gaps' array.")
        knowledge_gaps = tuple(dict.fromkeys(gap.strip() for gap in raw_gaps if isinstance(gap, str) and gap.strip()))
        if not established_findings:
            raise ValueError("No established finding cited a retrieved source.")
        if not analytical_rationale:
            raise ValueError("Expected a non-empty analytical rationale.")

        synthesis = LiteratureSynthesis(
            established_findings=established_findings,
            contradictions=contradictions,
            knowledge_gaps=knowledge_gaps,
            analytical_rationale=analytical_rationale,
        )
        logger.info(
            "Literature synthesis produced %d findings, %d contradictions, and %d gaps.",
            len(synthesis.established_findings),
            len(synthesis.contradictions),
            len(synthesis.knowledge_gaps),
        )
        return synthesis, None
    except (json.JSONDecodeError, AttributeError, ValueError) as exc:
        logger.error(
            "Could not parse literature synthesis response: %s",
            response,
            exc_info=True,
        )
        return None, f"Literature synthesis failed: {exc}"


def format_literature_synthesis(
    synthesis: LiteratureSynthesis,
) -> str:
    """Format the structured review for hypothesis generation prompts."""

    def format_finding(finding: LiteratureFinding) -> str:
        chunk_ids = [str(ref.get("chunk_id")) for ref in finding.evidence_refs if ref.get("chunk_id")]
        provenance = f" Evidence chunks: {', '.join(chunk_ids)}" if chunk_ids else ""
        return f"- {finding.claim} Sources: {', '.join(finding.source_ids)}{provenance}"

    findings = "\n".join(format_finding(finding) for finding in synthesis.established_findings)
    contradictions = "\n".join(format_finding(finding) for finding in synthesis.contradictions)
    gaps = "\n".join(f"- {gap}" for gap in synthesis.knowledge_gaps)
    return (
        f"Established findings:\n{findings}\n\n"
        f"Contradictions:\n{contradictions or '- None identified'}\n\n"
        f"Knowledge gaps:\n{gaps or '- None identified'}\n\n"
        "Analytical rationale:\n"
        f"{synthesis.analytical_rationale}"
    )


def call_llm_for_debate_refinement(
    prompt: str,
    num_hypotheses: int,
    temperature: float,
    model: str | None = None,
) -> tuple[list[Dict] | None, str | None]:
    """Refine a candidate set during one scientific-debate turn."""

    refined = call_llm_for_generation(
        prompt,
        num_hypotheses=num_hypotheses,
        temperature=temperature,
        model=model,
    )
    errors = [str(item.get("text", "Unknown debate error")) for item in refined if item.get("title") == "Error"]
    if errors:
        return None, f"Scientific debate refinement failed: {errors[0]}"
    if len(refined) != num_hypotheses:
        return None, (
            f"Scientific debate refinement failed: expected {num_hypotheses} hypotheses, received {len(refined)}."
        )
    return refined, None
