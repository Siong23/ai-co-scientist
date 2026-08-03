import json
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List

# Import necessary components from other modules
from .agents_modules.evolution import EvolutionAgent
from .agents_modules.generation import GenerationAgent
from .agents_modules.meta_review import MetaReviewAgent
from .agents_modules.proximity import ProximityAgent
from .agents_modules.ranking import RankingAgent
from .agents_modules.reflection import ReflectionAgent
from .agents_modules.supervisor import SupervisorAgent
from .models import ContextMemory, Hypothesis, ResearchGoal
from .rag_retriever import (
    ArxivRAGRetriever,
    EvidenceAspect,
    SearchQueryPlan,
    format_documents_for_prompt,
    serialize_documents,
)
from .utils import (
    call_llm,
    generate_unique_id,
    generate_visjs_data,
    logger,  # Use the logger configured in utils
    similarity_score,
)

# --- Agent-Specific LLM Calls (Moved from main.py/utils.py for better cohesion) ---


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
        starts = [
            index
            for index in (cleaned.find("["), cleaned.find("{"))
            if index >= 0
        ]
        if not starts:
            raise ValueError("No JSON object or array was found.")
        hypotheses_data, _ = json.JSONDecoder().raw_decode(
            cleaned[min(starts) :]
        )

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
                {
                    aliases.get(key, key): value
                    for key, value in item.items()
                }
                if isinstance(item, dict)
                else item
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
        f"return an array containing exactly {num_hypotheses} objects. "
        "Every object must contain exactly these five keys: 'title', "
        "'hypothesis', 'rationale', 'feasibility', and 'source_ids'. The "
        "first four values must be strings and 'source_ids' must be an array "
        "of exact Source ID strings from the retrieved context. If the "
        "retrieved context is insufficient or not directly relevant, return "
        'exactly {"error": "The retrieved context is insufficient to generate '
        'grounded hypotheses."}.'
    )
    full_prompt = f"{prompt}\n\n{schema_instruction}"

    response = call_llm(full_prompt, temperature=temperature, model=model)
    logger.info("LLM generation response: %s", response)

    if response.startswith("Error:"):
        logger.error("LLM generation call failed: %s", response)
        return [{"title": "Error", "text": response}]

    try:
        return _parse_generation_response(response)
    except ValueError as first_error:
        logger.warning(
            "Initial generation output was not valid structured JSON; "
            "requesting one format-only repair: %s",
            first_error,
        )

    repair_prompt = (
        "Reformat the candidate response below to satisfy the JSON schema. "
        "Preserve its scientific meaning and exact source IDs. Do not add "
        "facts, citations, explanations, or Markdown. If a required field is "
        "absent and cannot be recovered, return the specified error object.\n\n"
        f"{schema_instruction}\n\nCandidate response:\n{response}"
    )
    repaired_response = call_llm(
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
) -> tuple[SearchQueryPlan | None, str | None]:
    """Create a goal-faithful plan with hard and optional search dimensions."""

    query_example = ", ".join(
        f'"query {index}"' for index in range(1, query_count + 1)
    )
    base_prompt = f"""
You are a goal-faithful scientific search planner. Decompose the user's
research goal before rewriting it into precise arXiv queries.

Generate exactly {query_count} distinct, concise search queries that cover
the goal's explicit requirements and useful optional exploration directions.
Remove request words such as "brief", "describe", and "explain". Preserve named
entities and add useful scientific synonyms where appropriate.

Return 1 to 5 explicit_requirements. Every requirement must include a
goal_quote copied verbatim from the user's goal. The quote itself becomes the
hard evidence requirement, so choose the shortest span that preserves the
user's meaning. Do not add datasets, metrics, populations, failure modes,
mechanisms, perturbations, or outcomes that the user did not request. Use short
stable snake_case IDs. Generic instructions such as "generate testable
hypotheses" are not evidence requirements.

Each goal_quote must contain at most 16 whitespace-separated words. Never put
an entire comparison, question, or causal claim into one requirement. For a
comparison goal, separately quote the focal method, comparator, domain, and
requested outcomes when they are explicitly present.

Return 0 to 5 exploration_directions. These may suggest useful search angles,
synonyms, or neighboring literature, but they are optional and must never
become evidence gates.

Also return required_terms containing only indispensable named entities,
locations, organisms, materials, diseases, technologies, or their close
synonyms. Do not use generic words such as "history", "study", or "model" as
required terms. A retrieved paper will be rejected unless its title or abstract
contains at least one required term.

Return only valid JSON with this exact shape:
{{
  "queries": [{query_example}],
  "required_terms": ["entity", "synonym"],
  "explicit_requirements": [
    {{"id": "short_id", "goal_quote": "verbatim words from the user's goal"}}
  ],
  "exploration_directions": ["optional search direction"]
}}

Research goal:
{research_goal}
""".strip()

    def parse_response(response: str) -> SearchQueryPlan:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]

        payload = json.loads(cleaned_response.strip())
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
                "Expected 'queries', 'required_terms', "
                "'explicit_requirements', and "
                "'exploration_directions' arrays."
            )

        normalized_queries = tuple(
            dict.fromkeys(query.strip() for query in queries if isinstance(query, str) and query.strip())
        )
        normalized_terms = tuple(
            dict.fromkeys(term.strip() for term in required_terms if isinstance(term, str) and term.strip())
        )
        if len(normalized_queries) != query_count:
            raise ValueError(f"Expected exactly {query_count} unique search queries.")
        if not normalized_terms:
            raise ValueError("Expected at least one required entity term.")

        explicit_requirements: list[EvidenceAspect] = []
        seen_aspect_ids: set[str] = set()
        for raw_aspect in raw_requirements:
            if not isinstance(raw_aspect, dict):
                continue
            aspect_id = str(raw_aspect.get("id", "")).strip()
            goal_quote = str(raw_aspect.get("goal_quote", "")).strip()
            normalized_goal = " ".join(
                research_goal.casefold().split()
            )
            normalized_quote = " ".join(goal_quote.casefold().split())
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]{1,39}", aspect_id)
                or not normalized_quote
                or normalized_quote not in normalized_goal
                or len(goal_quote.split()) > 16
                or aspect_id in seen_aspect_ids
            ):
                continue
            seen_aspect_ids.add(aspect_id)
            explicit_requirements.append(
                EvidenceAspect(
                    aspect_id=aspect_id,
                    description=goal_quote,
                )
            )
        if not 1 <= len(explicit_requirements) <= 5:
            raise ValueError(
                "Expected 1 to 5 unique explicit requirements with "
                "verbatim goal quotes."
            )
        exploration_directions = tuple(
            dict.fromkeys(
                direction.strip()
                for direction in raw_directions
                if isinstance(direction, str) and direction.strip()
            )
        )
        if len(exploration_directions) > 5:
            raise ValueError(
                "Expected no more than 5 exploration directions."
            )

        return SearchQueryPlan(
            queries=normalized_queries,
            required_terms=normalized_terms,
            explicit_requirements=tuple(explicit_requirements),
            exploration_directions=exploration_directions,
        )

    correction = ""
    for attempt in range(2):
        prompt = base_prompt + correction
        response = call_llm(
            prompt,
            temperature=0.0,
            model=model,
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
    """Resolve a model-emitted arXiv ID to one unique retrieved source."""

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
    """Suggest sources that directly support the goal without gating coverage."""

    aspect_text = "\n".join(
        f"- {aspect.aspect_id}: {aspect.description}"
        for aspect in explicit_requirements
    )
    prompt = f"""
You are a strict relevance grader for scientific literature retrieval.

Keep a retrieved source when it directly and substantively supports at least
one explicit requirement below. A source does not need to cover the entire research
goal by itself; the next stage checks collective coverage. Keyword overlap, a
shared country or entity name, or an incidental use of words such as "history"
is not enough. Exclude lexical collisions, analogies, and papers about a
different domain. Return every directly relevant source, not only the single
best match. Never include an irrelevant source merely to increase the number
of selected sources. If uncertain, exclude it.

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
    response = call_llm(
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
            "Evidence relevance grader selected %d/%d sources: %s",
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

    @property
    def sufficient(self) -> bool:
        return not self.missing_aspect_ids


def call_llm_for_evidence_coverage(
    research_goal: str,
    explicit_requirements: tuple[EvidenceAspect, ...],
    retrieved_context: str,
    available_source_ids: set[str],
    model: str | None = None,
    max_gap_queries: int = 5,
) -> tuple[EvidenceCoverage | None, str | None]:
    """Check collective evidence coverage and propose corrective searches."""

    aspect_text = "\n".join(
        f"- {aspect.aspect_id}: {aspect.description}"
        for aspect in explicit_requirements
    )
    prompt = f"""
You are an evidence-coverage auditor for scientific hypothesis generation.

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
    response = call_llm(
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
            raise ValueError(
                "Expected 'aspect_coverage' and 'gap_queries' arrays."
            )

        known_aspect_ids = {
            aspect.aspect_id for aspect in explicit_requirements
        }
        aspect_source_ids: dict[str, tuple[str, ...]] = {
            aspect_id: () for aspect_id in known_aspect_ids
        }
        for item in raw_coverage:
            if not isinstance(item, dict):
                continue
            aspect_id = str(item.get("aspect_id", "")).strip()
            raw_source_ids = item.get("source_ids")
            if (
                aspect_id not in known_aspect_ids
                or not isinstance(raw_source_ids, list)
            ):
                continue
            valid_ids = _resolve_retrieved_source_ids(
                raw_source_ids,
                available_source_ids,
            )
            aspect_source_ids[aspect_id] = tuple(
                dict.fromkeys(
                    (*aspect_source_ids[aspect_id], *valid_ids)
                )
            )

        missing_aspect_ids = tuple(
            aspect.aspect_id
            for aspect in explicit_requirements
            if not aspect_source_ids[aspect.aspect_id]
        )
        gap_queries = tuple(
            dict.fromkeys(
                query.strip()
                for query in raw_gap_queries
                if isinstance(query, str) and query.strip()
            )
        )[:max_gap_queries]
        reason = str(payload.get("reason", "")).strip()

        coverage = EvidenceCoverage(
            aspect_source_ids=aspect_source_ids,
            missing_aspect_ids=missing_aspect_ids,
            gap_queries=gap_queries,
            reason=reason,
        )
        logger.info(
            "Evidence coverage sufficient=%s missing=%s map=%s reason=%s",
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


@dataclass(frozen=True)
class LiteratureFinding:
    """One source-grounded claim in the literature synthesis."""

    claim: str
    source_ids: tuple[str, ...]


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
) -> tuple[LiteratureSynthesis | None, str | None]:
    """Build a citation-validated literature review before generation."""

    requirement_text = "\n".join(
        f"- {requirement.aspect_id}: {requirement.description}"
        for requirement in explicit_requirements
    )
    direction_text = "\n".join(
        f"- {direction}" for direction in exploration_directions
    )
    prompt = f"""
You are preparing the literature review and analytical rationale for a
scientific Generation agent.

Use only the retrieved sources below. Extract what the sources actually
establish, identify source-supported contradictions, and identify genuine
knowledge gaps. Do not treat the research goal or optional exploration
directions as evidence. Do not introduce facts, datasets, metrics, mechanisms,
or results absent from the retrieved text.

The analytical rationale may connect established findings into promising
research directions, but it must clearly distinguish established evidence from
new inference. Optional exploration directions may guide analysis but are not
requirements and need not be present in the literature.

Return only valid JSON:
{{
  "established_findings": [
    {{"claim": "source-supported finding", "source_ids": ["exact Source ID"]}}
  ],
  "contradictions": [
    {{"claim": "source-supported contradiction", "source_ids": ["exact Source ID"]}}
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
    response = call_llm(
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
                if valid_source_ids:
                    findings.append(
                        LiteratureFinding(
                            claim=claim,
                            source_ids=tuple(valid_source_ids),
                        )
                    )
            return tuple(findings)

        established_findings = validated_findings(
            "established_findings"
        )
        contradictions = validated_findings("contradictions")
        raw_gaps = payload.get("knowledge_gaps")
        analytical_rationale = str(
            payload.get("analytical_rationale", "")
        ).strip()
        if not isinstance(raw_gaps, list):
            raise ValueError("Expected a 'knowledge_gaps' array.")
        knowledge_gaps = tuple(
            dict.fromkeys(
                gap.strip()
                for gap in raw_gaps
                if isinstance(gap, str) and gap.strip()
            )
        )
        if not established_findings:
            raise ValueError(
                "No established finding cited a retrieved source."
            )
        if not analytical_rationale:
            raise ValueError("Expected a non-empty analytical rationale.")

        synthesis = LiteratureSynthesis(
            established_findings=established_findings,
            contradictions=contradictions,
            knowledge_gaps=knowledge_gaps,
            analytical_rationale=analytical_rationale,
        )
        logger.info(
            "Literature synthesis produced %d findings, %d "
            "contradictions, and %d gaps.",
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

    findings = "\n".join(
        f"- {finding.claim} Sources: {', '.join(finding.source_ids)}"
        for finding in synthesis.established_findings
    )
    contradictions = "\n".join(
        f"- {finding.claim} Sources: {', '.join(finding.source_ids)}"
        for finding in synthesis.contradictions
    )
    gaps = "\n".join(
        f"- {gap}" for gap in synthesis.knowledge_gaps
    )
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
    errors = [
        str(item.get("text", "Unknown debate error"))
        for item in refined
        if item.get("title") == "Error"
    ]
    if errors:
        return None, f"Scientific debate refinement failed: {errors[0]}"
    if len(refined) != num_hypotheses:
        return None, (
            "Scientific debate refinement failed: expected "
            f"{num_hypotheses} hypotheses, received {len(refined)}."
        )
    return refined, None


# Updated signature to accept temperature
def call_llm_for_reflection(hypothesis_text: str, temperature: float = 0.5, model: str | None = None) -> Dict:
    """Calls LLM for reviewing a hypothesis, handling JSON parsing."""
    logger.info("LLM reflection called with temperature: %.2f", temperature)
    prompt = (
        f"Review the following hypothesis and provide a novelty assessment (HIGH, MEDIUM, or LOW), "
        f"a feasibility assessment (HIGH, MEDIUM, or LOW), a comment, and a list of relevant references in JSON format:\n\n"
        f"Hypothesis: {hypothesis_text}\n\n"
        f"For references, provide arXiv IDs (e.g., '2301.12345'), DOIs, or paper titles with venues that are relevant to this hypothesis. "
        f"Do not provide PubMed IDs (PMIDs) unless this is specifically a biomedical/life sciences hypothesis.\n\n"
        f"Return the response as a JSON object with the following keys: 'novelty_review', 'feasibility_review', 'comment', 'references'."
    )
    # Pass the received temperature down to the actual LLM call
    response = call_llm(prompt, temperature=temperature, model=model)
    logger.info("LLM reflection response for hypothesis: %s", response)

    if response.startswith("Error:"):
        logger.error(f"LLM reflection call failed: {response}")
        return {
            "novelty_review": "Not reviewed",
            "feasibility_review": "Not reviewed",
            "comment": f"LLM review failed: {response}",
            "references": [],
        }

    # Default values
    review_data = {
        "novelty_review": "MEDIUM",
        "feasibility_review": "MEDIUM",
        "comment": "Could not parse LLM response.",
        "references": [],
    }

    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        parsed_data = json.loads(response)

        # Update defaults with parsed data, performing basic validation
        novelty = parsed_data.get("novelty_review", "MEDIUM").upper()
        if novelty in ["HIGH", "MEDIUM", "LOW"]:
            review_data["novelty_review"] = novelty
        else:
            logger.warning("Invalid novelty review value received: %s", novelty)

        feasibility = parsed_data.get("feasibility_review", "MEDIUM").upper()
        if feasibility in ["HIGH", "MEDIUM", "LOW"]:
            review_data["feasibility_review"] = feasibility
        else:
            logger.warning("Invalid feasibility review value received: %s", feasibility)

        review_data["comment"] = parsed_data.get("comment", "No comment provided.")
        # review_data["references"] = parsed_data.get("references", [])
        # if not isinstance(review_data["references"], list):
        #     logger.warning("Invalid references format received: %s", review_data["references"])
        #     review_data["references"] = []
        references = parsed_data.get("references", [])
        if isinstance(references, list):
            review_data["references"] = references
        else:
            logger.warning("Invalid references format received: %s", review_data["references"])
            review_data["references"] = []

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        logger.warning("Error parsing LLM reflection response: %s", response, exc_info=True)
        review_data["comment"] = f"Could not parse LLM response: {e}"  # Update comment with error

    logger.info("Parsed reflection data: %s", review_data)
    return review_data


# --- Ranking Helpers (Moved from main.py) ---


def format_references(references):
    if not references:
        return "No references provided."

    formatted = []

    for ref in references:
        if isinstance(ref, dict):
            title = ref.get("title", "Unknown title")
            authors = ref.get("authors", "")
            year = ref.get("year", "")

            formatted.append(f"{title} ({authors}, {year})")
        else:
            formatted.append(str(ref))

    return "\n".join(formatted)


def run_pairwise_debate(hypoA: Hypothesis, hypoB: Hypothesis, research_goal: ResearchGoal) -> tuple[Hypothesis, str]:
    """Compares two hypotheses based on novelty and feasibility scores."""

    # def score(h: Hypothesis) -> int:
    #     mapping = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, None: 0, "ERROR": 0}  # Handle ERROR case
    #     score_novelty = mapping.get(h.novelty_review, 0) if isinstance(h.novelty_review, str) else 0
    #     score_feasibility = mapping.get(h.feasibility_review, 0) if isinstance(h.feasibility_review, str) else 0
    #     return score_novelty + score_feasibility

    # scoreA = score(hypoA)
    # scoreB = score(hypoB)

    # if scoreA > scoreB:
    #     winner = hypoA
    # elif scoreB > scoreA:
    #     winner = hypoB
    # else:
    #     winner = random.choice([hypoA, hypoB])  # Tie-breaker

    # logger.info(
    #     "Debate: %s (score %d) vs %s (score %d) => Winner: %s",
    #     hypoA.hypothesis_id,
    #     scoreA,
    #     hypoB.hypothesis_id,
    #     scoreB,
    #     winner.hypothesis_id,
    # )
    # return winner

    reviewA = f"""
    Novelty Review:
    {hypoA.novelty_review}

    Feasibility Review:
    {hypoA.feasibility_review}

    Reviewer Comments:
    {chr(10).join(map(str, hypoA.review_comments))}

    References:
    {format_references(hypoA.references)}
    """

    reviewB = f"""
    Novelty Review:
    {hypoB.novelty_review}

    Feasibility Review:
    {hypoB.feasibility_review}

    Reviewer Comments:
    {chr(10).join(map(str, hypoB.review_comments))}

    References:
    {format_references(hypoB.references)}
    """

    # Format constraints into readable bullet points
    considerations = (
        "\n".join(f"- {k}: {v}" for k, v in research_goal.constraints.items()) if research_goal.constraints else "None"
    )

    prompt = f"""
    You are an expert evaluator tasked with comparing two hypotheses.

    Evaluate the two provided hypotheses (Hypothesis 1 and Hypothesis 2)
    and determine which one is superior based on the specified {research_goal.idea_attributes}.
    Provide a concise rationale for your selection, concluding with the phrase "better hypothesis: <1 or 2>".

    Goal:
    {research_goal.description}

    Evaluation Criteria:
    {research_goal.preferences}

    Considerations:
    {considerations}

    Each hypothesis includes an independent review.
    These reviews may contain numerical scores.
    Disregard these scores in your comparative analysis,
    as they may not be directly comparable across reviews.

    Hypothesis 1:
    {hypoA.text}

    Hypothesis 2:
    {hypoB.text}

    Review of Hypothesis 1:
    {reviewA}

    Review of Hypothesis 2:
    {reviewB}

    Reasoning and conclusion (end with "better hypothesis: <1 or 2>"): 
    """

    response = call_llm(
        prompt,
        temperature=0.2,
        model=research_goal.llm_model,
    )

    logger.info("Pairwise ranking response:\n%s", response)

    try:
        winner_index = parse_pairwise_result(response)

        winner = hypoA if winner_index == 1 else hypoB

    except ValueError:
        logger.warning(
            "Could not parse LLM ranking response.\n%s",
            response,
        )

        winner = random.choice([hypoA, hypoB])

    return winner, response

    # match = re.search(
    #     r"better\s+(?:idea|hypothesis)\s*:\s*([12])",
    #     response,
    #     re.IGNORECASE,
    # )

    # if match:
    #     winner = hypoA if match.group(1) == "1" else hypoB
    # else:
    #     logger.warning("Could not parse LLM ranking. Using random winner.")
    #     winner = random.choice([hypoA, hypoB])

    # return winner, response


def parse_pairwise_result(response: str) -> int:
    """
    Parse the LLM response and return the winning hypothesis index (1 or 2).
    Raises:
        ValueError: if no winner can be identified.
    """
    patterns = [
        r"better\s+(?:idea|hypothesis)\s*:\s*([12])",
        r"winner\s*:\s*([12])",
        r"better\s+(?:idea|hypothesis)\s+is\s+([12])",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))

    raise ValueError("Could not determine winner.")


def update_elo(winner: Hypothesis, loser: Hypothesis, k_factor: int):
    """Updates Elo scores after a comparison, using provided k_factor."""
    # k_factor is now passed as an argument
    ratingA = winner.elo_score
    ratingB = loser.elo_score
    expectedA = 1 / (1 + math.pow(10, (ratingB - ratingA) / 400))
    expectedB = 1 - expectedA  # Or 1 / (1 + math.pow(10, (ratingA - ratingB) / 400))
    winner.elo_score = ratingA + k_factor * (1 - expectedA)
    loser.elo_score = ratingB + k_factor * (0 - expectedB)  # Loser's score update
    logger.info(
        "Updated Elo: Winner %s -> %.2f, Loser %s -> %.2f",
        winner.hypothesis_id,
        winner.elo_score,
        loser.hypothesis_id,
        loser.elo_score,
    )


# --- Evolution Helper (Moved from main.py) ---


def combine_hypotheses(hypoA: Hypothesis, hypoB: Hypothesis) -> Hypothesis:
    """Combines two hypotheses into a new one."""
    new_id = generate_unique_id("E")  # Use utility function
    combined_title = f"Combined: {hypoA.title} & {hypoB.title}"
    # Consider a more sophisticated combination prompt/logic if needed
    combined_text = f"Combination of:\n1. {hypoA.text}\n2. {hypoB.text}"
    logger.info("Combining hypotheses %s and %s into %s", hypoA.hypothesis_id, hypoB.hypothesis_id, new_id)
    new_hypothesis = Hypothesis(new_id, combined_title, combined_text)
    new_hypothesis.parent_ids = [hypoA.hypothesis_id, hypoB.hypothesis_id]
    new_hypothesis.evidence_source_ids = list(
        dict.fromkeys(
            hypoA.evidence_source_ids + hypoB.evidence_source_ids
        )
    )
    return new_hypothesis


###############################################################################
# Agent Implementations
###############################################################################




__all__ = [
    "ArxivRAGRetriever",
    "ContextMemory",
    "EvidenceAspect",
    "EvolutionAgent",
    "GenerationAgent",
    "Hypothesis",
    "LiteratureFinding",
    "LiteratureSynthesis",
    "MetaReviewAgent",
    "ProximityAgent",
    "RankingAgent",
    "ReflectionAgent",
    "ResearchGoal",
    "SearchQueryPlan",
    "SupervisorAgent",
    "call_llm",
    "call_llm_for_debate_refinement",
    "call_llm_for_evidence_coverage",
    "call_llm_for_generation",
    "call_llm_for_literature_synthesis",
    "call_llm_for_relevance_filter",
    "call_llm_for_reflection",
    "call_llm_for_search_queries",
    "combine_hypotheses",
    "format_documents_for_prompt",
    "format_literature_synthesis",
    "format_references",
    "generate_unique_id",
    "generate_visjs_data",
    "logger",
    "parse_pairwise_result",
    "run_pairwise_debate",
    "serialize_documents",
    "similarity_score",
    "update_elo",
]


__all__ = [
    "ArxivRAGRetriever",
    "ContextMemory",
    "EvidenceAspect",
    "EvolutionAgent",
    "GenerationAgent",
    "Hypothesis",
    "LiteratureFinding",
    "LiteratureSynthesis",
    "MetaReviewAgent",
    "ProximityAgent",
    "RankingAgent",
    "ReflectionAgent",
    "ResearchGoal",
    "SearchQueryPlan",
    "SupervisorAgent",
    "call_llm",
    "call_llm_for_debate_refinement",
    "call_llm_for_evidence_coverage",
    "call_llm_for_generation",
    "call_llm_for_literature_synthesis",
    "call_llm_for_relevance_filter",
    "call_llm_for_reflection",
    "call_llm_for_search_queries",
    "combine_hypotheses",
    "format_documents_for_prompt",
    "format_literature_synthesis",
    "format_references",
    "generate_unique_id",
    "generate_visjs_data",
    "logger",
    "parse_pairwise_result",
    "run_pairwise_debate",
    "serialize_documents",
    "similarity_score",
    "update_elo",
]
