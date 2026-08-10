"""Hypothesis generation agent."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from ..config import config
from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..paper_library import ChromaPaperLibrary
from ..rag_retriever import (
    ArxivRAGRetriever,
    EvidenceAspect,
    SearchQueryPlan,
    format_documents_for_prompt,
    serialize_documents,
)
from ._compat import _legacy

RESEARCH_PLANNER_SYSTEM_PROMPT = """You are the Research Planning component of a research-oriented RAG system.

Your job is NOT to answer the user's question and NOT to generate web
search queries yet.

First, analyze the user's research goal and construct a concise research plan.

Determine:

1. What the user is ultimately trying to learn or decide.
2. What information is required to satisfy that goal.
3. Which sub-questions need to be investigated.
4. Which claims require current/external evidence.
5. Important entities, technologies, dates, constraints, or terminology.
6. Whether the question requires comparison, causal analysis,
   fact verification, discovery, or multi-hop research.
7. What evidence would constitute a satisfactory answer.
8. Any ambiguity that could materially affect the research.

Do not provide the final answer.
Do not generate search queries.
Do not expose private chain-of-thought.

Return only the following JSON:

{
  "research_goal": "...",
  "research_type": "...",
  "key_entities": [],
  "constraints": [],
  "sub_questions": [],
  "evidence_requirements": [],
  "freshness_requirement": "...",
  "ambiguities": [],
  "search_strategy": "..."
}"""


QUERY_REWRITER_SYSTEM_PROMPT = """You are the Web Search Query Rewriter in a research-oriented RAG system.

You receive:

1. The original user request.
2. A structured research plan produced by the Research Planner.

Your job is to generate high-quality web search queries that maximize
the probability of retrieving evidence needed to satisfy the research goal.

Do NOT answer the user's question.

For each research sub-question:

- Generate a focused search query.
- Prefer specific entities and technical terminology over vague language.
- Add date/version information when freshness matters.
- Generate separate queries when different evidence types are required.
- Avoid simply copying the user's original wording.
- Avoid overly long natural-language questions when keyword-oriented
  searches would retrieve better results.
- Prefer primary or authoritative sources when appropriate.
- Do not combine unrelated sub-questions into one query.

Return JSON:

{
  "queries": [
    {
      "query": "...",
      "purpose": "...",
      "sub_question": "...",
      "preferred_sources": [],
      "freshness": "..."
    }
  ]
}"""


HYPOTHESIS_AUDITOR_SYSTEM_PROMPT = """You are the Hypothesis Critic and Novelty Auditor in a research-oriented RAG system.

Your task is to make each generated hypothesis reliable before it leaves the
Generation Agent. Compare every candidate directly with the supplied retrieved
sources. Do not use outside knowledge and do not invent citations.

For each candidate:

1. Verify that every Source ID exists in the supplied evidence.
2. Check whether the cited sources actually entail each established statement
   in the rationale. A proposed relationship may remain an explicitly labeled
   hypothesis, but it must not be presented as an established fact.
   Decompose the rationale into atomic factual claims. Match the semantic
   subject, component, metric, setting, and relationship against exact evidence
   chunks. A number attached to a different component is unsupported even when
   the same number occurs elsewhere in the evidence.
3. Identify the closest retrieved prior art and determine whether the proposed
   contribution substantially duplicates it.
4. Judge whether the candidate synthesizes a genuine unresolved interaction
   across papers instead of merely combining keywords.
5. Require a clear, plausible intermediate mechanism from intervention to
   predicted outcome.
6. Require operational falsifiability: intervention, baseline, measurable
   outcome, and a result that would reject the hypothesis.
7. Remove unsupported precision. Exact percentages, thresholds, latencies, or
   performance improvements must occur in the retrieved evidence; otherwise
   replace them with non-fabricated measurable comparisons.

Revise a repairable candidate before scoring it. Scores must describe the
final revised version. Reject a candidate that cannot be repaired without
unsupported evidence or whose core novelty is already present in prior art.
Use a 0-to-10 scale for every score, where 10 is strongest. Do not use decimal
fractions on a 0-to-1 scale. The draft_unsupported fields record problems found
in the original candidate. The remaining_unsupported fields must describe only
problems still present in final_hypothesis after revision; return empty arrays
when the final version has fixed them.
Do not expose private chain-of-thought; provide concise audit findings only.

Return only valid JSON:

{
  "audited_hypotheses": [
    {
      "candidate_index": 0,
      "scores": {
        "evidence_validity": 0,
        "claim_evidence_entailment": 0,
        "novelty_against_prior_art": 0,
        "cross_paper_synthesis": 0,
        "mechanistic_plausibility": 0,
        "operational_falsifiability": 0,
        "unsupported_specificity": 0
      },
      "claim_assessments": [
        {
          "claim_id": "claim_1",
          "claim": "one atomic established factual statement from the final rationale",
          "source_id": "exact supplied Source ID",
          "chunk_ids": ["exact supplied chunk ID"],
          "evidence_spans": ["short exact span copied from the supplied evidence"],
          "support_status": "entailed | partially_supported | unsupported | contradicted",
          "reason": "brief subject/metric/setting comparison"
        }
      ],
      "closest_prior_art": [
        {
          "source_id": "exact supplied Source ID",
          "overlap": "concise overlap",
          "remaining_novelty": "concise unresolved contribution"
        }
      ],
      "draft_unsupported_claims": [],
      "draft_unsupported_numbers": [],
      "remaining_unsupported_claims": [],
      "remaining_unsupported_numbers": [],
      "verdict": "accept | revise | reject",
      "revision_instruction": "concise explanation",
      "final_hypothesis": {
        "title": "...",
        "hypothesis": "...",
        "rationale": "...",
        "feasibility": "include metric, baseline, and rejection criterion",
        "source_ids": ["exact supplied Source ID"],
        "evidence_refs": ["exact supplied chunk ID"]
      }
    }
  ]
}"""


class GenerationAgent:
    """Generate hypotheses grounded in multi-source academic retrieval."""

    def __init__(
        self,
        minimum_relevant_sources: int | None = None,
        corrective_retrieval_rounds: int | None = None,
        debate_rounds: int | None = None,
        audit_enabled: bool | None = None,
        paper_library: ChromaPaperLibrary | None = None,
    ) -> None:
        self.rag_retriever = ArxivRAGRetriever(
            minimum_relevant_sources=minimum_relevant_sources,
            corrective_retrieval_rounds=corrective_retrieval_rounds,
            generation_debate_rounds=debate_rounds,
        )
        self.debate_rounds = max(
            0,
            min(5, self.rag_retriever.generation_debate_rounds),
        )
        rag_config = config.get("rag", {})
        self.audit_enabled = (
            bool(rag_config.get("hypothesis_audit_enabled", False))
            if audit_enabled is None and debate_rounds is None
            else bool(audit_enabled)
        )
        self.paper_library = paper_library or ChromaPaperLibrary(embeddings=self.rag_retriever.embeddings)

    def _retrieve_scientific_sources(
        self,
        research_goal: ResearchGoal,
        query_plan: SearchQueryPlan,
        rerank_query: str | None = None,
    ):
        return self.rag_retriever.retrieve(
            rerank_query or research_goal.description,
            query_plan,
        )

    def _retrieve_original_scientific_sources(self, research_goal: ResearchGoal):
        """Run the first retrieval stage with the user's unmodified goal."""

        return self.rag_retriever.retrieve_original_goal(research_goal.description)

    def _enrich_with_full_text(
        self,
        documents,
        research_goal: ResearchGoal,
        query_plan: SearchQueryPlan | None = None,
    ):
        """Use relevant paper bodies when available without blocking generation."""

        try:
            evidence_queries = _legacy.build_evidence_queries(
                research_goal.description,
                query_plan.explicit_requirements if query_plan is not None else (),
            )
            return self.paper_library.enrich_documents(documents, evidence_queries)
        except Exception as exc:
            _legacy.logger.warning(
                "Paper download/vector indexing failed; continuing with abstracts: %s",
                _legacy.redact_secrets(str(exc)),
            )
            return list(documents)

    @staticmethod
    def _ensure_evidence_metadata(documents):
        """Give abstract fallbacks explicit evidence identities and modes."""

        normalized = []
        for document in documents:
            metadata = dict(document.metadata)
            source_id = str(metadata.get("source_id", ""))
            refs = metadata.get("evidence_refs")
            if not isinstance(refs, list) or not refs:
                refs = [
                    {
                        "source_id": source_id,
                        "chunk_id": f"abstract:{source_id}",
                        "section": "Abstract",
                        "page": None,
                        "evidence_type": "abstract_only",
                    }
                ]
                metadata["evidence_refs"] = refs
            metadata.setdefault("evidence_status", "abstract_only")
            metadata.setdefault("evidence_mode", "abstract_only")
            metadata.setdefault("full_text_indexed", False)
            metadata.setdefault("full_text_chunks_used", 0)
            normalized.append(type(document)(page_content=document.page_content, metadata=metadata))
        return normalized

    @staticmethod
    def _available_evidence_refs(documents) -> dict[str, dict]:
        refs: dict[str, dict] = {}
        for document in documents:
            for ref in document.metadata.get("evidence_refs", []):
                if isinstance(ref, dict) and ref.get("chunk_id"):
                    refs[str(ref["chunk_id"])] = dict(ref)
        return refs

    @staticmethod
    def _build_minimal_fallback_plan(research_goal: str) -> SearchQueryPlan:
        """Keep usable original evidence when LLM query planning fails."""

        normalized_goal = research_goal.strip()
        return SearchQueryPlan(
            queries=(normalized_goal,),
            required_terms=(),
            explicit_requirements=(
                EvidenceAspect(
                    aspect_id="goal_scope",
                    description=normalized_goal,
                ),
            ),
        )

    @staticmethod
    def _merge_retrieved_documents(*document_groups):
        """Merge retrieval rounds while deduplicating arXiv versions."""

        merged = []
        seen_ids: set[str] = set()
        for documents in document_groups:
            for document in documents:
                source_id = str(document.metadata.get("source_id", ""))
                canonical_id = re.sub(
                    r"v\d+$",
                    "",
                    source_id,
                    flags=re.IGNORECASE,
                )
                if not canonical_id or canonical_id in seen_ids:
                    continue
                seen_ids.add(canonical_id)
                merged.append(document)
        return merged

    def _run_scientific_debate(
        self,
        research_goal: ResearchGoal,
        query_plan: SearchQueryPlan,
        synthesis: _legacy.LiteratureSynthesis,
        hypotheses: list[Dict],
    ) -> list[Dict]:
        """Refine candidates through a short, stateful expert debate."""

        if self.debate_rounds == 0 or not hypotheses or any(item.get("title") == "Error" for item in hypotheses):
            return hypotheses

        roles = (
            "evidence and research-goal alignment reviewer",
            "skeptical methods and falsifiability reviewer",
            "integrating domain expert",
        )
        current_hypotheses = hypotheses
        synthesis_text = _legacy.format_literature_synthesis(synthesis)
        optional_directions = "\n".join(f"- {direction}" for direction in query_plan.exploration_directions)
        for round_index in range(self.debate_rounds):
            role = roles[round_index % len(roles)]
            debate_prompt = f"""
You are the {role} in turn {round_index + 1} of
{self.debate_rounds} of a simulated scientific debate.

Collaboratively refine the candidate hypotheses for the user's exact research
goal. Critically examine factual grounding, alignment, novelty, utility,
specificity, falsifiability, limitations, and practical feasibility. Remove or
rewrite unsupported factual premises. Preserve bold new inference when it is
clearly presented as a hypothesis rather than established fact.

The optional exploration directions below may inspire refinement but are not
requirements. Do not expand the user's goal or introduce new mandatory
datasets, metrics, mechanisms, populations, or outcomes.

Return exactly {len(current_hypotheses)} refined hypotheses. Keep each
hypothesis self-contained. Use only Source IDs present in the literature
review, and retain citations for every established premise.

Research goal:
{research_goal.description}

Constraints:
{research_goal.constraints}

Optional exploration directions:
{optional_directions or "- None"}

Literature review and analytical rationale:
{synthesis_text}

Candidate hypotheses from the preceding discussion:
{json.dumps(current_hypotheses, ensure_ascii=False)}

Your refined contribution:
""".strip()
            refined, debate_error = _legacy.call_llm_for_debate_refinement(
                debate_prompt,
                num_hypotheses=len(current_hypotheses),
                temperature=research_goal.generation_temperature,
                model=research_goal.llm_model,
            )
            if debate_error or refined is None:
                _legacy.logger.warning(
                    "Keeping the last valid hypotheses after debate round %d failed: %s",
                    round_index + 1,
                    debate_error,
                )
                break
            current_hypotheses = refined

        return current_hypotheses

    def generate_new_hypotheses(
        self,
        research_goal: ResearchGoal,
        context: ContextMemory,
    ) -> Tuple[List[Hypothesis], List[str]]:
        """Retrieve external evidence, then generate hypotheses."""

        num_to_generate = research_goal.num_hypotheses
        gen_temp = research_goal.generation_temperature
        query_plan, rewrite_error = _legacy.call_llm_for_search_queries(
            research_goal.description,
            model=getattr(research_goal, "query_rewrite_model", getattr(research_goal, "llm_model", None)),
            query_count=self.rag_retriever.query_count,
            research_planner_prompt=RESEARCH_PLANNER_SYSTEM_PROMPT,
            query_rewriter_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
        )

        # The structured research plan and rewritten queries are produced before
        # any normal web/literature search. The original goal remains a useful
        # fallback query when planning or rewriting is unavailable.
        try:
            candidate_documents = self._retrieve_original_scientific_sources(research_goal)
        except Exception as exc:
            _legacy.logger.error("Original-goal retrieval failed: %s", exc, exc_info=True)
            candidate_documents = []

        if (rewrite_error or query_plan is None) and not candidate_documents:
            context.last_retrieved_sources = []
            error = rewrite_error or "Query rewriting failed."
            _legacy.logger.error(error)
            return [], [error]
        if rewrite_error or query_plan is None:
            _legacy.logger.warning(
                "%s Continuing with %d original-goal candidate(s) and a minimal fallback plan.",
                rewrite_error or "Query rewriting failed.",
                len(candidate_documents),
            )
            query_plan = self._build_minimal_fallback_plan(research_goal.description)

        _legacy.logger.info(
            "Query rewriting produced queries=%s required_terms=%s explicit_requirements=%s exploration_directions=%s",
            query_plan.queries,
            query_plan.required_terms,
            query_plan.explicit_requirements,
            query_plan.exploration_directions,
        )

        expanded_retrieval_attempted = False
        if not candidate_documents:
            try:
                candidate_documents = self._retrieve_scientific_sources(research_goal, query_plan)
                expanded_retrieval_attempted = True
            except Exception as exc:
                _legacy.logger.error("Expanded RAG retrieval failed: %s", exc, exc_info=True)
                return [], [f"Expanded RAG retrieval failed: {exc}"]

        retrieved_documents = []
        shortlisted_documents = []
        coverage = None
        corrective_round = 0
        fallback_attempted = False
        while True:
            candidate_context = format_documents_for_prompt(candidate_documents)
            candidate_source_ids = {str(document.metadata["source_id"]) for document in candidate_documents}
            relevant_source_ids, relevance_error = _legacy.call_llm_for_relevance_filter(
                research_goal.description,
                candidate_context,
                candidate_source_ids,
                model=research_goal.llm_model,
                explicit_requirements=(query_plan.explicit_requirements),
            )
            if relevance_error or relevant_source_ids is None:
                _legacy.logger.warning(
                    "Abstract candidate filtering was unavailable; discovery coverage "
                    "will conservatively inspect all %d candidate source(s): %s",
                    len(candidate_documents),
                    relevance_error or "no relevance result",
                )
                shortlisted_documents = list(candidate_documents)
            else:
                selected_ids = set(relevant_source_ids)
                shortlisted_documents = [
                    document
                    for document in candidate_documents
                    if str(document.metadata.get("source_id", "")) in selected_ids
                ]
                _legacy.logger.info(
                    "Abstract candidate filter retained %d/%d sources: %s",
                    len(shortlisted_documents),
                    len(candidate_documents),
                    relevant_source_ids,
                )

            discovery_context = format_documents_for_prompt(shortlisted_documents)
            discovery_source_ids = {
                str(document.metadata["source_id"]) for document in shortlisted_documents
            }
            coverage, coverage_error = _legacy.call_llm_for_evidence_coverage(
                research_goal.description,
                query_plan.explicit_requirements,
                discovery_context,
                discovery_source_ids,
                model=research_goal.llm_model,
                max_gap_queries=self.rag_retriever.query_count,
            )
            if coverage_error or coverage is None:
                context.last_retrieved_sources = []
                error = coverage_error or "Evidence coverage grading failed."
                _legacy.logger.error(error)
                return [], [error]

            if coverage.sufficient:
                break

            if not expanded_retrieval_attempted:
                _legacy.logger.info("Original-goal retrieval was insufficient; starting expanded-query retrieval.")
                try:
                    expanded_documents = self._retrieve_scientific_sources(research_goal, query_plan)
                except Exception as exc:
                    _legacy.logger.error("Expanded RAG retrieval failed: %s", exc, exc_info=True)
                    return [], [f"Expanded RAG retrieval failed: {exc}"]
                expanded_retrieval_attempted = True
                candidate_documents = self._merge_retrieved_documents(candidate_documents, expanded_documents)
                continue

            if corrective_round >= (self.rag_retriever.corrective_retrieval_rounds):
                if not fallback_attempted:
                    fallback_attempted = True
                    missing_aspects = [
                        aspect
                        for aspect in query_plan.explicit_requirements
                        if aspect.aspect_id in coverage.missing_aspect_ids
                    ]
                    fallback_queries = tuple(
                        dict.fromkeys(
                            [
                                *coverage.gap_queries,
                                *(aspect.description for aspect in missing_aspects),
                            ]
                        )
                    )
                    fallback_plan = SearchQueryPlan(
                        queries=fallback_queries or query_plan.queries,
                        required_terms=(),
                        explicit_requirements=query_plan.explicit_requirements,
                        exploration_directions=query_plan.exploration_directions,
                    )
                    try:
                        fallback_documents = self.rag_retriever.retrieve_fallback(
                            research_goal.description,
                            fallback_plan,
                        )
                    except Exception as exc:
                        _legacy.logger.error(
                            "Supplementary search fallback failed: %s",
                            _legacy.redact_secrets(str(exc)),
                        )
                        fallback_documents = []
                    if fallback_documents:
                        candidate_documents = self._merge_retrieved_documents(
                            candidate_documents,
                            fallback_documents,
                        )
                        continue

                missing_descriptions = [
                    aspect.description.rstrip(".")
                    for aspect in query_plan.explicit_requirements
                    if aspect.aspect_id in coverage.missing_aspect_ids
                ]
                error = (
                    "Retrieved evidence is insufficient after "
                    f"{corrective_round} corrective retrieval "
                    "round(s) and supplementary-search fallback. "
                    "Missing explicit requirements: "
                    + "; ".join(missing_descriptions)
                    + ". Hypothesis generation was not executed."
                )
                _legacy.logger.error(error)
                context.last_retrieved_sources = []
                return [], [error]

            missing_aspects = [
                aspect for aspect in query_plan.explicit_requirements if aspect.aspect_id in coverage.missing_aspect_ids
            ]
            corrective_queries = tuple(
                dict.fromkeys(
                    [
                        *coverage.gap_queries,
                        *(aspect.description for aspect in missing_aspects),
                    ]
                )
            )
            gap_plan = SearchQueryPlan(
                queries=corrective_queries,
                # Gap queries are already targeted at a missing requirement.
                # Reusing the initial entity filter can discard comparator-only
                # or domain-only papers before the relevance grader sees them.
                required_terms=(),
                explicit_requirements=query_plan.explicit_requirements,
                exploration_directions=(query_plan.exploration_directions),
            )
            _legacy.logger.info(
                "Corrective retrieval round %d for missing explicit requirements=%s queries=%s",
                corrective_round + 1,
                coverage.missing_aspect_ids,
                corrective_queries,
            )
            try:
                gap_documents = self._retrieve_scientific_sources(
                    research_goal,
                    gap_plan,
                    rerank_query=" ".join(corrective_queries),
                )
            except Exception as exc:
                _legacy.logger.error(
                    "Corrective RAG retrieval failed: %s",
                    exc,
                    exc_info=True,
                )
                return [], [f"Corrective RAG retrieval failed: {exc}"]
            corrective_round += 1
            candidate_documents = self._merge_retrieved_documents(
                candidate_documents,
                gap_documents,
            )

        coverage_source_ids = {
            source_id for source_ids in coverage.aspect_source_ids.values() for source_id in source_ids
        }
        coverage_documents = [
            document
            for document in shortlisted_documents
            if str(document.metadata["source_id"]) in coverage_source_ids
        ]
        minimum_sources = self.rag_retriever.minimum_relevant_sources
        if len(coverage_documents) < minimum_sources:
            error = (
                f"Discovery coverage confirmed {len(coverage_documents)} "
                "supporting source(s), but at least "
                f"{minimum_sources} are required. Hypothesis generation "
                "was not executed."
            )
            _legacy.logger.error(error)
            context.last_retrieved_sources = []
            return [], [error]

        # Put discovery-coverage sources first, then use the remaining relevant
        # shortlist up to the configurable acquisition budget. Prompt size is
        # independently bounded by passage retrieval.
        prioritized_ids = [str(document.metadata["source_id"]) for document in coverage_documents]
        prioritized_ids.extend(
            str(document.metadata["source_id"])
            for document in shortlisted_documents
            if str(document.metadata["source_id"]) not in prioritized_ids
        )
        documents_by_id = {
            str(document.metadata["source_id"]): document for document in shortlisted_documents
        }
        retrieved_documents = [documents_by_id[source_id] for source_id in prioritized_ids]
        retrieved_documents = self._enrich_with_full_text(
            retrieved_documents,
            research_goal,
            query_plan,
        )
        retrieved_documents = self._ensure_evidence_metadata(retrieved_documents)
        retrieved_context = format_documents_for_prompt(retrieved_documents)
        available_evidence_refs = self._available_evidence_refs(retrieved_documents)

        has_full_text_evidence = any(
            ref.get("evidence_type") == "full_text"
            for ref in available_evidence_refs.values()
        )
        if has_full_text_evidence:
            full_text_coverage, full_text_coverage_error = (
                _legacy.call_llm_for_full_text_evidence_coverage(
                    research_goal.description,
                    query_plan.explicit_requirements,
                    retrieved_context,
                    available_evidence_refs,
                    model=research_goal.llm_model,
                    max_gap_queries=self.rag_retriever.query_count,
                )
            )
            if full_text_coverage_error or full_text_coverage is None:
                context.last_retrieved_sources = serialize_documents(retrieved_documents)
                error = full_text_coverage_error or "Full-text evidence coverage failed."
                _legacy.logger.error(error)
                return [], [error]
            if not full_text_coverage.sufficient:
                missing_descriptions = [
                    aspect.description
                    for aspect in query_plan.explicit_requirements
                    if aspect.aspect_id in full_text_coverage.missing_aspect_ids
                ]
                context.last_retrieved_sources = serialize_documents(retrieved_documents)
                return [], [
                    "Full-text evidence is insufficient for: "
                    + "; ".join(missing_descriptions)
                    + ". Hypothesis generation abstained."
                ]
            evidence_coverage = full_text_coverage
        else:
            abstract_refs = {
                aspect_id: tuple(
                    available_evidence_refs[f"abstract:{source_id}"]
                    for source_id in source_ids
                    if f"abstract:{source_id}" in available_evidence_refs
                )
                for aspect_id, source_ids in coverage.aspect_source_ids.items()
            }
            evidence_coverage = _legacy.EvidenceCoverage(
                aspect_source_ids=coverage.aspect_source_ids,
                missing_aspect_ids=coverage.missing_aspect_ids,
                gap_queries=coverage.gap_queries,
                reason="Abstract-only fallback: " + coverage.reason,
                aspect_evidence_refs=abstract_refs,
                stage="full_text_fallback",
            )

        context.last_retrieved_sources = serialize_documents(retrieved_documents)
        coverage_map = "\n".join(
            (
                f"- {aspect.description}: "
                + ", ".join(
                    str(ref.get("chunk_id"))
                    for ref in evidence_coverage.aspect_evidence_refs.get(aspect.aspect_id, ())
                )
            )
            for aspect in query_plan.explicit_requirements
        )

        allowed_source_ids = {str(document.metadata["source_id"]) for document in retrieved_documents}

        synthesis, synthesis_error = _legacy.call_llm_for_literature_synthesis(
            research_goal.description,
            query_plan.explicit_requirements,
            query_plan.exploration_directions,
            retrieved_context,
            allowed_source_ids,
            model=research_goal.llm_model,
            available_evidence_refs=available_evidence_refs,
        )
        if synthesis_error or synthesis is None:
            context.last_retrieved_sources = []
            error = synthesis_error or "Literature synthesis failed."
            _legacy.logger.error(error)
            return [], [error]
        synthesis_text = _legacy.format_literature_synthesis(synthesis)
        optional_directions = "\n".join(f"- {direction}" for direction in query_plan.exploration_directions)

        prompt = (
            "You are an expert tasked with formulating novel and robust "
            "scientific hypotheses for an audience of domain experts.\n\n"
            f"Goal:\n{research_goal.description}\n\n"
            "Criteria for a strong hypothesis:\n"
            "- Precisely align with the user's goal and constraints.\n"
            "- Be plausible, novel, specific, falsifiable, feasible, and "
            "safe.\n"
            "- Explicitly acknowledge relevant contradictions or "
            "limitations.\n\n"
            f"Constraints:\n{research_goal.constraints}\n\n"
            "Existing hypotheses to avoid duplicating:\n"
            f"{list(context.hypotheses.keys())}\n\n"
            "Explicit requirements validated against the literature:\n"
            f"{coverage_map}\n\n"
            "Optional exploration directions (inspiration only, not "
            "requirements):\n"
            f"{optional_directions or '- None'}\n\n"
            "Literature review and analytical rationale:\n"
            f"{synthesis_text}\n\n"
            "Retrieved articles available for citation:\n"
            f"{retrieved_context}\n\n"
            "Use the literature review as the factual foundation. Do not "
            "introduce factual claims, statistics, events, or established "
            "mechanisms absent from the retrieved evidence.\n"
            "A hypothesis may propose a new mechanism or outcome. Clearly "
            "label that part as new inference, and explain how it follows "
            "from established findings rather than presenting it as fact.\n"
            "If the evidence is insufficient or not directly relevant, "
            "do not generate hypotheses; return the specified error object.\n"
            f"Otherwise, propose up to {num_to_generate} concise, novel, feasible, "
            "specific, and experimentally testable hypotheses.\n"
            "Use this output structure for every item:\n"
            "- title: a short descriptive name.\n"
            "- hypothesis: one clear, testable claim.\n"
            "- rationale: why the claim follows from the retrieved evidence "
            "and why it matters.\n"
            "- feasibility: a concise practical method for testing the claim, "
            "including measurable outcomes where supported.\n"
            "- source_ids: the exact retrieved Source IDs supporting it.\n"
            "- evidence_refs: exact chunk IDs supporting established rationale statements.\n"
            "Return these six fields and no additional prose "
            "sections inside each item.\n"
            "Include only exact Source IDs present in the retrieved evidence. "
            "Do not invent Source IDs. Every hypothesis must cite the specific "
            "retrieved sources supporting it in source_ids; cite more than one "
            "source when the claim combines evidence from multiple papers.\n"
            "Every evidence_refs value must be an exact chunk_id present in the "
            "retrieved evidence. Abstract chunk IDs may support only statements "
            "explicitly present in that abstract.\n"
        )

        raw_output = _legacy.call_llm_for_generation(
            prompt,
            num_hypotheses=num_to_generate,
            temperature=gen_temp,
            model=research_goal.llm_model,
        )
        raw_output = self._run_scientific_debate(
            research_goal,
            query_plan,
            synthesis,
            raw_output,
        )

        context.last_hypothesis_audits = []
        if self.audit_enabled and raw_output:
            validation_config = config.get("validation", {})
            audit_context_limit = max(
                2000,
                int(validation_config.get("audit_max_evidence_chars", 9000)),
            )
            accepted_output: list[dict] = [
                candidate for candidate in raw_output if candidate.get("title") == "Error"
            ]
            verified_count = 0
            saw_candidate = False
            for candidate_index, candidate in enumerate(raw_output):
                if candidate.get("title") == "Error":
                    continue
                saw_candidate = True
                candidate_source_ids = _legacy._resolve_retrieved_source_ids(
                    candidate.get("source_ids", []),
                    allowed_source_ids,
                )
                candidate_documents = [
                    document
                    for document in retrieved_documents
                    if str(document.metadata.get("source_id")) in candidate_source_ids
                ] or retrieved_documents
                candidate_context = format_documents_for_prompt(candidate_documents)
                if len(candidate_context) > audit_context_limit:
                    candidate_context = (
                        candidate_context[:audit_context_limit]
                        + "\n[Evidence context truncated at the configured audit limit.]"
                    )
                audits, audit_error = _legacy.call_llm_for_hypothesis_audit(
                    research_goal.description,
                    [candidate],
                    candidate_context,
                    allowed_source_ids,
                    model=research_goal.llm_model,
                    system_prompt=HYPOTHESIS_AUDITOR_SYSTEM_PROMPT,
                    available_evidence_ref_ids=set(available_evidence_refs),
                    available_evidence_refs=available_evidence_refs,
                )
                if audit_error or not audits:
                    report = {
                        "candidate_index": candidate_index,
                        "scores": {},
                        "weighted_score": None,
                        "closest_prior_art": [],
                        "unsupported_claims": [],
                        "unsupported_numbers": [],
                        "claim_assessments": [],
                        "warnings": [audit_error or "Hypothesis audit failed."],
                        "revision_instruction": "",
                        "verdict": "UNVERIFIED",
                        "hard_failures": ["Audit output could not be verified."],
                    }
                    context.last_hypothesis_audits.append(report)
                    _legacy.logger.error(
                        "Hypothesis candidate %d is UNVERIFIED: %s",
                        candidate_index,
                        audit_error or "missing audit result",
                    )
                    continue

                audit = audits[0]
                audit["candidate_index"] = candidate_index
                audit["audit_report"]["candidate_index"] = candidate_index
                context.last_hypothesis_audits.append(audit["audit_report"])
                if not audit["passed"] or audit["final_hypothesis"] is None:
                    _legacy.logger.warning(
                        "Hypothesis candidate %d rejected by grounding audit: %s",
                        candidate_index,
                        audit["audit_report"]["hard_failures"],
                    )
                    continue
                accepted_output.append(
                    {
                        **audit["final_hypothesis"],
                        "_audit_report": audit["audit_report"],
                    }
                )
                verified_count += 1
            raw_output = accepted_output
            if saw_candidate and verified_count == 0:
                return [], [
                    "All generated hypotheses were rejected or left unverified by the grounding audit."
                ]

        if not raw_output:
            return [], [
                "Generation abstained because no sufficiently grounded hypothesis was produced."
            ]

        new_hypos: List[Hypothesis] = []
        errors: List[str] = []

        for idea in raw_output:
            if idea.get("title") == "Error":
                error_text = str(idea.get("text", "Unknown generation error"))
                _legacy.logger.error(
                    "Hypothesis generation failed: %s",
                    error_text,
                )
                errors.append(error_text)
                continue

            claimed_source_ids = idea.get(
                "source_ids",
                [],
            )

            if not isinstance(claimed_source_ids, list):
                claimed_source_ids = []

            valid_source_ids = _legacy._resolve_retrieved_source_ids(
                claimed_source_ids,
                allowed_source_ids,
            )

            # Reject hypotheses whose citations were not retrieved.
            if not valid_source_ids:
                error = f"Generated hypothesis has no valid retrieved source IDs: {idea.get('title', 'Untitled')}"
                _legacy.logger.warning(error)
                errors.append(error)
                continue

            hypo_id = _legacy.generate_unique_id("G")

            while hypo_id in context.hypotheses:
                hypo_id = _legacy.generate_unique_id("G")

            hypothesis = Hypothesis(
                hypo_id,
                str(idea["title"]).strip(),
                (
                    f"Hypothesis: {str(idea['hypothesis']).strip()}\n\n"
                    f"Rationale: {str(idea['rationale']).strip()}\n\n"
                    f"Feasibility: {str(idea['feasibility']).strip()}"
                ),
            )
            hypothesis.evidence_source_ids = valid_source_ids
            raw_evidence_refs = idea.get("evidence_refs", [])
            hypothesis.evidence_refs = [
                str(value).strip()
                for value in raw_evidence_refs
                if isinstance(value, str)
                and str(value).strip() in available_evidence_refs
            ] if isinstance(raw_evidence_refs, list) else []
            audit_report = idea.get("_audit_report")
            if isinstance(audit_report, dict):
                hypothesis.audit_report = audit_report
                hypothesis.audit_score = audit_report.get("weighted_score")
                hypothesis.audit_verdict = audit_report.get("verdict")

            _legacy.logger.info(
                "Generated RAG-grounded hypothesis: %s",
                hypothesis.to_dict(),
            )
            new_hypos.append(hypothesis)

        return new_hypos, errors
