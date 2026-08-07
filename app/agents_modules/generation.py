"""Hypothesis generation agent."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from ..models import ContextMemory, Hypothesis, ResearchGoal
from ..rag_retriever import (
    ArxivRAGRetriever,
    EvidenceAspect,
    SearchQueryPlan,
    format_documents_for_prompt,
    serialize_documents,
)
from ._compat import _legacy


class GenerationAgent:
    """Generate hypotheses grounded in multi-source academic retrieval."""

    def __init__(
        self,
        minimum_relevant_sources: int | None = None,
        corrective_retrieval_rounds: int | None = None,
        debate_rounds: int | None = None,
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
        try:
            candidate_documents = self._retrieve_original_scientific_sources(research_goal)
        except Exception as exc:
            _legacy.logger.error("Original-goal retrieval failed: %s", exc, exc_info=True)
            candidate_documents = []

        query_plan, rewrite_error = _legacy.call_llm_for_search_queries(
            research_goal.description,
            model=research_goal.llm_model,
            query_count=self.rag_retriever.query_count,
        )
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
                    "Evidence relevance grading was unavailable; coverage "
                    "will still audit all %d candidate source(s): %s",
                    len(candidate_documents),
                    relevance_error or "no relevance result",
                )
                relevant_source_ids = []
            else:
                _legacy.logger.info(
                    "RAG candidate count=%d relevance suggestions=%s",
                    len(candidate_documents),
                    relevant_source_ids,
                )

            coverage, coverage_error = _legacy.call_llm_for_evidence_coverage(
                research_goal.description,
                query_plan.explicit_requirements,
                candidate_context,
                candidate_source_ids,
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
        retrieved_documents = [
            document for document in candidate_documents if str(document.metadata["source_id"]) in coverage_source_ids
        ]
        minimum_sources = self.rag_retriever.minimum_relevant_sources
        if len(retrieved_documents) < minimum_sources:
            error = (
                f"RAG coverage auditing confirmed {len(retrieved_documents)} "
                "supporting arXiv source(s), but at least "
                f"{minimum_sources} are required. Hypothesis generation "
                "was not executed."
            )
            _legacy.logger.error(error)
            context.last_retrieved_sources = []
            return [], [error]

        context.last_retrieved_sources = serialize_documents(retrieved_documents)
        retrieved_context = format_documents_for_prompt(retrieved_documents)
        coverage_map = "\n".join(
            (f"- {aspect.description}: " + ", ".join(coverage.aspect_source_ids[aspect.aspect_id]))
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
            f"Otherwise, propose {num_to_generate} concise, novel, feasible, "
            "specific, and experimentally testable hypotheses.\n"
            "Use this output structure for every item:\n"
            "- title: a short descriptive name.\n"
            "- hypothesis: one clear, testable claim.\n"
            "- rationale: why the claim follows from the retrieved evidence "
            "and why it matters.\n"
            "- feasibility: a concise practical method for testing the claim, "
            "including measurable outcomes where supported.\n"
            "- source_ids: the exact retrieved Source IDs supporting it.\n"
            "Return exactly these five fields and no additional prose "
            "sections inside each item.\n"
            "Include only exact Source IDs present in the retrieved evidence. "
            "Do not invent Source IDs. Every hypothesis must cite the specific "
            "retrieved sources supporting it in source_ids; cite more than one "
            "source when the claim combines evidence from multiple papers.\n"
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

            _legacy.logger.info(
                "Generated RAG-grounded hypothesis: %s",
                hypothesis.to_dict(),
            )
            new_hypos.append(hypothesis)

        return new_hypos, errors
