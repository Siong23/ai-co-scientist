"""Compatibility facade for the modular agent implementation.

Agent orchestration and helper implementations live in app.agents_modules.
This module keeps the historical import surface stable for the UI, tests, and
downstream callers.
"""

from .agents_modules.evolution import EvolutionAgent
from .agents_modules.evolution_helpers import combine_hypotheses
from .agents_modules.generation import GenerationAgent
from .agents_modules.generation_helpers import (
    EvidenceCoverage,
    LiteratureFinding,
    LiteratureSynthesis,
    _canonical_arxiv_id,  # noqa: F401
    _parse_generation_response,  # noqa: F401
    _resolve_retrieved_source_id,  # noqa: F401
    _resolve_retrieved_source_ids,  # noqa: F401
    call_llm_for_debate_refinement,
    call_llm_for_evidence_coverage,
    call_llm_for_generation,
    call_llm_for_literature_synthesis,
    call_llm_for_relevance_filter,
    call_llm_for_search_queries,
    format_literature_synthesis,
)
from .agents_modules.meta_review import MetaReviewAgent
from .agents_modules.proximity import ProximityAgent
from .agents_modules.ranking import RankingAgent
from .agents_modules.ranking_helpers import (
    format_references,
    parse_pairwise_result,
    run_pairwise_debate,
    update_elo,
    update_elo_tie,
)
from .agents_modules.reflection import ReflectionAgent
from .agents_modules.reflection_helpers import call_llm_for_reflection
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
    logger,
    redact_secrets,
    similarity_score,
)

__all__ = [
    "ArxivRAGRetriever",
    "ContextMemory",
    "EvidenceAspect",
    "EvidenceCoverage",
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
    "redact_secrets",
    "run_pairwise_debate",
    "serialize_documents",
    "similarity_score",
    "update_elo",
    "update_elo_tie",
]
