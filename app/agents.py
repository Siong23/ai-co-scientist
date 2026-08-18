"""Compatibility facade for the modular agent implementation.

Agent orchestration and helper implementations live in app.agents_modules.
This module keeps the historical import surface stable for the UI, tests, and
downstream callers.
"""

from .agents_modules.evolution import EvolutionAgent
from .agents_modules.evolution_helpers import (
    EVOLUTION_STRATEGIES,
    build_evolution_prompt,
    call_llm_for_evolution,
    combine_hypotheses,
    create_evolved_hypothesis,
    parse_evolution_response,
)
from .agents_modules.generation import GenerationAgent
from .agents_modules.generation_helpers import (
    EvidenceCoverage,
    FocusArea,
    LiteratureFinding,
    LiteratureSynthesis,
    _canonical_arxiv_id,  # noqa: F401
    _parse_generation_response,  # noqa: F401
    _resolve_retrieved_source_id,  # noqa: F401
    _resolve_retrieved_source_ids,  # noqa: F401
    call_llm_for_debate_refinement,
    call_llm_for_evidence_coverage,
    call_llm_for_focus_area_identification,
    call_llm_for_generation,
    call_llm_for_hypothesis_audit,
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
from .agents_modules.reflection_helpers import (
    call_llm_for_hypothesis_revision,
    call_llm_for_reflection,
)
from .agents_modules.supervisor import SupervisorAgent
from .agents_modules.supervisor_planner import (
    SUPERVISOR_ACTIONS,
    SupervisorAction,
    SupervisorDecision,
    SupervisorPlanner,
    assess_supervisor_state,
    build_supervisor_planning_prompt,
    decide_action_heuristically,
    parse_supervisor_decision,
)
from .evidence import EvidenceChunk, EvidenceDocument, EvidenceSource
from .models import ContextMemory, Hypothesis, ResearchGoal
from .rag_retriever import (
    ArxivRAGRetriever,
    EvidenceAspect,
    ProvisionalHypothesis,
    ResearchRetriever,
    SearchQuery,
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
    "EvidenceChunk",
    "EvidenceDocument",
    "EvidenceSource",
    "EvidenceCoverage",
    "EvolutionAgent",
    "EVOLUTION_STRATEGIES",
    "FocusArea",
    "GenerationAgent",
    "Hypothesis",
    "LiteratureFinding",
    "LiteratureSynthesis",
    "MetaReviewAgent",
    "ProximityAgent",
    "ProvisionalHypothesis",
    "RankingAgent",
    "ReflectionAgent",
    "ResearchGoal",
    "ResearchRetriever",
    "SUPERVISOR_ACTIONS",
    "SearchQuery",
    "SearchQueryPlan",
    "SupervisorAction",
    "SupervisorDecision",
    "SupervisorPlanner",
    "SupervisorAgent",
    "assess_supervisor_state",
    "build_supervisor_planning_prompt",
    "call_llm",
    "call_llm_for_debate_refinement",
    "call_llm_for_evidence_coverage",
    "call_llm_for_focus_area_identification",
    "call_llm_for_evolution",
    "call_llm_for_generation",
    "call_llm_for_hypothesis_audit",
    "call_llm_for_hypothesis_revision",
    "call_llm_for_literature_synthesis",
    "call_llm_for_relevance_filter",
    "call_llm_for_reflection",
    "call_llm_for_search_queries",
    "combine_hypotheses",
    "build_evolution_prompt",
    "create_evolved_hypothesis",
    "decide_action_heuristically",
    "format_documents_for_prompt",
    "format_literature_synthesis",
    "format_references",
    "generate_unique_id",
    "generate_visjs_data",
    "logger",
    "parse_pairwise_result",
    "parse_evolution_response",
    "parse_supervisor_decision",
    "redact_secrets",
    "run_pairwise_debate",
    "serialize_documents",
    "similarity_score",
    "update_elo",
    "update_elo_tie",
]
