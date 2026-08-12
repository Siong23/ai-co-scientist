import json
import threading
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from app.agents import (
    EvidenceCoverage,
    GenerationAgent,
    LiteratureFinding,
    LiteratureSynthesis,
    call_llm_for_evidence_coverage,
    call_llm_for_hypothesis_audit,
    call_llm_for_literature_synthesis,
    call_llm_for_relevance_filter,
    call_llm_for_search_queries,
    combine_hypotheses,
)
from app.agents_modules.generation_helpers import (
    ResearchActionDecision,
    call_llm_for_research_action,
)
from app.config import config
from app.models import ContextMemory, Hypothesis, ResearchGoal
from app.rag_retriever import (
    ArxivRAGRetriever,
    EvidenceAspect,
    SearchQuery,
    SearchQueryPlan,
    format_documents_for_grading,
    reciprocal_rank_fusion,
    serialize_documents,
)


@pytest.fixture(autouse=True)
def _disable_live_original_goal_search_for_generation_agent(monkeypatch):
    """Keep generation tests offline; source-stage behavior is tested separately."""

    monkeypatch.setattr(GenerationAgent, "_retrieve_original_scientific_sources", lambda *_: [])


def _paper(
    arxiv_id: str,
    title: str,
    abstract: str,
) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": ["Researcher"],
        "published": "2020-01-01",
        "primary_category": "econ.GN",
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def test_production_generation_defaults_enable_audit_and_agentic_research():
    agent = GenerationAgent()

    assert agent.audit_enabled is True
    assert agent.agentic_research_enabled is True
    assert GenerationAgent(agentic_research_enabled=False).agentic_research_enabled is False


def test_research_action_controller_can_find_in_known_web_page():
    coverage = EvidenceCoverage(
        aspect_source_ids={"goal_scope": ("web:docs",)},
        missing_aspect_ids=(),
        gap_queries=(),
        reason="Covered.",
    )
    synthesis = LiteratureSynthesis(
        established_findings=(),
        contradictions=(),
        knowledge_gaps=("The exact limit needs verification.",),
        analytical_rationale="Inspect the primary documentation.",
    )
    response = json.dumps(
        {
            "action": "FIND_IN_PAGE",
            "queries": [],
            "source_ids": ["web:docs", "web:invented", "arXiv:paper"],
            "target": "What exact context limit does the documentation state?",
            "reason": "The known official page should contain the exact value.",
        }
    )

    with patch("app.agents.call_llm", return_value=response) as mock_llm:
        decision, error = call_llm_for_research_action(
            "Verify the documented model context limit.",
            synthesis,
            coverage,
            [],
            available_sources=[
                {
                    "source_id": "web:docs",
                    "source_family": "web",
                    "document_type": "official_docs",
                    "title": "Model documentation",
                    "url": "https://docs.example.org/model",
                    "domain": "docs.example.org",
                },
                {
                    "source_id": "arXiv:paper",
                    "source_family": "academic",
                    "title": "A paper",
                },
            ],
        )

    assert error is None
    assert decision is not None
    assert decision.action == "FIND_IN_PAGE"
    assert decision.source_ids == ("web:docs",)
    assert decision.queries == ()
    controller_prompt = mock_llm.call_args.args[0]
    assert "OPEN_URL" in controller_prompt
    assert "SEARCH_PRIMARY_SOURCE" in controller_prompt
    assert "Search queries maximize discovery recall" in controller_prompt


def test_agentic_search_reranks_against_information_need_not_joined_queries():
    agent = GenerationAgent(agentic_research_enabled=True)
    coverage = EvidenceCoverage(
        aspect_source_ids={"goal_scope": ("arXiv:known",)},
        missing_aspect_ids=(),
        gap_queries=(),
        reason="Covered.",
    )
    synthesis = LiteratureSynthesis(
        established_findings=(),
        contradictions=(),
        knowledge_gaps=("The mechanism is unresolved.",),
        analytical_rationale="Target the unresolved mechanism.",
    )
    current_document = Document(
        page_content="Known evidence",
        metadata={
            "source_id": "arXiv:known",
            "source_type": "academic",
            "title": "Known evidence",
            "abstract": "Known evidence.",
        },
    )
    target = "What evidence shows hierarchical retrieval reduces long-context degradation?"
    decision = ResearchActionDecision(
        action="SEARCH",
        queries=(
            "hierarchical retrieval long context",
            "retrieval degradation benchmark",
            "Qwen context length",
        ),
        target=target,
        reason="Resolve the mechanism-specific evidence need.",
    )

    with (
        patch.object(agent, "_analyze_assumptions", return_value=[]),
        patch(
            "app.agents_modules.generation.call_llm_for_research_action",
            return_value=(decision, None),
        ),
        patch.object(
            agent,
            "_retrieve_scientific_sources",
            return_value=[],
        ) as mock_retrieve,
    ):
        agent._run_agentic_research(
            ResearchGoal("Study long-context degradation."),
            SearchQueryPlan(
                queries=("long-context retrieval",),
                required_terms=(),
                explicit_requirements=(EvidenceAspect("goal_scope", "long-context degradation"),),
            ),
            coverage,
            [current_document],
            synthesis,
        )

    assert mock_retrieve.call_args.kwargs["rerank_query"] == target
    action_plan = mock_retrieve.call_args.args[1]
    assert action_plan.query_texts == decision.queries
    assert all(query.sub_question == target for query in action_plan.queries)


def _query_plan_payload(
    goal: str = "brief describe the malaysia history",
    requirements: list[dict] | None = None,
    query_count: int = 5,
) -> str:
    if requirements is None:
        requirements = [
            {
                "id": "goal_scope",
                "goal_quote": goal.strip(),
            }
        ]
    return json.dumps(
        {
            "queries": [
                "Malaysia colonial history",
                "British Malaya decolonization",
                "Malaysia independence history",
                "Malaysia post-independence development",
                "Malaya political economic history",
                "Malaysia social history",
                "Malaysia economic development history",
                "Malaysia political institutions history",
            ][:query_count],
            "required_terms": ["Malaysia", "Malaya"],
            "explicit_requirements": requirements,
            "exploration_directions": ["Compare alternative historical interpretations."],
        }
    )


def _research_plan_payload(goal: str = "brief describe the malaysia history") -> str:
    return json.dumps(
        {
            "research_goal": goal,
            "research_type": "discovery and synthesis",
            "key_entities": ["Malaysia", "Malaya"],
            "constraints": [],
            "sub_questions": ["What evidence describes Malaysia's history?"],
            "evidence_requirements": ["Authoritative historical sources"],
            "freshness_requirement": "No special freshness requirement",
            "ambiguities": [],
            "search_strategy": "Search primary and scholarly historical sources.",
        }
    )


def _relevance_payload(*source_ids: str) -> str:
    return json.dumps(
        {
            "relevant_source_ids": list(source_ids),
            "reason": "Selected sources directly support the goal.",
        }
    )


def _coverage_payload(
    *source_ids: str,
    gap_queries: tuple[str, ...] = (),
    aspect_ids: tuple[str, ...] = ("goal_scope",),
) -> str:
    return json.dumps(
        {
            "aspect_coverage": [
                {
                    "aspect_id": aspect_id,
                    "source_ids": list(source_ids),
                }
                for aspect_id in aspect_ids
            ],
            "gap_queries": list(gap_queries),
            "reason": "Coverage assessment.",
        }
    )


def _synthesis_payload(*source_ids: str) -> str:
    return json.dumps(
        {
            "established_findings": [
                {
                    "claim": "Retrieved evidence establishes the factual premise.",
                    "source_ids": list(source_ids),
                }
            ],
            "contradictions": [],
            "knowledge_gaps": ["The proposed relationship remains untested."],
            "analytical_rationale": ("The established premise motivates a new testable inference."),
        }
    )


def test_query_rewriting_uses_selected_model_and_zero_temperature():
    with patch(
        "app.agents.call_llm",
        side_effect=[
            _research_plan_payload(),
            _query_plan_payload(query_count=5),
        ],
    ) as mock_call:
        plan, error = call_llm_for_search_queries(
            "brief describe the malaysia history",
            model="chosen-model",
        )

    assert error is None
    assert plan is not None
    assert len(plan.queries) == 5
    assert plan.required_terms == ("Malaysia", "Malaya")
    assert [aspect.aspect_id for aspect in plan.explicit_requirements] == ["goal_scope"]
    assert [aspect.description for aspect in plan.explicit_requirements] == ["brief describe the malaysia history"]
    assert plan.exploration_directions == ("Compare alternative historical interpretations.",)
    assert mock_call.call_count == 2
    assert all(call.kwargs["temperature"] == 0.0 for call in mock_call.call_args_list)
    assert all(call.kwargs["model"] == "chosen-model" for call in mock_call.call_args_list)
    planner_prompt = mock_call.call_args_list[0].args[0]
    normalized_planner_prompt = " ".join(planner_prompt.split())
    assert "brief describe the malaysia history" in normalized_planner_prompt
    planner_system_prompt = mock_call.call_args_list[0].kwargs["system_prompt"]
    assert "Research Planning component" in planner_system_prompt
    assert "Do not generate search queries" in planner_system_prompt

    rewriter_prompt = " ".join(mock_call.call_args_list[1].args[0].split())
    assert "STRUCTURED RESEARCH PLAN" in rewriter_prompt
    assert "discovery and synthesis" in rewriter_prompt
    rewriter_system_prompt = mock_call.call_args_list[1].kwargs["system_prompt"]
    assert "Search Planner" in rewriter_system_prompt
    assert "goal_quote copied verbatim" in rewriter_system_prompt
    assert "must never become evidence gates" in " ".join(rewriter_system_prompt.split())


def test_query_rewriter_accepts_structured_query_objects():
    rewritten = json.loads(_query_plan_payload(query_count=5))
    rewritten["queries"] = [
        {
            "query": query,
            "purpose": "Find supporting evidence",
            "sub_question": "What evidence addresses the goal?",
            "source_type": "official",
            "preferred_domains": ["https://Example.org/docs"],
            "freshness": "year",
            "evidence_requirement_id": "goal_scope",
        }
        for query in rewritten["queries"]
    ]

    with patch(
        "app.agents.call_llm",
        side_effect=[_research_plan_payload(), json.dumps(rewritten)],
    ):
        plan, error = call_llm_for_search_queries(
            "brief describe the malaysia history",
            query_count=5,
        )

    assert error is None
    assert plan is not None
    assert plan.queries[0] == SearchQuery(
        query="Malaysia colonial history",
        purpose="Find supporting evidence",
        sub_question="What evidence addresses the goal?",
        source_type="official",
        preferred_domains=("example.org",),
        freshness="year",
        evidence_requirement_id="goal_scope",
    )
    assert len(plan.queries) == 5


def test_query_rewriter_uses_literature_oriented_evidence_needs():
    goal = (
        "Develop an AI-driven resource allocation framework to reduce latency "
        "and improve throughput in dense 5G networks."
    )
    requirements = [
        {
            "id": "resource_allocation",
            "goal_quote": "AI-driven resource allocation framework",
            "evidence_need": "AI-driven resource allocation methods for dense 5G networks",
        },
        {
            "id": "network_outcomes",
            "goal_quote": "reduce latency and improve throughput",
            "evidence_need": "latency and throughput outcomes in dense 5G networks",
        },
    ]

    with patch(
        "app.agents.call_llm",
        side_effect=[
            _research_plan_payload(goal),
            _query_plan_payload(goal, requirements=requirements, query_count=5),
        ],
    ):
        plan, error = call_llm_for_search_queries(goal, query_count=5)

    assert error is None
    assert plan is not None
    assert [aspect.description for aspect in plan.explicit_requirements] == [
        "AI-driven resource allocation methods for dense 5G networks",
        "latency and throughput outcomes in dense 5G networks",
    ]


def _audit_payload(
    final_hypothesis: dict | None,
    *,
    scores: dict | None = None,
    unsupported_claims: list[str] | None = None,
    unsupported_numbers: list[str] | None = None,
    verdict: str = "accept",
) -> str:
    return json.dumps(
        {
            "audited_hypotheses": [
                {
                    "candidate_index": 0,
                    "scores": scores
                    or {
                        "evidence_validity": 8,
                        "claim_evidence_entailment": 8,
                        "novelty_against_prior_art": 8,
                        "cross_paper_synthesis": 8,
                        "mechanistic_plausibility": 8,
                        "operational_falsifiability": 8,
                        "unsupported_specificity": 8,
                    },
                    "closest_prior_art": [
                        {
                            "source_id": "arXiv:1111.1111",
                            "overlap": "Shared baseline.",
                            "remaining_novelty": "New interaction.",
                        }
                    ],
                    "unsupported_claims": unsupported_claims or [],
                    "unsupported_numbers": unsupported_numbers or [],
                    "verdict": verdict,
                    "revision_instruction": "Keep only the supported novelty.",
                    "final_hypothesis": final_hypothesis,
                }
            ]
        }
    )


def test_hypothesis_auditor_revises_and_passes_a_grounded_candidate():
    final_hypothesis = {
        "title": "Audited hypothesis",
        "hypothesis": "Method A will outperform the baseline under congestion.",
        "rationale": "The cited source grounds the baseline limitation.",
        "feasibility": "Compare Method A with the baseline and reject the claim if latency is not lower.",
        "source_ids": ["1111.1111v2"],
    }
    with patch(
        "app.agents.call_llm",
        return_value=_audit_payload(
            final_hypothesis,
            scores={
                "evidence_validity": 0.8,
                "claim_evidence_entailment": 0.8,
                "novelty_against_prior_art": 0.8,
                "cross_paper_synthesis": 0.8,
                "mechanistic_plausibility": 0.8,
                "operational_falsifiability": 0.8,
                "unsupported_specificity": 0.8,
            },
            unsupported_claims=["Unsupported wording in the original draft."],
            unsupported_numbers=["10% in the original draft"],
            verdict="revise",
        ),
    ) as mock_call:
        audits, error = call_llm_for_hypothesis_audit(
            "Improve network performance.",
            [{"title": "Draft"}],
            "Source ID: arXiv:1111.1111\nTitle: Prior work\nAbstract: Baseline limitation.",
            {"arXiv:1111.1111"},
            model="audit-model",
            system_prompt="auditor system prompt",
        )

    assert error is None
    assert audits is not None
    assert audits[0]["passed"] is True
    assert audits[0]["final_hypothesis"]["source_ids"] == ["arXiv:1111.1111"]
    assert audits[0]["audit_report"]["weighted_score"] == 80.0
    assert audits[0]["audit_report"]["draft_unsupported_claims"]
    assert audits[0]["audit_report"]["unsupported_claims"] == []
    assert mock_call.call_args.kwargs == {
        "temperature": 0.0,
        "model": "audit-model",
        "system_prompt": "auditor system prompt",
        "max_tokens": 3072,
        "reasoning": "off",
    }


def test_balanced_hypothesis_auditor_keeps_numeric_target_with_warning():
    final_hypothesis = {
        "title": "Invented precision",
        "hypothesis": "The method will improve throughput by 10%.",
        "rationale": "The source describes the method without that number.",
        "feasibility": "Compare it with a baseline and measure throughput.",
        "source_ids": ["arXiv:1111.1111"],
    }
    with patch("app.agents.call_llm", return_value=_audit_payload(final_hypothesis)):
        audits, error = call_llm_for_hypothesis_audit(
            "Improve network performance.",
            [{"title": "Draft"}],
            "Source ID: arXiv:1111.1111\nAbstract: The method improves throughput.",
            {"arXiv:1111.1111"},
        )

    assert error is None
    assert audits is not None
    assert audits[0]["passed"] is True
    assert audits[0]["audit_report"]["unsupported_numbers"] == ["10%"]
    assert audits[0]["audit_report"]["scores"]["unsupported_specificity"] == 5.0
    assert audits[0]["audit_report"]["warnings"]
    assert audits[0]["audit_report"]["hard_failures"] == []
    assert "proposed experimental targets" in " ".join(
        audits[0]["audit_report"]["warnings"]
    )


def test_strict_hypothesis_auditor_rejects_unsupported_numeric_target(monkeypatch):
    monkeypatch.setitem(config["rag"], "hypothesis_audit_mode", "strict")
    final_hypothesis = {
        "title": "Invented precision",
        "hypothesis": "The method will improve throughput by 10%.",
        "rationale": "The source describes the method without that number.",
        "feasibility": "Compare it with a baseline and measure throughput.",
        "source_ids": ["arXiv:1111.1111"],
    }

    with patch("app.agents.call_llm", return_value=_audit_payload(final_hypothesis)):
        audits, error = call_llm_for_hypothesis_audit(
            "Improve network performance.",
            [{"title": "Draft"}],
            "Source ID: arXiv:1111.1111\nAbstract: The method improves throughput.",
            {"arXiv:1111.1111"},
        )

    assert error is None
    assert audits is not None
    assert audits[0]["passed"] is False
    assert audits[0]["audit_report"]["mode"] == "strict"
    assert "unsupported numerical claims" in " ".join(
        audits[0]["audit_report"]["hard_failures"]
    )


def test_hypothesis_auditor_treats_quality_score_and_model_verdict_as_advisory():
    final_hypothesis = {
        "title": "Grounded but exploratory",
        "hypothesis": "The method may outperform the baseline under congestion.",
        "rationale": "The source establishes the baseline limitation; the improvement remains a hypothesis.",
        "feasibility": "Compare both methods and reject the hypothesis if latency is not lower.",
        "source_ids": ["arXiv:1111.1111"],
    }
    scores = {
        "evidence_validity": 6,
        "claim_evidence_entailment": 6,
        "novelty_against_prior_art": 4,
        "cross_paper_synthesis": 6,
        "mechanistic_plausibility": 6,
        "operational_falsifiability": 6,
        "unsupported_specificity": 6,
    }

    with patch(
        "app.agents.call_llm",
        return_value=_audit_payload(
            final_hypothesis,
            scores=scores,
            verdict="reject",
        ),
    ):
        audits, error = call_llm_for_hypothesis_audit(
            "Improve network performance.",
            [{"title": "Draft"}],
            "Source ID: arXiv:1111.1111\nAbstract: The baseline has a documented limitation.",
            {"arXiv:1111.1111"},
        )

    assert error is None
    assert audits is not None
    assert audits[0]["passed"] is True
    assert audits[0]["audit_report"]["verdict"] == "PASS_WITH_WARNINGS"
    assert audits[0]["audit_report"]["hard_failures"] == []
    warnings = " ".join(audits[0]["audit_report"]["warnings"])
    assert "Novelty score is below 5/10" in warnings
    assert "Weighted audit score is below 70/100" in warnings
    assert "model auditor recommended rejection" in warnings


def test_generation_returns_only_hypotheses_that_pass_the_audit_gate():
    source_id = "arXiv:1111.1111"
    document = Document(
        page_content=(
            "Source ID: arXiv:1111.1111\nTitle: Prior work\nAbstract: Evidence about the method and baseline."
        ),
        metadata={
            "source_id": source_id,
            "arxiv_id": "1111.1111",
            "title": "Prior work",
            "abstract": "Evidence about the method and baseline.",
        },
    )
    plan = SearchQueryPlan(
        queries=("method baseline",),
        required_terms=(),
        explicit_requirements=(EvidenceAspect("method", "the requested method"),),
    )
    coverage = EvidenceCoverage(
        aspect_source_ids={"method": (source_id,)},
        missing_aspect_ids=(),
        gap_queries=(),
        reason="Covered.",
    )
    synthesis = LiteratureSynthesis(
        established_findings=(LiteratureFinding("The baseline has a limitation.", (source_id,)),),
        contradictions=(),
        knowledge_gaps=("The new interaction is untested.",),
        analytical_rationale="The gap supports a testable comparison.",
    )
    draft = {
        "title": "Draft",
        "hypothesis": "Draft claim.",
        "rationale": "Draft rationale.",
        "feasibility": "Draft method.",
        "source_ids": [source_id],
    }
    final = {
        "title": "Audited",
        "hypothesis": "The method will outperform the baseline under congestion.",
        "rationale": "Prior work grounds the baseline limitation; the improvement remains a hypothesis.",
        "feasibility": "Compare against the baseline and reject the claim if latency is not lower.",
        "source_ids": [source_id],
    }
    audit = {
        "candidate_index": 0,
        "passed": True,
        "final_hypothesis": final,
        "audit_report": {
            "scores": {},
            "weighted_score": 82.0,
            "closest_prior_art": [],
            "unsupported_claims": [],
            "unsupported_numbers": [],
            "revision_instruction": "Clarify novelty.",
            "verdict": "PASS",
            "hard_failures": [],
        },
    }
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=True,
        agentic_research_enabled=False,
    )
    with (
        patch("app.agents.call_llm_for_search_queries", return_value=(plan, None)),
        patch.object(agent, "_retrieve_scientific_sources", return_value=[document]),
        patch("app.agents.call_llm_for_relevance_filter", return_value=([source_id], None)),
        patch("app.agents.call_llm_for_evidence_coverage", return_value=(coverage, None)),
        patch("app.agents.call_llm_for_literature_synthesis", return_value=(synthesis, None)),
        patch("app.agents.call_llm_for_generation", return_value=[draft]),
        patch("app.agents.call_llm_for_hypothesis_audit", return_value=([audit], None)),
    ):
        context = ContextMemory()
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("Improve network performance.", num_hypotheses=1),
            context,
        )

    assert errors == []
    assert len(hypotheses) == 1
    assert hypotheses[0].title == "Audited"
    assert hypotheses[0].audit_verdict == "PASS"
    assert hypotheses[0].audit_score == 82.0
    assert context.last_hypothesis_audits == [audit["audit_report"]]


def test_query_rewriting_allows_no_required_entity_terms():
    payload = json.dumps(
        {
            "queries": ["one", "two", "three", "four", "five"],
            "required_terms": [],
            "explicit_requirements": [
                {
                    "id": "goal_scope",
                    "goal_quote": "scientific creativity",
                }
            ],
            "exploration_directions": [],
        }
    )

    with patch("app.agents.call_llm", return_value=payload):
        plan, error = call_llm_for_search_queries("Improve scientific creativity")

    assert error is None
    assert plan is not None
    assert plan.required_terms == ()


def test_query_rewriting_rejects_invalid_or_incomplete_json():
    invalid_payloads = [
        "not json",
        '{"queries": ["only one"], "required_terms": ["Malaysia"]}',
        json.dumps(
            {
                "queries": [
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                ],
                "required_terms": [],
            }
        ),
    ]

    for payload in invalid_payloads:
        with patch("app.agents.call_llm", return_value=payload):
            plan, error = call_llm_for_search_queries("goal")

        assert plan is None
        assert error is not None
        assert error.startswith("Query rewriting failed:")


def test_query_rewriting_rejects_hard_requirement_absent_from_goal():
    payload = json.dumps(
        {
            "queries": ["one", "two", "three", "four", "five"],
            "required_terms": ["concept bottleneck"],
            "explicit_requirements": [
                {
                    "id": "invented_condition",
                    "goal_quote": "adversarial perturbations",
                }
            ],
            "exploration_directions": [],
        }
    )

    with patch("app.agents.call_llm", return_value=payload):
        plan, error = call_llm_for_search_queries("Compare concept bottleneck models with Grad-CAM.")

    assert plan is None
    assert error is not None
    assert "verbatim goal quotes" in error


def test_query_rewriting_retries_a_composite_requirement_as_atomic_quotes():
    goal = (
        "Generate testable hypotheses about whether concept bottleneck models "
        "improve the interpretability and reliability of deep-learning-based "
        "medical image classification compared with post-hoc explanation "
        "methods such as SHAP and Grad-CAM."
    )
    composite = json.dumps(
        {
            "queries": ["one", "two", "three", "four", "five"],
            "required_terms": ["concept bottleneck", "Grad-CAM"],
            "explicit_requirements": [
                {
                    "id": "whole_goal",
                    "goal_quote": (
                        "concept bottleneck models improve the "
                        "interpretability and reliability of "
                        "deep-learning-based medical image classification "
                        "compared with post-hoc explanation methods such as "
                        "SHAP and Grad-CAM"
                    ),
                }
            ],
            "exploration_directions": [],
        }
    )
    corrected = json.dumps(
        {
            "queries": ["one", "two", "three", "four", "five"],
            "required_terms": ["concept bottleneck", "Grad-CAM"],
            "explicit_requirements": [
                {
                    "id": "focal_method",
                    "goal_quote": "concept bottleneck models",
                },
                {
                    "id": "domain",
                    "goal_quote": ("deep-learning-based medical image classification"),
                },
                {
                    "id": "comparator",
                    "goal_quote": ("post-hoc explanation methods such as SHAP and Grad-CAM"),
                },
                {
                    "id": "outcomes",
                    "goal_quote": "interpretability and reliability",
                },
            ],
            "exploration_directions": [],
        }
    )

    with patch(
        "app.agents.call_llm",
        side_effect=[composite, corrected],
    ) as mock_call:
        plan, error = call_llm_for_search_queries(goal)

    assert error is None
    assert plan is not None
    assert [item.aspect_id for item in plan.explicit_requirements] == [
        "focal_method",
        "domain",
        "comparator",
        "outcomes",
    ]
    assert mock_call.call_count == 2
    second_prompt = " ".join(mock_call.call_args_list[1].args[0].split())
    assert "previous response was invalid" in second_prompt
    assert "Atomize long or composite goal quotes" in second_prompt


def test_query_rewriting_failure_stops_when_original_retrieval_is_empty():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    with (
        patch(
            "app.agents.call_llm",
            return_value="Error: LM Studio unavailable",
        ),
        patch.object(
            agent.rag_retriever,
            "retrieve",
        ) as mock_retrieve,
    ):
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("Malaysia history"),
            ContextMemory(),
        )

    assert hypotheses == []
    assert errors == ["Query rewriting failed: Error: LM Studio unavailable"]
    mock_retrieve.assert_not_called()


def test_query_rewriting_failure_uses_original_candidates():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    document = Mock()
    document.page_content = "Source ID: arXiv:1234.5678\nAbstract: relevant evidence"
    document.metadata = {
        "source_id": "arXiv:1234.5678",
        "arxiv_id": "1234.5678",
        "title": "Relevant evidence",
        "abstract": "Evidence for improving scientific creativity.",
    }
    generation_payload = json.dumps(
        [
            {
                "title": "Fallback-grounded hypothesis",
                "hypothesis": "A grounded relationship can be tested.",
                "rationale": "The original evidence supports the premise.",
                "feasibility": "Evaluate the relationship empirically.",
                "source_ids": ["arXiv:1234.5678"],
            }
        ]
    )

    with (
        patch.object(
            GenerationAgent,
            "_retrieve_original_scientific_sources",
            return_value=[document],
        ),
        patch.object(agent.rag_retriever, "retrieve") as mock_retrieve,
        patch(
            "app.agents.call_llm",
            side_effect=[
                "Error: planner unavailable",
                _relevance_payload("arXiv:1234.5678"),
                _coverage_payload("arXiv:1234.5678"),
                _synthesis_payload("arXiv:1234.5678"),
                generation_payload,
            ],
        ) as mock_llm,
    ):
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("Improve scientific creativity", num_hypotheses=1),
            ContextMemory(),
        )

    assert errors == []
    assert len(hypotheses) == 1
    assert hypotheses[0].evidence_source_ids == ["arXiv:1234.5678"]
    mock_retrieve.assert_not_called()
    coverage_prompt = mock_llm.call_args_list[2].args[0]
    assert "goal_scope: Improve scientific creativity" in coverage_prompt


def test_rag_defaults_keep_more_candidate_evidence():
    retriever = ArxivRAGRetriever()

    assert retriever.results_per_query == 20
    assert retriever.top_k == 10
    assert retriever.max_abstract_chars == 4000


def test_grading_context_is_bounded_and_keeps_every_source():
    documents = [
        Document(
            page_content="unused full source text",
            metadata={
                "source_id": f"arXiv:0000.0000{i}",
                "title": f"Paper {i}",
                "abstract": f"MARKER_{i} " + ("evidence " * 1000),
                "published": "2026-01-01",
                "primary_category": "cs.NI",
            },
        )
        for i in range(10)
    ]

    context = format_documents_for_grading(
        documents,
        max_abstract_chars=1600,
        max_total_chars=12000,
    )

    assert len(context) <= 12000
    assert context.count("<source id=") == 10
    assert all(f"MARKER_{i}" in context for i in range(10))
    assert "unused full source text" not in context


def test_grading_context_exposes_provenance_and_retrieval_intent():
    document = Document(
        page_content="unused full source text",
        metadata={
            "source_id": "web:official",
            "source_family": "web",
            "document_type": "official_docs",
            "provider": "tavily",
            "title": "Official model documentation",
            "canonical_url": "https://docs.example.org/models/context",
            "domain": "docs.example.org",
            "source_authority": "official first-party documentation",
            "published_at": "2026-01-01",
            "updated_at": "2026-07-01",
            "sub_question": "What is the supported context window?",
            "retrieval_query": "Example model context window official docs",
            "freshness": "year",
            "summary": "The supported context window is documented here.",
        },
    )

    context = format_documents_for_grading([document])

    assert "URL: https://docs.example.org/models/context" in context
    assert "Domain: docs.example.org" in context
    assert "Provider: tavily" in context
    assert "Source type: web" in context
    assert "Document type: official_docs" in context
    assert "Authority: official first-party documentation" in context
    assert "Published: 2026-01-01" in context
    assert "Updated: 2026-07-01" in context
    assert "Retrieved for: What is the supported context window?" in context
    assert "Search query: Example model context window official docs" in context
    assert "Requested freshness: year" in context
    assert "Content: The supported context window is documented here." in context


def test_web_evidence_serialization_preserves_canonical_fields():
    document = Document(
        page_content="Web evidence",
        metadata={
            "source_id": "web:guidance",
            "source_type": "web",
            "provider": "tavily",
            "title": "Official guidance",
            "url": "https://agency.example/guidance",
            "canonical_url": "https://agency.example/guidance",
            "domain": "agency.example",
            "summary": "Current technical guidance.",
            "content": "Current technical guidance.",
            "published_at": "2026-01-01",
            "updated_at": "2026-02-01",
            "page_type": "government_guidance",
            "source_authority": "official",
        },
    )

    serialized = serialize_documents([document])[0]

    assert serialized["source_type"] == "web"
    assert serialized["url"] == "https://agency.example/guidance"
    assert serialized["domain"] == "agency.example"
    assert serialized["summary"] == "Current technical guidance."
    assert serialized["source_authority"] == "official"
    assert serialized["arxiv_id"] is None


def test_relevance_grader_keeps_only_known_directly_relevant_sources():
    available_ids = {
        "arXiv:2001.03488v1",
        "arXiv:0912.1838v1",
    }
    payload = _relevance_payload(
        "2001.03488",
        "arXiv:9999.99999",
    )

    with patch(
        "app.agents.call_llm",
        return_value=payload,
    ) as mock_call:
        selected_ids, error = call_llm_for_relevance_filter(
            "Malaysia economic history",
            "retrieved context",
            available_ids,
            model="chosen-model",
        )

    assert error is None
    assert selected_ids == ["arXiv:2001.03488v1"]
    assert mock_call.call_args.kwargs == {
        "temperature": 0.0,
        "model": "chosen-model",
        "max_tokens": 512,
        "reasoning": "off",
    }
    grader_prompt = mock_call.call_args.args[0]
    assert "mixed research evidence" in grader_prompt
    assert "Keyword\noverlap" in grader_prompt
    assert "Exclude lexical collisions" in grader_prompt
    assert "Do not reject a source merely because it is a web page" in grader_prompt
    assert "untrusted evidence data" in grader_prompt
    assert "relevance, authority, freshness" in grader_prompt
    assert "Judge authority from the URL, domain, provider" in grader_prompt


def test_source_id_resolution_rejects_ambiguous_retrieved_versions():
    with patch(
        "app.agents.call_llm",
        return_value=_relevance_payload("arXiv:2001.03488"),
    ):
        selected_ids, error = call_llm_for_relevance_filter(
            "A scientific goal",
            "retrieved context",
            {
                "arXiv:2001.03488v1",
                "arXiv:2001.03488v2",
            },
        )

    assert error is None
    assert selected_ids == []


def test_source_id_resolution_accepts_exact_semantic_scholar_id():
    with patch(
        "app.agents.call_llm",
        return_value=_relevance_payload("s2:paper-id"),
    ):
        selected_ids, error = call_llm_for_relevance_filter(
            "research goal",
            "retrieved context",
            {"s2:paper-id"},
        )

    assert error is None
    assert selected_ids == ["s2:paper-id"]


def test_coverage_grader_ignores_unknown_sources_and_finds_missing_aspects():
    aspects = (
        EvidenceAspect("intervention", "Concept bottleneck models."),
        EvidenceAspect("comparator", "SHAP or Grad-CAM comparison."),
    )
    payload = json.dumps(
        {
            "aspect_coverage": [
                {
                    "aspect_id": "intervention",
                    "source_ids": ["2205.15480"],
                },
                {
                    "aspect_id": "comparator",
                    "source_ids": ["arXiv:9999.99999"],
                },
            ],
            "gap_queries": ["medical imaging CBM SHAP Grad-CAM"],
            "reason": "Comparator evidence is missing.",
        }
    )

    with patch("app.agents.call_llm", return_value=payload):
        coverage, coverage_error = call_llm_for_evidence_coverage(
            "Compare CBMs with SHAP and Grad-CAM.",
            aspects,
            "retrieved context",
            {"arXiv:2205.15480v2"},
        )

    assert coverage_error is None
    assert coverage is not None
    assert coverage.sufficient is False
    assert coverage.missing_aspect_ids == ("comparator",)
    assert coverage.gap_queries == ("medical imaging CBM SHAP Grad-CAM",)


def test_literature_synthesis_keeps_only_findings_with_retrieved_sources():
    aspects = (EvidenceAspect("core_topic", "The user-stated core topic."),)
    payload = json.dumps(
        {
            "established_findings": [
                {
                    "claim": "Supported premise.",
                    "source_ids": ["arXiv:2205.15480v2"],
                },
                {
                    "claim": "Unsupported premise.",
                    "source_ids": ["arXiv:9999.99999"],
                },
            ],
            "contradictions": [],
            "knowledge_gaps": ["A direct comparison remains unresolved."],
            "analytical_rationale": ("The supported premise motivates a testable comparison."),
        }
    )

    with patch("app.agents.call_llm", return_value=payload):
        synthesis, synthesis_error = call_llm_for_literature_synthesis(
            "Compare two methods.",
            aspects,
            ("Optional neighboring method.",),
            "retrieved context",
            {"arXiv:2205.15480v2"},
        )

    assert synthesis_error is None
    assert synthesis is not None
    assert [finding.claim for finding in synthesis.established_findings] == ["Supported premise."]
    assert synthesis.established_findings[0].source_ids == ("arXiv:2205.15480v2",)


def test_reciprocal_rank_fusion_deduplicates_versions_and_rewards_recurrence():
    recurring_v1 = _paper(
        "2001.03488v1",
        "Malaysia SAM",
        "Malaysia evidence",
    )
    recurring_v2 = _paper(
        "2001.03488v2",
        "Malaysia SAM updated",
        "Malaysia updated evidence",
    )
    other = _paper(
        "9999.00001v1",
        "Other",
        "Other evidence",
    )

    fused = reciprocal_rank_fusion(
        [
            [recurring_v1, other],
            [other, recurring_v2],
            [recurring_v1],
        ],
        k=60,
    )

    assert len(fused) == 2
    assert fused[0].source_id == "arXiv:2001.03488v1"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_multi_query_retrieval_filters_irrelevant_history_papers():
    malaysia = _paper(
        "2001.03488v1",
        "Income Distribution in Malaysia",
        "A study of public expenditure in Malaysia.",
    )
    duplicate = _paper(
        "2001.03488v2",
        "Income Distribution in Malaysia",
        "Updated evidence about Malaysia.",
    )
    context_history = _paper(
        "0912.1838v1",
        "A Brief History of Context",
        "Context-aware systems in computer science.",
    )
    quantum_history = _paper(
        "2103.05280v1",
        "Consistent Histories Interpretation",
        "A history of quantum mechanics.",
    )
    retriever = ArxivRAGRetriever(
        query_count=5,
        results_per_query=6,
        top_k=4,
    )
    retriever.arxiv = Mock()
    retriever.semantic_scholar = None
    retriever.springer = None
    retriever.arxiv.search_papers.side_effect = [
        [malaysia, context_history],
        [duplicate, quantum_history],
        [],
        [malaysia],
        [],
    ]
    query_plan = SearchQueryPlan(
        queries=("q1", "q2", "q3", "q4", "q5"),
        required_terms=("Malaysia", "Malaya"),
    )

    fake_store = Mock()

    def return_indexed_documents(*args, **kwargs):
        return fake_store.add_documents.call_args.kwargs["documents"]

    fake_store.similarity_search.side_effect = return_indexed_documents
    with patch(
        "app.rag_retriever.InMemoryVectorStore",
        return_value=fake_store,
    ):
        documents = retriever.retrieve(
            "brief describe the malaysia history",
            query_plan,
        )

    assert retriever.arxiv.search_papers.call_count == 5
    assert len(documents) == 1
    assert documents[0].metadata["source_id"] == ("arXiv:2001.03488v1")
    indexed_text = documents[0].page_content
    assert "Malaysia" in indexed_text
    assert "Context-aware" not in indexed_text
    assert "quantum mechanics" not in indexed_text


def test_retrieval_returns_empty_when_strict_filter_removes_every_paper():
    retriever = ArxivRAGRetriever(query_count=5)
    retriever.arxiv = Mock()
    retriever.semantic_scholar = None
    retriever.springer = None
    retriever.arxiv.search_papers.return_value = [
        _paper(
            "2103.05280v1",
            "Consistent Histories Interpretation",
            "A history of quantum mechanics.",
        )
    ]
    query_plan = SearchQueryPlan(
        queries=("q1", "q2", "q3", "q4", "q5"),
        required_terms=("Malaysia", "Malaya"),
    )

    with patch("app.rag_retriever.InMemoryVectorStore") as mock_store:
        documents = retriever.retrieve(
            "brief describe the malaysia history",
            query_plan,
        )

    assert documents == []
    mock_store.assert_not_called()


def test_targeted_corrective_retrieval_can_skip_initial_entity_filter():
    comparator_paper = _paper(
        "2401.00001v1",
        "Grad-CAM for Medical Imaging",
        "A study of Grad-CAM explanations in medical classifiers.",
    )
    retriever = ArxivRAGRetriever(
        query_count=1,
        results_per_query=3,
        top_k=2,
    )
    retriever.arxiv = Mock()
    retriever.semantic_scholar = None
    retriever.springer = None
    retriever.arxiv.search_papers.return_value = [comparator_paper]
    query_plan = SearchQueryPlan(
        queries=("Grad-CAM medical image classification",),
        required_terms=(),
    )
    fake_store = Mock()

    def return_indexed_documents(*args, **kwargs):
        return fake_store.add_documents.call_args.kwargs["documents"]

    fake_store.similarity_search.side_effect = return_indexed_documents
    with patch(
        "app.rag_retriever.InMemoryVectorStore",
        return_value=fake_store,
    ):
        documents = retriever.retrieve(
            "Grad-CAM medical image classification",
            query_plan,
        )

    assert len(documents) == 1
    assert documents[0].metadata["source_id"] == ("arXiv:2401.00001v1")


def test_retrieval_tries_original_goal_in_semantic_scholar_before_rewritten_queries():
    direct_paper = _paper(
        "s2:direct-paper",
        "Direct goal result",
        "Evidence about lightweight security monitoring at 5G MEC sites.",
    )
    retriever = ArxivRAGRetriever(query_count=2, top_k=1)
    retriever.semantic_scholar = Mock()
    retriever.semantic_scholar.search_papers.return_value = [direct_paper]
    retriever.springer = None
    retriever.arxiv = Mock()
    retriever.arxiv.search_papers.return_value = []
    fake_store = Mock()
    fake_store.similarity_search.side_effect = lambda *args, **kwargs: fake_store.add_documents.call_args.kwargs[
        "documents"
    ]

    with patch("app.rag_retriever.InMemoryVectorStore", return_value=fake_store):
        documents = retriever.retrieve_original_goal("lightweight security monitoring 5G MEC")

    retriever.semantic_scholar.search_papers.assert_called_once_with(query="lightweight security monitoring 5G MEC")
    retriever.arxiv.search_papers.assert_called_once()
    assert documents[0].metadata["source_id"] == "s2:direct-paper"


def test_web_retrieval_uses_web_schema_without_paper_identifiers():
    web_result = {
        "source_id": "web:official-guidance",
        "source_type": "web",
        "provider": "tavily",
        "title": "Official MEC security guidance",
        "url": "https://example.org/mec-security",
        "canonical_url": "https://example.org/mec-security",
        "domain": "example.org",
        "page_type": "technical_guidance",
        "snippet": "Lightweight monitoring guidance for constrained MEC nodes.",
        "content": "Lightweight monitoring guidance for constrained MEC nodes.",
        "published_at": "2026-01-10",
        "updated_at": "2026-02-12",
        "search_score": 0.9,
        "pdf_url": None,
    }
    retriever = ArxivRAGRetriever(query_count=1, top_k=1, minimum_relevant_sources=1)
    retriever.arxiv = Mock()
    retriever.arxiv.search_papers.return_value = []
    retriever.semantic_scholar = None
    retriever.springer = None
    retriever.elsevier = None
    retriever.tavily = Mock(is_configured=True, last_error_status=None)
    retriever.tavily.search.return_value = [web_result]
    retriever.tavily.extract.return_value = {
        "https://example.org/mec-security": (
            "Extracted lightweight monitoring evidence for constrained MEC nodes."
        )
    }
    fake_store = Mock()
    fake_store.similarity_search.side_effect = lambda *args, **kwargs: fake_store.add_documents.call_args.kwargs[
        "documents"
    ]

    with patch("app.rag_retriever.InMemoryVectorStore", return_value=fake_store):
        documents = retriever.retrieve(
            "lightweight MEC security",
            SearchQueryPlan(
                queries=(
                    SearchQuery(
                        query="MEC security guidance",
                        purpose="Find first-party operational guidance",
                        sub_question="What does the official guidance recommend?",
                        source_type="official",
                        preferred_domains=("example.org",),
                        freshness="year",
                        evidence_requirement_id="guidance",
                    ),
                ),
                required_terms=(),
            ),
        )

    retriever.tavily.search.assert_called_once_with(
        query="MEC security guidance",
        include_domains=("example.org",),
        time_range="year",
    )
    retriever.arxiv.search_papers.assert_not_called()
    retriever.tavily.extract.assert_called_once_with(
        urls=["https://example.org/mec-security"],
        query="What does the official guidance recommend?",
    )
    assert len(documents) == 1
    assert documents[0].metadata["source_type"] == "web"
    assert documents[0].metadata["source_family"] == "web"
    assert documents[0].metadata["document_type"] == "official_docs"
    assert documents[0].metadata["source_id"] == "web:official-guidance"
    assert documents[0].metadata["domain"] == "example.org"
    assert documents[0].metadata["arxiv_id"] is None
    assert documents[0].metadata["content_extracted"] is True
    assert documents[0].metadata["full_text_available"] is True
    assert documents[0].metadata["document_rerank_score"] == 1.0
    assert documents[0].metadata["purpose"] == "Find first-party operational guidance"
    assert documents[0].metadata["planned_source_type"] == "official"
    assert documents[0].metadata["freshness"] == "year"
    assert documents[0].metadata["evidence_requirement_id"] == "guidance"
    assert "Extracted lightweight monitoring evidence" in documents[0].page_content
    assert "Web content:" in documents[0].page_content
    serialized = serialize_documents(documents)[0]
    assert serialized["sub_question"] == "What does the official guidance recommend?"
    assert serialized["preferred_domains"] == ["example.org"]
    assert serialized["evidence_requirement_id"] == "guidance"
    assert serialized["document_type"] == "official_docs"
    assert serialized["full_text_available"] is True
    assert serialized["chunk_rerank_score"] == 1.0


def test_open_web_documents_extracts_known_url_for_find_in_page_target():
    retriever = ArxivRAGRetriever(query_count=1, top_k=1)
    retriever.tavily = Mock(is_configured=True)
    retriever.tavily.extract.return_value = {
        "https://docs.example.org/model": "The exact supported limit is 128K tokens."
    }
    document = Document(
        page_content="Search discovery",
        metadata={
            "source_id": "web:docs",
            "source_type": "web",
            "source_family": "web",
            "document_type": "official_docs",
            "provider": "tavily",
            "title": "Model documentation",
            "url": "https://docs.example.org/model",
            "canonical_url": "https://docs.example.org/model",
            "domain": "docs.example.org",
            "summary": "Model documentation.",
            "content_extracted": False,
        },
    )
    target = "What exact context limit does the documentation state?"

    opened = retriever.open_web_documents([document], target)

    retriever.tavily.extract.assert_called_once_with(
        urls=["https://docs.example.org/model"],
        query=target,
    )
    assert len(opened) == 1
    assert opened[0].metadata["source_id"] == "web:docs"
    assert opened[0].metadata["sub_question"] == target
    assert opened[0].metadata["full_text_available"] is True
    assert "128K tokens" in opened[0].page_content


def test_extracted_web_chunks_are_reranked_by_sub_question_and_globally_bounded():
    retriever = ArxivRAGRetriever(query_count=1, top_k=2)
    retriever.max_web_evidence_chunks = 2
    retriever.tavily = Mock(is_configured=True)
    retriever.tavily.extract.return_value = {
        "https://example.org/one": (
            "<chunk 1> background noise\n"
            "<chunk 2> alpha mechanism directly answers the question"
        ),
        "https://example.org/two": (
            "<chunk 1> unrelated navigation\n"
            "<chunk 2> alpha benchmark provides decisive evidence"
        ),
    }
    selected = [
        Document(
            page_content="Search snippet",
            metadata={
                "source_id": f"web:{index}",
                "source_type": "web",
                "source_family": "web",
                "document_type": "official_docs",
                "provider": "tavily",
                "title": f"Source {index}",
                "url": url,
                "canonical_url": url,
                "domain": "example.org",
                "summary": "Discovery snippet",
                "content": "",
                "content_extracted": False,
                "sub_question": "Which alpha evidence answers the question?",
                "retrieval_query": "alpha evidence",
                "document_rerank_score": 0.8 - index / 10,
            },
        )
        for index, url in enumerate(
            ("https://example.org/one", "https://example.org/two"),
            start=1,
        )
    ]

    class ChunkVectorStore:
        def __init__(self, *args, **kwargs):
            self.documents = []

        def add_documents(self, documents, ids):
            self.documents = list(documents)

        def similarity_search_with_score(self, query, k):
            assert query == "Which alpha evidence answers the question?"
            return sorted(
                (
                    (document, 0.95 if "alpha" in document.page_content else 0.1)
                    for document in self.documents
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:k]

    with patch("app.rag_retriever.InMemoryVectorStore", ChunkVectorStore):
        chunks = retriever._extract_selected_web_documents(
            selected,
            "original research goal",
        )

    assert len(chunks) == 2
    assert all("alpha" in chunk.metadata["content"] for chunk in chunks)
    assert all(chunk.metadata["chunk_rerank_score"] == 0.95 for chunk in chunks)
    assert all(chunk.metadata["full_text_available"] is True for chunk in chunks)
    assert {chunk.metadata["parent_source_id"] for chunk in chunks} == {
        "web:1",
        "web:2",
    }
    assert all(chunk.metadata["document_type"] == "official_docs" for chunk in chunks)


def test_web_search_snippet_is_not_evidence_when_extract_fails():
    web_result = {
        "source_id": "web:search-only",
        "source_type": "web",
        "provider": "tavily",
        "title": "Search-only result",
        "url": "https://example.org/search-only",
        "canonical_url": "https://example.org/search-only",
        "domain": "example.org",
        "snippet": "A promising but unverified search snippet.",
        "content": "",
        "content_extracted": False,
    }
    retriever = ArxivRAGRetriever(query_count=1, top_k=1)
    retriever.tavily = Mock(is_configured=True)
    retriever.tavily.extract.return_value = {}
    fake_store = Mock()
    fake_store.similarity_search.side_effect = lambda *args, **kwargs: (
        fake_store.add_documents.call_args.kwargs["documents"]
    )

    with patch("app.rag_retriever.InMemoryVectorStore", return_value=fake_store):
        documents = retriever._rank_documents(
            "research goal",
            SearchQueryPlan(queries=("query",), required_terms=()),
            [[web_result]],
        )

    assert documents == []
    retriever.tavily.extract.assert_called_once_with(
        urls=["https://example.org/search-only"],
        query="research goal",
    )


def test_expanded_retrieval_searches_different_sources_concurrently():
    both_sources_started = threading.Event()
    start_lock = threading.Lock()
    started_sources = 0

    class BlockingSearchSource:
        last_error_status = None

        def __init__(self, paper):
            self.paper = paper

        def search_papers(self, query, **kwargs):
            nonlocal started_sources
            with start_lock:
                started_sources += 1
                if started_sources == 2:
                    both_sources_started.set()
            assert both_sources_started.wait(timeout=1)
            return [self.paper]

    retriever = ArxivRAGRetriever(query_count=1, top_k=2)
    retriever.arxiv = BlockingSearchSource(
        _paper("2401.00001v1", "arXiv result", "Shared concurrent evidence."),
    )
    retriever.semantic_scholar = BlockingSearchSource(
        _paper("s2:paper-id", "Semantic Scholar result", "Shared concurrent evidence."),
    )
    retriever.springer = None
    retriever.elsevier = None
    retriever.tavily = None
    query_plan = SearchQueryPlan(queries=("concurrent query",), required_terms=())
    fake_store = Mock()
    fake_store.similarity_search.side_effect = lambda *args, **kwargs: fake_store.add_documents.call_args.kwargs[
        "documents"
    ]

    with patch("app.rag_retriever.InMemoryVectorStore", return_value=fake_store):
        documents = retriever.retrieve("research goal", query_plan)

    assert both_sources_started.is_set()
    assert {document.metadata["source_id"] for document in documents} == {
        "arXiv:2401.00001v1",
        "s2:paper-id",
    }
    assert {stat["source"] for stat in retriever.last_search_stats} == {"arXiv", "Semantic Scholar"}
    assert all(stat["status"] == "ok" for stat in retriever.last_search_stats)


def test_structured_queries_route_to_academic_and_news_providers():
    retriever = ArxivRAGRetriever(query_count=2, top_k=1)
    retriever.arxiv = Mock(last_error_status=None)
    retriever.arxiv.search_papers.return_value = []
    retriever.semantic_scholar = Mock(last_error_status=None)
    retriever.semantic_scholar.search_papers.return_value = []
    retriever.springer = None
    retriever.elsevier = None
    retriever.tavily = Mock(is_configured=True, last_error_status=None)
    retriever.tavily.search.return_value = []
    query_plan = SearchQueryPlan(
        queries=(
            SearchQuery(
                query="hierarchical retrieval long context paper",
                sub_question="Which mechanisms have academic support?",
                purpose="Prior academic evidence",
                source_type="academic",
            ),
            SearchQuery(
                query="Qwen long context benchmark update",
                sub_question="What changed in recent benchmark reporting?",
                purpose="Recent competing evidence",
                source_type="news",
                freshness="month",
            ),
        ),
        required_terms=(),
    )

    assert retriever.retrieve("Qwen long-context degradation", query_plan) == []

    retriever.arxiv.search_papers.assert_called_once_with(
        query="hierarchical retrieval long context paper",
        max_results=retriever.results_per_query,
        sort_by="relevance",
    )
    retriever.semantic_scholar.search_papers.assert_called_once_with(
        query="hierarchical retrieval long context paper"
    )
    retriever.tavily.search.assert_called_once_with(
        query="Qwen long context benchmark update",
        time_range="month",
        topic="news",
    )


def test_pdf_promotion_does_not_evict_direct_web_evidence():
    web_result = _paper(
        "tavily:web-result",
        "Web result",
        "Relevant search-only evidence.",
    )
    web_result.update({"pdf_url": None, "source": "tavily"})
    arxiv_result = _paper(
        "2401.00001v1",
        "Downloadable arXiv result",
        "Relevant downloadable evidence.",
    )
    arxiv_result["source"] = "arxiv"
    retriever = ArxivRAGRetriever(query_count=1, top_k=1, minimum_relevant_sources=1)
    retriever.minimum_downloadable_sources = 1
    retriever.tavily = Mock(is_configured=True)
    retriever.tavily.extract.return_value = {
        web_result["arxiv_url"]: "Extracted directly relevant web evidence."
    }
    fake_store = Mock()
    fake_store.similarity_search.side_effect = lambda *args, **kwargs: fake_store.add_documents.call_args.kwargs[
        "documents"
    ]

    with patch("app.rag_retriever.InMemoryVectorStore", return_value=fake_store):
        documents = retriever._rank_documents(
            "research goal",
            SearchQueryPlan(queries=("query",), required_terms=()),
            [[web_result, arxiv_result]],
        )

    assert [document.metadata["source_id"] for document in documents] == ["tavily:web-result"]
    assert documents[0].metadata["source_type"] == "web"


def test_strict_generation_evidence_gate_keeps_web_content_and_indexed_academic_sources():
    indexed = Document(
        page_content="Indexed evidence",
        metadata={
            "source_id": "arXiv:1111.1111",
            "source_type": "academic",
            "full_text_indexed": True,
        },
    )
    web_content = Document(
        page_content="Web content",
        metadata={
            "source_id": "web:guidance",
            "source_type": "web",
            "content_extracted": True,
            "full_text_indexed": False,
        },
    )
    search_only_web = Document(
        page_content="Search snippet only",
        metadata={
            "source_id": "web:search-only",
            "source_type": "web",
            "content_extracted": False,
            "full_text_indexed": False,
        },
    )
    academic_abstract_only = Document(
        page_content="Academic abstract only",
        metadata={
            "source_id": "arXiv:2222.2222",
            "source_type": "academic",
            "full_text_indexed": False,
        },
    )

    class StrictPaperLibrary:
        enabled = True
        require_indexed_sources_for_generation = True

        @staticmethod
        def enrich_documents(documents, _query):
            return list(documents)

    agent = GenerationAgent(paper_library=StrictPaperLibrary())

    assert agent._prepare_candidate_documents(
        [indexed, web_content, search_only_web, academic_abstract_only],
        ResearchGoal("Use downloadable evidence"),
    ) == [indexed, web_content]


def test_retrieval_stops_arxiv_batch_after_rate_limit():
    retriever = ArxivRAGRetriever(query_count=3, top_k=1)
    retriever.semantic_scholar = Mock()
    retriever.semantic_scholar.search_papers.side_effect = [[], [], [], []]
    retriever.springer = None
    retriever.arxiv = Mock()
    retriever.arxiv.search_papers.return_value = []
    retriever.arxiv.last_error_status = 429
    query_plan = SearchQueryPlan(
        queries=("rewritten one", "rewritten two", "rewritten three"),
        required_terms=(),
    )

    with patch("app.rag_retriever.InMemoryVectorStore"):
        retriever.retrieve("original goal", query_plan)

    assert retriever.arxiv.search_papers.call_count == 1


def test_retrieval_stops_semantic_scholar_batch_after_rate_limit():
    retriever = ArxivRAGRetriever(query_count=3, top_k=1)
    retriever.semantic_scholar = Mock()
    retriever.semantic_scholar.search_papers.return_value = []
    retriever.semantic_scholar.last_error_status = 429
    retriever.springer = None
    retriever.arxiv = Mock()
    retriever.arxiv.search_papers.return_value = []
    query_plan = SearchQueryPlan(
        queries=("rewritten one", "rewritten two", "rewritten three"),
        required_terms=(),
    )

    with patch("app.rag_retriever.InMemoryVectorStore"):
        retriever.retrieve("original goal", query_plan)

    # The remaining rewritten queries are skipped after the first rate-limit signal.
    assert retriever.semantic_scholar.search_papers.call_count == 1


def test_semantic_scholar_fallback_fuses_and_reranks_results():
    semantic_scholar_paper = _paper(
        "s2:paper-id",
        "Semantic Scholar result",
        "Evidence from a paper outside arXiv.",
    )
    retriever = ArxivRAGRetriever(query_count=2, top_k=1)
    retriever.semantic_scholar = Mock()
    retriever.semantic_scholar.search_papers.side_effect = [
        [semantic_scholar_paper],
        [semantic_scholar_paper],
    ]
    query_plan = SearchQueryPlan(
        queries=("first query", "second query"),
        required_terms=(),
    )
    fake_store = Mock()
    fake_store.similarity_search.side_effect = lambda *args, **kwargs: fake_store.add_documents.call_args.kwargs[
        "documents"
    ]

    with patch(
        "app.rag_retriever.InMemoryVectorStore",
        return_value=fake_store,
    ):
        documents = retriever.retrieve_fallback("research goal", query_plan)

    assert retriever.semantic_scholar.search_papers.call_count == 2
    assert len(documents) == 1
    assert documents[0].metadata["source_id"] == "s2:paper-id"
    assert fake_store.similarity_search.call_args.kwargs["k"] == 1


def test_springer_fallback_fuses_evidence_for_generation():
    springer_result = _paper(
        "springer:doi:10.1007/s00123-024",
        "Springer nature evidence",
        "Evidence from Springer Nature scientific literature.",
    )
    retriever = ArxivRAGRetriever(query_count=1, top_k=1)
    retriever.semantic_scholar = None
    retriever.springer = Mock()
    retriever.springer.is_configured = True
    retriever.springer.search_papers.return_value = [springer_result]
    query_plan = SearchQueryPlan(queries=("springer query",), required_terms=())
    fake_store = Mock()
    fake_store.similarity_search.side_effect = lambda *args, **kwargs: fake_store.add_documents.call_args.kwargs[
        "documents"
    ]

    with patch("app.rag_retriever.InMemoryVectorStore", return_value=fake_store):
        documents = retriever.retrieve_fallback("research goal", query_plan)

    retriever.springer.search_papers.assert_called_once_with(query="springer query")
    assert documents[0].metadata["source_id"] == "springer:doi:10.1007/s00123-024"


def test_generation_prompt_contains_retrieved_abstract_and_source_id():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    query_plan = _query_plan_payload()
    paper = _paper(
        "2001.03488v1",
        "Income Distribution in Malaysia",
        "UNIQUE_MALAYSIA_EVIDENCE about public expenditure.",
    )
    generation_payload = json.dumps(
        [
            {
                "title": "Grounded hypothesis",
                "hypothesis": "A testable Malaysia hypothesis.",
                "rationale": "The retrieved Malaysia evidence supports it.",
                "feasibility": "Test it with a controlled comparison.",
                "source_ids": ["arXiv:2001.03488v1"],
            }
        ]
    )
    agent.rag_retriever.arxiv = Mock()
    agent.rag_retriever.semantic_scholar = None
    agent.rag_retriever.springer = None
    agent.rag_retriever.elsevier = None
    agent.rag_retriever.tavily = None
    agent.rag_retriever.arxiv.search_papers.side_effect = [
        [paper],
        [paper],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    fake_store = Mock()

    def return_indexed_documents(*args, **kwargs):
        return fake_store.add_documents.call_args.kwargs["documents"]

    fake_store.similarity_search.side_effect = return_indexed_documents

    with (
        patch(
            "app.rag_retriever.InMemoryVectorStore",
            return_value=fake_store,
        ),
        patch(
            "app.agents.call_llm",
            side_effect=[
                query_plan,
                _relevance_payload("arXiv:2001.03488v1"),
                _coverage_payload("arXiv:2001.03488v1"),
                _synthesis_payload("arXiv:2001.03488v1"),
                generation_payload,
            ],
        ) as mock_llm,
    ):
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal(
                "brief describe the malaysia history",
                num_hypotheses=1,
            ),
            ContextMemory(),
        )

    generation_prompt = mock_llm.call_args_list[4].args[0]
    assert "UNIQUE_MALAYSIA_EVIDENCE" in generation_prompt
    assert "arXiv:2001.03488v1" in generation_prompt
    assert "Literature review and analytical rationale" in generation_prompt
    assert "Explicit requirements validated" in generation_prompt
    assert "Optional exploration directions" in generation_prompt
    assert "evidence coverage stage has already verified" in generation_prompt
    assert "do not return an error object" in generation_prompt
    assert "- hypothesis: one clear, testable claim." in generation_prompt
    assert "- rationale: why the claim follows" in generation_prompt
    assert "- feasibility: a concise practical method" in generation_prompt
    assert hypotheses[0].text == (
        "Hypothesis: A testable Malaysia hypothesis.\n\n"
        "Rationale: The retrieved Malaysia evidence supports it.\n\n"
        "Feasibility: Test it with a controlled comparison."
    )
    assert len(hypotheses) == 1
    assert hypotheses[0].evidence_source_ids == ["arXiv:2001.03488v1"]
    assert errors == []


def test_generation_stops_when_model_reports_insufficient_context():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    document = Mock()
    document.page_content = "Source ID: arXiv:1234.5678\nAbstract: limited evidence"
    document.metadata = {
        "source_id": "arXiv:1234.5678",
        "arxiv_id": "1234.5678",
        "title": "Limited evidence",
        "abstract": "limited evidence",
    }
    insufficient_payload = json.dumps(
        {"error": ("The retrieved context is insufficient to generate grounded hypotheses.")}
    )

    with (
        patch(
            "app.agents.call_llm",
            side_effect=[
                _query_plan_payload("Any scientific topic"),
                _relevance_payload("arXiv:1234.5678"),
                _coverage_payload("arXiv:1234.5678"),
                _synthesis_payload("arXiv:1234.5678"),
                insufficient_payload,
            ],
        ),
        patch.object(
            GenerationAgent,
            "_retrieve_scientific_sources",
            return_value=[document],
        ),
    ):
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("Any scientific topic"),
            ContextMemory(),
        )

    assert hypotheses == []
    assert errors == ["The retrieved context is insufficient to generate grounded hypotheses."]


def test_empty_relevance_suggestion_does_not_override_complete_coverage():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    document = Mock()
    document.page_content = "Source ID: arXiv:0912.1838v1\nAbstract: history of context"
    document.metadata = {
        "source_id": "arXiv:0912.1838v1",
        "arxiv_id": "0912.1838v1",
        "title": "A Brief History of Context",
        "abstract": "Context-aware computer systems.",
    }

    generation_payload = json.dumps(
        [
            {
                "title": "Coverage-grounded hypothesis",
                "hypothesis": "A grounded relationship can be tested.",
                "rationale": "The coverage-confirmed evidence supports it.",
                "feasibility": "Evaluate the relationship empirically.",
                "source_ids": ["0912.1838"],
            }
        ]
    )

    with (
        patch(
            "app.agents.call_llm",
            side_effect=[
                _query_plan_payload("Malaysia history"),
                _relevance_payload(),
                _coverage_payload("0912.1838"),
                _synthesis_payload("0912.1838"),
                generation_payload,
            ],
        ) as mock_llm,
        patch.object(
            GenerationAgent,
            "_retrieve_scientific_sources",
            return_value=[document],
        ),
    ):
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("Malaysia history"),
            ContextMemory(),
        )

    assert errors == []
    assert len(hypotheses) == 1
    assert hypotheses[0].evidence_source_ids == ["arXiv:0912.1838v1"]
    assert mock_llm.call_count == 5


def test_generation_uses_collective_coverage_and_excludes_unmapped_sources():
    goal = "Compare GNN intrusion detection with conventional DNN baselines in 5G under adversarial robustness"
    requirements = [
        {"id": "method", "goal_quote": "GNN intrusion detection"},
        {
            "id": "baseline",
            "goal_quote": "conventional DNN baselines",
        },
        {"id": "domain", "goal_quote": "5G"},
        {
            "id": "robustness",
            "goal_quote": "adversarial robustness",
        },
    ]
    document_specs = [
        (
            "2101.00001v1",
            "GNN Intrusion Detection",
            "GNN_UNIQUE supports graph-based intrusion detection.",
        ),
        (
            "2102.00002v2",
            "Adversarial Robustness for Network Defenses",
            "ROBUSTNESS_UNIQUE studies adversarial robustness.",
        ),
        (
            "2103.00003v1",
            "Intrusion Detection in 5G Networks",
            "FIVE_G_UNIQUE studies intrusion detection in 5G.",
        ),
        (
            "2104.00004v3",
            "Conventional Deep Neural Intrusion Baselines",
            "DNN_UNIQUE evaluates conventional DNN baselines.",
        ),
        (
            "2105.00005v1",
            "Unrelated Graph Keyword Collision",
            "IRRELEVANT_UNIQUE must never reach synthesis or generation.",
        ),
    ]
    documents = []
    for arxiv_id, title, abstract in document_specs:
        document = Mock()
        document.page_content = f"Source ID: arXiv:{arxiv_id}\nTitle: {title}\nAbstract: {abstract}"
        document.metadata = {
            "source_id": f"arXiv:{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
        }
        documents.append(document)

    coverage = EvidenceCoverage(
        aspect_source_ids={
            "method": ("arXiv:2101.00001v1",),
            "robustness": ("arXiv:2102.00002v2",),
            "domain": ("arXiv:2103.00003v1",),
            "baseline": ("arXiv:2104.00004v3",),
        },
        missing_aspect_ids=(),
        gap_queries=(),
        reason="Different papers collectively cover every explicit facet.",
    )
    generation_payload = json.dumps(
        [
            {
                "title": "Cross-facet IDS hypothesis",
                "hypothesis": "A GNN defense may improve robust 5G IDS.",
                "rationale": "Each explicit facet has retrieved support.",
                "feasibility": "Compare the methods under controlled attacks.",
                "source_ids": [
                    "2101.00001",
                    "arXiv:2102.00002",
                    "2103.00003",
                    "arXiv:2104.00004",
                ],
            }
        ]
    )
    candidate_contexts = []

    def audit_all_candidates(
        research_goal,
        explicit_requirements,
        retrieved_context,
        available_source_ids,
        **kwargs,
    ):
        candidate_contexts.append(retrieved_context)
        assert len(available_source_ids) == 5
        return coverage, None

    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    with (
        patch(
            "app.agents.call_llm",
            side_effect=[
                _query_plan_payload(goal, requirements=requirements),
                _synthesis_payload(
                    "2101.00001",
                    "2102.00002",
                    "2103.00003",
                    "2104.00004",
                ),
                generation_payload,
            ],
        ) as mock_llm,
        patch.object(
            agent,
            "_retrieve_scientific_sources",
            return_value=documents,
        ),
        patch(
            "app.agents.call_llm_for_relevance_filter",
            return_value=([], None),
        ),
        patch(
            "app.agents.call_llm_for_evidence_coverage",
            side_effect=audit_all_candidates,
        ),
    ):
        context = ContextMemory()
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal(goal, num_hypotheses=1),
            context,
        )

    assert errors == []
    assert len(hypotheses) == 1
    assert hypotheses[0].evidence_source_ids == [
        "arXiv:2101.00001v1",
        "arXiv:2102.00002v2",
        "arXiv:2103.00003v1",
        "arXiv:2104.00004v3",
    ]
    assert "IRRELEVANT_UNIQUE" in candidate_contexts[0]
    synthesis_prompt = mock_llm.call_args_list[1].args[0]
    generation_prompt = mock_llm.call_args_list[2].args[0]
    assert "IRRELEVANT_UNIQUE" not in synthesis_prompt
    assert "IRRELEVANT_UNIQUE" not in generation_prompt
    assert "GNN_UNIQUE" in synthesis_prompt
    assert "GNN_UNIQUE" in generation_prompt
    assert len(context.last_retrieved_sources) == 4
    assert {source["source_id"] for source in context.last_retrieved_sources} == {
        "arXiv:2101.00001v1",
        "arXiv:2102.00002v2",
        "arXiv:2103.00003v1",
        "arXiv:2104.00004v3",
    }


def test_generation_can_enforce_a_configured_minimum_source_count():
    agent = GenerationAgent(
        minimum_relevant_sources=3,
        debate_rounds=0,
    )
    documents = []
    for arxiv_id in ("1111.1111", "2222.2222"):
        document = Mock()
        document.page_content = f"Source ID: arXiv:{arxiv_id}\nAbstract: directly relevant"
        document.metadata = {
            "source_id": f"arXiv:{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": f"Relevant evidence {arxiv_id}",
            "abstract": "Directly relevant evidence.",
        }
        documents.append(document)

    with (
        patch(
            "app.agents.call_llm",
            side_effect=[
                _query_plan_payload("A focused scientific goal"),
                _relevance_payload("arXiv:1111.1111", "arXiv:2222.2222"),
                _coverage_payload(
                    "arXiv:1111.1111",
                    "arXiv:2222.2222",
                ),
            ],
        ) as mock_llm,
        patch.object(
            GenerationAgent,
            "_retrieve_scientific_sources",
            return_value=documents,
        ),
    ):
        context = ContextMemory()
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("A focused scientific goal"),
            context,
        )

    assert hypotheses == []
    assert errors == [
        "RAG coverage auditing confirmed 2 supporting indexed source(s), "
        "but at least 3 are required. Hypothesis generation "
        "was not executed."
    ]
    assert context.last_retrieved_sources == []
    assert mock_llm.call_count == 3


def test_evolved_hypothesis_inherits_parent_evidence_sources():
    first = Hypothesis("G1", "First", "First hypothesis")
    first.evidence_source_ids = ["arXiv:1111.1111", "arXiv:2222.2222"]
    second = Hypothesis("G2", "Second", "Second hypothesis")
    second.evidence_source_ids = ["arXiv:2222.2222", "arXiv:3333.3333"]

    evolved = combine_hypotheses(first, second)

    assert evolved.evidence_source_ids == [
        "arXiv:1111.1111",
        "arXiv:2222.2222",
        "arXiv:3333.3333",
    ]


def test_generation_rejects_source_id_outside_retrieved_top_k():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    generation_payload = json.dumps(
        [
            {
                "title": "Unsupported hypothesis",
                "hypothesis": "Uses an unreturned paper.",
                "rationale": "The unsupported source appears relevant.",
                "feasibility": "Test the unsupported claim.",
                "source_ids": ["arXiv:9999.99999"],
            }
        ]
    )
    document = Mock()
    document.page_content = "Malaysia evidence"
    document.metadata = {
        "source_id": "arXiv:2001.03488v1",
        "arxiv_id": "2001.03488v1",
        "title": "Malaysia evidence",
        "abstract": "Malaysia evidence",
    }

    with (
        patch(
            "app.agents.call_llm",
            side_effect=[
                _query_plan_payload("Malaysia history"),
                _relevance_payload("arXiv:2001.03488v1"),
                _coverage_payload("arXiv:2001.03488v1"),
                _synthesis_payload("arXiv:2001.03488v1"),
                generation_payload,
            ],
        ),
        patch.object(
            GenerationAgent,
            "_retrieve_scientific_sources",
            return_value=[document],
        ),
    ):
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("Malaysia history", num_hypotheses=1),
            ContextMemory(),
        )

    assert hypotheses == []
    assert errors == ["Generated hypothesis has no valid retrieved source IDs: Unsupported hypothesis"]


def test_missing_evidence_triggers_corrective_retrieval_before_generation():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        corrective_retrieval_rounds=2,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    subject_document = Mock()
    subject_document.page_content = "Source ID: arXiv:1111.1111\nAbstract: Evidence about the subject."
    subject_document.metadata = {
        "source_id": "arXiv:1111.1111",
        "arxiv_id": "1111.1111",
        "title": "Subject evidence",
        "abstract": "Evidence about the subject.",
    }
    outcome_document = Mock()
    outcome_document.page_content = "Source ID: arXiv:2222.2222\nAbstract: Evidence about the requested outcome."
    outcome_document.metadata = {
        "source_id": "arXiv:2222.2222",
        "arxiv_id": "2222.2222",
        "title": "Outcome evidence",
        "abstract": "Evidence about the requested outcome.",
    }
    first_coverage = EvidenceCoverage(
        aspect_source_ids={
            "subject_scope": ("arXiv:1111.1111",),
            "requested_outcome": (),
        },
        missing_aspect_ids=("requested_outcome",),
        gap_queries=("targeted requested outcome evidence",),
        reason="Outcome evidence is missing.",
    )
    complete_coverage = EvidenceCoverage(
        aspect_source_ids={
            "subject_scope": ("arXiv:1111.1111",),
            "requested_outcome": ("arXiv:2222.2222",),
        },
        missing_aspect_ids=(),
        gap_queries=(),
        reason="All aspects are covered.",
    )
    generation_payload = json.dumps(
        [
            {
                "title": "Correctively grounded hypothesis",
                "hypothesis": "A grounded relationship can be tested.",
                "rationale": "Both evidence dimensions are represented.",
                "feasibility": "Compare measurable outcomes.",
                "source_ids": [
                    "arXiv:1111.1111",
                    "arXiv:2222.2222",
                ],
            }
        ]
    )

    with (
        patch(
            "app.agents.call_llm",
            side_effect=[
                _query_plan_payload(
                    "A multi-aspect scientific goal",
                    requirements=[
                        {
                            "id": "subject_scope",
                            "goal_quote": "multi-aspect",
                        },
                        {
                            "id": "requested_outcome",
                            "goal_quote": "scientific goal",
                        },
                    ],
                ),
                _synthesis_payload(
                    "arXiv:1111.1111",
                    "arXiv:2222.2222",
                ),
                generation_payload,
            ],
        ) as mock_llm,
        patch.object(
            agent,
            "_retrieve_scientific_sources",
            side_effect=[
                [subject_document],
                [outcome_document],
            ],
        ) as mock_retrieve,
        patch(
            "app.agents.call_llm_for_relevance_filter",
            side_effect=[
                (["arXiv:1111.1111"], None),
                (
                    [
                        "arXiv:1111.1111",
                        "arXiv:2222.2222",
                    ],
                    None,
                ),
            ],
        ),
        patch(
            "app.agents.call_llm_for_evidence_coverage",
            side_effect=[
                (first_coverage, None),
                (complete_coverage, None),
            ],
        ) as mock_coverage,
    ):
        context = ContextMemory()
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("A multi-aspect scientific goal", num_hypotheses=1),
            context,
        )

    assert errors == []
    assert len(hypotheses) == 1
    assert hypotheses[0].evidence_source_ids == [
        "arXiv:1111.1111",
        "arXiv:2222.2222",
    ]
    assert mock_retrieve.call_count == 2
    assert mock_retrieve.call_args_list[1].kwargs["rerank_query"] == "scientific goal"
    gap_plan = mock_retrieve.call_args_list[1].args[1]
    assert gap_plan.query_texts == (
        "targeted requested outcome evidence",
        "scientific goal",
    )
    assert gap_plan.required_terms == ()
    assert len(context.last_retrieved_sources) == 2
    second_coverage_context = mock_coverage.call_args_list[1].args[2]
    assert "Evidence about the subject" in second_coverage_context
    assert "Evidence about the requested outcome" in second_coverage_context
    generation_prompt = mock_llm.call_args_list[2].args[0]
    assert "Evidence about the subject" in generation_prompt
    assert "Evidence about the requested outcome" in generation_prompt


def test_generation_stops_when_corrective_retrieval_cannot_fill_gap():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        corrective_retrieval_rounds=1,
        debate_rounds=0,
    )
    document = Mock()
    document.page_content = "Source ID: arXiv:1111.1111\nAbstract: Evidence about only one aspect."
    document.metadata = {
        "source_id": "arXiv:1111.1111",
        "arxiv_id": "1111.1111",
        "title": "Partial evidence",
        "abstract": "Evidence about only one aspect.",
    }
    incomplete_coverage = EvidenceCoverage(
        aspect_source_ids={
            "subject_scope": ("arXiv:1111.1111",),
            "requested_outcome": (),
        },
        missing_aspect_ids=("requested_outcome",),
        gap_queries=(),
        reason="Outcome evidence remains missing.",
    )

    with (
        patch(
            "app.agents.call_llm",
            return_value=_query_plan_payload(
                "A multi-aspect scientific goal",
                requirements=[
                    {
                        "id": "subject_scope",
                        "goal_quote": "multi-aspect",
                    },
                    {
                        "id": "requested_outcome",
                        "goal_quote": "scientific goal",
                    },
                ],
            ),
        ) as mock_llm,
        patch.object(
            agent,
            "_retrieve_scientific_sources",
            side_effect=[[document], []],
        ) as mock_retrieve,
        patch(
            "app.agents.call_llm_for_relevance_filter",
            return_value=(["arXiv:1111.1111"], None),
        ),
        patch(
            "app.agents.call_llm_for_evidence_coverage",
            return_value=(incomplete_coverage, None),
        ),
        patch.object(
            agent.rag_retriever,
            "retrieve_fallback",
            side_effect=RuntimeError("fallback unavailable"),
        ) as mock_fallback,
    ):
        context = ContextMemory()
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("A multi-aspect scientific goal", num_hypotheses=1),
            context,
        )

    assert hypotheses == []
    assert len(errors) == 1
    assert "insufficient after 1 corrective retrieval round(s)" in errors[0]
    assert "scientific goal" in errors[0]
    assert "Hypothesis generation was not executed" in errors[0]
    assert context.last_retrieved_sources == []
    assert mock_llm.call_count == 1
    gap_plan = mock_retrieve.call_args_list[1].args[1]
    assert gap_plan.query_texts == ("scientific goal",)
    assert gap_plan.required_terms == ()
    mock_fallback.assert_called_once()


def test_semantic_scholar_fallback_can_fill_gap_after_arxiv_is_exhausted():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        corrective_retrieval_rounds=0,
        debate_rounds=0,
        audit_enabled=False,
        agentic_research_enabled=False,
    )
    arxiv_document = Mock()
    arxiv_document.page_content = "Source ID: arXiv:1111.1111\nAbstract: Evidence about the subject."
    arxiv_document.metadata = {
        "source_id": "arXiv:1111.1111",
        "arxiv_id": "1111.1111",
        "title": "Subject evidence",
        "abstract": "Evidence about the subject.",
    }
    fallback_document = Mock()
    fallback_document.page_content = "Source ID: s2:outcome\nAbstract: Evidence about the requested outcome."
    fallback_document.metadata = {
        "source_id": "s2:outcome",
        "arxiv_id": "s2:outcome",
        "title": "Outcome evidence",
        "abstract": "Evidence about the requested outcome.",
    }
    incomplete_coverage = EvidenceCoverage(
        aspect_source_ids={
            "subject_scope": ("arXiv:1111.1111",),
            "requested_outcome": (),
        },
        missing_aspect_ids=("requested_outcome",),
        gap_queries=("targeted outcome evidence",),
        reason="Outcome evidence is missing.",
    )
    complete_coverage = EvidenceCoverage(
        aspect_source_ids={
            "subject_scope": ("arXiv:1111.1111",),
            "requested_outcome": ("s2:outcome",),
        },
        missing_aspect_ids=(),
        gap_queries=(),
        reason="All aspects are covered.",
    )
    generation_payload = json.dumps(
        [
            {
                "title": "Fallback-grounded hypothesis",
                "hypothesis": "A grounded relationship can be tested.",
                "rationale": "Both evidence dimensions are represented.",
                "feasibility": "Compare measurable outcomes.",
                "source_ids": ["arXiv:1111.1111", "s2:outcome"],
            }
        ]
    )

    with (
        patch(
            "app.agents.call_llm",
            side_effect=[
                _query_plan_payload(
                    "A multi-aspect scientific goal",
                    requirements=[
                        {"id": "subject_scope", "goal_quote": "multi-aspect"},
                        {
                            "id": "requested_outcome",
                            "goal_quote": "scientific goal",
                        },
                    ],
                ),
                _synthesis_payload("arXiv:1111.1111", "s2:outcome"),
                generation_payload,
            ],
        ),
        patch.object(
            agent,
            "_retrieve_scientific_sources",
            return_value=[arxiv_document],
        ),
        patch.object(
            agent.rag_retriever,
            "retrieve_fallback",
            return_value=[fallback_document],
        ) as mock_fallback,
        patch(
            "app.agents.call_llm_for_relevance_filter",
            side_effect=[
                (["arXiv:1111.1111"], None),
                (["arXiv:1111.1111", "s2:outcome"], None),
            ],
        ),
        patch(
            "app.agents.call_llm_for_evidence_coverage",
            side_effect=[
                (incomplete_coverage, None),
                (complete_coverage, None),
            ],
        ),
    ):
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("A multi-aspect scientific goal", num_hypotheses=1),
            ContextMemory(),
        )

    assert errors == []
    assert hypotheses[0].evidence_source_ids == ["arXiv:1111.1111", "s2:outcome"]
    fallback_plan = mock_fallback.call_args.args[1]
    assert fallback_plan.query_texts == (
        "targeted outcome evidence",
        "scientific goal",
    )
    assert fallback_plan.required_terms == ()


def test_generation_debate_runs_three_stateful_refinement_turns():
    agent = GenerationAgent(debate_rounds=3)
    query_plan = SearchQueryPlan(
        queries=("query",),
        required_terms=("method",),
        explicit_requirements=(
            EvidenceAspect(
                "core_comparison",
                "Compare the methods requested by the user.",
            ),
        ),
        exploration_directions=("Optional robustness analysis.",),
    )
    synthesis = LiteratureSynthesis(
        established_findings=(
            LiteratureFinding(
                claim="A retrieved premise.",
                source_ids=("arXiv:1111.1111",),
            ),
        ),
        contradictions=(),
        knowledge_gaps=("The direct comparison is unresolved.",),
        analytical_rationale="The premise motivates a comparison.",
    )
    initial = [
        {
            "title": "Initial",
            "hypothesis": "Initial claim.",
            "rationale": "Initial rationale.",
            "feasibility": "Initial method.",
            "source_ids": ["arXiv:1111.1111"],
        }
    ]

    def refined(title):
        return [
            {
                "title": title,
                "hypothesis": f"{title} claim.",
                "rationale": f"{title} rationale.",
                "feasibility": f"{title} method.",
                "source_ids": ["arXiv:1111.1111"],
            }
        ]

    with patch(
        "app.agents.call_llm_for_debate_refinement",
        side_effect=[
            (refined("Evidence refined"), None),
            (refined("Methods refined"), None),
            (refined("Integrated"), None),
        ],
    ) as mock_debate:
        result = agent._run_scientific_debate(
            ResearchGoal("Compare the requested methods."),
            query_plan,
            synthesis,
            initial,
        )

    assert result[0]["title"] == "Integrated"
    assert mock_debate.call_count == 3
    assert "turn 1 of" in mock_debate.call_args_list[0].args[0]
    assert "Candidate hypotheses from the preceding discussion" in (mock_debate.call_args_list[1].args[0])
    final_debate_prompt = " ".join(mock_debate.call_args_list[2].args[0].split())
    assert "may inspire refinement but are not requirements" in (final_debate_prompt)


def test_generation_debate_keeps_last_valid_turn_when_next_turn_fails():
    agent = GenerationAgent(debate_rounds=3)
    query_plan = SearchQueryPlan(
        queries=("query",),
        required_terms=("method",),
        explicit_requirements=(EvidenceAspect("core_topic", "The requested topic."),),
    )
    synthesis = LiteratureSynthesis(
        established_findings=(
            LiteratureFinding(
                claim="A retrieved premise.",
                source_ids=("arXiv:1111.1111",),
            ),
        ),
        contradictions=(),
        knowledge_gaps=(),
        analytical_rationale="A grounded rationale.",
    )
    initial = [
        {
            "title": "Initial",
            "hypothesis": "Initial claim.",
            "rationale": "Initial rationale.",
            "feasibility": "Initial method.",
            "source_ids": ["arXiv:1111.1111"],
        }
    ]
    first_refinement = [{**initial[0], "title": "First valid turn"}]

    with patch(
        "app.agents.call_llm_for_debate_refinement",
        side_effect=[
            (first_refinement, None),
            (None, "invalid JSON"),
        ],
    ):
        result = agent._run_scientific_debate(
            ResearchGoal("Study the requested topic."),
            query_plan,
            synthesis,
            initial,
        )

    assert result == first_refinement
