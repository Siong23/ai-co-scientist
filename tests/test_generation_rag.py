import json
from unittest.mock import Mock, patch

from app.agents import (
    EvidenceCoverage,
    GenerationAgent,
    LiteratureFinding,
    LiteratureSynthesis,
    call_llm_for_evidence_coverage,
    call_llm_for_literature_synthesis,
    call_llm_for_relevance_filter,
    call_llm_for_search_queries,
    combine_hypotheses,
)
from app.models import ContextMemory, Hypothesis, ResearchGoal
from app.rag_retriever import (
    ArxivRAGRetriever,
    EvidenceAspect,
    SearchQueryPlan,
    reciprocal_rank_fusion,
)


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


def _query_plan_payload(
    goal: str = "brief describe the malaysia history",
    requirements: list[dict] | None = None,
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
            ],
            "required_terms": ["Malaysia", "Malaya"],
            "explicit_requirements": requirements,
            "exploration_directions": [
                "Compare alternative historical interpretations."
            ],
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
            "knowledge_gaps": [
                "The proposed relationship remains untested."
            ],
            "analytical_rationale": (
                "The established premise motivates a new testable inference."
            ),
        }
    )


def test_query_rewriting_uses_selected_model_and_zero_temperature():
    with patch(
        "app.agents.call_llm",
        return_value=_query_plan_payload(),
    ) as mock_call:
        plan, error = call_llm_for_search_queries(
            "brief describe the malaysia history",
            model="chosen-model",
        )

    assert error is None
    assert plan is not None
    assert len(plan.queries) == 5
    assert plan.required_terms == ("Malaysia", "Malaya")
    assert [
        aspect.aspect_id
        for aspect in plan.explicit_requirements
    ] == ["goal_scope"]
    assert [
        aspect.description
        for aspect in plan.explicit_requirements
    ] == ["brief describe the malaysia history"]
    assert plan.exploration_directions == (
        "Compare alternative historical interpretations.",
    )
    assert mock_call.call_args.kwargs == {
        "temperature": 0.0,
        "model": "chosen-model",
    }
    planner_prompt = mock_call.call_args.args[0]
    normalized_planner_prompt = " ".join(planner_prompt.split())
    assert "goal_quote copied verbatim" in normalized_planner_prompt
    assert (
        "must never become evidence gates"
        in normalized_planner_prompt
    )


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
        plan, error = call_llm_for_search_queries(
            "Compare concept bottleneck models with Grad-CAM."
        )

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
                    "goal_quote": (
                        "deep-learning-based medical image classification"
                    ),
                },
                {
                    "id": "comparator",
                    "goal_quote": (
                        "post-hoc explanation methods such as SHAP and "
                        "Grad-CAM"
                    ),
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


def test_query_rewriting_failure_stops_before_retrieval():
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
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
    }
    grader_prompt = mock_call.call_args.args[0]
    assert "Keyword overlap" in grader_prompt
    assert "Exclude lexical collisions" in grader_prompt


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
    assert coverage.gap_queries == (
        "medical imaging CBM SHAP Grad-CAM",
    )


def test_literature_synthesis_keeps_only_findings_with_retrieved_sources():
    aspects = (
        EvidenceAspect("core_topic", "The user-stated core topic."),
    )
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
            "analytical_rationale": (
                "The supported premise motivates a testable comparison."
            ),
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
    assert [finding.claim for finding in synthesis.established_findings] == [
        "Supported premise."
    ]
    assert synthesis.established_findings[0].source_ids == (
        "arXiv:2205.15480v2",
    )


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
    assert fused[0]["arxiv_id"] == "2001.03488v1"
    assert fused[0]["_rrf_score"] > fused[1]["_rrf_score"]


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
    assert documents[0].metadata["source_id"] == (
        "arXiv:2401.00001v1"
    )


def test_generation_prompt_contains_retrieved_abstract_and_source_id():
    agent = GenerationAgent(debate_rounds=0)
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
    agent.rag_retriever.arxiv.search_papers.side_effect = [
        [paper],
        [paper],
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
    assert "If the evidence is insufficient or not directly relevant" in generation_prompt
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
    goal = (
        "Compare GNN intrusion detection with conventional DNN baselines "
        "in 5G under adversarial robustness"
    )
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
        document.page_content = (
            f"Source ID: arXiv:{arxiv_id}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )
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
    assert {
        source["source_id"] for source in context.last_retrieved_sources
    } == {
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
        document.page_content = (
            f"Source ID: arXiv:{arxiv_id}\nAbstract: directly relevant"
        )
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
        "RAG coverage auditing confirmed 2 supporting arXiv source(s), "
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
    )
    subject_document = Mock()
    subject_document.page_content = (
        "Source ID: arXiv:1111.1111\n"
        "Abstract: Evidence about the subject."
    )
    subject_document.metadata = {
        "source_id": "arXiv:1111.1111",
        "arxiv_id": "1111.1111",
        "title": "Subject evidence",
        "abstract": "Evidence about the subject.",
    }
    outcome_document = Mock()
    outcome_document.page_content = (
        "Source ID: arXiv:2222.2222\n"
        "Abstract: Evidence about the requested outcome."
    )
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
    assert mock_retrieve.call_args_list[1].kwargs["rerank_query"] == (
        "targeted requested outcome evidence scientific goal"
    )
    gap_plan = mock_retrieve.call_args_list[1].args[1]
    assert gap_plan.queries == (
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
    document.page_content = (
        "Source ID: arXiv:1111.1111\n"
        "Abstract: Evidence about only one aspect."
    )
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
    assert gap_plan.queries == ("scientific goal",)
    assert gap_plan.required_terms == ()


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
    assert "Candidate hypotheses from the preceding discussion" in (
        mock_debate.call_args_list[1].args[0]
    )
    final_debate_prompt = " ".join(
        mock_debate.call_args_list[2].args[0].split()
    )
    assert "may inspire refinement but are not requirements" in (
        final_debate_prompt
    )


def test_generation_debate_keeps_last_valid_turn_when_next_turn_fails():
    agent = GenerationAgent(debate_rounds=3)
    query_plan = SearchQueryPlan(
        queries=("query",),
        required_terms=("method",),
        explicit_requirements=(
            EvidenceAspect("core_topic", "The requested topic."),
        ),
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
