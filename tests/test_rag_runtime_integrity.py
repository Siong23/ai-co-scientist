"""Offline regressions for relevance gates, latency and runtime diagnostics."""

import json
import logging
from threading import Barrier
from unittest.mock import Mock, patch

import numpy as np
from langchain_core.documents import Document

from app.agents_modules.generation import GenerationAgent
from app.agents_modules.generation_helpers import EvidenceCoverage
from app.agents_modules.supervisor import SupervisorAgent
from app.models import ContextMemory, Hypothesis, PairwiseDecision, ResearchGoal
from app.rag_retriever import (
    EvidenceAspect,
    SearchQueryPlan,
    SharedSentenceTransformerEmbeddings,
    format_documents_for_grading,
)
from app.runtime_logging import configure_runtime_logging
from app.utils import _format_lmstudio_error, _native_request_timeout, _openai_timeout, call_llm


def goal_plan():
    return SearchQueryPlan(
        queries=("5G slice bandwidth traffic spikes",),
        required_terms=(),
        explicit_requirements=(EvidenceAspect("scope", "5G slice bandwidth under traffic spikes"),),
    )


def test_initial_search_overlaps_planning_when_models_can_run_concurrently(monkeypatch):
    monkeypatch.setitem(
        __import__("app.agents_modules.generation", fromlist=["config"]).config,
        "serialize_lmstudio_model_calls",
        False,
    )
    agent = GenerationAgent()
    rendezvous = Barrier(2, timeout=3)
    plan = goal_plan()
    goal = ResearchGoal("5G slice bandwidth under traffic spikes")
    document = Document(page_content="5G evidence", metadata={"source_id": "source1"})

    def search(received_goal):
        assert received_goal is goal
        rendezvous.wait()
        return [document]

    def planning(*args, **kwargs):
        assert args[0] == goal.description
        rendezvous.wait()
        return plan, None

    with (
        patch.object(agent, "_retrieve_original_scientific_sources", side_effect=search),
        patch(
            "app.agents_modules.generation.call_llm_for_search_queries",
            side_effect=planning,
        ),
    ):
        assert agent._plan_and_retrieve_initial(goal) == (plan, None, [document])


def test_initial_search_serializes_lmstudio_chat_and_embedding_models(monkeypatch):
    module = __import__("app.agents_modules.generation", fromlist=["config"])
    monkeypatch.setitem(module.config, "use_lmstudio_embeddings", True)
    monkeypatch.setitem(module.config, "serialize_lmstudio_model_calls", True)
    agent = GenerationAgent()
    plan = goal_plan()
    goal = ResearchGoal("5G slice bandwidth under traffic spikes")
    document = Document(page_content="5G evidence", metadata={"source_id": "source1"})
    events = []

    def planning(*_args, **_kwargs):
        events.append("planning")
        return plan, None

    def search(received_goal):
        assert received_goal is goal
        assert events == ["planning"]
        events.append("retrieval")
        return [document]

    with (
        patch.object(agent, "_retrieve_original_scientific_sources", side_effect=search),
        patch("app.agents_modules.generation.call_llm_for_search_queries", side_effect=planning),
    ):
        assert agent._plan_and_retrieve_initial(goal) == (plan, None, [document])

    assert events == ["planning", "retrieval"]


def test_coverage_failure_cannot_manufacture_support():
    agent = GenerationAgent(agentic_research_enabled=False)
    context = ContextMemory()
    document = Document(page_content="Unverified result", metadata={"source_id": "source1"})
    with (
        patch.object(agent, "_plan_and_retrieve_initial", return_value=(goal_plan(), None, [document])),
        patch.object(
            agent,
            "_grade_candidate_evidence",
            return_value=(["source1"], None, None, "coverage timeout"),
        ),
        patch("app.agents_modules.generation.call_llm_for_generation") as generate,
    ):
        hypotheses, errors = agent.generate_new_hypotheses(ResearchGoal("5G slice bandwidth"), context)
    assert not hypotheses and errors == ["coverage timeout"]
    assert context.last_retrieved_sources == []
    generate.assert_not_called()


def test_minimum_source_count_cannot_be_filled_with_unrelated_candidates():
    agent = GenerationAgent(agentic_research_enabled=False)
    agent.rag_retriever.minimum_relevant_sources = 2
    documents = [
        Document(page_content="5G bandwidth evidence", metadata={"source_id": "related"}),
        Document(page_content="Unrelated domain", metadata={"source_id": "unrelated"}),
    ]
    coverage = EvidenceCoverage(
        aspect_source_ids={"scope": ("related",)}, missing_aspect_ids=(), gap_queries=(), reason="Covered"
    )
    with (
        patch.object(agent, "_plan_and_retrieve_initial", return_value=(goal_plan(), None, documents)),
        patch.object(
            agent,
            "_grade_candidate_evidence",
            return_value=(["related"], None, coverage, None),
        ),
        patch("app.agents_modules.generation.call_llm_for_literature_synthesis") as synthesize,
    ):
        hypotheses, errors = agent.generate_new_hypotheses(ResearchGoal("5G slice bandwidth"), ContextMemory())
    assert not hypotheses
    assert "at least 2" in errors[0]
    synthesize.assert_not_called()


def test_full_text_requirement_cannot_fall_back_when_every_download_fails():
    document = Document(page_content="Only an abstract", metadata={"source_id": "source1"})
    library = Mock(enabled=True, require_indexed_sources_for_generation=True)
    library.enrich_documents.return_value = [document]
    agent = GenerationAgent(paper_library=library)
    assert agent._prepare_candidate_documents([document], ResearchGoal("goal")) == []


def test_empty_evidence_does_not_spend_llm_calls_on_grading():
    with patch("app.agents.call_llm") as llm:
        result = GenerationAgent()._grade_candidate_evidence(ResearchGoal("goal"), goal_plan(), "", set())
    assert result[2].missing_aspect_ids == ("scope",)
    llm.assert_not_called()


def test_synthesis_failure_preserves_validated_retrieval_diagnostics():
    source_id = "arXiv:2205.15480v2"
    document = Document(
        page_content="Closed-loop allocation evidence",
        metadata={
            "source_id": source_id,
            "title": "Closed-loop allocation",
            "abstract": "Closed-loop allocation evidence",
        },
    )
    coverage = EvidenceCoverage(
        aspect_source_ids={"scope": (source_id,)},
        missing_aspect_ids=(),
        gap_queries=(),
        reason="Covered",
    )
    library = Mock(enabled=False, require_indexed_sources_for_generation=False)
    library.enrich_documents.side_effect = lambda documents, *_args: list(documents)
    agent = GenerationAgent(
        minimum_relevant_sources=1,
        audit_enabled=False,
        paper_library=library,
        agentic_research_enabled=False,
    )
    with (
        patch.object(
            agent,
            "_plan_and_retrieve_initial",
            return_value=(goal_plan(), None, [document]),
        ),
        patch.object(
            agent,
            "_grade_candidate_evidence",
            return_value=([source_id], None, coverage, None),
        ),
        patch(
            "app.agents_modules.generation.call_llm_for_literature_synthesis",
            return_value=(None, "Literature synthesis failed after format repair: invalid JSON"),
        ),
        patch("app.agents_modules.generation.call_llm_for_generation") as generate,
    ):
        context = ContextMemory()
        hypotheses, errors = agent.generate_new_hypotheses(
            ResearchGoal("5G slice bandwidth under traffic spikes"),
            context,
        )

    assert hypotheses == []
    assert errors == ["Literature synthesis failed after format repair: invalid JSON"]
    assert [source["source_id"] for source in context.last_retrieved_sources] == [source_id]
    diagnostics = context.last_generation_diagnostics
    assert diagnostics["evidence_retrieval"]["status"] == "completed"
    assert diagnostics["literature_synthesis"]["status"] == "failed"
    assert diagnostics["hypothesis_generation"]["status"] == "not_executed"
    assert diagnostics["evidence_consumed"] is False
    generate.assert_not_called()


def test_failed_empty_generation_is_not_repeated_until_budget_exhaustion():
    supervisor = SupervisorAgent(mode="dynamic")
    supervisor.generation_agent.generate_new_hypotheses = Mock(return_value=([], ["No verified relevant evidence"]))
    result = supervisor.run_dynamic_cycle(ResearchGoal("goal"), ContextMemory(), max_steps=10, planner_mode="heuristic")
    supervisor.generation_agent.generate_new_hypotheses.assert_called_once()
    assert result["finalization"]["status"] == "generation_failed"
    assert result["supervisor_state"]["status"] == "incomplete"


def test_empty_literature_rationale_continues_through_the_supervised_cycle():
    """The reported 5G response must recover before Reflection and Ranking."""

    source_id = "arXiv:2205.15480v2"
    plan = SearchQueryPlan(
        queries=("5G slice bandwidth traffic spikes",),
        required_terms=(),
        explicit_requirements=(
            EvidenceAspect(
                "scope",
                "Closed-loop 5G slice bandwidth allocation during traffic spikes.",
            ),
        ),
    )
    document = Document(
        page_content=(
            f"Source ID: {source_id}\n"
            "Title: Closed-loop 5G allocation\n"
            "Abstract: Closed-loop allocation improves responsiveness."
        ),
        metadata={
            "source_id": source_id,
            "title": "Closed-loop 5G allocation",
            "abstract": "Closed-loop allocation improves responsiveness.",
        },
    )
    coverage = EvidenceCoverage(
        aspect_source_ids={"scope": (source_id,)},
        missing_aspect_ids=(),
        gap_queries=(),
        reason="The validated source covers the explicit 5G requirement.",
    )
    synthesis_payload = json.dumps(
        {
            "established_findings": [
                {
                    "claim": "Closed-loop allocation improves responsiveness.",
                    "source_ids": [source_id],
                }
            ],
            "contradictions": [],
            "knowledge_gaps": ["Performance during abrupt traffic spikes is unresolved."],
            "analytical_rationale": "",
        }
    )
    generation_payload = json.dumps(
        [
            {
                "title": "Feedback allocation A",
                "hypothesis": "A closed-loop controller can improve spike response.",
                "rationale": "This is a new inference from the validated finding and gap.",
                "feasibility": "Compare it with a static allocation baseline.",
                "source_ids": [source_id],
            },
            {
                "title": "Feedback allocation B",
                "hypothesis": "A second feedback policy can improve spike recovery.",
                "rationale": "This is a distinct testable inference from the same gap.",
                "feasibility": "Measure recovery against a static allocation baseline.",
                "source_ids": [source_id],
            },
        ]
    )

    paper_library = Mock(enabled=False, require_indexed_sources_for_generation=False)
    paper_library.enrich_documents.side_effect = lambda documents, *_args: list(documents)
    generation_agent = GenerationAgent(
        minimum_relevant_sources=1,
        debate_rounds=0,
        audit_enabled=False,
        paper_library=paper_library,
        agentic_research_enabled=False,
    )
    generation_agent._plan_and_retrieve_initial = Mock(return_value=(plan, None, [document]))
    generation_agent._grade_candidate_evidence = Mock(return_value=([source_id], None, coverage, None))

    supervisor = SupervisorAgent(mode="dynamic")
    supervisor.generation_agent = generation_agent
    evolved = Hypothesis(
        "E1",
        "Evolved feedback allocation",
        "Hypothesis: An evolved feedback policy remains testable.",
    )
    evolved.parent_ids = ["G-parent"]
    evolved.evidence_source_ids = [source_id]
    evolved.evidence_sources = [
        {
            "source_id": source_id,
            "title": "Closed-loop 5G allocation",
            "abstract": "Closed-loop allocation improves responsiveness.",
        }
    ]

    def evolve_once(context, _research_goal):
        context.last_evolution_attempts = [{"strategy": "grounding", "status": "accepted", "reason": "test"}]
        return [evolved]

    supervisor.evolution_agent.evolve_hypotheses = Mock(side_effect=evolve_once)
    reflection_calls = []

    def reflect(hypothesis, *_args, **_kwargs):
        reflection_calls.append(hypothesis.hypothesis_id)
        return {
            "novelty_review": "HIGH",
            "feasibility_review": "HIGH",
            "alignment_score": 8,
            "novelty_score": 8,
            "feasibility_score": 8,
            "plausibility_score": 8,
            "testability_score": 8,
            "evidence_quality_score": 8,
            "expected_research_value_score": 8,
            "strengths": ["Grounded and testable."],
            "weaknesses": [],
            "recommendation": "ACCEPT",
            "sub_claims": ["The proposed controller improves spike response."],
            "comment": "Accept for pairwise comparison.",
            "references": [source_id],
        }

    def assess_claims(hypothesis, **_kwargs):
        return {
            "claims": [
                {
                    "claim": "The proposed controller improves spike response.",
                    "status": "SUPPORTED",
                    "confidence": 8.0,
                    "supporting_evidence": [{"source_id": source_id}],
                    "contradictory_evidence": [],
                }
            ],
            "overall_confidence": 8.0,
        }

    ranking_calls = []

    def rank_pair(hypothesis_a, hypothesis_b, _research_goal):
        ranking_calls.append((hypothesis_a.hypothesis_id, hypothesis_b.hypothesis_id))
        return PairwiseDecision(
            hypothesis_a_id=hypothesis_a.hypothesis_id,
            hypothesis_b_id=hypothesis_b.hypothesis_id,
            outcome="A",
            scores_a={"quality": 8.0},
            scores_b={"quality": 7.0},
            decisive_criteria=["evidence"],
            confidence=8,
            reasoning="Both candidates were independently reviewed.",
        )

    def proximity(context, **_kwargs):
        hypothesis_ids = list(context.hypotheses)
        result = {
            "graph": {"adjacency_graph": {}, "nodes": [], "edges": []},
            "clusters": {hypothesis_id: index for index, hypothesis_id in enumerate(hypothesis_ids)},
            "cluster_members": {},
            "near_duplicates": [],
            "diversity_score": 1.0,
        }
        context.proximity_analysis = result
        return result

    supervisor.proximity_agent.get_proximity_analysis = Mock(side_effect=proximity)
    supervisor.meta_review_agent.summarize_and_feedback = Mock(
        return_value={
            "meta_review_critique": ["Cross-hypothesis review completed."],
            "research_overview": {"suggested_next_steps": []},
        }
    )

    with (
        patch("app.agents.call_llm", side_effect=[synthesis_payload, generation_payload]),
        patch(
            "app.agents_modules.reflection.call_llm_for_reflection",
            side_effect=reflect,
        ),
        patch(
            "app.agents_modules.reflection.evaluate_claims",
            side_effect=assess_claims,
        ),
        patch(
            "app.agents_modules.ranking.run_pairwise_debate",
            side_effect=rank_pair,
        ),
    ):
        context = ContextMemory()
        result = supervisor.run_dynamic_cycle(
            ResearchGoal(
                "Develop a closed-loop multi-agent AI framework to dynamically "
                "allocate 5G slice bandwidth during traffic spikes",
                num_hypotheses=2,
            ),
            context,
            max_steps=10,
            planner_mode="heuristic",
        )

    actions = [decision["action"] for decision in result["supervisor_decisions"]]
    assert result["finalization"]["status"] != "generation_failed"
    assert len(context.hypotheses) >= 2
    assert reflection_calls
    assert ranking_calls
    assert actions.index("REFLECT") < actions.index("RANK")
    assert "EVOLVE" in actions
    assert "PROXIMITY" in actions
    assert "META_REVIEW" in actions
    assert result["steps"]["generation"]["stages"]["literature_synthesis"]["status"] == "warning"
    assert result["warnings"] == [
        "Literature synthesis omitted analytical_rationale; a conservative "
        "rationale was constructed from validated findings and gaps."
    ]


def test_grading_uses_retrieved_passages_instead_of_abstract_when_indexed():
    document = Document(
        page_content="wrapper",
        metadata={
            "source_id": "source1",
            "abstract": "Generic background",
            "full_text_indexed": True,
            "evidence_refs": [{"evidence_type": "full_text", "text": "Measured latency during traffic spikes"}],
        },
    )
    digest = format_documents_for_grading([document])
    assert "Measured latency during traffic spikes" in digest
    assert "Generic background" not in digest


def test_document_embedding_cache_reuses_duplicates_and_returns_copies():
    model = Mock()
    model.encode.side_effect = lambda texts, **kw: np.array([[len(text), 1.0] for text in texts])
    adapter = SharedSentenceTransformerEmbeddings()
    with patch("app.rag_retriever.get_sentence_transformer_model", return_value=model):
        first = adapter.embed_documents(["old", "old", "other"])
        first[0][0] = 999
        second = adapter.embed_documents(["old", "new"])
    assert model.encode.call_args_list[0].args[0] == ["old", "other"]
    assert model.encode.call_args_list[1].args[0] == ["new"]
    assert second == [[3, 1.0], [3, 1.0]]


def test_document_embedding_cache_is_bounded_and_invalidated_on_model_change():
    models = [Mock(), Mock()]
    for model in models:
        model.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 2))
    adapter = SharedSentenceTransformerEmbeddings()
    adapter._embedding_cache_size = 1
    with patch("app.rag_retriever.get_sentence_transformer_model", return_value=models[0]):
        assert len(adapter.embed_documents(["a", "b"])) == 2
        assert len(adapter._embedding_cache) == 1
    with patch("app.rag_retriever.get_sentence_transformer_model", return_value=models[1]):
        adapter.embed_documents(["b"])
    models[1].encode.assert_called_once()


def test_runtime_log_is_bounded_idempotent_and_redacts_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-placeholder-sensitive")
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        path = configure_runtime_logging(tmp_path)
        assert configure_runtime_logging(tmp_path) == path
        added = [handler for handler in root.handlers if handler not in before]
        assert len(added) == 1
        assert added[0].maxBytes == 5_000_000 and added[0].backupCount == 3
        try:
            raise RuntimeError("test-placeholder-sensitive")
        except RuntimeError:
            logging.getLogger("aicoscientist").exception("Provider failed: %s", "test-placeholder-sensitive")
        content = path.read_text(encoding="utf-8")
        assert "test-placeholder-sensitive" not in content
        assert "REDACTED" in content and "RuntimeError" in content
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_server_error_keeps_redacted_body_for_diagnosis(monkeypatch):
    import requests

    monkeypatch.setenv("LMSTUDIO_API_KEY", "test-credential-placeholder")
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"KV cache allocation failed test-credential-placeholder"}'
    error = requests.HTTPError("500 Server Error", response=response)
    message = _format_lmstudio_error(error, "chosen-model")
    assert "KV cache allocation failed" in message
    assert "test-credential-placeholder" not in message


def test_native_server_errors_fall_back_to_openai_compatible_chat(monkeypatch):
    import requests

    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"native inference engine failed"}'
    completion = Mock()
    completion.choices = [Mock(message=Mock(content="fallback result"))]
    client = Mock()
    client.chat.completions.create.return_value = completion
    monkeypatch.setitem(__import__("app.utils", fromlist=["config"]).config, "lmstudio_native_server_error_retries", 1)
    with (
        patch("app.utils.requests.post", side_effect=requests.HTTPError("500", response=response)) as native,
        patch("app.utils.OpenAI", return_value=client),
        patch("app.utils.time.sleep"),
    ):
        result = call_llm("goal", model="model", reasoning="off")
    assert result == "fallback result"
    assert native.call_count == 2
    client.chat.completions.create.assert_called_once()


def test_lmstudio_timeouts_separate_fast_connect_from_long_read(monkeypatch):
    module = __import__("app.utils", fromlist=["config"])
    monkeypatch.setitem(module.config, "llm_request_timeout_seconds", 300)
    monkeypatch.setitem(module.config, "lmstudio_connect_timeout_seconds", 10)
    assert _native_request_timeout() == (10.0, 300.0)
    timeout = _openai_timeout()
    assert timeout.connect == 10.0
    assert timeout.read == 300.0
