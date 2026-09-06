"""Offline regressions for relevance gates, latency and runtime diagnostics."""

import logging
from threading import Barrier
from unittest.mock import Mock, patch

import numpy as np
from langchain_core.documents import Document

from app.agents_modules.generation import GenerationAgent
from app.agents_modules.generation_helpers import EvidenceCoverage
from app.agents_modules.supervisor import SupervisorAgent
from app.models import ContextMemory, ResearchGoal
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


def test_failed_empty_generation_is_not_repeated_until_budget_exhaustion():
    supervisor = SupervisorAgent(mode="dynamic")
    supervisor.generation_agent.generate_new_hypotheses = Mock(return_value=([], ["No verified relevant evidence"]))
    result = supervisor.run_dynamic_cycle(ResearchGoal("goal"), ContextMemory(), max_steps=10, planner_mode="heuristic")
    supervisor.generation_agent.generate_new_hypotheses.assert_called_once()
    assert result["finalization"]["status"] == "generation_failed"
    assert result["supervisor_state"]["status"] == "incomplete"


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
