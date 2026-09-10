from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from app import search_backoff


class Provider:
    api_key = ""
    last_error_status = None
    last_error_kind = ""

    def search_papers(self):
        pass


@pytest.fixture(autouse=True)
def isolated_cooldowns(monkeypatch):
    monkeypatch.setattr(search_backoff, "_states", {})


@pytest.mark.parametrize("status,delay", [(429, 60), (503, 60), (432, 300), (401, 300)])
def test_cooldown_shared_across_instances_and_expires(monkeypatch, status, delay):
    now = [100.0]
    monkeypatch.setattr(search_backoff.time, "monotonic", lambda: now[0])
    source = Provider()
    source.last_error_status = status
    search = Mock(return_value=[[]])
    assert search_backoff.guarded_search(source, search) == ([[]], False)
    assert search_backoff.guarded_search(Provider(), search) == ([], True)
    assert search.call_count == 1
    now[0] += delay
    assert search_backoff.guarded_search(Provider(), search) == ([[]], False)
    assert search.call_count == 2


def test_concurrent_agents_do_not_repeat_failed_request():
    def search():
        return [[]]

    sources = [Provider() for _ in range(8)]
    for source in sources:
        source.last_error_status = 429
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda source: search_backoff.guarded_search(source, search), sources))
    assert sum(not skipped for _, skipped in results) == 1


@pytest.mark.parametrize("status,kind", [(None, ""), (404, "provider_error")])
def test_empty_results_and_query_specific_errors_allow_other_queries(status, kind):
    source = Provider()
    source.last_error_status, source.last_error_kind = status, kind
    search = Mock(return_value=[[]])
    for _ in range(2):
        assert search_backoff.guarded_search(source, search) == ([[]], False)
    assert search.call_count == 2


def test_changed_credentials_can_retry_immediately():
    source = Provider()
    source.last_error_status = 432
    search_backoff.guarded_search(source, lambda: [])
    replacement = Provider()
    replacement.api_key = "offline-test-credential"
    assert search_backoff.guarded_search(replacement, lambda: [["evidence"]]) == ([["evidence"]], False)


def test_arxiv_timeout_is_classified_and_cooled_down():
    import requests

    from app.tools.arxiv_search import ArxivSearchTool

    source = ArxivSearchTool()
    source.client.results = Mock(side_effect=requests.ReadTimeout("read timed out"))
    assert search_backoff.guarded_search(source, lambda: source.search_papers("5G slicing")) == ([], False)
    assert source.last_error_kind == "timeout"
    assert search_backoff.guarded_search(source, lambda: source.search_papers("5G slicing")) == ([], True)
    assert source.client.results.call_count == 1


def test_provider_failures_reach_cycle_warnings():
    from app.agents import SupervisorAgent
    from app.models import ContextMemory, ResearchGoal

    supervisor = SupervisorAgent()
    supervisor.generation_agent.generate_new_hypotheses = Mock(return_value=([], []))
    supervisor.generation_agent.rag_retriever.last_search_stats = [
        {"source": "Tavily", "status": "provider_error"},
        {"source": "Tavily", "status": "cooldown"},
        {"source": "Springer", "status": "ok"},
    ]
    details = {}
    supervisor.step_generation(ResearchGoal(description="5G slicing"), ContextMemory(), Mock(), details)
    assert len(details["warnings"]) == 1
    assert "Tavily" in details["warnings"][0]
    assert "Springer" not in details["warnings"][0]
    assert details["steps"]["generation"]["warnings"] == details["warnings"]
