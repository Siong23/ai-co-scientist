"""Offline regression tests for literature retrieval, evidence coverage, and query planning fixes.

Covers:
- Partial query completion without discarding valid queries when intents are missing.
- Separation of original requirements from optional exploration directions.
- Joint evidence coverage across multiple sources.
- Full-text retrieval fixes (Nature.com allowlist and lane fallback in search_many).
- Service error classification, cooldowns, and recovery (Elsevier, Tavily, Springer, Backoff).
- Minimal fallback plan decomposition and retrieval diversity.
- Query sanitization for contradictory evidence searches.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from app.agents_modules.generation import GenerationAgent
from app.agents_modules.generation_helpers import (
    EvidenceCoverage,
    call_llm_for_search_queries,
)
from app.agents_modules.reflection_helpers import _sanitize_claim_query
from app.paper_library import ChromaPaperLibrary, PaperChunk
from app.rag_retriever import EvidenceAspect, ResearchRetriever, SearchQuery, SearchQueryPlan
from app.search_backoff import guarded_search
from app.tools.elsevier_search import ElsevierSearchTool
from app.tools.springer_search import SpringerSearchTool
from app.tools.tavily_search import TavilySearchTool

# ---------------------------------------------------------------------------
# 1. Partial query completion without discarding valid queries
# ---------------------------------------------------------------------------


def test_partial_query_completion_synthesizes_missing_prior_art():
    """When the LLM query plan returns support and counterevidence but misses prior_art,

    it must NOT discard the plan; it should synthesize a targeted prior_art query.
    """
    goal = "Develop a closed-loop multi-agent AI framework to dynamically allocate 5G slice bandwidth during traffic spikes"

    plan_response = """
    {
      "queries": [
        {
          "query": "5G slice bandwidth dynamic allocation multi-agent reinforcement learning",
          "purpose": "Find support for dynamic slice allocation",
          "sub_question": "How can multi-agent RL allocate bandwidth?",
          "source_type": "academic",
          "search_intent": "support"
        },
        {
          "query": "5G network slicing traffic spikes latency SLA violations",
          "purpose": "Find counterevidence or bottlenecks during spikes",
          "sub_question": "What are the latency bottlenecks during traffic spikes?",
          "source_type": "academic",
          "search_intent": "counterevidence"
        },
        {
          "query": "closed-loop autonomous network resource allocation framework",
          "purpose": "General architecture support",
          "sub_question": "What closed-loop architectures exist?",
          "source_type": "academic",
          "search_intent": "support"
        }
      ],
      "required_terms": ["5G", "bandwidth", "slice"],
      "explicit_requirements": [
        {
          "id": "slice_bandwidth",
          "goal_quote": "dynamically allocate 5G slice bandwidth",
          "evidence_need": "methods for dynamically allocating bandwidth to 5G slices"
        }
      ],
      "exploration_directions": ["hierarchical control", "action masking"]
    }
    """

    planner_response = """
    {
      "research_goal": "Develop a closed-loop multi-agent AI framework to dynamically allocate 5G slice bandwidth during traffic spikes",
      "research_type": "methodological",
      "key_entities": ["5G", "slice bandwidth", "traffic spikes"],
      "constraints": [],
      "sub_questions": ["How to allocate bandwidth dynamically?"],
      "evidence_requirements": ["Bandwidth allocation methods"],
      "freshness_requirement": "none",
      "ambiguities": [],
      "search_strategy": "academic",
      "provisional_hypotheses": [
        {
          "hypothesis_id": "primary_hypothesis",
          "role": "primary",
          "statement": "Multi-agent coordination enables sub-second slice bandwidth reallocation during traffic spikes",
          "goal_quote": "dynamically allocate 5G slice bandwidth"
        },
        {
          "hypothesis_id": "alternative_hypothesis",
          "role": "alternative",
          "statement": "Centralized heuristic control outperforms distributed agents under sudden traffic bursts",
          "goal_quote": "traffic spikes"
        },
        {
          "hypothesis_id": "null_hypothesis",
          "role": "null",
          "statement": "Multi-agent coordination overhead negates dynamic allocation gains during traffic spikes",
          "goal_quote": "traffic spikes"
        }
      ]
    }
    """

    with patch("app.agents.call_llm", side_effect=[planner_response, plan_response]):
        plan, error = call_llm_for_search_queries(goal, query_count=5)

    assert error is None
    assert plan is not None
    intents = {q.search_intent for q in plan.queries}
    assert "support" in intents
    assert "counterevidence" in intents
    assert "prior_art" in intents, "Missing prior_art query should have been synthesized"
    # Ensure original valid queries were retained
    assert any("multi-agent reinforcement learning" in q.query for q in plan.queries)


# ---------------------------------------------------------------------------
# 2. Separation of original requirements from optional directions
# ---------------------------------------------------------------------------


def test_separation_of_requirements_from_optional_directions():
    """Exploration directions must remain optional and not become required terms or explicit requirements."""
    goal = "Develop a closed-loop multi-agent AI framework to dynamically allocate 5G slice bandwidth during traffic spikes"

    plan = SearchQueryPlan(
        queries=(
            SearchQuery(query="5G slice bandwidth allocation", search_intent="support"),
            SearchQuery(query="5G slicing traffic spike latency", search_intent="counterevidence"),
            SearchQuery(query="5G slice resource management prior art", search_intent="prior_art"),
        ),
        required_terms=("5G", "bandwidth"),
        explicit_requirements=(
            EvidenceAspect(
                aspect_id="bandwidth_allocation",
                description="dynamic allocation of 5G slice bandwidth",
                goal_quote="dynamically allocate 5G slice bandwidth",
            ),
        ),
        exploration_directions=("hierarchical reinforcement learning", "digital twin simulation"),
    )

    assert plan.explicit_requirements[0].goal_quote in goal
    # Required terms only contain core entities, not optional exploration directions
    assert "hierarchical reinforcement learning" not in plan.required_terms
    assert "digital twin simulation" not in plan.required_terms

    # Explicit requirements only contain user goal aspects
    aspect_ids = {a.aspect_id for a in plan.explicit_requirements}
    assert "hierarchical reinforcement learning" not in aspect_ids
    assert len(plan.exploration_directions) == 2


# ---------------------------------------------------------------------------
# 3. Joint coverage from multiple sources
# ---------------------------------------------------------------------------


def test_joint_coverage_from_multiple_sources():
    """Multiple distinct sources should jointly cover different explicit requirements."""
    req_coordination = EvidenceAspect(
        aspect_id="coordination",
        description="multi-agent coordination mechanisms",
        goal_quote="multi-agent AI framework",
    )
    req_bandwidth = EvidenceAspect(
        aspect_id="bandwidth",
        description="5G slice bandwidth allocation during traffic spikes",
        goal_quote="dynamically allocate 5G slice bandwidth during traffic spikes",
    )

    # Source A covers coordination; Source B covers bandwidth allocation
    coverage = EvidenceCoverage(
        aspect_source_ids={
            req_coordination.aspect_id: ("arXiv:2401.00001",),
            req_bandwidth.aspect_id: ("springer:10.1007/example-paper",),
        },
        missing_aspect_ids=(),
        gap_queries=(),
        reason="Joint coverage across multi-agent coordination and 5G bandwidth allocation.",
    )

    assert coverage.sufficient is True
    assert len(coverage.missing_aspect_ids) == 0
    assert "coordination" in coverage.aspect_source_ids
    assert "bandwidth" in coverage.aspect_source_ids
    # Multiple different sources provide joint grounding
    all_sources = {s for s_list in coverage.aspect_source_ids.values() for s in s_list}
    assert len(all_sources) == 2


# ---------------------------------------------------------------------------
# 4. Full-text retrieval fixes: Nature allowlist and lane fallback
# ---------------------------------------------------------------------------


def test_nature_com_allowed_in_pdf_hosts():
    """Nature.com hosts must be accepted by the PDF validator for Springer Nature 10.1038 papers."""
    library = ChromaPaperLibrary(enabled=True)
    assert "www.nature.com" in library.allowed_pdf_hosts
    assert "nature.com" in library.allowed_pdf_hosts
    # Should not raise ValueError
    library._validate_pdf_url("https://www.nature.com/articles/s41598-026-40237-8.pdf")
    library._validate_pdf_url("https://nature.com/articles/s41598-026-40237-8.pdf")

    # Untrusted domain still raises
    with pytest.raises(ValueError, match="PDF host is not allowed"):
        library._validate_pdf_url("https://malicious-site.example.com/paper.pdf")


def test_search_many_falls_back_when_lane_has_no_committed_sources():
    """When a requirement lane has no committed papers (e.g. initial round),

    search_many should fall back to searching all indexed sources instead of returning [].
    """
    library = ChromaPaperLibrary(enabled=True)
    library.retrieval_workers = 1

    chunk = PaperChunk(
        source_id="springer:10.1007/test-source",
        title="Test 5G Slicing Paper",
        page=1,
        text="Dynamic bandwidth allocation for 5G slices under spike traffic.",
        chunk_id="chunk-1",
    )

    with patch.object(library, "search", return_value=[chunk]) as mock_search:
        # Lane 'goal_scope' has no sources, but 'source_ids' has the indexed source
        results = library.search_many(
            queries=["5G slice bandwidth"],
            source_ids=["springer:10.1007/test-source"],
            top_k=5,
            source_ids_by_requirement={"goal_scope": []},
        )

    # Must fall back to searching the available source_ids instead of returning empty
    assert len(results) > 0
    assert results[0].source_id == "springer:10.1007/test-source"
    mock_search.assert_called_once_with("5G slice bandwidth", ["springer:10.1007/test-source"], 5)


# ---------------------------------------------------------------------------
# 5. Service error classification, cooldowns, and normal recovery
# ---------------------------------------------------------------------------


def test_elsevier_scopus_clears_error_status_on_success():
    """Elsevier Scopus must have last_error_status=None on HTTP 200 so _provider_status does not report provider_error."""
    tool = ElsevierSearchTool()
    tool.api_key = "test-key"

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "search-results": {
            "entry": [
                {
                    "dc:title": "5G Slice Bandwidth Allocation",
                    "dc:description": "Abstract of the paper.",
                    "eid": "2-s2.0-123456",
                }
            ]
        }
    }

    with patch("requests.get", return_value=mock_response):
        papers = tool.search_papers("5G slice")

    assert len(papers) == 1
    assert tool.last_error_status is None, "last_error_status must be None on HTTP 200 success"

    # Verify provider_status treats it as ok
    status = ResearchRetriever._provider_status(tool, 1, 1, 1)
    assert status == "ok", f"Expected 'ok' status, got {status!r}"


def test_tavily_search_classifies_432_as_quota_or_plan_rejection():
    """Tavily search must classify HTTP 432 (Plan Limit Exceeded) as quota_or_plan_rejection."""
    tool = TavilySearchTool()
    tool.api_key = "test-key"

    mock_response = Mock()
    mock_response.status_code = 432
    http_error = requests.HTTPError("432 Client Error: Plan Limit Exceeded")
    http_error.response = mock_response

    with patch("requests.post", side_effect=http_error):
        results = tool.search("5G slice bandwidth")

    assert results == []
    assert tool.last_error_status == 432
    assert tool.last_error_kind == "quota_or_plan_rejection"


def test_springer_search_tracks_error_status_and_kind():
    """Springer search must track last_error_status and last_error_kind on failure."""
    tool = SpringerSearchTool()
    tool.api_key = "test-key"

    mock_response = Mock()
    mock_response.status_code = 429
    http_error = requests.HTTPError("429 Too Many Requests")
    http_error.response = mock_response

    with patch("requests.get", side_effect=http_error):
        results = tool.search_papers("5G slice bandwidth")

    assert results == []
    assert tool.last_error_status == 429
    assert tool.last_error_kind == "rate_limited"


def test_guarded_search_does_not_cooldown_on_http_400():
    """A client syntax error (HTTP 400) must NOT trigger a 30s provider cooldown for subsequent queries."""
    source = Mock()
    source.api_key = "unique-test-key-400"
    source.last_error_status = 400
    source.last_error_kind = "provider_error"

    call_count = 0

    def mock_search():
        nonlocal call_count
        call_count += 1
        return []

    # First search returns HTTP 400
    results, in_cooldown = guarded_search(source, mock_search, provider_name="test_400")
    assert in_cooldown is False

    # Second search should NOT be in cooldown because 400 is query-specific
    results2, in_cooldown2 = guarded_search(source, mock_search, provider_name="test_400")
    assert in_cooldown2 is False
    assert call_count == 2


# ---------------------------------------------------------------------------
# 6. Minimal fallback plan decomposition and diversity
# ---------------------------------------------------------------------------


def test_minimal_fallback_plan_decomposes_goal_and_diversifies_queries():
    """Fallback plan must not simply repeat the raw goal; it must decompose aspects and diversify queries."""
    goal = "Develop a closed-loop multi-agent AI framework to dynamically allocate 5G slice bandwidth during traffic spikes"

    plan = GenerationAgent._build_minimal_fallback_plan(goal)

    assert plan is not None
    # Multiple diverse queries: goal, prior_art, support, counterevidence
    intents = {q.search_intent for q in plan.queries}
    assert "goal" in intents
    assert "prior_art" in intents
    assert "support" in intents
    assert "counterevidence" in intents
    assert len(plan.queries) >= 4

    # The goal was decomposed into multiple aspects
    assert len(plan.explicit_requirements) >= 2
    for aspect in plan.explicit_requirements:
        assert aspect.goal_quote in goal, f"Aspect quote '{aspect.goal_quote}' must be a substring of goal"


# ---------------------------------------------------------------------------
# 7. Query sanitization for contradictory evidence searches
# ---------------------------------------------------------------------------


def test_sanitize_claim_query_removes_punctuation_and_bounds_length():
    """_sanitize_claim_query must strip punctuation, brackets, and excessive words that cause 404/400 errors."""
    raw_claim = (
        "Integrating LLMs into telecommunication systems (e.g., 5G/6G) offers potential for dynamic "
        "network reconfiguration, automated slice management, and intelligent closed-loop control."
    )

    sanitized = _sanitize_claim_query(raw_claim, max_words=10)

    # Must not contain periods, commas, or parentheses
    assert "." not in sanitized
    assert "," not in sanitized
    assert "(" not in sanitized
    assert ")" not in sanitized
    assert "/" not in sanitized
    words = sanitized.split()
    assert len(words) <= 10
    assert len(words) >= 4
