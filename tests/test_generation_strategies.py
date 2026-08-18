"""Offline tests for the focus_area and raw_idea generation strategies.

Paper reference: Gottweis et al. 2026 (Co-Scientist, Nature SI §2.1).
  - using focus areas:  13.8% GPQA contribution (highest single strategy).
  - using raw ideas:    2.66% GPQA contribution (high-diversity speculative).

All LLM calls are mocked; no network or API key is required.
"""

import json
from unittest.mock import patch

import pytest

from app.agents_modules.generation_helpers import (
    GENERATION_STRATEGIES,
    FocusArea,
    LiteratureFinding,
    LiteratureSynthesis,
    call_llm_for_focus_area_identification,
    generation_strategies_for_count,
    generation_strategy_instruction,
)

# ---------------------------------------------------------------------------
# GENERATION_STRATEGIES membership
# ---------------------------------------------------------------------------


def test_generation_strategies_contains_focus_area():
    assert "focus_area" in GENERATION_STRATEGIES


def test_generation_strategies_contains_raw_idea():
    assert "raw_idea" in GENERATION_STRATEGIES


def test_generation_strategies_still_contains_original_four():
    for strategy in ("literature_grounded", "contradiction_driven", "conditional_hop", "cross_paper_synthesis"):
        assert strategy in GENERATION_STRATEGIES, f"Missing original strategy: {strategy}"


# ---------------------------------------------------------------------------
# generation_strategy_instruction
# ---------------------------------------------------------------------------


def test_strategy_instruction_raw_idea_mentions_diversity_or_speculative():
    instruction = generation_strategy_instruction("raw_idea")
    assert "diversity" in instruction.lower() or "speculative" in instruction.lower()


def test_strategy_instruction_focus_area_with_focus_area_object():
    fa = FocusArea(
        area_id="fa_1",
        description="long-context efficiency under sparse attention",
        rationale="No retrieved paper addresses attention sparsity beyond 32k tokens.",
    )
    instruction = generation_strategy_instruction("focus_area", focus_area=fa)
    assert "long-context efficiency" in instruction
    assert "sparse attention" in instruction
    assert "No retrieved paper" in instruction


def test_strategy_instruction_focus_area_falls_back_when_none():
    fallback = generation_strategy_instruction("focus_area", focus_area=None)
    expected = generation_strategy_instruction("literature_grounded")
    assert fallback == expected


def test_strategy_instruction_unknown_raises():
    with pytest.raises(ValueError, match="Unknown generation strategy"):
        generation_strategy_instruction("invented_strategy")


# ---------------------------------------------------------------------------
# generation_strategies_for_count
# ---------------------------------------------------------------------------


def test_strategies_for_count_cycles_through_all():
    n = len(GENERATION_STRATEGIES)
    strategies = generation_strategies_for_count(n * 2)
    for s in GENERATION_STRATEGIES:
        assert strategies.count(s) >= 2, f"Strategy '{s}' appeared fewer than 2 times"


def test_strategies_for_count_zero_returns_empty():
    assert generation_strategies_for_count(0) == ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthesis():
    return LiteratureSynthesis(
        established_findings=(LiteratureFinding(claim="A is true.", source_ids=("src:1",)),),
        contradictions=(),
        knowledge_gaps=("Mechanism X is unknown.",),
        analytical_rationale="Evidence motivates further research.",
    )


# ---------------------------------------------------------------------------
# call_llm_for_focus_area_identification — happy path
# ---------------------------------------------------------------------------


def test_focus_area_identification_happy_path():
    llm_response = json.dumps(
        {
            "focus_areas": [
                {
                    "area_id": "fa_1",
                    "description": "Impact of sparse attention on retrieval",
                    "rationale": "No retrieved paper addresses this interaction.",
                    "suggested_queries": ["sparse attention retrieval"],
                },
                {
                    "area_id": "fa_2",
                    "description": "Cross-modal grounding in low-resource settings",
                    "rationale": "Only one paper touches this topic.",
                    "suggested_queries": ["cross-modal grounding low resource"],
                },
            ]
        }
    )
    with patch("app.agents.call_llm", return_value=llm_response):
        areas, error = call_llm_for_focus_area_identification(
            research_goal="How do sparse attention mechanisms affect RAG retrieval?",
            synthesis=_make_synthesis(),
            available_source_ids={"src:1", "src:2"},
            max_areas=2,
        )
    assert error is None
    assert len(areas) == 2
    assert areas[0].area_id == "fa_1"
    assert "sparse attention" in areas[0].description
    assert areas[1].area_id == "fa_2"


def test_focus_area_identification_caps_at_max_areas():
    llm_response = json.dumps(
        {
            "focus_areas": [
                {"area_id": f"fa_{i}", "description": f"Area {i}", "rationale": f"Rationale {i}"} for i in range(1, 6)
            ]
        }
    )
    with patch("app.agents.call_llm", return_value=llm_response):
        areas, error = call_llm_for_focus_area_identification("any goal", _make_synthesis(), {"src:1"}, max_areas=2)
    assert error is None
    assert len(areas) == 2


def test_focus_area_identification_max_areas_capped_at_five():
    llm_response = json.dumps(
        {
            "focus_areas": [
                {"area_id": f"fa_{i}", "description": f"Area {i}", "rationale": f"Rationale {i}"} for i in range(1, 10)
            ]
        }
    )
    with patch("app.agents.call_llm", return_value=llm_response):
        areas, error = call_llm_for_focus_area_identification("any goal", _make_synthesis(), {"src:1"}, max_areas=10)
    assert error is None
    assert len(areas) <= 5


# ---------------------------------------------------------------------------
# call_llm_for_focus_area_identification — error handling
# ---------------------------------------------------------------------------


def test_focus_area_identification_llm_error_returns_empty():
    with patch("app.agents.call_llm", return_value="Error: API key missing"):
        areas, error = call_llm_for_focus_area_identification("any goal", _make_synthesis(), {"src:1"})
    assert areas == []
    assert error is not None
    assert "Focus area identification failed" in error


def test_focus_area_identification_bad_json_returns_error():
    with patch("app.agents.call_llm", return_value="not json at all"):
        areas, error = call_llm_for_focus_area_identification("any goal", _make_synthesis(), {"src:1"})
    assert areas == []
    assert error is not None


def test_focus_area_identification_missing_array_key_returns_error():
    with patch("app.agents.call_llm", return_value=json.dumps({"wrong_key": []})):
        areas, error = call_llm_for_focus_area_identification("any goal", _make_synthesis(), {"src:1"})
    assert areas == []
    assert error is not None


def test_focus_area_identification_skips_items_without_description_or_rationale():
    llm_response = json.dumps(
        {
            "focus_areas": [
                {"area_id": "fa_1", "description": "", "rationale": "Some rationale"},
                {"area_id": "fa_2", "description": "Valid area", "rationale": ""},
                {"area_id": "fa_3", "description": "Good area", "rationale": "Good rationale"},
            ]
        }
    )
    with patch("app.agents.call_llm", return_value=llm_response):
        areas, error = call_llm_for_focus_area_identification("any goal", _make_synthesis(), {"src:1"}, max_areas=5)
    assert error is None
    assert len(areas) == 1
    assert areas[0].area_id == "fa_3"


def test_focus_area_identification_fenced_json_is_parsed():
    payload = json.dumps({"focus_areas": [{"area_id": "fa_1", "description": "Area", "rationale": "Rationale"}]})
    llm_response = f"```json\n{payload}\n```"
    with patch("app.agents.call_llm", return_value=llm_response):
        areas, error = call_llm_for_focus_area_identification("any goal", _make_synthesis(), {"src:1"})
    assert error is None
    assert len(areas) == 1


# ---------------------------------------------------------------------------
# FocusArea dataclass
# ---------------------------------------------------------------------------


def test_focus_area_is_immutable():
    fa = FocusArea(area_id="fa_1", description="D", rationale="R")
    with pytest.raises((AttributeError, TypeError)):
        fa.description = "changed"  # type: ignore[misc]


def test_focus_area_default_suggested_queries_is_empty_tuple():
    fa = FocusArea(area_id="fa_1", description="D", rationale="R")
    assert fa.suggested_queries == ()
