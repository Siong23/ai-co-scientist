import pytest

from app.agents import call_llm_for_generation


@pytest.mark.integration
def test_lmstudio_generation_success():
    """Exercise the configured local LM Studio server when explicitly requested."""
    result = call_llm_for_generation("Test prompt for LM Studio success", num_hypotheses=2, temperature=0.7)
    assert isinstance(result, list)
    assert all(h.get("title") != "Error" for h in result)
