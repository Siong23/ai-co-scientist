"""Offline tests of the LLM boundary: parsing and error propagation.

The LM Studio call goes through the OpenAI SDK client in app.utils.call_llm;
these tests mock that client so no local server or network is needed.
They replace the coverage of the deleted FastAPI-era tests/test_api.py.
"""

import json
from unittest.mock import MagicMock, patch

import app.utils as utils
from app.agents import call_llm_for_generation, call_llm_for_reflection


def _completion(content: str):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    return completion


def test_generation_happy_path_parses_hypotheses():
    payload = json.dumps(
        [
            {"title": "Hypothesis A", "text": "Perovskite tandem cells."},
            {"title": "Hypothesis B", "text": "Bifacial panel coatings."},
        ]
    )
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion(payload)
        result = call_llm_for_generation("test goal", num_hypotheses=2, temperature=0.7)

    assert [h["title"] for h in result] == ["Hypothesis A", "Hypothesis B"]
    assert all("text" in h for h in result)


def test_generation_handles_markdown_fenced_json():
    payload = '```json\n[{"title": "T", "text": "X"}]\n```'
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion(payload)
        result = call_llm_for_generation("test goal")

    assert result == [{"title": "T", "text": "X"}]


def test_generation_passes_selected_model_to_llm_boundary():
    payload = '[{"title": "T", "text": "X"}]'
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = call_llm_for_generation("test goal", model="selected-local-model")

    assert result == [{"title": "T", "text": "X"}]
    assert mock_call.call_args.kwargs["model"] == "selected-local-model"


def test_401_propagates_as_error_hypothesis():
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = Exception(
            "Error code: 401 - No auth credentials found"
        )
        result = call_llm_for_generation("test goal", num_hypotheses=2)

    assert len(result) == 1
    assert result[0]["title"] == "Error"
    assert "authentication failed" in result[0]["text"].lower()


def test_reflection_error_returns_not_reviewed():
    # call_llm is imported into app.agents' namespace, so patch it there.
    with patch("app.agents.call_llm", return_value="Error: API call failed"):
        review = call_llm_for_reflection("some hypothesis")

    assert review["novelty_review"] == "Not reviewed"
    assert review["feasibility_review"] == "Not reviewed"
    assert review["references"] == []


def test_reflection_passes_selected_model_to_llm_boundary():
    payload = json.dumps(
        {
            "novelty_review": "HIGH",
            "feasibility_review": "MEDIUM",
            "comment": "Looks plausible.",
            "references": [],
        }
    )
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        review = call_llm_for_reflection("some hypothesis", model="selected-local-model")

    assert review["novelty_review"] == "HIGH"
    assert mock_call.call_args.kwargs["model"] == "selected-local-model"
