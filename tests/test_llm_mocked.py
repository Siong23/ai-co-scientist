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
            {
                "title": "Hypothesis A",
                "hypothesis": "Perovskite tandem cells improve efficiency.",
                "rationale": "Retrieved evidence supports tandem designs.",
                "feasibility": "Compare tandem and baseline cells.",
                "source_ids": ["arXiv:1111.1111"],
            },
            {
                "title": "Hypothesis B",
                "hypothesis": "Bifacial panel coatings improve yield.",
                "rationale": "Retrieved evidence supports bifacial capture.",
                "feasibility": "Measure coated and uncoated panels.",
                "source_ids": ["arXiv:2222.2222"],
            },
        ]
    )
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion(payload)
        result = call_llm_for_generation("test goal", num_hypotheses=2, temperature=0.7)

    assert [h["title"] for h in result] == ["Hypothesis A", "Hypothesis B"]
    assert all(
        {"hypothesis", "rationale", "feasibility", "source_ids"}.issubset(h)
        for h in result
    )


def test_generation_handles_markdown_fenced_json():
    expected = {
        "title": "T",
        "hypothesis": "H",
        "rationale": "R",
        "feasibility": "F",
        "source_ids": ["arXiv:1234.5678"],
    }
    payload = f"```json\n{json.dumps([expected])}\n```"
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion(payload)
        result = call_llm_for_generation("test goal")

    assert result == [expected]


def test_generation_accepts_wrapped_json_and_common_field_aliases():
    payload = json.dumps(
        {
            "hypotheses": [
                {
                    "Title": "T",
                    "Hypothesis": "H",
                    "Rationale": "R",
                    "Feasibility": "F",
                    "Source IDs": ["arXiv:1234.5678"],
                }
            ]
        }
    )
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = call_llm_for_generation("test goal", num_hypotheses=1)

    assert result == [
        {
            "title": "T",
            "hypothesis": "H",
            "rationale": "R",
            "feasibility": "F",
            "source_ids": ["arXiv:1234.5678"],
        }
    ]
    assert mock_call.call_count == 1


def test_generation_repairs_unparsable_output_once():
    repaired = json.dumps(
        [
            {
                "title": "T",
                "hypothesis": "H",
                "rationale": "R",
                "feasibility": "F",
                "source_ids": ["arXiv:1234.5678"],
            }
        ]
    )
    with patch(
        "app.agents.call_llm",
        side_effect=["Here are the hypotheses: not JSON", repaired],
    ) as mock_call:
        result = call_llm_for_generation(
            "test goal",
            num_hypotheses=1,
            temperature=0.7,
            model="selected-local-model",
        )

    assert result[0]["title"] == "T"
    assert mock_call.call_count == 2
    repair_call = mock_call.call_args_list[1]
    assert repair_call.kwargs == {
        "temperature": 0.0,
        "model": "selected-local-model",
    }
    assert "format" in repair_call.args[0].lower()


def test_generation_parses_insufficient_context_error():
    payload = json.dumps({"error": ("The retrieved context is insufficient to generate grounded hypotheses.")})
    with patch("app.agents.call_llm", return_value=payload):
        result = call_llm_for_generation("test goal")

    assert result == [
        {
            "title": "Error",
            "text": ("The retrieved context is insufficient to generate grounded hypotheses."),
        }
    ]


def test_generation_passes_selected_model_to_llm_boundary():
    expected = {
        "title": "T",
        "hypothesis": "H",
        "rationale": "R",
        "feasibility": "F",
        "source_ids": ["arXiv:1234.5678"],
    }
    payload = json.dumps([expected])
    with patch("app.agents.call_llm", return_value=payload) as mock_call:
        result = call_llm_for_generation("test goal", model="selected-local-model")

    assert result == [expected]
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
