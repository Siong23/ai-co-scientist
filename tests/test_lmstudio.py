"""Offline tests for the LM Studio integration boundary."""

from unittest.mock import MagicMock, patch

import pytest

import app.utils as utils
from app.utils import call_llm, classify_llm_error, fetch_lmstudio_models


def _completion(content: str = "LOCAL RESPONSE"):
    completion = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    completion.choices = [choice]
    return completion


def test_environment_overrides_lmstudio_configuration(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://local-server:9999/v1/")
    monkeypatch.setenv("LMSTUDIO_MODEL", "local/model")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "local-secret")

    assert utils.get_lmstudio_base_url() == "http://local-server:9999/v1"
    assert utils.get_lmstudio_model() == "local/model"
    assert utils.get_lmstudio_api_key() == "local-secret"


def test_fetch_lmstudio_models_is_sorted_and_deduplicated(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1/")
    response = MagicMock()
    response.json.return_value = {"data": [{"id": "model-b"}, {"id": "model-a"}, {"id": "model-b"}, {"missing": "id"}]}

    with patch.object(utils.requests, "get", return_value=response) as mock_get:
        models = fetch_lmstudio_models()

    assert models == ["model-a", "model-b"]
    mock_get.assert_called_once_with(
        "http://localhost:1234/v1/models",
        headers={},
        timeout=utils.config.get("lmstudio_model_list_timeout_seconds", 10),
    )
    response.raise_for_status.assert_called_once()


def test_fetch_lmstudio_models_uses_optional_auth(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_API_KEY", "secret")
    response = MagicMock()
    response.json.return_value = {"data": []}

    with patch.object(utils.requests, "get", return_value=response) as mock_get:
        fetch_lmstudio_models()

    assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer secret"}


def test_fetch_lmstudio_models_failure_is_offline_safe(monkeypatch, caplog):
    secret = "secret-that-must-not-leak"
    monkeypatch.setenv("LMSTUDIO_API_KEY", secret)
    with patch.object(utils.requests, "get", side_effect=RuntimeError(f"connection failed {secret}")):
        assert fetch_lmstudio_models() == []

    assert secret not in caplog.text


def test_call_llm_uses_local_openai_compatible_api(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "secret")
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion()
        result = call_llm("prompt", temperature=0.2, model="selected-model")

    assert result == "LOCAL RESPONSE"
    mock_openai.assert_called_once_with(
        base_url="http://localhost:1234/v1",
        api_key="secret",
        max_retries=0,
        timeout=utils.config.get("llm_request_timeout_seconds", 180),
    )
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="selected-model",
        messages=[{"role": "user", "content": "prompt"}],
        temperature=0.2,
    )


def test_missing_model_short_circuits_without_network(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)
    monkeypatch.setitem(utils.config, "llm_model", "")
    with patch.object(utils, "OpenAI") as mock_openai:
        result = call_llm("prompt")

    assert result == "Error: LLM model not configured."
    mock_openai.assert_not_called()


@pytest.mark.parametrize(
    "provider_error, category, expected_fragment",
    [
        ("Error code: 401 - Unauthorized", "Missing or invalid API key", "authentication failed"),
        ("Request timed out", "Model provider timed out", "timed out"),
        ("Error code: 404 - model not found", "Model unavailable or delisted", "model unavailable"),
        ("Connection refused", "LM Studio unavailable", "Could not connect"),
        ("unexpected local server error", "LLM/API error", "LM Studio call failed"),
    ],
)
def test_call_llm_surfaces_actionable_errors(provider_error, category, expected_fragment):
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = RuntimeError(provider_error)
        result = call_llm("prompt", model="local-model")

    assert expected_fragment.lower() in result.lower()
    assert classify_llm_error(result) == category


def test_call_llm_handles_empty_completion():
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = MagicMock(choices=[])
        assert call_llm("prompt", model="local-model") == "Error: LM Studio returned no completion choices."

    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion("")
        assert call_llm("prompt", model="local-model") == "Error: LM Studio returned an empty response."


def test_lmstudio_key_is_redacted_from_error_and_logs(monkeypatch, caplog):
    secret = "lmstudio-secret-canary"
    monkeypatch.setenv("LMSTUDIO_API_KEY", secret)
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = RuntimeError(f"server echoed {secret}")
        result = call_llm("prompt", model="local-model")

    assert secret not in result
    assert secret not in caplog.text
    assert "***REDACTED***" in result
