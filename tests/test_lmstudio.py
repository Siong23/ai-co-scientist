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
        max_tokens=utils.config.get("llm_default_max_tokens", 2048),
    )


def test_call_llm_sends_an_explicit_system_prompt():
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion()
        call_llm(
            "user request",
            model="selected-model",
            system_prompt="planner instructions",
        )

    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="selected-model",
        messages=[
            {"role": "system", "content": "planner instructions"},
            {"role": "user", "content": "user request"},
        ],
        temperature=0.7,
        max_tokens=utils.config.get("llm_default_max_tokens", 2048),
    )


def test_call_llm_honors_an_explicit_output_limit():
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _completion()
        call_llm("prompt", model="selected-model", max_tokens=321)

    assert mock_openai.return_value.chat.completions.create.call_args.kwargs["max_tokens"] == 321


def test_call_llm_falls_back_when_model_rejects_reasoning_control(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    native_response = MagicMock(status_code=400)
    native_response.text = "Model does not expose reasoning configuration."
    native_response.raise_for_status.side_effect = utils.requests.HTTPError(
        "400 Client Error",
        response=native_response,
    )
    with (
        patch.object(utils.requests, "post", return_value=native_response) as mock_post,
        patch.object(utils, "OpenAI") as mock_openai,
    ):
        mock_openai.return_value.chat.completions.create.return_value = _completion('{"ok": true}')
        result = call_llm(
            "return JSON",
            temperature=0.0,
            model="selected-model",
            system_prompt="JSON only",
            max_tokens=64,
            reasoning="off",
        )

    assert result == '{"ok": true}'
    mock_post.assert_called_once()
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="selected-model",
        messages=[
            {"role": "system", "content": "JSON only"},
            {"role": "user", "content": "return JSON"},
        ],
        temperature=0.0,
        max_tokens=64,
    )


def test_call_llm_retries_one_transient_native_server_error(monkeypatch):
    monkeypatch.setitem(utils.config, "lmstudio_native_server_error_retries", 1)
    monkeypatch.setitem(utils.config, "lmstudio_native_retry_backoff_seconds", 0.25)
    server_error_response = MagicMock(status_code=500)
    failed_response = MagicMock()
    failed_response.raise_for_status.side_effect = utils.requests.HTTPError(
        "500 Server Error",
        response=server_error_response,
    )
    successful_response = MagicMock()
    successful_response.json.return_value = {
        "output": [{"type": "message", "content": '{"ok": true}'}],
    }

    with (
        patch.object(
            utils.requests,
            "post",
            side_effect=[failed_response, successful_response],
        ) as mock_post,
        patch.object(utils.time, "sleep") as mock_sleep,
    ):
        result = call_llm(
            "return JSON",
            model="selected-model",
            reasoning="on",
        )

    assert result == '{"ok": true}'
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(0.25)


def test_call_llm_does_not_retry_native_client_errors(monkeypatch):
    monkeypatch.setitem(utils.config, "lmstudio_native_server_error_retries", 1)
    client_error_response = MagicMock(status_code=400)
    failed_response = MagicMock()
    failed_response.raise_for_status.side_effect = utils.requests.HTTPError(
        "400 Client Error",
        response=client_error_response,
    )

    with (
        patch.object(utils.requests, "post", return_value=failed_response) as mock_post,
        patch.object(utils.time, "sleep") as mock_sleep,
    ):
        result = call_llm(
            "return JSON",
            model="selected-model",
            reasoning="on",
        )

    assert "400 Client Error" in result
    mock_post.assert_called_once()
    mock_sleep.assert_not_called()


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
        (
            "Context size has been exceeded",
            "Model context window exceeded",
            "LM Studio call failed",
        ),
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
