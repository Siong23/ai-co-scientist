"""Offline tests for the LM Studio integration boundary."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import app.utils as utils
from app.utils import call_llm, classify_llm_error, fetch_lmstudio_models

# conftest replaces encode with an offline stub for every test; keep the real one.
_LMSTUDIO_ENCODE = utils.LMStudioSentenceTransformer.encode


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


def test_embedding_server_defaults_to_the_chat_server(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://chat-server:1234/v1/")
    monkeypatch.setitem(utils.config, "lmstudio_embedding_base_url", None)

    assert utils.get_lmstudio_embedding_base_url() == "http://chat-server:1234/v1"


def test_embedding_server_can_differ_from_the_chat_server(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://chat-server:1234/v1")
    monkeypatch.setitem(utils.config, "lmstudio_embedding_base_url", "http://embedding-config:1234/v1/")
    assert utils.get_lmstudio_embedding_base_url() == "http://embedding-config:1234/v1"

    monkeypatch.setenv("LMSTUDIO_EMBEDDING_BASE_URL", "http://embedding-env:1234/v1/")
    assert utils.get_lmstudio_embedding_base_url() == "http://embedding-env:1234/v1"
    assert utils.get_lmstudio_base_url() == "http://chat-server:1234/v1"


def test_embeddings_are_requested_from_the_embedding_server(monkeypatch):
    monkeypatch.setattr(utils.LMStudioSentenceTransformer, "encode", _LMSTUDIO_ENCODE)
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://chat-server:1234/v1")
    monkeypatch.setenv("LMSTUDIO_EMBEDDING_BASE_URL", "http://embedding-server:1234/v1")
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[3.0, 4.0])])
        vector = utils.LMStudioSentenceTransformer("embedding-model").encode("text")

    assert mock_openai.call_args.kwargs["base_url"] == "http://embedding-server:1234/v1"
    mock_openai.return_value.embeddings.create.assert_called_once_with(model="embedding-model", input=["text"])
    assert vector.tolist() == pytest.approx([0.6, 0.8])


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
    mock_openai.assert_called_once()
    client_args = mock_openai.call_args.kwargs
    assert client_args["base_url"] == "http://localhost:1234/v1"
    assert client_args["api_key"] == "secret"
    assert client_args["max_retries"] == 0
    assert client_args["timeout"].connect == utils.config.get("lmstudio_connect_timeout_seconds", 10)
    assert client_args["timeout"].read == utils.config.get("llm_request_timeout_seconds", 180)
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


def _native_response(output):
    response = MagicMock()
    response.json.return_value = {"output": output}
    return response


def test_native_reasoning_only_output_is_an_error_not_the_answer():
    """A budget spent on reasoning must not hand callers the reasoning as code."""

    output = [{"type": "reasoning", "content": "First I will load the CSV, then..."}]
    with patch.object(utils.requests, "post", return_value=_native_response(output)):
        result = call_llm("write the script", model="local-model", reasoning="on")

    assert result == utils.REASONING_ONLY_ERROR
    assert "load the CSV" not in result


def test_native_answer_excludes_reasoning_text():
    output = [
        {"type": "reasoning", "content": "Let me think about the imports."},
        {"type": "message", "content": "import torch"},
    ]
    with patch.object(utils.requests, "post", return_value=_native_response(output)):
        assert call_llm("write the script", model="local-model", reasoning="on") == "import torch"


def test_native_fallback_still_reads_non_reasoning_items():
    """Output items of other types still carry the answer when no message exists."""

    output = [
        {"type": "reasoning", "content": "scratch work"},
        {"type": "output_text", "text": '{"ok": true}'},
    ]
    with patch.object(utils.requests, "post", return_value=_native_response(output)):
        assert call_llm("return JSON", model="local-model", reasoning="on") == '{"ok": true}'


def test_native_output_that_reaches_the_token_limit_is_logged(caplog):
    response = _native_response([{"type": "message", "content": '{"findings": ['}])
    response.json.return_value["stats"] = {"total_output_tokens": 64}
    with caplog.at_level("WARNING", logger=utils.logger.name):
        with patch.object(utils.requests, "post", return_value=response):
            result = call_llm("return JSON", model="local-model", max_tokens=64, reasoning="off")

    assert result == '{"findings": ['
    assert "stopped at max_output_tokens=64" in caplog.text


def test_native_output_below_the_token_limit_is_not_logged(caplog):
    response = _native_response([{"type": "message", "content": '{"ok": true}'}])
    response.json.return_value["stats"] = {"total_output_tokens": 12}
    with caplog.at_level("WARNING", logger=utils.logger.name):
        with patch.object(utils.requests, "post", return_value=response):
            call_llm("return JSON", model="local-model", max_tokens=64, reasoning="off")

    assert "max_output_tokens" not in caplog.text


def test_compatible_output_cut_by_length_is_logged(caplog):
    completion = _completion('{"findings": [')
    completion.choices[0].finish_reason = "length"
    with caplog.at_level("WARNING", logger=utils.logger.name):
        with patch.object(utils, "OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = completion
            call_llm("return JSON", model="local-model", max_tokens=32)

    assert "stopped at max_output_tokens=32" in caplog.text


def test_compatible_api_does_not_return_reasoning_content_as_the_answer():
    completion = _completion("")
    completion.choices[0].message.reasoning_content = "The user wants JSON, so I should..."
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = completion
        assert call_llm("return JSON", model="local-model") == utils.REASONING_ONLY_ERROR


@pytest.fixture
def single_model_server(monkeypatch):
    """Keep one model per server, starting from a process that has used none."""

    monkeypatch.setitem(utils.config, "lmstudio_single_model_per_server", True)
    monkeypatch.setattr(utils, "_lmstudio_server_slots", {})
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://lm-server:1234/v1")


def _loaded_models(*loaded):
    """A GET /api/v1/models response in which each (key, instance_id) is loaded."""

    response = MagicMock()
    response.json.return_value = {
        "models": [{"key": key, "loaded_instances": [{"id": instance_id}]} for key, instance_id in loaded]
    }
    return response


def _unloaded_ids(mock_post):
    return [
        call.kwargs["json"]["instance_id"]
        for call in mock_post.call_args_list
        if call.args[0] == "http://lm-server:1234/api/v1/models/unload"
    ]


def test_a_new_model_first_unloads_every_other_model_on_the_server(single_model_server):
    loaded = _loaded_models(("chat-model", "chat-model"), ("qwen/qwen3.6-27b", "qwen/qwen3.6-27b"))
    with (
        patch.object(utils.requests, "get", return_value=loaded) as mock_get,
        patch.object(utils.requests, "post") as mock_post,
        patch.object(utils, "OpenAI") as mock_openai,
    ):
        mock_openai.return_value.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[1.0, 0.0])])
        _LMSTUDIO_ENCODE(utils.LMStudioSentenceTransformer("embedding-model"), "text")

    mock_get.assert_called_once()
    assert mock_get.call_args.args[0] == "http://lm-server:1234/api/v1/models"
    assert _unloaded_ids(mock_post) == ["chat-model", "qwen/qwen3.6-27b"]
    mock_openai.return_value.embeddings.create.assert_called_once()


def test_the_model_in_use_is_neither_unloaded_nor_rechecked(single_model_server):
    loaded = _loaded_models(("chat-model", "chat-model@q8_k_xl"))
    with (
        patch.object(utils.requests, "get", return_value=loaded) as mock_get,
        patch.object(utils.requests, "post") as mock_post,
        patch.object(utils, "OpenAI") as mock_openai,
    ):
        mock_openai.return_value.chat.completions.create.return_value = _completion()
        assert call_llm("first", model="chat-model") == "LOCAL RESPONSE"
        assert call_llm("second", model="chat-model") == "LOCAL RESPONSE"

    mock_get.assert_called_once()
    assert _unloaded_ids(mock_post) == []


def test_a_model_switch_waits_for_running_requests_and_goes_first(single_model_server):
    """New requests for the loaded model queue behind a waiting switch."""

    slot_for = utils._lmstudio_model_slot
    base_url = "http://lm-server:1234/v1"
    entered = []
    events = []

    def unload_after_release(*_args, **kwargs):
        events.append(f"unload {kwargs['json']['instance_id']}")
        return MagicMock()

    def request(model):
        with slot_for(base_url, model):
            entered.append(model)

    def wait_until_waiting(model):
        slot = utils._lmstudio_server_slots["http://lm-server:1234"]
        deadline = time.monotonic() + 5
        while model not in slot.waiting:
            assert time.monotonic() < deadline, f"{model} never started waiting"
            time.sleep(0.01)

    with (
        patch.object(utils.requests, "get", side_effect=lambda *a, **k: _loaded_models(*current)),
        patch.object(utils.requests, "post", side_effect=unload_after_release),
    ):
        current = []
        with slot_for(base_url, "model-a"):
            current = [("model-a", "model-a")]
            switch = threading.Thread(target=request, args=("model-b",))
            switch.start()
            wait_until_waiting("model-b")
            same_model = threading.Thread(target=request, args=("model-a",))
            same_model.start()
            wait_until_waiting("model-a")
            time.sleep(0.1)
            assert entered == []
            events.append("model-a finished")
        switch.join(timeout=5)
        same_model.join(timeout=5)

    assert entered == ["model-b", "model-a"]
    assert events[:2] == ["model-a finished", "unload model-a"]


def test_a_failed_model_load_unloads_other_models_and_retries_once(single_model_server):
    responses = iter([_loaded_models(), _loaded_models(("other-model", "other-model"))])
    with (
        patch.object(utils.requests, "get", side_effect=lambda *a, **k: next(responses)),
        patch.object(utils.requests, "post") as mock_post,
        patch.object(utils, "OpenAI") as mock_openai,
    ):
        mock_openai.return_value.chat.completions.create.side_effect = [
            RuntimeError('Error code: 400 - Failed to load model "chat-model". Error: out of memory'),
            _completion(),
        ]
        result = call_llm("prompt", model="chat-model")

    assert result == "LOCAL RESPONSE"
    assert mock_openai.return_value.chat.completions.create.call_count == 2
    assert _unloaded_ids(mock_post) == ["other-model"]


def test_other_errors_are_not_retried_with_a_model_switch(single_model_server):
    with (
        patch.object(utils.requests, "get", return_value=_loaded_models()) as mock_get,
        patch.object(utils, "OpenAI") as mock_openai,
    ):
        mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("Connection refused")
        result = call_llm("prompt", model="chat-model")

    assert "Could not connect" in result
    mock_openai.return_value.chat.completions.create.assert_called_once()
    mock_get.assert_called_once()


def test_lmstudio_key_is_redacted_from_error_and_logs(monkeypatch, caplog):
    secret = "lmstudio-secret-canary"
    monkeypatch.setenv("LMSTUDIO_API_KEY", secret)
    with patch.object(utils, "OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.side_effect = RuntimeError(f"server echoed {secret}")
        result = call_llm("prompt", model="local-model")

    assert secret not in result
    assert secret not in caplog.text
    assert "***REDACTED***" in result
