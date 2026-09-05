import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import config loading function and config object
from .config import config

# --- Logging Setup ---
# Configure a root logger or a specific logger for the app
# Using a basic configuration here, can be enhanced
logging.basicConfig(
    level=config.get("logging_level", logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("aicoscientist")  # Use a specific name for the app logger

# Optional: Add file handler based on config (if needed globally)
# log_filename_base = config.get('log_file_name', 'app')
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# file_handler = logging.FileHandler(f"{log_filename_base}_{timestamp}.txt")
# formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


# --- LM Studio Integration ---
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LMSTUDIO_API_KEY = "lm-studio"
_execution_budget_lock = threading.RLock()
_execution_deadline: float | None = None
_execution_cancel_event: threading.Event | None = None


@contextmanager
def execution_budget(deadline: float, cancel_event: threading.Event):
    """Apply one cycle deadline to every LLM request made by this worker."""

    global _execution_cancel_event, _execution_deadline
    with _execution_budget_lock:
        previous_deadline = _execution_deadline
        previous_cancel_event = _execution_cancel_event
        _execution_deadline = float(deadline)
        _execution_cancel_event = cancel_event
    try:
        yield
    finally:
        with _execution_budget_lock:
            _execution_deadline = previous_deadline
            _execution_cancel_event = previous_cancel_event


def execution_cancelled() -> bool:
    """Return whether the current cycle has exhausted or cancelled its budget."""

    with _execution_budget_lock:
        cancel_event = _execution_cancel_event
        deadline = _execution_deadline
    return bool(cancel_event is not None and cancel_event.is_set()) or bool(
        deadline is not None and time.monotonic() >= deadline
    )


def _request_timeout() -> float:
    """Bound a provider call by both its configured timeout and cycle deadline."""

    configured = max(0.1, float(config.get("llm_request_timeout_seconds", 180)))
    with _execution_budget_lock:
        deadline = _execution_deadline
    if deadline is None:
        return configured
    return max(0.1, min(configured, deadline - time.monotonic()))


def get_lmstudio_base_url() -> str:
    """Return the OpenAI-compatible LM Studio API base URL."""
    value = os.getenv("LMSTUDIO_BASE_URL") or config.get("lmstudio_base_url") or DEFAULT_LMSTUDIO_BASE_URL
    return str(value).rstrip("/")


def get_lmstudio_native_chat_url() -> str:
    """Return LM Studio's native chat endpoint beside the configured /v1 API."""

    base_url = get_lmstudio_base_url()
    server_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    return f"{server_url}/api/v1/chat"


def get_lmstudio_api_key() -> str:
    """Return the optional LM Studio API key or the SDK placeholder value."""
    return os.getenv("LMSTUDIO_API_KEY") or DEFAULT_LMSTUDIO_API_KEY


def get_lmstudio_model(model: Optional[str] = None) -> str:
    """Resolve an explicit, environment, or configured local model identifier."""
    return model or os.getenv("LMSTUDIO_MODEL") or config.get("llm_model", "")


# --- Secret Redaction ---
def redact_secrets(text: str) -> str:
    """Remove provider credentials from logs and user-facing errors."""
    redacted = str(text)
    for variable in ("LMSTUDIO_API_KEY", "ELSEVIER_API_KEY", "ELSEVIER_INST_TOKEN"):
        secret = os.getenv(variable)
        if secret:
            redacted = redacted.replace(secret, "***REDACTED***")
    return redacted


def _lmstudio_headers() -> Dict[str, str]:
    api_key = os.getenv("LMSTUDIO_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def fetch_lmstudio_models() -> List[str]:
    """Return sorted model IDs currently exposed by LM Studio.

    Failures are logged and converted to an empty list so the UI can still
    start with the configured model as a fallback.
    """
    models_url = f"{get_lmstudio_base_url()}/models"
    try:
        response = requests.get(
            models_url,
            headers=_lmstudio_headers(),
            timeout=config.get("lmstudio_model_list_timeout_seconds", 10),
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        return sorted({item.get("id") for item in models if isinstance(item, dict) and item.get("id")})
    except Exception as exc:
        logger.warning("Could not fetch LM Studio models: %s", redact_secrets(str(exc)))
        return []


# --- Error Classification ---
def classify_llm_error(error_text: str) -> str:
    """Map a local LLM error string to a short, user-actionable category."""
    text = (error_text or "").lower()
    if any(
        marker in text
        for marker in (
            "context size has been exceeded",
            "context length exceeded",
            "maximum context length",
            "context window exceeded",
        )
    ):
        return "Model context window exceeded"
    if "authentication failed" in text or "401" in text or "unauthorized" in text:
        return "Missing or invalid API key"
    if "timed out" in text or "timeout" in text:
        return "Model provider timed out"
    if "rate limit" in text or "429" in text:
        return "Rate limited by the model provider"
    if "model unavailable" in text or "model not found" in text or "invalid model" in text or "404" in text:
        return "Model unavailable or delisted"
    if "could not connect" in text or "connection refused" in text or "connection reset" in text:
        return "LM Studio unavailable"
    if "could not parse" in text or "invalid json" in text:
        return "Model returned unparsable output"
    if "model not configured" in text:
        return "LLM model not configured"
    if (
        "retrieved evidence is insufficient" in text
        or "rag retrieval found no usable" in text
        or "missing explicit requirements" in text
    ):
        return "Insufficient retrieved evidence"
    if (
        "rejected by the novelty and grounding audit" in text
        or "rejected or left unverified by the grounding audit" in text
    ):
        return "Hypothesis quality gate rejected all candidates"
    return "LLM/API error"


def _format_lmstudio_error(exc: Exception, model: str) -> str:
    error = redact_secrets(str(exc))
    lowered = error.lower()
    if "401" in error or "unauthorized" in lowered or "authentication" in lowered:
        return "Error: LM Studio authentication failed. Check LMSTUDIO_API_KEY."
    if "timeout" in lowered or "timed out" in lowered:
        return f"Error: LM Studio request timed out for model '{model}'. Details: {error}"
    if "404" in error or "model not found" in lowered or "invalid model" in lowered:
        return (
            f"Error: LM Studio model unavailable ('{model}'). Load or select the model in LM Studio. Details: {error}"
        )
    if any(marker in lowered for marker in ("connection refused", "connection reset", "failed to connect")):
        return (
            f"Error: Could not connect to LM Studio at {get_lmstudio_base_url()}. "
            f"Start the LM Studio server and verify LMSTUDIO_BASE_URL. Details: {error}"
        )
    return f"Error: LM Studio call failed: {error}"


def call_llm(
    prompt: str,
    temperature: float = 0.7,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[str] = None,
) -> str:
    """Call LM Studio, using its native API when reasoning is configured."""
    selected_model = get_lmstudio_model(model)
    if not selected_model:
        logger.error("LM Studio model is not configured.")
        return "Error: LLM model not configured."

    if execution_cancelled():
        return "Error: Cycle execution cancelled after reaching its time limit."

    started_at = time.perf_counter()
    try:
        output_token_limit = max_tokens
        if output_token_limit is None:
            output_token_limit = int(config.get("llm_default_max_tokens", 8192))
        output_token_limit = max(1, int(output_token_limit))

        if reasoning is not None:
            payload = {
                "model": selected_model,
                "input": prompt,
                "temperature": temperature,
                "max_output_tokens": output_token_limit,
                "reasoning": reasoning,
                "store": False,
                "stream": False,
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            retry_count = max(
                0,
                int(config.get("lmstudio_native_server_error_retries", 1)),
            )
            for attempt in range(retry_count + 1):
                try:
                    response = requests.post(
                        get_lmstudio_native_chat_url(),
                        headers=_lmstudio_headers(),
                        json=payload,
                        timeout=_request_timeout(),
                    )
                    response.raise_for_status()
                    response_payload = response.json()
                    output = response_payload.get("output", []) if isinstance(response_payload, dict) else []
                    content = "\n".join(
                        str(item.get("content", ""))
                        for item in output
                        if isinstance(item, dict) and item.get("type") == "message" and item.get("content")
                    ).strip()
                    if not content:
                        content = "\n".join(
                            str(item.get("content") or item.get("text") or "")
                            for item in output
                            if isinstance(item, dict) and (item.get("content") or item.get("text"))
                        ).strip()
                    if not content:
                        return "Error: LM Studio returned an empty response."
                    return content
                except Exception as exc:
                    response = getattr(exc, "response", None)
                    status_code = getattr(response, "status_code", None)
                    retryable = isinstance(status_code, int) and 500 <= status_code < 600 and attempt < retry_count
                    if not retryable:
                        raise
                    logger.warning(
                        "LM Studio native chat returned HTTP %d; retrying once.",
                        status_code,
                    )
                    time.sleep(
                        max(
                            0.0,
                            float(
                                config.get(
                                    "lmstudio_native_retry_backoff_seconds",
                                    1.0,
                                )
                            ),
                        )
                    )

        client = OpenAI(
            base_url=get_lmstudio_base_url(),
            api_key=get_lmstudio_api_key(),
            max_retries=0,
            timeout=_request_timeout(),
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        completion = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=output_token_limit,
        )
        if not completion.choices:
            return "Error: LM Studio returned no completion choices."
        message = completion.choices[0].message
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not content.strip():
            reasoning_content = getattr(message, "reasoning_content", None)
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                content = reasoning_content
            else:
                content = ""
        content = content.strip()
        if not content:
            return "Error: LM Studio returned an empty response."
        return content
    except Exception as exc:
        if execution_cancelled():
            return "Error: Cycle execution cancelled after reaching its time limit."
        error = _format_lmstudio_error(exc, selected_model)
        logger.error("%s", error)
        return error
    finally:
        logger.info(
            "LLM call completed model=%s prompt_chars=%d max_output_tokens=%s elapsed_ms=%d cancelled=%s",
            selected_model,
            len(prompt),
            max_tokens,
            int((time.perf_counter() - started_at) * 1000),
            execution_cancelled(),
        )


# --- ID Generation ---
def generate_unique_id(prefix="H") -> str:
    """Generates a unique identifier string."""
    return f"{prefix}{random.randint(1000, 9999)}"


# --- VIS.JS Graph Data Generation ---
def generate_visjs_data(adjacency_graph: Dict) -> Dict[str, list]:
    """Generates node and edge data lists for vis.js graph (for JSON serialization)."""
    nodes = []
    edges = []

    if not isinstance(adjacency_graph, dict):
        logger.error(f"Invalid adjacency_graph type: {type(adjacency_graph)}. Expected dict.")
        return {"nodes": [], "edges": []}

    for node_id, connections in adjacency_graph.items():
        nodes.append({"id": node_id, "label": node_id})
        if isinstance(connections, list):
            for connection in connections:
                if isinstance(connection, dict) and "similarity" in connection and "other_id" in connection:
                    similarity_val = connection.get("similarity")
                    if isinstance(similarity_val, (int, float)) and similarity_val > 0.2:
                        edges.append(
                            {
                                "from": node_id,
                                "to": connection["other_id"],
                                "label": f"{similarity_val:.2f}",
                                "arrows": "to",
                            }
                        )
                else:
                    logger.warning(f"Skipping invalid connection format for node {node_id}: {connection}")
        else:
            logger.warning(f"Skipping invalid connections format for node {node_id}: {connections}")

    return {"nodes": nodes, "edges": edges}


# --- Similarity Calculation ---
_sentence_transformer_model = None


class LMStudioSentenceTransformer:
    """SentenceTransformer-compatible interface backed by LM Studio /v1/embeddings API."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.get(
            "sentence_transformer_model", "qwen/text-embedding-qwen3-embedding-8b"
        )

    def encode(
        self,
        sentences,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = False,
    ):
        is_single = isinstance(sentences, str)
        input_texts = [sentences] if is_single else list(sentences)

        client = OpenAI(
            base_url=get_lmstudio_base_url(),
            api_key=get_lmstudio_api_key(),
            max_retries=0,
            timeout=config.get("llm_request_timeout_seconds", 180),
        )
        response = client.embeddings.create(
            model=self.model_name,
            input=input_texts,
        )
        embeddings = [data.embedding for data in response.data]
        np_embeddings = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(np_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            np_embeddings = np_embeddings / norms

        if is_single:
            result = np_embeddings[0]
            if convert_to_tensor:
                try:
                    import torch

                    return torch.tensor(result)
                except ImportError:
                    return result
            return result
        else:
            if convert_to_tensor:
                try:
                    import torch

                    return torch.tensor(np_embeddings)
                except ImportError:
                    return np_embeddings
            return np_embeddings


def get_sentence_transformer_model():
    """Loads and returns a singleton instance of the sentence transformer model."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        model_name = config.get("sentence_transformer_model", "all-MiniLM-L6-v2")
        if config.get("use_lmstudio_embeddings", False):
            logger.info(f"Using LM Studio API embeddings for model: {model_name}...")
            _sentence_transformer_model = LMStudioSentenceTransformer(model_name)
        else:
            try:
                logger.info(f"Loading sentence transformer model: {model_name}...")
                _sentence_transformer_model = SentenceTransformer(model_name)
                logger.info("Sentence transformer model loaded successfully.")
            except ImportError:
                logger.error(
                    "Failed to import sentence_transformers. Please install it: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model '{model_name}': {e}")
                raise  # Re-raise after logging
    return _sentence_transformer_model


def similarity_score(textA: str, textB: str) -> float:
    """Calculates cosine similarity between two texts using sentence embeddings."""
    try:
        if not textA.strip() or not textB.strip():
            logger.warning("Empty string provided to similarity_score.")
            return 0.0

        model = get_sentence_transformer_model()
        if model is None:  # Check if model loading failed previously
            return 0.0  # Or handle error appropriately

        embedding_a = model.encode(textA, convert_to_tensor=True)
        embedding_b = model.encode(textB, convert_to_tensor=True)

        # Ensure embeddings are 2D numpy arrays for cosine_similarity
        embedding_a_np = embedding_a.cpu().numpy().reshape(1, -1)
        embedding_b_np = embedding_b.cpu().numpy().reshape(1, -1)

        similarity = cosine_similarity(embedding_a_np, embedding_b_np)[0][0]

        # Clamp the value between 0.0 and 1.0
        similarity = float(np.clip(similarity, 0.0, 1.0))

        # logger.debug(f"Similarity score: {similarity:.4f}") # Use debug level
        return similarity
    except Exception as e:
        logger.error(f"Error calculating similarity score: {e}", exc_info=True)  # Log traceback
        return 0.0  # Return 0 on error instead of 0.5
