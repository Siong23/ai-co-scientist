"""Shared offline-test safeguards."""

import numpy as np
import pytest
import requests

from app.config import config

_PROVIDER_CREDENTIALS = (
    "LMSTUDIO_API_KEY",
    "LMSTUDIO_BASE_URL",
    "LMSTUDIO_MODEL",
    "SEMANTIC_SCHOLAR_API_KEY",
    "SPRINGER_API_KEY",
    "SPRINGER_OPEN_ACCESS_API_KEY",
    "SPRINGER_META_API_KEY",
    "ELSEVIER_API_KEY",
    "ELSEVIER_INST_TOKEN",
    "TAVILY_API_KEY",
)


@pytest.fixture(autouse=True)
def disable_external_provider_credentials(monkeypatch, request):
    """Keep local .env credentials out of the canonical offline suite."""

    if request.node.get_closest_marker("network") or request.node.get_closest_marker("integration"):
        return
    for variable in _PROVIDER_CREDENTIALS:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture(autouse=True)
def disable_automatic_paper_downloads(monkeypatch):
    """The canonical test suite must never download PDFs or call embeddings."""

    paper_library_config = config.setdefault("paper_library", {})
    monkeypatch.setitem(paper_library_config, "enabled", False)


@pytest.fixture(autouse=True)
def disable_live_embeddings(monkeypatch, request):
    """Keep offline tests from waiting on the configured LM Studio server."""

    if request.node.get_closest_marker("network") or request.node.get_closest_marker("integration"):
        return

    from app import utils

    def encode_offline(
        _self,
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_tensor=False,
    ):
        del convert_to_numpy, normalize_embeddings, show_progress_bar
        is_single = isinstance(sentences, str)
        input_texts = [sentences] if is_single else list(sentences)
        embeddings = np.ones((len(input_texts), 2), dtype=np.float32)
        result = embeddings[0] if is_single else embeddings
        if convert_to_tensor:
            import torch

            return torch.tensor(result)
        return result

    monkeypatch.setattr(utils.LMStudioSentenceTransformer, "encode", encode_offline)


@pytest.fixture(autouse=True)
def block_unmocked_requests(monkeypatch, request):
    """Fail fast if an offline test accidentally reaches the network."""

    if request.node.get_closest_marker("network") or request.node.get_closest_marker("integration"):
        return

    def fail_request(*_args, **_kwargs):
        pytest.fail(
            "Offline test attempted an unmocked requests network call. "
            "Mock the HTTP boundary or mark the test as network/integration.",
            pytrace=False,
        )

    monkeypatch.setattr(requests.sessions.Session, "request", fail_request)
