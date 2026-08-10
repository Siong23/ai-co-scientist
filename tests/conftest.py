"""Shared offline-test safeguards."""

import pytest
import requests

from app.config import config


@pytest.fixture(autouse=True)
def disable_automatic_paper_downloads(monkeypatch):
    """The canonical test suite must never download PDFs or call embeddings."""

    paper_library_config = config.setdefault("paper_library", {})
    monkeypatch.setitem(paper_library_config, "enabled", False)


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
