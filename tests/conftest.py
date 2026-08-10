"""Shared offline-test safeguards."""

import pytest

from app.config import config


@pytest.fixture(autouse=True)
def disable_automatic_paper_downloads(monkeypatch):
    """The canonical test suite must never download PDFs or call embeddings."""

    paper_library_config = config.setdefault("paper_library", {})
    monkeypatch.setitem(paper_library_config, "enabled", False)
