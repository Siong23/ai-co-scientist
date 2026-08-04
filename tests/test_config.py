"""Tests for local environment configuration."""

import os

from app.config import load_environment


def test_load_environment_reads_dotenv_without_overriding_process_values(tmp_path, monkeypatch):
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("TAVILY_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert load_environment(dotenv_file)
    assert os.environ["TAVILY_API_KEY"] == "from-dotenv"

    monkeypatch.setenv("TAVILY_API_KEY", "from-process")
    load_environment(dotenv_file)
    assert os.environ["TAVILY_API_KEY"] == "from-process"
