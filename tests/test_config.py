import os

from app.config import load_environment


def test_load_environment_reads_dotenv_without_overriding_process_environment(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "TAVILY_API_KEY=dotenv-tavily\nELSEVIER_API_KEY=dotenv-elsevier\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAVILY_API_KEY", "process-tavily")
    monkeypatch.delenv("ELSEVIER_API_KEY", raising=False)

    assert load_environment(dotenv_path) is True
    assert os.environ["TAVILY_API_KEY"] == "process-tavily"
    assert os.environ["ELSEVIER_API_KEY"] == "dotenv-elsevier"
