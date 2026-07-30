from pathlib import Path

import yaml


def load_deploy_steps():
    workflow = yaml.safe_load(
        Path(".github/workflows/cicd.yml").read_text(encoding="utf-8")
    )
    return workflow["jobs"]["deploy"]["steps"]


def test_deploy_installs_dependencies_into_service_virtualenv():
    install_step = next(
        step
        for step in load_deploy_steps()
        if step.get("name") == "Install/update dependencies"
    )

    assert (
        "./venv/bin/python -m pip install -r requirements.txt"
        in install_step["run"]
    )
    assert "|| pip install" not in install_step["run"]


def test_deploy_waits_for_http_readiness_and_reports_service_logs():
    health_step = next(
        step
        for step in load_deploy_steps()
        if step.get("name") == "Verify application health"
    )

    assert (
        "curl --fail --silent http://127.0.0.1:7860/"
        in health_step["run"]
    )
    assert (
        "journalctl -u ai-co-scientist.service -n 100 --no-pager"
        in health_step["run"]
    )
