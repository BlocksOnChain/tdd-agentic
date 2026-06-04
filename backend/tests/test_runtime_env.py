"""Agent runtime environment detection and prompt injection."""

from __future__ import annotations

from backend.agents.runtime_env import (
    detect_agent_runtime,
    get_agent_runtime_agents_md_section,
    get_agent_runtime_prompt_section,
)


def test_detect_agent_runtime_has_core_keys() -> None:
    r = detect_agent_runtime()
    assert r["arch"]
    assert r["platform"]
    assert "os_pretty" in r


def test_prompt_section_documents_execution_host() -> None:
    section = get_agent_runtime_prompt_section()
    assert "AGENT EXECUTION ENVIRONMENT" in section
    assert "shell_run" in section
    assert "orchestrator backend container" in section
    assert detect_agent_runtime()["os_pretty"] in section


def test_agents_md_section_covers_distro_and_mongodb() -> None:
    md = get_agent_runtime_agents_md_section()
    assert "## Agent execution environment" in md
    assert "mongodb-memory-server" in md
    assert "MONGOMS_DISTRO" in md
    assert "docker compose" in md.lower()
