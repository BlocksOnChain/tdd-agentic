"""Tests for _build_specialist_input changes (canonical project_context)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.agents.runner import _build_specialist_input, SystemState


def _make_state(project_context: str = "Build a todo app", extra_messages: list | None = None) -> SystemState:
    msgs = list(extra_messages) if extra_messages else []
    return SystemState(project_context=project_context, messages=msgs)


def test_specialist_input_includes_pm_instruction() -> None:
    """The PM instruction message is included in specialist input."""
    pm_msg = HumanMessage(content="[from project_manager → backend_dev]\n{\"tik\":[\"uuid1\"]}\nImplement auth.")
    state = _make_state(extra_messages=[pm_msg])
    result = _build_specialist_input(state, "backend_dev")
    contents = [m.content for m in result]
    assert any("Implement auth" in c for c in contents)


def test_specialist_input_no_first_human_fallback() -> None:
    """The goal is NOT extracted from first_human when project_context is set."""
    # project_context is set, so _build_specialist_input should NOT add a
    # fallback "original project goal" message from the first human.
    goal_msg = HumanMessage(content="[original project goal]\nBuild a todo app")
    pm_msg = HumanMessage(content="[from project_manager → backend_dev]\n{}")
    state = _make_state(extra_messages=[goal_msg, pm_msg])
    result = _build_specialist_input(state, "backend_dev")

    # Should NOT contain the "[original project goal]" prefix
    contents = [m.content for m in result]
    assert not any("[original project goal]" in c for c in contents)


def test_specialist_input_empty_state() -> None:
    """Empty state returns empty list."""
    state = _make_state(extra_messages=[])
    result = _build_specialist_input(state, "backend_dev")
    assert result == []


def test_backend_dev_excludes_handoff_summaries() -> None:
    """backend_dev does not receive cross-agent handoff summaries."""
    pm_msg = HumanMessage(content="[from project_manager → backend_dev]\n{}")
    handoff_msg = HumanMessage(content="[from researcher → project_manager]\nResearch complete.")
    state = _make_state(extra_messages=[pm_msg, handoff_msg])
    result = _build_specialist_input(state, "backend_dev")
    contents = [m.content for m in result]
    # Should NOT include the handoff summary
    assert not any("[from researcher" in c for c in contents)


def test_non_dev_agent_gets_handoff_summaries() -> None:
    """Non-dev agents (like PM) receive handoff summaries."""
    pm_msg = HumanMessage(content="[from project_manager → project_manager]\n{}")
    handoff_msg = HumanMessage(content="[from researcher → project_manager]\nResearch complete.")
    state = _make_state(extra_messages=[pm_msg, handoff_msg])
    result = _build_specialist_input(state, "project_manager")
    contents = [m.content for m in result]
    # Should include the handoff summary
    assert any("[from researcher" in c for c in contents)
