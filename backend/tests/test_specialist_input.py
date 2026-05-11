"""Specialist focused-history assembly."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.agents.runner import _build_specialist_input
from backend.agents.state import SystemState


def test_skips_duplicate_goal_when_project_context_set() -> None:
    state = SystemState(
        project_context="Build a todo API",
        messages=[
            HumanMessage(content="Build a todo API"),
            HumanMessage(content="[from project_manager → researcher]\nResearch FastAPI."),
        ],
    )
    out = _build_specialist_input(state, "researcher")
    assert not any(
        isinstance(m, HumanMessage) and "[original project goal]" in str(m.content) for m in out
    )
