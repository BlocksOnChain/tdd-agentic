"""Specialist focused-history assembly."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.agents.runner import (
    build_specialist_focused_messages,
    format_pm_assignment_message,
    specialist_closing_instruction,
)
from backend.agents.state import SystemState


def test_skips_duplicate_goal_when_project_context_set_for_dev() -> None:
    state = SystemState(
        project_context="Build a todo API",
        messages=[
            HumanMessage(content="Build a todo API"),
            HumanMessage(content="[from project_manager → backend_dev]\nImplement API."),
        ],
    )
    out = build_specialist_focused_messages(state, "backend_dev")
    assert not any(
        isinstance(m, HumanMessage) and "[original project goal]" in str(m.content) for m in out
    )


def test_specialist_closing_instruction_varies_by_role_and_runtime() -> None:
    assert "docs/" in specialist_closing_instruction(role="researcher")
    assert "web_search" in specialist_closing_instruction(role="researcher", for_deep=True)
    assert "write_file" in specialist_closing_instruction(role="researcher", for_deep=True)
    assert "Kanban" not in specialist_closing_instruction(role="backend_dev")
    assert "Kanban" in specialist_closing_instruction(role="backend_dev", for_deep=True)


def test_researcher_keeps_original_goal_when_project_context_set() -> None:
    state = SystemState(
        project_context="Build a todo API",
        messages=[
            HumanMessage(content="Build a todo API"),
            HumanMessage(content="[from project_manager → researcher]\nResearch FastAPI."),
        ],
    )
    out = build_specialist_focused_messages(state, "researcher")
    assert any(
        isinstance(m, HumanMessage) and "[original project goal]" in str(m.content) for m in out
    )


def test_format_pm_assignment_surfaces_phase_and_body() -> None:
    message = format_pm_assignment_message(
        HumanMessage(
            content=(
                "[from project_manager → researcher]\n"
                '{"phase":"research"}\n'
                "Write docs/tech-stack.md and ingest into RAG."
            )
        )
    )
    text = str(message.content)
    assert text.startswith("[current PM assignment]")
    assert "phase: research" in text
    assert "Write docs/tech-stack.md" in text


def test_researcher_skips_prior_self_handoff_summary() -> None:
    state = SystemState(
        project_context="Build a todo API",
        messages=[
            HumanMessage(content="Build a todo API"),
            HumanMessage(
                content=(
                    "[from researcher → project_manager]\n"
                    "I am a 3rd year student looking for a summer internship."
                )
            ),
            HumanMessage(content="[from project_manager → researcher]\nRefresh docs/tech-stack.md."),
        ],
    )
    out = build_specialist_focused_messages(state, "researcher")
    assert not any(
        isinstance(m, HumanMessage) and "summer internship" in str(m.content) for m in out
    )
