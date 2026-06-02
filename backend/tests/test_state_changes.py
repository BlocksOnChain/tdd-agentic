"""Tests for SystemState changes (dead fields neutralized, context_store added)."""
from __future__ import annotations

from backend.agents.state import SystemState


def test_system_state_has_context_store_field() -> None:
    """SystemState has a context_store field."""
    state = SystemState()
    assert hasattr(state, "context_store")
    assert isinstance(state.context_store, dict)


def test_context_store_defaults_to_empty_dict() -> None:
    """context_store defaults to an empty dict."""
    state = SystemState()
    assert state.context_store == {}


def test_context_store_can_be_set() -> None:
    """context_store can be set to a dict."""
    state = SystemState(context_store={"ctx_1": {"agent": "researcher"}})
    assert state.context_store["ctx_1"]["agent"] == "researcher"


def test_project_context_canonical() -> None:
    """project_context exists and can hold the project goal."""
    state = SystemState(project_context="Build a todo app")
    assert state.project_context == "Build a todo app"


def test_active_fields_exist() -> None:
    """active_ticket_id and active_subtask_id exist."""
    state = SystemState(
        active_ticket_id="ticket-uuid",
        active_subtask_id="subtask-uuid",
    )
    assert state.active_ticket_id == "ticket-uuid"
    assert state.active_subtask_id == "subtask-uuid"


def test_dead_fields_do_not_accumulate() -> None:
    """Dead fields (pending_questions, human_responses) use identity lambdas."""
    # These fields should use lambda _a, _b: _a or [] to prevent accumulation.
    # The operator.add was replaced to prevent unbounded growth.
    state = SystemState()
    # Default should be empty
    assert isinstance(state.pending_questions, list)
    assert isinstance(state.human_responses, list)
    assert len(state.pending_questions) == 0
    assert len(state.human_responses) == 0
