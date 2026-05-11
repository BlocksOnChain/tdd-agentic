"""PM routing metadata and handoff formatting."""
from __future__ import annotations

from backend.agents.project_manager.supervisor import (
    RoutingDecision,
    _format_pm_handoff,
    _normalise_ticket_ids,
    _parse_routing,
)


def test_parses_ticket_ids_and_phase() -> None:
    text = (
        '{"next_agent": "backend_dev", "rationale": "work", '
        '"ticket_ids": ["7c4f842e-e29b-41d4-a716-446655440000"], '
        '"phase": "implement", "instructions": "Finish subtask 2."}'
    )
    decision = _parse_routing(text)
    assert decision is not None
    assert decision.ticket_ids == ["7c4f842e-e29b-41d4-a716-446655440000"]
    assert decision.phase == "implement"


def test_normalise_ticket_ids_from_instructions_fallback() -> None:
    uid = "7c4f842e-e29b-41d4-a716-446655440000"
    decision = RoutingDecision(
        next_agent="backend_lead",
        instructions=f"Plan ticket {uid}.",
    )
    assert _normalise_ticket_ids(decision) == [uid]


def test_format_pm_handoff_includes_json_header() -> None:
    uid = "7c4f842e-e29b-41d4-a716-446655440000"
    decision = RoutingDecision(
        next_agent="backend_lead",
        ticket_ids=[uid],
        phase="backend_planning",
        instructions="Add backend subtasks.",
    )
    body = _format_pm_handoff("backend_lead", decision)
    assert body.startswith("[from project_manager → backend_lead]")
    assert uid in body
    assert "backend_planning" in body
