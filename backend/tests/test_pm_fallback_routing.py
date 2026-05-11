"""DB-style fallback when the PM model omits valid routing JSON or chooses end too early."""
from __future__ import annotations

from types import SimpleNamespace

from backend.agents.project_manager.supervisor import _fallback_routing_decision
from backend.ticket_system.models import AgentRole, TicketStatus


def _st(**kwargs):
    defaults = dict(
        title="",
        description="",
        required_functionality="",
        assigned_to=AgentRole.BACKEND_DEV,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_fallback_routes_frontend_for_todo_ticket_with_client_hint_and_no_fe_subtask() -> None:
    t = SimpleNamespace(
        id="tid",
        status=TicketStatus.TODO,
        title="WebRTC Peer Connection Manager (Client)",
        description="",
        business_requirements=[],
        technical_requirements=[],
        questions=[],
        subtasks=[_st()],
    )
    d = _fallback_routing_decision([t])
    assert d is not None
    assert d.next_agent == "frontend_lead"
    assert d.ticket_ids == ["tid"]


def test_fallback_routes_backend_for_draft_without_subtasks() -> None:
    t = SimpleNamespace(
        id="draft-1",
        status=TicketStatus.DRAFT,
        title="New API",
        description="",
        business_requirements=[],
        technical_requirements=[],
        questions=[],
        subtasks=[],
    )
    d = _fallback_routing_decision([t])
    assert d is not None
    assert d.next_agent == "backend_lead"


def test_fallback_skips_when_frontend_subtasks_exist() -> None:
    t = SimpleNamespace(
        id="tid",
        status=TicketStatus.TODO,
        title="Dashboard (Client)",
        description="",
        business_requirements=[],
        technical_requirements=[],
        questions=[],
        subtasks=[_st(assigned_to=AgentRole.FRONTEND_DEV)],
    )
    assert _fallback_routing_decision([t]) is None


def test_fallback_skips_api_only_ticket_in_todo() -> None:
    t = SimpleNamespace(
        id="tid",
        status=TicketStatus.TODO,
        title="Add rate limiting middleware",
        description="",
        business_requirements=[],
        technical_requirements=[],
        questions=[],
        subtasks=[_st()],
    )
    assert _fallback_routing_decision([t]) is None


def test_fallback_none_when_all_tickets_done() -> None:
    t = SimpleNamespace(
        id="tid",
        status=TicketStatus.DONE,
        title="Done",
        description="",
        business_requirements=[],
        technical_requirements=[],
        questions=[],
        subtasks=[],
    )
    assert _fallback_routing_decision([t]) is None
