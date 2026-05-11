"""DB-style fallback when the PM model omits valid routing JSON or chooses end too early."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from backend.agents.project_manager import supervisor
from backend.agents.project_manager.supervisor import (
    RoutingDecision,
    _fallback_routing_decision,
    _infer_fallback_route,
    _pm_ticket_creation_nudge,
    _research_materials_ready,
    _research_phase_complete,
)
from backend.agents.session_memory import SessionMemory
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


def test_research_phase_complete_after_researcher_handoff() -> None:
    messages = [
        HumanMessage(content="Build a todo app"),
        HumanMessage(content="[from project_manager → researcher]\n{\"phase\":\"research\"}"),
        HumanMessage(content="[from researcher → project_manager]\nDocs ingested."),
    ]
    assert _research_phase_complete(messages) is True


def test_research_phase_not_complete_before_handoff() -> None:
    messages = [
        HumanMessage(content="Build a todo app"),
        HumanMessage(content="[from project_manager → researcher]\n{\"phase\":\"research\"}"),
    ]
    assert _research_phase_complete(messages) is False


def test_research_phase_not_complete_from_session_memory_without_handoff() -> None:
    memory = SessionMemory(
        last_route="researcher",
        recent_handoffs=["researcher: Docs ingested."],
    )
    assert _research_phase_complete([], memory) is False


class _FakeSession:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_infer_fallback_routes_pm_after_research_without_tickets(monkeypatch) -> None:
    async def _list_tickets(_db, project_id: str):
        assert project_id == "proj-1"
        return []

    monkeypatch.setattr(supervisor.service, "list_tickets", _list_tickets)
    monkeypatch.setattr(supervisor, "AsyncSessionLocal", _FakeSession)
    monkeypatch.setattr(
        supervisor,
        "list_research_markdown",
        lambda _pid: [{"path": "docs/tech-stack.md", "bytes": 120}],
    )

    decision = await _infer_fallback_route(
        "proj-1",
        messages=[
            HumanMessage(content="[from project_manager → researcher]\n{}"),
            HumanMessage(content="[from researcher → project_manager]\nDone."),
        ],
    )
    assert decision is not None
    assert decision.next_agent == "project_manager"


@pytest.mark.asyncio
async def test_infer_fallback_routes_researcher_before_research(monkeypatch) -> None:
    async def _list_tickets(_db, project_id: str):
        return []

    monkeypatch.setattr(supervisor.service, "list_tickets", _list_tickets)
    monkeypatch.setattr(supervisor, "AsyncSessionLocal", _FakeSession)

    decision = await _infer_fallback_route("proj-1", messages=[])
    assert decision is not None
    assert decision.next_agent == "researcher"


@pytest.mark.asyncio
async def test_infer_fallback_routes_researcher_when_docs_missing(monkeypatch) -> None:
    async def _list_tickets(_db, project_id: str):
        return []

    monkeypatch.setattr(supervisor.service, "list_tickets", _list_tickets)
    monkeypatch.setattr(supervisor, "AsyncSessionLocal", _FakeSession)
    monkeypatch.setattr(supervisor, "list_research_markdown", lambda _pid: [])

    decision = await _infer_fallback_route(
        "proj-1",
        messages=[
            HumanMessage(content="[from project_manager → researcher]\n{}"),
            HumanMessage(content="[from researcher → project_manager]\nDone."),
        ],
    )
    assert decision is not None
    assert decision.next_agent == "researcher"


def test_research_materials_ready_requires_substantive_docs(monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "list_research_markdown",
        lambda _pid: [{"path": "docs/tech-stack.md", "bytes": 10}],
    )
    assert _research_materials_ready("proj-1") is True
    monkeypatch.setattr(
        supervisor,
        "list_research_markdown",
        lambda _pid: [
            {"path": "AGENTS.md", "bytes": 10},
            {"path": "docs/README.md", "bytes": 10},
        ],
    )
    assert _research_materials_ready("proj-1") is False


def test_pm_ticket_creation_nudge_targets_project_manager() -> None:
    nudge = _pm_ticket_creation_nudge()
    assert nudge.startswith("[from project_manager → project_manager]")
    assert "list_tickets" in nudge
    assert "create_ticket" in nudge


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
