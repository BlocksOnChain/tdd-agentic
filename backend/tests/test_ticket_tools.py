"""Tests for PM ticket tool helpers and list_tickets etag caching."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.ticket_system.models import TicketStatus
from backend.tools.ticket_tools import (
    PM_TICKET_TOOLS,
    _ticket_roster_etag,
    get_ticket_summary,
    list_tickets,
)


def _ticket(
    *,
    tid: str,
    status: TicketStatus = TicketStatus.DRAFT,
    order_index: int = 0,
    updated_at: datetime | None = None,
    title: str = "Ticket",
    description: str = "",
    subtasks: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tid,
        title=title,
        description=description,
        status=status,
        order_index=order_index,
        updated_at=updated_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
        subtasks=subtasks or [],
    )


def test_pm_ticket_tools_includes_get_ticket_summary() -> None:
    tool_names = {t.name for t in PM_TICKET_TOOLS}
    assert "get_ticket_summary" in tool_names
    assert "list_tickets" in tool_names
    assert "get_ticket" in tool_names


def test_ticket_roster_etag_is_stable_for_same_roster() -> None:
    tickets = [
        _ticket(tid="b", order_index=1),
        _ticket(tid="a", order_index=0),
    ]
    assert _ticket_roster_etag(tickets) == _ticket_roster_etag(list(reversed(tickets)))


def test_ticket_roster_etag_changes_when_status_changes() -> None:
    base = [_ticket(tid="a", status=TicketStatus.DRAFT)]
    changed = [_ticket(tid="a", status=TicketStatus.IN_PROGRESS)]
    assert _ticket_roster_etag(base) != _ticket_roster_etag(changed)


def test_ticket_roster_etag_changes_when_updated_at_changes() -> None:
    early = [
        _ticket(
            tid="a",
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    ]
    late = [
        _ticket(
            tid="a",
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    ]
    assert _ticket_roster_etag(early) != _ticket_roster_etag(late)


@asynccontextmanager
async def _fake_session():
    yield AsyncMock()


@pytest.mark.asyncio
async def test_list_tickets_returns_etag_and_tickets() -> None:
    ticket = _ticket(tid="ticket-1", title="Auth middleware")
    list_item = {
        "id": "ticket-1",
        "title": "Auth middleware",
        "status": "draft",
        "order_index": 0,
        "subtask_count": 0,
        "description_preview": "",
        "hint": "Call get_ticket(<id>) for requirements and full subtask trees.",
    }
    with patch("backend.tools.ticket_tools.AsyncSessionLocal", _fake_session):
        with patch(
            "backend.tools.ticket_tools.service.list_tickets",
            AsyncMock(return_value=[ticket]),
        ):
            with patch(
                "backend.tools.ticket_tools.service.to_dict_ticket_list_item",
                return_value=list_item,
            ):
                raw = await list_tickets.ainvoke({"project_id": "proj-1"})
    payload = json.loads(raw)
    assert payload["etag"] == _ticket_roster_etag([ticket])
    assert payload["tickets"] == [list_item]


@pytest.mark.asyncio
async def test_list_tickets_returns_unchanged_when_etag_matches() -> None:
    ticket = _ticket(tid="ticket-1")
    etag = _ticket_roster_etag([ticket])
    with patch("backend.tools.ticket_tools.AsyncSessionLocal", _fake_session):
        with patch(
            "backend.tools.ticket_tools.service.list_tickets",
            AsyncMock(return_value=[ticket]),
        ):
            raw = await list_tickets.ainvoke(
                {"project_id": "proj-1", "since_last_check": etag}
            )
    payload = json.loads(raw)
    assert payload == {"unchanged": True, "etag": etag}


@pytest.mark.asyncio
async def test_list_tickets_returns_full_payload_when_etag_stale() -> None:
    ticket = _ticket(tid="ticket-1", status=TicketStatus.IN_PROGRESS)
    stale_etag = _ticket_roster_etag([_ticket(tid="ticket-1", status=TicketStatus.DRAFT)])
    list_item = {"id": "ticket-1", "status": "in_progress"}
    with patch("backend.tools.ticket_tools.AsyncSessionLocal", _fake_session):
        with patch(
            "backend.tools.ticket_tools.service.list_tickets",
            AsyncMock(return_value=[ticket]),
        ):
            with patch(
                "backend.tools.ticket_tools.service.to_dict_ticket_list_item",
                return_value=list_item,
            ):
                raw = await list_tickets.ainvoke(
                    {"project_id": "proj-1", "since_last_check": stale_etag}
                )
    payload = json.loads(raw)
    assert payload["etag"] == _ticket_roster_etag([ticket])
    assert payload["tickets"] == [list_item]
    assert "unchanged" not in payload


@pytest.mark.asyncio
async def test_get_ticket_summary_is_bound_tool() -> None:
    assert get_ticket_summary.name == "get_ticket_summary"
