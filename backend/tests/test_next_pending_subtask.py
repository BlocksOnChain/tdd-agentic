"""Unit tests for actionable subtask selection helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backend.ticket_system.models import AgentRole, SubtaskStatus
from backend.ticket_system.service import (
    ACTIONABLE_SUBTASK_STATUSES,
    _first_actionable_subtask,
    _subtask_status_priority,
)
from backend.tools.ticket_tools import _parse_dev_role


def _sub(
    *,
    sid: str,
    status: SubtaskStatus,
    order_index: int,
    assigned_to: AgentRole = AgentRole.BACKEND_DEV,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=sid,
        title=f"sub-{order_index}",
        status=status,
        order_index=order_index,
        assigned_to=assigned_to,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_subtask_status_priority_prefers_in_progress_then_blocked_then_pending() -> None:
    assert _subtask_status_priority(SubtaskStatus.IN_PROGRESS) < _subtask_status_priority(
        SubtaskStatus.BLOCKED
    )
    assert _subtask_status_priority(SubtaskStatus.BLOCKED) < _subtask_status_priority(
        SubtaskStatus.PENDING
    )


def test_first_actionable_subtask_resumes_in_progress_over_pending() -> None:
    subs = [
        _sub(sid="pending-1", status=SubtaskStatus.PENDING, order_index=0),
        _sub(sid="active-1", status=SubtaskStatus.IN_PROGRESS, order_index=5),
    ]
    picked = _first_actionable_subtask(subs)
    assert picked is not None
    assert picked["id"] == "active-1"
    assert picked["status"] == "in_progress"


def test_first_actionable_subtask_picks_blocked_before_pending() -> None:
    subs = [
        _sub(sid="pending-0", status=SubtaskStatus.PENDING, order_index=0),
        _sub(sid="blocked-1", status=SubtaskStatus.BLOCKED, order_index=1),
    ]
    picked = _first_actionable_subtask(subs)
    assert picked is not None
    assert picked["id"] == "blocked-1"


def test_first_actionable_subtask_ignores_done() -> None:
    subs = [_sub(sid="done-0", status=SubtaskStatus.DONE, order_index=0)]
    assert _first_actionable_subtask(subs) is None


def test_actionable_statuses_exclude_done() -> None:
    assert SubtaskStatus.DONE not in ACTIONABLE_SUBTASK_STATUSES


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("none", None),
        ("None", None),
        ("null", None),
        ("backend_dev", AgentRole.BACKEND_DEV),
    ],
)
def test_parse_dev_role(raw: str | None, expected: AgentRole | None) -> None:
    assert _parse_dev_role(raw) == expected


def test_parse_dev_role_rejects_non_dev_roles() -> None:
    with pytest.raises(ValueError, match="not a valid dev role"):
        _parse_dev_role("project_manager")
