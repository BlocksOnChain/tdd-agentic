"""Tests for the dev-agent subtask resolution gate.

Regression: backend_dev read files and ran partial tests but ended the turn
without marking the subtask done or blocked, handing a vague summary to the PM.
"""
from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from backend.agents.runner import (
    _engaged_subtask_id_from_tools,
    _subtask_resolution_outcome,
    _subtasks_marked_blocked,
    _subtasks_marked_done,
)


def _tm(name: str, payload: dict, cid: str = "c1") -> ToolMessage:
    return ToolMessage(content=json.dumps(payload), name=name, tool_call_id=cid)


def test_engaged_subtask_from_next_pending_in_project() -> None:
    msgs = [
        _tm(
            "next_pending_subtask_in_project",
            {
                "ticket": {"id": "t-1"},
                "subtask": {"id": "sub-1", "title": "Implement POST /submit"},
                "resume": False,
            },
        )
    ]
    assert _engaged_subtask_id_from_tools(msgs) == "sub-1"


def test_engaged_subtask_null_means_no_work() -> None:
    msgs = [_tm("next_pending_subtask_in_project", {"ticket": None, "subtask": None, "resume": False})]
    assert _engaged_subtask_id_from_tools(msgs) == None  # noqa: E711


def test_resolution_incomplete_when_subtask_not_closed() -> None:
    msgs = [
        _tm(
            "next_pending_subtask_in_project",
            {"subtask": {"id": "sub-1"}, "resume": False},
        ),
        _tm("update_subtask_status", {"id": "sub-1", "status": "in_progress"}),
        _tm("run_tests", {"ok": True, "exit_code": 0}),
    ]
    status, detail = _subtask_resolution_outcome(
        msgs, engaged_subtask_id="sub-1", step_exhausted=False, max_steps=30
    )
    assert status == "incomplete"
    assert "not marked done or blocked" in detail


def test_resolution_resolved_when_marked_done() -> None:
    msgs = [
        _tm("next_pending_subtask_in_project", {"subtask": {"id": "sub-1"}, "resume": False}),
        _tm("update_subtask_status", {"id": "sub-1", "status": "done"}),
    ]
    status, _ = _subtask_resolution_outcome(
        msgs, engaged_subtask_id="sub-1", step_exhausted=True, max_steps=30
    )
    assert status == "resolved"


def test_resolution_resolved_when_marked_blocked() -> None:
    msgs = [
        _tm("next_pending_subtask_in_project", {"subtask": {"id": "sub-1"}, "resume": False}),
        _tm("update_subtask_status", {"id": "sub-1", "status": "blocked"}),
    ]
    status, _ = _subtask_resolution_outcome(
        msgs, engaged_subtask_id="sub-1", step_exhausted=False, max_steps=30
    )
    assert status == "resolved"


def test_resolution_no_engagement_without_subtask_id() -> None:
    status, detail = _subtask_resolution_outcome(
        [], engaged_subtask_id=None, step_exhausted=False, max_steps=30
    )
    assert status == "no_engagement"
    assert detail == ""


def test_step_exhausted_detail_in_incomplete() -> None:
    msgs = [_tm("next_pending_subtask", {"subtask": {"id": "sub-1"}, "resume": True})]
    status, detail = _subtask_resolution_outcome(
        msgs, engaged_subtask_id="sub-1", step_exhausted=True, max_steps=20
    )
    assert status == "incomplete"
    assert "max_steps=20" in detail


def test_subtasks_marked_blocked_detects_blocked_status() -> None:
    msgs = [_tm("update_subtask_status", {"id": "sub-1", "status": "blocked"})]
    assert _subtasks_marked_blocked(msgs) == ["sub-1"]
    assert _subtasks_marked_done(msgs) == []
