"""Tests for the devops/infra completion verification gate.

Regression: devops reported "infrastructure complete" while run_tests and
shell_run were failing with exit 127 (jest/npm/node not found). The runner now
deterministically classifies the turn's run_tests outcome so a false completion
can be downgraded to BLOCKED.
"""
from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from backend.agents.runner import _subtasks_marked_done, _verification_outcome


def _tm(name: str, payload: dict, cid: str = "c1") -> ToolMessage:
    return ToolMessage(content=json.dumps(payload), name=name, tool_call_id=cid)


def test_run_tests_failure_is_classified_failed() -> None:
    msgs = [_tm("run_tests", {"ok": False, "exit_code": 127, "stderr": "jest: not found"})]
    status, detail = _verification_outcome(msgs)
    assert status == "failed"
    assert "127" in detail
    assert "jest: not found" in detail


def test_run_tests_success_is_classified_passed() -> None:
    msgs = [_tm("run_tests", {"ok": True, "exit_code": 0, "stdout": "1 passing"})]
    assert _verification_outcome(msgs) == ("passed", "")


def test_no_run_tests_is_classified_none() -> None:
    """Scaffolding files without running a test does not prove the infra works."""
    msgs = [_tm("fs_write", {"ok": True, "path": "src/config/db.js"})]
    assert _verification_outcome(msgs) == ("none", "")


def test_last_run_tests_wins() -> None:
    """A failing run followed by a passing run counts as passed (and vice versa)."""
    fail_then_pass = [
        _tm("run_tests", {"ok": False, "exit_code": 1}),
        _tm("run_tests", {"ok": True}),
    ]
    assert _verification_outcome(fail_then_pass)[0] == "passed"

    pass_then_fail = [
        _tm("run_tests", {"ok": True}),
        _tm("run_tests", {"ok": False, "exit_code": 1, "stderr": "1 failing"}),
    ]
    assert _verification_outcome(pass_then_fail)[0] == "failed"


def test_unparseable_run_tests_is_failed() -> None:
    """A run_tests result we can't parse must not be read as success."""
    bad = ToolMessage(content="<<not json>>", name="run_tests", tool_call_id="c1")
    assert _verification_outcome([bad])[0] == "failed"


def test_shell_run_failure_is_not_the_gate() -> None:
    """Plain shell_run is diagnostics/installs, not the verification gate."""
    msgs = [_tm("shell_run", {"ok": False, "exit_code": 127, "stderr": "npm: not found"})]
    # No run_tests present -> "none", not "failed" (shell_run alone isn't the gate).
    assert _verification_outcome(msgs) == ("none", "")


def test_subtasks_marked_done_detects_done_status() -> None:
    msgs = [
        _tm("update_subtask_status", {"id": "sub-1", "status": "in_progress"}),
        _tm("update_subtask_status", {"id": "sub-1", "status": "done"}),
        _tm("update_subtask_status", {"id": "sub-2", "status": "done"}),
    ]
    assert _subtasks_marked_done(msgs) == ["sub-1", "sub-2"]


def test_subtasks_marked_done_ignores_non_done() -> None:
    msgs = [_tm("update_subtask_status", {"id": "sub-1", "status": "blocked"})]
    assert _subtasks_marked_done(msgs) == []
