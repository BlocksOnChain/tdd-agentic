"""Lead → state.execution_plan → Coordinator hand-off.

Regression: ``state.execution_plan`` was written by the Lead and read by nobody.
The Coordinator's prompt told it to "read state.execution_plan", but graph state
is invisible to an LLM, so it received only the PM's one-line intent and invented
subtasks — silently discarding every RITE spec the Lead had written.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents.runner import (
    _execution_plan_persisted,
    _render_execution_plan_message,
)
from backend.agents.state import ExecutionPlan, SubtaskPlan, SystemState
from backend.agents.state import TestCaseInput as RiteSpec  # aliased: pytest tries to collect Test*


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        ticket_id="11111111-1111-1111-1111-111111111111",
        subtasks=[
            SubtaskPlan(
                title="Create JWT service",
                description="Sign and verify access tokens.",
                required_functionality="signJwt / verifyJwt",
                assigned_to="backend_dev",
                test_cases=[
                    RiteSpec(
                        given="a valid payload and secret",
                        should="return a verifiable token",
                        expected="a string with three dot-separated segments",
                    )
                ],
            ),
            SubtaskPlan(
                title="Create login page",
                description="Email + password form.",
                assigned_to="frontend_dev",
                test_cases=[
                    RiteSpec(
                        given="an empty password field",
                        should="disable the submit button",
                        expected="button.disabled === true",
                    )
                ],
            ),
        ],
    )


def test_rendered_plan_carries_every_subtask_and_spec() -> None:
    msg = _render_execution_plan_message(_plan())
    body = msg.content

    assert "[execution_plan]" in body
    payload = json.loads(body[body.index("{") : body.rindex("}") + 1])

    titles = [s["title"] for s in payload["subtasks"]]
    assert titles == ["Create JWT service", "Create login page"]
    assert [s["assigned_to"] for s in payload["subtasks"]] == ["backend_dev", "frontend_dev"]

    # The RITE specs are the whole point of the hand-off — they must survive.
    assert payload["subtasks"][0]["test_cases"][0]["expected"] == (
        "a string with three dot-separated segments"
    )
    assert payload["ticket_id"] == "11111111-1111-1111-1111-111111111111"


def test_missing_plan_tells_coordinator_to_stop_not_improvise() -> None:
    for empty in (None, ExecutionPlan(ticket_id=None, subtasks=[])):
        body = _render_execution_plan_message(empty).content
        assert "MISSING" in body
        assert "Do NOT invent subtasks" in body
        assert "save_execution_plan" in body  # explicitly told not to call it


def test_oversized_plan_drops_whole_subtasks_not_characters(monkeypatch) -> None:
    """Truncation must never produce invalid JSON the Coordinator half-persists."""
    import backend.agents.runner as runner

    monkeypatch.setattr(runner, "MAX_EXECUTION_PLAN_CHARS", 400)
    big = ExecutionPlan(
        ticket_id="t",
        subtasks=[
            SubtaskPlan(title=f"Subtask {i}", description="x" * 80, assigned_to="backend_dev")
            for i in range(10)
        ],
    )
    body = runner._render_execution_plan_message(big).content
    payload = json.loads(body[body.index("{") : body.rindex("}") + 1])

    assert 0 < len(payload["subtasks"]) < 10
    assert "omitted" in body
    # Every surviving subtask is intact, not cut mid-field.
    assert all(s["title"] and s["assigned_to"] for s in payload["subtasks"])


def test_plan_persisted_detection() -> None:
    ok = ToolMessage(
        content=json.dumps({"ticket_id": "t", "subtask_count": 2, "subtask_ids": ["a", "b"]}),
        name="save_execution_plan",
        tool_call_id="1",
    )
    empty = ToolMessage(
        content=json.dumps({"ticket_id": "t", "subtask_count": 0, "subtask_ids": []}),
        name="save_execution_plan",
        tool_call_id="2",
    )
    err = ToolMessage(
        content=json.dumps({"error": "No ticket with id 'x'"}),
        name="save_execution_plan",
        tool_call_id="3",
    )
    other = ToolMessage(content="{}", name="transition_ticket", tool_call_id="4")

    assert _execution_plan_persisted([ok]) is True
    assert _execution_plan_persisted([empty]) is False
    assert _execution_plan_persisted([err]) is False
    assert _execution_plan_persisted([other]) is False
    assert _execution_plan_persisted([err, ok]) is True


@pytest.mark.asyncio
async def test_coordinator_subgraph_sees_the_plan_and_clears_it(monkeypatch) -> None:
    """Drive the real Coordinator subgraph with a stub model and assert both halves."""
    import backend.agents.coordinator.subgraph as coord

    seen: dict = {}

    class _StubModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, *a, **kw):
            if "prompt" not in seen:
                seen["prompt"] = "\n\n".join(str(m.content) for m in messages)
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "save_execution_plan",
                            "args": {"project_id": "p", "ticket_id": "t", "subtasks": []},
                            "id": "call_1",
                        }
                    ],
                )
            return AIMessage(content="Persisted 2 subtasks under ticket t.")

    async def _fake_save(**kwargs):
        seen["saved"] = kwargs
        return json.dumps({"ticket_id": "t", "subtask_count": 2, "subtask_ids": ["a", "b"]})

    monkeypatch.setattr(coord, "coordinator_model", lambda: _StubModel())
    graph = coord.build_coordinator_subgraph()

    # Stub the persistence tool the compiled graph already bound.
    import backend.tools.persistence_tools as pt

    monkeypatch.setattr(pt.save_execution_plan, "coroutine", _fake_save)

    state = SystemState(
        project_id="p",
        project_context="build a todo app",
        execution_plan=_plan(),
        messages=[
            HumanMessage(
                content='[from project_manager → coordinator]\n{"t":"coordinator"}\nPersist the plan.'
            )
        ],
    )

    result = await graph.ainvoke(state)

    # 1. The Coordinator actually saw the Lead's subtasks.
    assert "Create JWT service" in seen["prompt"]
    assert "Create login page" in seen["prompt"]
    assert "a string with three dot-separated segments" in seen["prompt"]

    # 2. The plan is cleared once persisted, so it can't leak onto the next ticket.
    assert result["execution_plan"] is None
