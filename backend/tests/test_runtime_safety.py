"""Per-invocation isolation, replay safety, and bounded control flow."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from backend.agents.runner import build_specialist_subgraph
from backend.agents.state import SystemState


@tool
async def _pick_subtask(project_id: str, role: str) -> str:
    """Stand-in for next_pending_subtask_in_project."""
    return json.dumps({"subtask": {"id": "sub-1"}, "resume": False})


_pick_subtask.name = "next_pending_subtask_in_project"


@tool
async def _write_file(project_id: str, path: str, content: str) -> str:
    """Stand-in for a phase-2 code tool."""
    return json.dumps({"ok": True})


_write_file.name = "fs_write"


class _ScriptedModel:
    """Records the tool names it was bound with on each bind_tools call."""

    def __init__(self, script, bindings):
        self._script = list(script)
        self._bindings = bindings
        self._i = 0

    def bind_tools(self, tools):
        self._bindings.append({t.name for t in tools})
        return self

    async def ainvoke(self, messages, *a, **kw):
        msg = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return msg


class TestPhasedToolIsolation:
    """The code-tool gate must reset every turn.

    Regression: ``code_phase_active`` and ``tools_by_name`` lived in the factory
    closure, so the first turn that picked up a subtask unlocked the code tools
    permanently — for every later turn, every later run, and every other project
    sharing the compiled graph.
    """

    def _graph(self, bindings):
        script = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "next_pending_subtask_in_project",
                        "args": {"project_id": "p", "role": "backend_dev"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
        # A fresh model per turn, mirroring how the runner calls llm_factory().
        return build_specialist_subgraph(
            name="backend_dev",
            role="backend_dev",
            llm_factory=lambda: _ScriptedModel(script, bindings),
            tools=[_pick_subtask],
            code_tools=[_write_file],
            phased_code_tools=True,
            base_system_prompt="dev",
            max_steps=4,
        )

    @pytest.mark.asyncio
    async def test_first_bind_of_every_turn_excludes_code_tools(self) -> None:
        bindings: list[set[str]] = []
        graph = self._graph(bindings)
        state = SystemState(project_id="p", messages=[HumanMessage(content="go")])

        await graph.ainvoke(state)
        first_turn = list(bindings)

        bindings.clear()
        await graph.ainvoke(state)
        second_turn = list(bindings)

        # Turn 1: starts locked, unlocks after the subtask is fetched.
        assert "fs_write" not in first_turn[0]
        assert "fs_write" in first_turn[-1]

        # Turn 2 must start locked again — this is the actual regression.
        assert "fs_write" not in second_turn[0], (
            "code tools leaked from the previous turn; phase state is not per-invocation"
        )
        assert "fs_write" in second_turn[-1]

    @pytest.mark.asyncio
    async def test_separate_graphs_do_not_share_phase_state(self) -> None:
        a_bindings: list[set[str]] = []
        b_bindings: list[set[str]] = []
        state = SystemState(project_id="p", messages=[HumanMessage(content="go")])

        await self._graph(a_bindings).ainvoke(state)
        await self._graph(b_bindings).ainvoke(state)

        assert "fs_write" not in b_bindings[0]


class TestEndVetoBound:
    """A DB heuristic may override 'end' — but not forever."""

    @pytest.mark.asyncio
    async def _route(self, *, veto_count: int, fallback):
        from backend.agents.project_manager import supervisor as sup

        node = sup.build_project_manager_node()
        state = SystemState(
            project_id="p",
            messages=[HumanMessage(content="goal")],
            end_veto_count=veto_count,
        )
        model = MagicMock()
        model.bind_tools.return_value = model
        model.ainvoke = AsyncMock(return_value=AIMessage(content='{"next_agent": "end"}'))

        with (
            patch.object(sup, "pm_model", return_value=model),
            patch.object(sup, "with_retry", side_effect=lambda r: r),
            patch.object(sup, "_advance_in_review_to_todo", new=AsyncMock(return_value=[])),
            patch.object(sup, "_infer_fallback_route", new=AsyncMock(return_value=fallback)),
            patch.object(sup, "_validate_ticket_ids", new=AsyncMock(side_effect=lambda d, p: (d, None))),
            patch.object(sup, "emit", new=AsyncMock(return_value=SimpleNamespace(kind="x"))),
        ):
            return await node(state)

    @pytest.mark.asyncio
    async def test_first_veto_overrides_end(self) -> None:
        from backend.agents.project_manager.supervisor import RoutingDecision

        result = await self._route(
            veto_count=0,
            fallback=RoutingDecision(next_agent="lead", rationale="needs planning"),
        )
        assert result["next_agent"] == "lead"
        assert result["end_veto_count"] == 1

    @pytest.mark.asyncio
    async def test_veto_stops_at_the_cap(self) -> None:
        """Otherwise an unsatisfiable heuristic spins to the recursion limit."""
        from backend.agents.project_manager.supervisor import MAX_END_VETOES, RoutingDecision

        result = await self._route(
            veto_count=MAX_END_VETOES,
            fallback=RoutingDecision(next_agent="lead", rationale="still not satisfied"),
        )
        assert result["next_agent"] == "end"

    @pytest.mark.asyncio
    async def test_no_fallback_means_end_stands(self) -> None:
        result = await self._route(veto_count=0, fallback=None)
        assert result["next_agent"] == "end"


class TestPlanPersistenceIsIdempotent:
    """The Coordinator can be re-dispatched after a crash, retry, or interrupt."""

    def _session(self, ticket):
        db = AsyncMock()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=db)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    @pytest.mark.asyncio
    async def test_existing_subtasks_are_not_duplicated(self) -> None:
        from backend.tools import persistence_tools as pt

        ticket = SimpleNamespace(
            id="t1",
            subtasks=[SimpleNamespace(title="Create JWT service", order_index=0)],
        )
        created: list[str] = []

        async def _create(db, ticket_id, payload):
            created.append(payload.title)
            return SimpleNamespace(id=f"sub-{len(created)}")

        with (
            patch.object(pt, "AsyncSessionLocal", return_value=self._session(ticket)),
            patch.object(pt.service, "get_ticket", new=AsyncMock(return_value=ticket)),
            patch.object(pt.service, "create_subtask", new=_create),
        ):
            out = json.loads(
                await pt.save_execution_plan.ainvoke(
                    {
                        "project_id": "p",
                        "ticket_id": "t1",
                        "subtasks": [
                            {"title": "Create JWT service", "assigned_to": "backend_dev", "order_index": 0},
                            {"title": "Create login page", "assigned_to": "frontend_dev", "order_index": 1},
                        ],
                    }
                )
            )

        assert created == ["Create login page"]
        assert out["subtask_count"] == 1
        assert out["skipped_existing"] == ["Create JWT service"]

    @pytest.mark.asyncio
    async def test_bad_role_is_reported_not_silently_reassigned(self) -> None:
        """Defaulting an unknown assigned_to to backend_dev turned frontend work
        into backend work, and only surfaced later as a dev building the wrong thing."""
        from backend.tools import persistence_tools as pt

        ticket = SimpleNamespace(id="t1", subtasks=[])
        created: list[str] = []

        async def _create(db, ticket_id, payload):
            created.append(payload.assigned_to.value)
            return SimpleNamespace(id="sub-1")

        with (
            patch.object(pt, "AsyncSessionLocal", return_value=self._session(ticket)),
            patch.object(pt.service, "get_ticket", new=AsyncMock(return_value=ticket)),
            patch.object(pt.service, "create_subtask", new=_create),
        ):
            out = json.loads(
                await pt.save_execution_plan.ainvoke(
                    {
                        "project_id": "p",
                        "ticket_id": "t1",
                        "subtasks": [{"title": "Build UI", "assigned_to": "ui_dev"}],
                    }
                )
            )

        assert created == []
        assert out["subtask_count"] == 0
        assert out["rejected"][0]["title"] == "Build UI"
        assert "ui_dev" in out["rejected"][0]["reason"]
