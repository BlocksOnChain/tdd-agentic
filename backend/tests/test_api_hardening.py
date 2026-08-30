"""Workspace path containment, event-log batching, and WS project filtering."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from backend.api.events import Event, EventBus
from backend.api.routes.agents import _contained_path


class TestWorkspaceContainment:
    """`?path=../..` must not read outside the project workspace.

    The tool layer has always enforced this (code_tools._resolve); the HTTP
    endpoints joined the user-supplied path straight onto the workspace root.
    """

    @pytest.fixture
    def workspace(self, tmp_path):
        ws = tmp_path / "workspace" / "proj-1"
        (ws / "src").mkdir(parents=True)
        (ws / "src" / "app.ts").write_text("export const x = 1;")
        (tmp_path / "workspace" / "secret.txt").write_text("other project's data")
        (tmp_path / "outside.txt").write_text("host filesystem")
        return ws.resolve()

    @pytest.mark.parametrize(
        "path",
        [
            "..",
            "../",
            "../secret.txt",
            "../../outside.txt",
            "src/../../secret.txt",
            "../../../../../../etc/passwd",
        ],
    )
    def test_escapes_are_refused(self, workspace, path) -> None:
        with pytest.raises(HTTPException) as exc:
            _contained_path(workspace, path)
        assert exc.value.status_code == 400
        assert "escapes" in exc.value.detail

    @pytest.mark.parametrize("path", ["", ".", "src", "src/app.ts", "./src/app.ts"])
    def test_legitimate_paths_resolve(self, workspace, path) -> None:
        resolved = _contained_path(workspace, path)
        assert resolved == workspace or workspace in resolved.parents

    def test_absolute_path_outside_is_refused(self, workspace, tmp_path) -> None:
        with pytest.raises(HTTPException):
            _contained_path(workspace, str(tmp_path / "outside.txt"))


class TestEventBusPersistence:
    """Log writes are queued, not awaited on the agent's critical path."""

    @pytest.mark.asyncio
    async def test_publish_does_not_await_the_database(self, monkeypatch) -> None:
        writes: list[list[Event]] = []
        release = asyncio.Event()

        async def _slow_persist(events):
            await release.wait()
            writes.append(list(events))

        import backend.agent_logs.persist as persist_mod

        monkeypatch.setattr(persist_mod, "persist_agent_events", _slow_persist)

        bus = EventBus()
        # Publishing must return promptly even while the writer is blocked.
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    bus.publish(Event(type="agent", project_id="p", payload={"kind": "log"}))
                    for _ in range(200)
                )
            ),
            timeout=1.0,
        )
        assert writes == [], "publish should not have waited for the writer"

        release.set()
        await asyncio.wait_for(bus.flush(), timeout=2.0)
        await bus.close()

        assert sum(len(batch) for batch in writes) == 200
        assert len(writes) < 200, "events should be written in batches, not one at a time"

    @pytest.mark.asyncio
    async def test_non_agent_events_are_not_persisted(self, monkeypatch) -> None:
        writes: list[Event] = []

        async def _persist(events):
            writes.extend(events)

        import backend.agent_logs.persist as persist_mod

        monkeypatch.setattr(persist_mod, "persist_agent_events", _persist)

        bus = EventBus()
        await bus.publish(Event(type="ticket", project_id="p", payload={}))
        await bus.publish(Event(type="agent", project_id=None, payload={}))
        await bus.publish(Event(type="agent", project_id="p", payload={"kind": "route"}))
        await asyncio.wait_for(bus.flush(), timeout=2.0)
        await bus.close()

        assert len(writes) == 1
        assert writes[0].payload["kind"] == "route"


class TestEventBusFiltering:
    """A client watching one project shouldn't receive every project's traffic."""

    @pytest.mark.asyncio
    async def test_subscriber_receives_only_its_project(self, monkeypatch) -> None:
        import backend.agent_logs.persist as persist_mod

        monkeypatch.setattr(persist_mod, "persist_agent_events", lambda events: asyncio.sleep(0))

        bus = EventBus()
        received: list[str | None] = []

        async def _consume():
            async for event in bus.subscribe(project_id="mine"):
                received.append(event.payload.get("tag"))
                if len(received) == 2:
                    return

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0)  # let the subscriber register

        await bus.publish(Event(type="agent", project_id="theirs", payload={"tag": "no"}))
        await bus.publish(Event(type="agent", project_id="mine", payload={"tag": "yes-1"}))
        await bus.publish(Event(type="agent", project_id="theirs", payload={"tag": "no"}))
        await bus.publish(Event(type="agent", project_id="mine", payload={"tag": "yes-2"}))

        await asyncio.wait_for(consumer, timeout=2.0)
        await bus.close()

        assert received == ["yes-1", "yes-2"]

    @pytest.mark.asyncio
    async def test_unfiltered_subscriber_receives_everything(self, monkeypatch) -> None:
        import backend.agent_logs.persist as persist_mod

        monkeypatch.setattr(persist_mod, "persist_agent_events", lambda events: asyncio.sleep(0))

        bus = EventBus()
        received: list[str] = []

        async def _consume():
            async for event in bus.subscribe():
                received.append(event.payload["tag"])
                if len(received) == 2:
                    return

        consumer = asyncio.create_task(_consume())
        await asyncio.sleep(0)

        await bus.publish(Event(type="agent", project_id="a", payload={"tag": "a"}))
        await bus.publish(Event(type="agent", project_id="b", payload={"tag": "b"}))

        await asyncio.wait_for(consumer, timeout=2.0)
        await bus.close()

        assert received == ["a", "b"]
