"""In-memory pub/sub for realtime agent events broadcast over WebSockets.

This is intentionally lightweight: each connected client gets an asyncio
queue, and the publisher fan-outs every event to every queue. Replace with
Redis Streams for multi-process deployments.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    type: str  # "agent" | "ticket" | "interrupt" | "log"
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    project_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


_logger = logging.getLogger(__name__)

# How many un-persisted events to hold before dropping the oldest. Generous:
# a dev agent at 60 steps emits hundreds of events in a burst, and the writer
# drains far faster than the model produces them.
PERSIST_QUEUE_MAX = 10_000
# Rows per INSERT batch.
PERSIST_BATCH_SIZE = 50


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._lock = asyncio.Lock()
        self._persist_queue: asyncio.Queue[Event] | None = None
        self._writer: asyncio.Task | None = None
        self._dropped = 0

    # ----- persistence -----

    def _ensure_writer(self) -> asyncio.Queue[Event]:
        """Lazily start the background log writer, bound to the running loop."""
        if self._persist_queue is None:
            self._persist_queue = asyncio.Queue(maxsize=PERSIST_QUEUE_MAX)
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(self._drain_persist_queue(), name="agent-log-writer")
        return self._persist_queue

    async def _drain_persist_queue(self) -> None:
        """Write queued agent events to Postgres in batches, off the agent's path."""
        from backend.agent_logs.persist import persist_agent_events

        assert self._persist_queue is not None
        queue = self._persist_queue
        while True:
            first = await queue.get()
            batch = [first]
            while len(batch) < PERSIST_BATCH_SIZE:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                await persist_agent_events(batch)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                _logger.exception("persisting %d agent log(s) failed", len(batch))
            finally:
                for _ in batch:
                    queue.task_done()

    async def flush(self) -> None:
        """Wait for queued agent logs to be written (shutdown / tests)."""
        if self._persist_queue is not None:
            await self._persist_queue.join()

    async def close(self) -> None:
        await self.flush()
        if self._writer is not None and not self._writer.done():
            self._writer.cancel()
            try:
                await self._writer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._writer = None

    # ----- pub/sub -----

    async def publish(self, event: Event) -> None:
        """Fan out to subscribers and hand the event to the log writer.

        Persistence is queued, never awaited inline. It used to open a session,
        INSERT, and COMMIT before returning — on the agent's critical path, once
        per tool result, hundreds of times per dev turn.
        """
        async with self._lock:
            queues = list(self._subscribers)
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop on slow consumers rather than block the producer
                pass

        if event.type != "agent" or not event.project_id:
            return
        try:
            self._ensure_writer().put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:
                _logger.warning(
                    "agent-log queue full; dropped %d event(s) so far", self._dropped
                )
        except RuntimeError:
            # No running loop (sync context / teardown) — logging is best-effort.
            pass

    async def subscribe(self, project_id: str | None = None) -> AsyncIterator[Event]:
        """Yield events, optionally only those for ``project_id``.

        Unfiltered subscribers receive every project's traffic, so a client
        watching one project pays for all concurrent runs.
        """
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        async with self._lock:
            self._subscribers.add(q)
        try:
            while True:
                event = await q.get()
                if project_id is not None and event.project_id not in (None, project_id):
                    continue
                yield event
        finally:
            async with self._lock:
                self._subscribers.discard(q)


bus = EventBus()
