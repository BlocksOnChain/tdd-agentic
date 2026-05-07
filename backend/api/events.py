"""In-memory pub/sub for realtime agent events broadcast over WebSockets.

This is intentionally lightweight: each connected client gets an asyncio
queue, and the publisher fan-outs every event to every queue. Replace with
Redis Streams for multi-process deployments.
"""
from __future__ import annotations

import asyncio
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


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, event: Event) -> None:
        async with self._lock:
            queues = list(self._subscribers)
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop on slow consumers rather than block the producer
                pass
        # Persist agent events so the Logs UI can reload full history after refresh.
        # Lazy-import avoids import cycles during app startup.
        try:
            from backend.agent_logs.persist import persist_agent_event

            await persist_agent_event(event)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).exception("persist_agent_event failed")

    async def subscribe(self) -> AsyncIterator[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)
        async with self._lock:
            self._subscribers.add(q)
        try:
            while True:
                event = await q.get()
                yield event
        finally:
            async with self._lock:
                self._subscribers.discard(q)


bus = EventBus()
