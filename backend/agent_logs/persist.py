"""Persist agent-scoped events from the EventBus to ``agent_logs`` for replay."""
from __future__ import annotations

import logging
from typing import Any

from backend.api.events import Event
from backend.db.session import AsyncSessionLocal
from backend.ticket_system.models import AgentLog

logger = logging.getLogger(__name__)


def _payload_for_storage(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serialisable copy of the payload."""
    out: dict[str, Any] = {}
    for k, v in payload.items():
        try:
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, dict):
                out[k] = _payload_for_storage(v)
            elif isinstance(v, list):
                out[k] = [
                    x if isinstance(x, (str, int, float, bool)) or x is None else str(x)
                    for x in v
                ]
            else:
                out[k] = str(v)
        except Exception:  # noqa: BLE001
            out[k] = repr(v)
    return out


def _row_for(event: Event) -> AgentLog | None:
    """Map a bus event to an ``AgentLog`` row, or ``None`` if it isn't loggable."""
    if event.type != "agent" or not event.project_id:
        return None
    raw = event.payload
    if not isinstance(raw, dict):
        return None
    p = _payload_for_storage(raw)
    ticket_id = p.get("ticket_id")
    subtask_id = p.get("subtask_id")
    return AgentLog(
        project_id=event.project_id,
        agent=str(p.get("node") or p.get("agent") or "system")[:64],
        kind=str(p.get("kind") or "log")[:32],
        payload=p,
        ticket_id=str(ticket_id) if ticket_id else None,
        subtask_id=str(subtask_id) if subtask_id else None,
    )


async def persist_agent_events(events: list[Event]) -> None:
    """Write a batch of ``agent`` bus events in one transaction.

    Called only from the EventBus's background writer — never from an agent's
    tool loop. One session and one commit per batch instead of per event.
    """
    rows = [row for row in (_row_for(e) for e in events) if row is not None]
    if not rows:
        return
    try:
        async with AsyncSessionLocal() as db:
            db.add_all(rows)
            await db.commit()
    except Exception:
        logger.exception("failed to persist %d agent log(s)", len(rows))


async def persist_agent_event(event: Event) -> None:
    """Write a single ``agent`` bus event. Convenience wrapper over the batch path."""
    await persist_agent_events([event])
