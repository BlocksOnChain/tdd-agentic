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


async def persist_agent_event(event: Event) -> None:
    """Write one ``agent`` bus event to Postgres (best-effort, non-blocking graph)."""
    if event.type != "agent" or not event.project_id:
        return
    raw = event.payload
    if not isinstance(raw, dict):
        return
    p = _payload_for_storage(raw)
    agent = str(p.get("node") or p.get("agent") or "system")[:64]
    kind = str(p.get("kind") or "log")[:32]
    ticket_id = p.get("ticket_id")
    subtask_id = p.get("subtask_id")
    ticket_id_s = str(ticket_id) if ticket_id else None
    subtask_id_s = str(subtask_id) if subtask_id else None

    try:
        async with AsyncSessionLocal() as db:
            row = AgentLog(
                project_id=event.project_id,
                agent=agent,
                kind=kind,
                payload=p,
                ticket_id=ticket_id_s,
                subtask_id=subtask_id_s,
            )
            db.add(row)
            await db.commit()
    except Exception:
        logger.exception("failed to persist agent log for project %s", event.project_id)
