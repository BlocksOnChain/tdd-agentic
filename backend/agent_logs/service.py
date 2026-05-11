"""Read historical agent logs from Postgres."""
from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent_logs.display import format_agent_log_detail


async def list_agent_logs(
    db: AsyncSession,
    *,
    project_id: str,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Return newest-first rows; caller may reverse for chronological UI."""
    lim = max(1, min(limit, 10_000))
    result = await db.execute(
        text(
            """
            SELECT
              id,
              created_at,
              agent,
              kind,
              ticket_id,
              subtask_id,
              payload,
              (payload->>'checkpoint_id') AS checkpoint_id
            FROM agent_logs
            WHERE project_id = :project_id
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        {"project_id": project_id, "lim": lim},
    )
    rows = result.fetchall()
    out: list[dict[str, Any]] = []
    for (
        id_,
        created_at,
        agent,
        kind,
        ticket_id,
        subtask_id,
        payload,
        checkpoint_id,
    ) in rows:
        p: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {}
        if ticket_id is not None:
            p.setdefault("ticket_id", ticket_id)
        if subtask_id is not None:
            p.setdefault("subtask_id", subtask_id)
        out.append(
            {
                "id": id_,
                "created_at": created_at.isoformat() if created_at else None,
                "ts": created_at.timestamp() if created_at else None,
                "agent": agent,
                "kind": kind,
                "payload": p,
                "detail": format_agent_log_detail(kind, p),
                "checkpoint_id": checkpoint_id,
            }
        )
    return out
