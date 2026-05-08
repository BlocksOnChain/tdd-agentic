"""Read historical agent logs from Postgres."""
from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_MAX_PREVIEW_CHARS = 1200
_MAX_RATIONALE_CHARS = 800
_MAX_ERROR_CHARS = 1200

async def list_agent_logs(
    db: AsyncSession,
    *,
    project_id: str,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Return newest-first rows; caller may reverse for chronological UI."""
    lim = max(1, min(limit, 10_000))
    # Fast path: return a compact, bounded payload preview (no full JSON).
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
              left((payload->>'preview'), :preview_chars) AS preview,
              left((payload->>'rationale'), :rationale_chars) AS rationale,
              left((payload->>'error'), :error_chars) AS error,
              (payload->>'checkpoint_id') AS checkpoint_id
            FROM agent_logs
            WHERE project_id = :project_id
            ORDER BY created_at DESC
            LIMIT :lim
            """
        ),
        {
            "project_id": project_id,
            "lim": lim,
            "preview_chars": _MAX_PREVIEW_CHARS,
            "rationale_chars": _MAX_RATIONALE_CHARS,
            "error_chars": _MAX_ERROR_CHARS,
        },
    )
    rows = result.fetchall()
    out: list[dict[str, Any]] = []
    for (id_, created_at, agent, kind, ticket_id, subtask_id, preview, rationale, error, checkpoint_id) in rows:
        p: dict[str, Any] = {}
        if ticket_id is not None:
            p["ticket_id"] = ticket_id
        if subtask_id is not None:
            p["subtask_id"] = subtask_id
        if preview:
            p["preview"] = preview
        if rationale:
            p["rationale"] = rationale
        if error:
            p["error"] = error
        out.append(
            {
                "id": id_,
                "created_at": created_at.isoformat() if created_at else None,
                "ts": created_at.timestamp() if created_at else None,
                "agent": agent,
                "kind": kind,
                "payload": p,
                "checkpoint_id": checkpoint_id,
            }
        )
    return out
