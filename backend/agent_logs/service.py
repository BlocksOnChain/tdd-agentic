"""Read historical agent logs from Postgres."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.ticket_system.models import AgentLog


async def list_agent_logs(
    db: AsyncSession,
    *,
    project_id: str,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Return newest-first rows; caller may reverse for chronological UI."""
    lim = max(1, min(limit, 10_000))
    stmt = (
        select(AgentLog)
        .where(AgentLog.project_id == project_id)
        .order_by(AgentLog.created_at.desc())
        .limit(lim)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    out: list[dict[str, Any]] = []
    for r in rows:
        p = dict(r.payload or {})
        cp = p.get("checkpoint_id")
        out.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "ts": r.created_at.timestamp() if r.created_at else None,
                "agent": r.agent,
                "kind": r.kind,
                "payload": p,
                "checkpoint_id": str(cp) if cp is not None else None,
            }
        )
    return out
