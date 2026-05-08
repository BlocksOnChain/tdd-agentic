"""In-memory TTL + singleflight for checkpoint list API.

Repeated GET /agents/checkpoints would each compile LangGraph + walk history —
under load stacked polls multiply CPU/RAM until the backend recovers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from backend.agents.checkpointer import get_checkpointer
from backend.agents.graph import build_root_graph
from backend.config import get_settings

logger = logging.getLogger(__name__)

_cache: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
_locks: dict[str, asyncio.Lock] = {}


def _lock(pid: str) -> asyncio.Lock:
    return _locks.setdefault(pid, asyncio.Lock())


def invalidate_checkpoint_list_cache(project_id: str | None = None) -> None:
    """Drop cached checkpoint lists after agent actions or globally."""
    global _cache
    if project_id is None:
        _cache.clear()
        return
    dead = [k for k in _cache if k[0] == project_id]
    for k in dead:
        del _cache[k]


async def _fetch_checkpoints(project_id: str, limit: int) -> dict[str, Any]:
    config = {"configurable": {"thread_id": project_id}}
    out: list[dict[str, Any]] = []
    async with get_checkpointer() as saver:
        graph = build_root_graph(checkpointer=saver)
        async for snap in graph.aget_state_history(config):
            cfg_conf = (snap.config or {}).get("configurable") or {}
            meta = snap.metadata or {}
            writes = meta.get("writes") or {}
            wrote_nodes = list(writes.keys()) if isinstance(writes, dict) else []
            out.append(
                {
                    "checkpoint_id": cfg_conf.get("checkpoint_id"),
                    "parent_checkpoint_id": (
                        (snap.parent_config or {}).get("configurable", {}).get("checkpoint_id")
                        if snap.parent_config
                        else None
                    ),
                    "created_at": getattr(snap, "created_at", None),
                    "source": meta.get("source"),
                    "step": meta.get("step"),
                    "wrote_nodes": wrote_nodes,
                    "next": list(snap.next or []),
                }
            )
            if len(out) >= limit:
                break
    return {"checkpoints": out}


async def get_checkpoints_list(
    project_id: str,
    *,
    limit: int,
    force_refresh: bool,
) -> dict[str, Any]:
    settings = get_settings()
    ttl = max(0.1, settings.checkpoints_cache_ttl_seconds)
    key = (project_id, limit)
    now = time.monotonic()

    if not force_refresh:
        hit = _cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]

    async with _lock(project_id):
        now = time.monotonic()
        if not force_refresh:
            hit = _cache.get(key)
            if hit is not None and hit[0] > now:
                return hit[1]

        logger.debug(
            "checkpoint list fetch project_id=%s limit=%s force=%s",
            project_id[:8],
            limit,
            force_refresh,
        )
        data = await _fetch_checkpoints(project_id, limit)
        _cache[key] = (time.monotonic() + ttl, data)
        return data
