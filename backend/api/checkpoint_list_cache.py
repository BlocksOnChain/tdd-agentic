"""In-memory TTL + singleflight for checkpoint list API.

Repeated GET /agents/checkpoints would each compile LangGraph + walk history —
under load stacked polls multiply CPU/RAM until the backend recovers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from backend.agents.checkpointer import get_checkpointer, get_pool
from backend.agents.graph import build_root_graph
from backend.config import get_settings

logger = logging.getLogger(__name__)

_cache: dict[tuple[str, int], tuple[float, dict[str, Any], float]] = {}
_locks: dict[str, asyncio.Lock] = {}

# Cache safety limits (best-effort; prevents unbounded growth under polling).
# We keep these small because the payload is "UI summary" only.
_MAX_CACHE_ITEMS = 128
_MAX_CACHE_BYTES = 2_000_000  # ~2MB


def _lock(pid: str) -> asyncio.Lock:
    return _locks.setdefault(pid, asyncio.Lock())

def _estimate_bytes(value: Any) -> int:
    """Approximate in-memory size of a JSON-like structure.

    We avoid json.dumps here to keep this cheap and to not allocate huge strings.
    """
    try:
        if value is None:
            return 4
        if isinstance(value, (bool, int, float)):
            return 16
        if isinstance(value, str):
            return len(value) * 2
        if isinstance(value, dict):
            return 64 + sum(_estimate_bytes(k) + _estimate_bytes(v) for k, v in value.items())
        if isinstance(value, list):
            return 64 + sum(_estimate_bytes(v) for v in value)
        return 128
    except Exception:
        return 256


def _cache_bytes_total() -> int:
    return int(sum(v[2] for v in _cache.values()))


def _evict_cache_if_needed() -> None:
    """Evict expired entries first, then oldest-expiring until within bounds."""
    now = time.monotonic()
    expired = [k for k, (exp, _, __) in _cache.items() if exp <= now]
    for k in expired:
        _cache.pop(k, None)

    # If still too big, evict by soonest expiration (oldest-ish usage).
    while _cache and (len(_cache) > _MAX_CACHE_ITEMS or _cache_bytes_total() > _MAX_CACHE_BYTES):
        oldest_key = min(_cache.items(), key=lambda kv: kv[1][0])[0]
        _cache.pop(oldest_key, None)


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
    out: list[dict[str, Any]] = []
    try:
        # Fast path: query checkpoint metadata directly from Postgres without
        # loading the full checkpoint JSONB/channel blobs or compiling LangGraph.
        pool = await get_pool()
        async with pool.connection() as conn:  # type: ignore[attr-defined]
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                      checkpoint->>'id' AS checkpoint_id,
                      checkpoint->>'ts' AS ts,
                      metadata
                    FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = ''
                    ORDER BY (checkpoint->>'ts') DESC NULLS LAST
                    LIMIT %s
                    """,
                    (project_id, limit),
                    binary=True,
                )
                rows = await cur.fetchall()
        for checkpoint_id, ts, meta in rows:
            meta = meta or {}
            writes = meta.get("writes") or {}
            wrote_nodes = list(writes.keys()) if isinstance(writes, dict) else []
            out.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": None,
                    # Frontend expects a value new Date(...) can parse; ISO-8601 is fine.
                    "created_at": ts,
                    "source": meta.get("source"),
                    "step": meta.get("step"),
                    "wrote_nodes": wrote_nodes,
                    "next": [],
                }
            )
    except Exception as exc:
        logger.warning(
            "checkpoint list fast-path FAILED project_id=%s limit=%s: %s",
            project_id[:8],
            limit,
            str(exc),
        )

    if out:
        return {"checkpoints": out}

    # Slow fallback: compile LangGraph and walk history (may be expensive / may fail).
    config = {"configurable": {"thread_id": project_id}}
    try:
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
                            (snap.parent_config or {})
                            .get("configurable", {})
                            .get("checkpoint_id")
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
    except Exception as exc:
        # Returning an empty list is safer than crashing the API (frontend will keep working).
        logger.exception(
            "checkpoint list fetch FAILED project_id=%s limit=%s: %s",
            project_id[:8],
            limit,
            str(exc),
        )
        return {"checkpoints": [], "error": "checkpoint_history_unavailable"}


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
        _evict_cache_if_needed()
        hit = _cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]

    async with _lock(project_id):
        now = time.monotonic()
        if not force_refresh:
            _evict_cache_if_needed()
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
        # Even when fetch fails, cache briefly to avoid hammering the DB
        # on every UI poll.
        effective_ttl = min(ttl, 5.0) if data.get("error") else ttl
        approx = float(_estimate_bytes(data))
        # Don't cache unexpectedly huge payloads.
        if approx <= (_MAX_CACHE_BYTES / 2):
            _cache[key] = (time.monotonic() + effective_ttl, data, approx)
        _evict_cache_if_needed()
        return data
