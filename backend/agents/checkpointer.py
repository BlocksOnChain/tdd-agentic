"""LangGraph checkpointer setup using AsyncPostgresSaver.

We expose a context-manager helper that yields an initialized saver. The
FastAPI app holds a single connection pool for the lifetime of the process
and reuses it across graph invocations.

A custom ``JsonPlusSerializer`` is configured with our application's
state types whitelisted under ``allowed_msgpack_modules`` so LangGraph
doesn't emit deprecation warnings on every checkpoint round-trip and so
strict-mode (``LANGGRAPH_STRICT_MSGPACK=true``) keeps working.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg_pool import AsyncConnectionPool

from backend.config import get_settings

# (module, qualname) tuples for every class our graph state contains that
# needs to round-trip through the checkpoint serializer. Add new entries
# here whenever you introduce a new Pydantic state model.
_ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    ("backend.agents.state", "SystemState"),
    ("backend.agents.state", "AgentEvent"),
    ("backend.agents.session_memory", "SessionMemory"),
]


def _build_serde() -> JsonPlusSerializer:
    """Return a serializer that knows about our application state types.

    ``allowed_msgpack_modules`` whitelists application classes for
    deserialization through the msgpack code path, suppressing
    LangGraph's pending-deprecation warning on every load.
    """
    return JsonPlusSerializer(
        allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES,
    )


_pool: AsyncConnectionPool | None = None


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = AsyncConnectionPool(
            conninfo=settings.checkpointer_url,
            max_size=20,
            min_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await _pool.open()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def delete_thread_checkpoints(thread_id: str) -> None:
    """Remove LangGraph checkpoint rows for a project thread."""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (thread_id,),
            )
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (thread_id,),
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (thread_id,),
            )


@asynccontextmanager
async def get_checkpointer():
    """Async context manager that yields an initialized AsyncPostgresSaver.

    On first use this also runs ``setup()`` which creates the checkpoint
    tables if missing — safe to call repeatedly.
    """
    pool = await get_pool()
    saver = AsyncPostgresSaver(pool, serde=_build_serde())
    await saver.setup()
    yield saver
