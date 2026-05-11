"""Routes for starting, resuming, and inspecting LangGraph agent runs."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import NoResultFound
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from backend.agents.checkpointer import delete_thread_checkpoints, get_checkpointer
from backend.agents.graph import build_root_graph
from backend.agents.observability import callbacks_for
from backend.agents.state import SystemState
from backend.api.checkpoint_list_cache import (
    get_checkpoints_list,
    invalidate_checkpoint_list_cache,
)
from backend.api.events import Event, bus
from backend.agent_logs.service import list_agent_logs
from backend.db.session import AsyncSessionLocal
from backend.ticket_system import service as ticket_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Per-project background graph tasks so we can cancel a run on demand.
RUNNING_TASKS: dict[str, asyncio.Task] = {}

# Hard ceiling for how long we wait for a single task to honour a cancel
# request before giving up and letting it leak. LangGraph + httpx normally
# respond within a few seconds; if it takes longer than this, something is
# wedged and continuing to wait helps no one.
_STOP_GRACE_SECONDS = 10.0


async def _cancel_task(project_id: str, task: asyncio.Task, timeout: float = _STOP_GRACE_SECONDS) -> bool:
    """Cancel a single task and wait for it to settle. Returns True if it stopped in time."""
    if task.done():
        return True
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(_swallow(task)), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "agent task for project %s did not cancel within %.1fs; abandoning",
            project_id,
            timeout,
        )
        return False
    return True


async def _swallow(task: asyncio.Task) -> None:
    """Await a task, suppressing CancelledError + any final exception."""
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def cancel_all_running_tasks(timeout: float = _STOP_GRACE_SECONDS) -> None:
    """Cancel every in-flight agent run. Used by lifespan shutdown so the
    process can exit/reload cleanly instead of blocking uvicorn's drain.
    """
    if not RUNNING_TASKS:
        return
    pending = list(RUNNING_TASKS.items())
    logger.info("cancelling %d in-flight agent task(s) on shutdown", len(pending))
    await asyncio.gather(
        *(_cancel_task(pid, t, timeout) for pid, t in pending),
        return_exceptions=True,
    )
    RUNNING_TASKS.clear()


async def _spawn(project_id: str, coro) -> None:
    """Schedule a graph run, cancelling any previous one for the same project."""
    prior = RUNNING_TASKS.get(project_id)
    if prior is not None and not prior.done():
        await _cancel_task(project_id, prior)

    async def _wrapper() -> None:
        try:
            await coro
        except asyncio.CancelledError:
            await bus.publish(
                Event(
                    type="agent",
                    project_id=project_id,
                    payload={"kind": "stopped", "node": "system"},
                )
            )
            raise
        finally:
            if RUNNING_TASKS.get(project_id) is task:
                RUNNING_TASKS.pop(project_id, None)

    task = asyncio.create_task(_wrapper(), name=f"agent-run:{project_id}")
    RUNNING_TASKS[project_id] = task

# ===== Transient error retry policy =====
# Errors whose string representation matches one of these substrings are
# treated as transient: we'll wait and resume the graph from the last
# checkpoint instead of giving up.
TRANSIENT_ERROR_MARKERS = (
    "429",
    "rate_limit",
    "rate limit",
    "overloaded",
    "503",
    "502",
    "504",
    "timeout",
    "timed out",
    "connection reset",
    "connection error",
)
MAX_AUTO_RETRIES = 3
INITIAL_RETRY_DELAY_S = 5.0


def _is_transient(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(marker in s for marker in TRANSIENT_ERROR_MARKERS)


class StartRunRequest(BaseModel):
    project_id: str
    goal: str
    project_context: str = ""
    fresh: bool = False


class ResumeRequest(BaseModel):
    project_id: str
    response: str


def _checkpoint_id(snapshot: Any) -> str | None:
    if snapshot is None or snapshot.config is None:
        return None
    cfg = snapshot.config.get("configurable") or {}
    return cfg.get("checkpoint_id")


async def _stream_once(graph, project_id: str, payload: Any, config: dict) -> None:
    """Drive the graph until it yields or completes; raises on error.

    After each update we tag the event with the *current* checkpoint_id so
    the UI can offer a "Resume from here" action per log entry.
    """
    last_checkpoint_id: str | None = None
    async for chunk in graph.astream(payload, config=config, stream_mode="updates"):
        snapshot = await graph.aget_state(config)
        checkpoint_id = _checkpoint_id(snapshot)
        if checkpoint_id and checkpoint_id != last_checkpoint_id:
            invalidate_checkpoint_list_cache(project_id)
            last_checkpoint_id = checkpoint_id
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                await _broadcast_interrupts(project_id, update)
                continue
            await bus.publish(
                Event(
                    type="agent",
                    project_id=project_id,
                    payload={
                        "node": node_name,
                        "update": _safe(update),
                        "checkpoint_id": checkpoint_id,
                    },
                )
            )
    snapshot = await graph.aget_state(config)
    if snapshot is not None:
        checkpoint_id = _checkpoint_id(snapshot)
        if checkpoint_id and checkpoint_id != last_checkpoint_id:
            invalidate_checkpoint_list_cache(project_id)
        for task in snapshot.tasks:
            for itp in getattr(task, "interrupts", []) or []:
                await _broadcast_interrupts(project_id, itp)


async def _run_graph(project_id: str, initial_payload: Any) -> None:
    """Background task that runs the graph with auto-retry on transient errors.

    ``initial_payload`` is either a ``SystemState`` (fresh start), a
    ``Command`` (HITL resume), or ``None`` (resume from last checkpoint).
    """
    config = {
        "configurable": {"thread_id": project_id},
        "recursion_limit": 250,
        "callbacks": callbacks_for(project_id),
    }
    async with get_checkpointer() as saver:
        graph = build_root_graph(checkpointer=saver)
        payload: Any = initial_payload
        delay = INITIAL_RETRY_DELAY_S

        for attempt in range(MAX_AUTO_RETRIES + 1):
            try:
                await _stream_once(graph, project_id, payload, config)
                return
            except Exception as exc:
                transient = _is_transient(exc)
                will_retry = transient and attempt < MAX_AUTO_RETRIES
                await bus.publish(
                    Event(
                        type="agent",
                        project_id=project_id,
                        payload={
                            "error": str(exc),
                            "kind": "transient_error" if will_retry else "crash",
                            "attempt": attempt + 1,
                            "will_retry_in_seconds": delay if will_retry else None,
                        },
                    )
                )
                if not will_retry:
                    return
                await asyncio.sleep(delay)
                delay *= 2
                # After the first failure, resume from the checkpoint with None
                payload = None


async def _broadcast_interrupts(project_id: str, update: Any) -> None:
    """Translate LangGraph interrupt payloads into ``interrupt`` events."""
    items = update if isinstance(update, list) else [update]
    for item in items:
        value = item
        if hasattr(item, "value"):
            value = item.value
        if isinstance(value, dict):
            await bus.publish(
                Event(
                    type="interrupt",
                    project_id=project_id,
                    payload={
                        "kind": value.get("kind", "ask_human"),
                        "question": value.get("question", str(value)),
                        "ticket_id": value.get("ticket_id"),
                        "asked_by": value.get("asked_by", "agent"),
                    },
                )
            )
        else:
            await bus.publish(
                Event(
                    type="interrupt",
                    project_id=project_id,
                    payload={"kind": "ask_human", "question": str(value)},
                )
            )


async def _resume_graph(project_id: str, response: str) -> None:
    """Resume a paused graph after the human answered an interrupt()."""
    await _run_graph(project_id, Command(resume=response))


async def _retry_graph(project_id: str) -> None:
    """Re-invoke the graph from the last checkpoint without any new input."""
    await _run_graph(project_id, None)


async def _resume_from_checkpoint(project_id: str, checkpoint_id: str) -> None:
    """Branch off and resume the graph from a specific historical checkpoint."""
    config = {
        "configurable": {"thread_id": project_id, "checkpoint_id": checkpoint_id},
        "recursion_limit": 250,
        "callbacks": callbacks_for(project_id),
    }
    delay = INITIAL_RETRY_DELAY_S
    async with get_checkpointer() as saver:
        graph = build_root_graph(checkpointer=saver)
        for attempt in range(MAX_AUTO_RETRIES + 1):
            try:
                await _stream_once(graph, project_id, None, config)
                return
            except Exception as exc:
                transient = _is_transient(exc)
                will_retry = transient and attempt < MAX_AUTO_RETRIES
                await bus.publish(
                    Event(
                        type="agent",
                        project_id=project_id,
                        payload={
                            "error": str(exc),
                            "kind": "transient_error" if will_retry else "crash",
                            "attempt": attempt + 1,
                            "will_retry_in_seconds": delay if will_retry else None,
                        },
                    )
                )
                if not will_retry:
                    return
                await asyncio.sleep(delay)
                delay *= 2


def _safe(value: Any) -> Any:
    """Best-effort JSON-safe rendering for stream updates."""
    try:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {k: _safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
    except Exception:
        return repr(value)


@router.get("/logs/{project_id}")
async def get_agent_logs(project_id: str, limit: int = 5000) -> dict[str, Any]:
    """Return persisted agent-bus events for a project (oldest first).

    Mirrors what the Logs panel shows: every ``agent`` typed EventBus publish.
    """
    async with AsyncSessionLocal() as db:
        rows = await list_agent_logs(db, project_id=project_id, limit=limit)
    rows.reverse()
    return {"logs": rows}


@router.get("/logs/item/{log_id}")
async def get_agent_log_item(log_id: str) -> dict[str, Any]:
    """Return a single persisted agent log row + optional checkpoint metadata.

    This is used by the frontend "click log line" popup to fetch full details
    on demand without loading large payloads for the whole log list.
    """
    async with AsyncSessionLocal() as db:
        row = (
            await db.execute(
                text(
                    """
                    SELECT
                      id,
                      project_id,
                      created_at,
                      agent,
                      kind,
                      payload,
                      (payload->>'checkpoint_id') AS checkpoint_id
                    FROM agent_logs
                    WHERE id = :id
                    LIMIT 1
                    """
                ),
                {"id": log_id},
            )
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Log not found")
        (id_, project_id, created_at, agent, kind, payload, checkpoint_id) = row

        checkpoint: dict[str, Any] | None = None
        if project_id and checkpoint_id:
            # Lightweight checkpoint metadata lookup. Avoid loading giant checkpoints.
            ck = (
                await db.execute(
                    text(
                        """
                        SELECT
                          checkpoint->>'id' AS checkpoint_id,
                          checkpoint->>'ts' AS ts,
                          pg_column_size(checkpoint) AS checkpoint_bytes,
                          metadata
                        FROM checkpoints
                        WHERE thread_id = :thread_id
                          AND checkpoint_ns = ''
                          AND checkpoint->>'id' = :checkpoint_id
                        LIMIT 1
                        """
                    ),
                    {"thread_id": project_id, "checkpoint_id": checkpoint_id},
                )
            ).fetchone()
            if ck is not None:
                (cid, ts, cbytes, meta) = ck
                checkpoint = {
                    "checkpoint_id": cid,
                    "created_at": ts,
                    "bytes": int(cbytes) if cbytes is not None else None,
                    "metadata": meta or {},
                }

        return {
            "log": {
                "id": id_,
                "project_id": project_id,
                "created_at": created_at.isoformat() if created_at else None,
                "ts": created_at.timestamp() if created_at else None,
                "agent": agent,
                "kind": kind,
                "payload": payload or {},
                "checkpoint_id": checkpoint_id,
            },
            "checkpoint": checkpoint,
        }


async def _ensure_project_exists(project_id: str) -> None:
    async with AsyncSessionLocal() as db:
        try:
            await ticket_service.get_project(db, project_id)
        except NoResultFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/start")
async def start_run(payload: StartRunRequest):
    await _ensure_project_exists(payload.project_id)
    invalidate_checkpoint_list_cache(payload.project_id)
    if payload.fresh:
        await delete_thread_checkpoints(payload.project_id)
    initial = SystemState(
        project_id=payload.project_id,
        project_context=payload.project_context or payload.goal,
        messages=[HumanMessage(content=payload.goal)],
    )
    await _spawn(payload.project_id, _run_graph(payload.project_id, initial))
    return {"status": "started", "project_id": payload.project_id, "fresh": payload.fresh}


@router.post("/resume")
async def resume_run(payload: ResumeRequest):
    await _ensure_project_exists(payload.project_id)
    invalidate_checkpoint_list_cache(payload.project_id)
    await _spawn(payload.project_id, _resume_graph(payload.project_id, payload.response))
    return {"status": "resumed", "project_id": payload.project_id}


class RetryRequest(BaseModel):
    project_id: str


@router.post("/retry")
async def retry_run(payload: RetryRequest):
    """Resume a project's graph from its last persisted checkpoint."""
    await _ensure_project_exists(payload.project_id)
    invalidate_checkpoint_list_cache(payload.project_id)
    await _spawn(payload.project_id, _retry_graph(payload.project_id))
    return {"status": "retrying", "project_id": payload.project_id}


class StopRequest(BaseModel):
    project_id: str


@router.post("/stop")
async def stop_run(payload: StopRequest):
    """Cancel any in-flight graph execution for the given project.

    Waits up to ~10s for the task to honour the cancel so the response
    accurately reflects whether the run actually stopped.
    """
    invalidate_checkpoint_list_cache(payload.project_id)
    task = RUNNING_TASKS.get(payload.project_id)
    if task is None or task.done():
        return {"status": "not_running", "project_id": payload.project_id}
    stopped = await _cancel_task(payload.project_id, task)
    return {
        "status": "stopped" if stopped else "stopping",
        "project_id": payload.project_id,
        "settled": stopped,
    }


class ResumeFromRequest(BaseModel):
    project_id: str
    checkpoint_id: str


@router.post("/resume_from")
async def resume_from(payload: ResumeFromRequest):
    """Resume the graph from a specific historical checkpoint."""
    await _ensure_project_exists(payload.project_id)
    invalidate_checkpoint_list_cache(payload.project_id)
    await _spawn(
        payload.project_id,
        _resume_from_checkpoint(payload.project_id, payload.checkpoint_id),
    )
    return {
        "status": "resuming",
        "project_id": payload.project_id,
        "checkpoint_id": payload.checkpoint_id,
    }


@router.get("/checkpoints/{project_id}")
async def list_checkpoints(
    project_id: str,
    limit: int = 100,
    force: bool = False,
) -> dict:
    """Return the project's checkpoint history (latest first).

    Responses are TTL-cached server-side unless ``force=1`` to avoid piled-up
    LangGraph compilations while the UI polls.

    Each item is a flattened ``StateSnapshot`` with the ``checkpoint_id``,
    creation timestamp, the writes that produced it, and the next nodes
    queued to run from that point.
    """
    return await get_checkpoints_list(project_id, limit=limit, force_refresh=force)


@router.get("/state/{project_id}")
async def get_state(project_id: str) -> dict:
    config = {"configurable": {"thread_id": project_id}}
    async with get_checkpointer() as saver:
        graph = build_root_graph(checkpointer=saver)
        snapshot = await graph.aget_state(config)
        if snapshot is None or snapshot.values is None:
            raise HTTPException(status_code=404, detail="No state for project")
        return {
            "values": _safe(snapshot.values),
            "next": list(snapshot.next),
            "tasks": [
                {"name": t.name, "interrupts": _safe(getattr(t, "interrupts", []))}
                for t in snapshot.tasks
            ],
        }


@router.get("/interrupts/{project_id}")
async def list_interrupts(project_id: str) -> dict:
    """Return any pending interrupts for the given project's thread."""
    config = {"configurable": {"thread_id": project_id}}
    async with get_checkpointer() as saver:
        graph = build_root_graph(checkpointer=saver)
        snapshot = await graph.aget_state(config)
        if snapshot is None:
            return {"interrupts": []}
        all_interrupts: list[Any] = []
        for task in snapshot.tasks:
            for itp in getattr(task, "interrupts", []) or []:
                all_interrupts.append(_safe(itp))
        return {"interrupts": all_interrupts}
