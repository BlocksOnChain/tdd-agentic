"""WebSocket endpoint that streams every EventBus event to connected clients."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.events import bus

router = APIRouter()


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, project_id: str | None = None) -> None:
    """Stream events. Pass ``?project_id=...`` to receive only that project's.

    Without the filter a client watching one project receives every concurrent
    run's traffic and has to discard it client-side.
    """
    await websocket.accept()
    try:
        # Send hello so the client knows the channel is live
        await websocket.send_text(
            json.dumps({"type": "hello", "payload": {"ok": True, "project_id": project_id}})
        )
        async for event in bus.subscribe(project_id=project_id):
            try:
                await websocket.send_text(json.dumps(event.to_json()))
            except (WebSocketDisconnect, asyncio.CancelledError):
                break
    except WebSocketDisconnect:
        return
