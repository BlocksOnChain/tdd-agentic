"""WebSocket endpoint that streams every EventBus event to connected clients."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.events import bus

router = APIRouter()


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        # Send hello so the client knows the channel is live
        await websocket.send_text(json.dumps({"type": "hello", "payload": {"ok": True}}))
        async for event in bus.subscribe():
            try:
                await websocket.send_text(json.dumps(event.to_json()))
            except (WebSocketDisconnect, asyncio.CancelledError):
                break
    except WebSocketDisconnect:
        return
