"""Shared utilities for agent subgraphs.

Provides helpers for invoking a tool-calling LLM, recording events, and
producing partial state updates that merge correctly with the root state's
reducers.
"""
from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt

from backend.agents.state import AgentEvent
from backend.api.events import Event, bus


async def emit(agent: str, kind: str, payload: dict[str, Any], project_id: str | None = None) -> AgentEvent:
    """Persist an event to the EventBus and return an AgentEvent for state."""
    await bus.publish(
        Event(type="agent", project_id=project_id, payload={"agent": agent, "kind": kind, **payload})
    )
    return AgentEvent(agent=agent, kind=kind, payload=payload, timestamp=time.time())


def _format_tool_error(exc: BaseException, tool_name: str | None) -> str:
    """Convert an exception (often a Pydantic ValidationError) into an
    actionable correction prompt for the LLM, so it knows what fields to
    fix on the retry.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        lines: list[str] = [
            f"TOOL_ERROR (validation): your call to '{tool_name}' had invalid arguments. "
            "Fix the issues below and call the tool again with ALL required fields.",
        ]
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "invalid")
            lines.append(f"  - {loc}: {msg}")
        return "\n".join(lines)
    return f"TOOL_ERROR: {exc!r}"


async def run_tool_calls(
    ai_message: AIMessage, tools_by_name: dict[str, BaseTool]
) -> list[ToolMessage]:
    """Execute tool calls in an AI message and return the resulting ToolMessages."""
    results: list[ToolMessage] = []
    for call in ai_message.tool_calls or []:
        name = call.get("name")
        args = call.get("args", {})
        call_id = call.get("id")
        tool = tools_by_name.get(name)
        if tool is None:
            content = f"TOOL_ERROR: tool '{name}' is not registered for this agent."
        else:
            try:
                content = await tool.ainvoke(args)
            except GraphInterrupt:
                # Must propagate so LangGraph can pause the graph and surface
                # HITL to the client. Never fold into ToolMessage JSON.
                raise
            except Exception as exc:  # noqa: BLE001
                content = _format_tool_error(exc, name)
        results.append(
            ToolMessage(content=str(content), name=name or "", tool_call_id=call_id or "")
        )
    return results


def last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None
