"""Root LangGraph state for the multi-agent system.

Important: everything in this state is persisted into LangGraph checkpoints.
Keep it small and bounded; large ever-growing arrays will bloat checkpoints and
can make resume-from-checkpoint fail.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class AgentEvent(BaseModel):
    """Lightweight event record for streaming to the monitoring client."""

    agent: str
    kind: str  # "log" | "tool_call" | "tool_result" | "interrupt" | "handoff"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float | None = None


class SystemState(BaseModel):
    """Root state object shared across the orchestration graph."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    messages: Annotated[list, add_messages] = Field(default_factory=list)

    project_id: str | None = None
    project_context: str = ""

    active_ticket_id: str | None = None
    active_subtask_id: str | None = None

    pending_questions: Annotated[list[str], operator.add] = Field(default_factory=list)
    human_responses: Annotated[list[str], operator.add] = Field(default_factory=list)

    next_agent: str | None = None  # supervisor routing decision

    # NOTE: events are already persisted in the DB (agent_logs) and streamed via WS.
    # We keep only a small tail in checkpoint state to avoid checkpoint bloat.
    events: Annotated[list[AgentEvent], lambda a, b: (list(a) + list(b))[-400:]] = Field(
        default_factory=list
    )


__all__ = ["SystemState", "AgentEvent"]
