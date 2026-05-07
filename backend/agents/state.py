"""Root LangGraph state for the multi-agent system.

Uses Pydantic BaseModel with `extra="forbid"` per 2026 best practices, plus the
`add_messages` reducer for the conversation channel and explicit reducers for
list aggregation across parallel branches.
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
    events: Annotated[list[AgentEvent], operator.add] = Field(default_factory=list)


__all__ = ["SystemState", "AgentEvent"]
