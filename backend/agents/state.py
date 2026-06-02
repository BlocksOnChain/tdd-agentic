"""Root LangGraph state for the multi-agent system.

Important: everything in this state is persisted into LangGraph checkpoints.
Keep it small and bounded; large ever-growing arrays will bloat checkpoints and
can make resume-from-checkpoint fail.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from backend.agents.message_reducer import add_messages_trimmed


class AgentEvent(BaseModel):
    """Lightweight event record for streaming to the monitoring client."""

    agent: str
    kind: str  # "log" | "tool_call" | "tool_result" | "interrupt" | "handoff"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float | None = None


class SystemState(BaseModel):
    """Root state object shared across the orchestration graph."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    messages: Annotated[list, add_messages_trimmed] = Field(default_factory=list)

    project_id: str | None = None
    project_context: str = ""

    active_ticket_id: str | None = None
    active_subtask_id: str | None = None

    # Legacy fields retained for older checkpoint compat — no longer written.
    # operator.add would accumulate unbounded; use identity lambda to drop new writes.
    pending_questions: Annotated[list[str], lambda _a, _b: _a or []] = Field(default_factory=list)
    human_responses: Annotated[list[str], lambda _a, _b: _a or []] = Field(default_factory=list)

    next_agent: str | None = None  # supervisor routing decision

    # Streamed via WS and stored in agent_logs; not duplicated in checkpoints.
    events: Annotated[list[AgentEvent], lambda _a, _b: []] = Field(default_factory=list)

    # Cross-agent context store (keyed "ctx_N" → ContextEntry).
    # Not persisted in checkpoints — cleared between graph runs.
    context_store: dict[str, Any] = Field(default_factory=dict)


__all__ = ["SystemState", "AgentEvent"]
