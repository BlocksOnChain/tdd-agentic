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


class TestCaseInput(BaseModel):
    """A single RITE-format test case in a plan."""

    model_config = ConfigDict(extra="forbid")

    given: str
    should: str
    expected: str
    test_type: str = "unit"
    notes: str = ""


class SubtaskPlan(BaseModel):
    """A single subtask in an execution plan (Lead's output)."""

    title: str
    description: str = ""
    required_functionality: str = ""
    test_cases: list[TestCaseInput] | None = None
    assigned_to: str  # backend_dev, frontend_dev, devops, qa


class ExecutionPlan(BaseModel):
    """Execution plan produced by Lead agent, consumed by Coordinator."""

    ticket_id: str | None = None  # None if creating new ticket
    subtasks: list[SubtaskPlan] = Field(default_factory=list)


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

    # Execution plan produced by Lead, to be persisted by Coordinator
    execution_plan: ExecutionPlan | None = Field(default=None)


__all__ = ["SystemState", "AgentEvent", "ExecutionPlan", "SubtaskPlan", "TestCaseInput"]
