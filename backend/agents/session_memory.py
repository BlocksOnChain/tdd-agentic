"""Structured session narrative merged into checkpoints alongside message trim.

See docs/deep-agents-integration-review.md — ``session_memory`` holds what aggressive
human-message trimming would otherwise lose: last route, capped handoff excerpts, etc.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_RECENT_HANDOFFS = 8
MAX_BLOCKERS = 12
MAX_HUMAN_DECISIONS = 8


class SessionMemory(BaseModel):
    """Rolling orchestration context stored in ``SystemState`` (checkpointed)."""

    model_config = ConfigDict(extra="forbid")

    goal: str = ""
    last_route: str | None = None
    active_ticket_id: str | None = None
    active_subtask_id: str | None = None
    recent_handoffs: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    human_decisions: list[str] = Field(default_factory=list)


def _cap_list(items: list[str], cap: int) -> list[str]:
    if len(items) <= cap:
        return items
    return items[-cap:]


def merge_session_memory(
    left: SessionMemory | None,
    right: SessionMemory | None,
) -> SessionMemory:
    """Merge specialist / PM updates into existing session memory.

    Scalars: non-empty ``goal`` and non-``None`` ids / ``last_route`` from ``right``
    overwrite. Lists: appended then capped.
    """
    base = left.model_copy(deep=True) if left is not None else SessionMemory()
    if right is None:
        return base
    r = right

    if r.goal:
        base.goal = r.goal
    if r.last_route is not None:
        base.last_route = r.last_route
    if r.active_ticket_id is not None:
        base.active_ticket_id = r.active_ticket_id
    if r.active_subtask_id is not None:
        base.active_subtask_id = r.active_subtask_id
    if r.recent_handoffs:
        base.recent_handoffs = _cap_list(
            [*base.recent_handoffs, *r.recent_handoffs],
            MAX_RECENT_HANDOFFS,
        )
    if r.blockers:
        base.blockers = _cap_list([*base.blockers, *r.blockers], MAX_BLOCKERS)
    if r.human_decisions:
        base.human_decisions = _cap_list(
            [*base.human_decisions, *r.human_decisions],
            MAX_HUMAN_DECISIONS,
        )
    return base


__all__ = [
    "MAX_RECENT_HANDOFFS",
    "SessionMemory",
    "merge_session_memory",
]
