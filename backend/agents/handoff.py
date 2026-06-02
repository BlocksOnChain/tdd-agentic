"""Structured inter-agent handoff protocol.

Replaces the natural-language handoff strings with a compact,
machine-parseable format that contains only what the next agent
needs to look up.

Encoding: short JSON keys to minimise per-handoff token cost.

  t  → target agent name
  p  → phase (research|backend_planning|frontend_planning|infrastructure|implement|review|qa)
  tik → ticket UUIDs
  sid → subtask UUIDs
  c  → context_refs (pointer IDs into ContextStore)
  f  → flags (e.g. requires_research, has_unanswered_questions)
  i  → intent (one-line summary, max 200 chars)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    RESEARCH = "research"
    BACKEND_PLANNING = "backend_planning"
    FRONTEND_PLANNING = "frontend_planning"
    INFRASTRUCTURE = "infrastructure"
    IMPLEMENT = "implement"
    REVIEW = "review"
    QA = "qa"

    @classmethod
    def coerce(cls, value: "Phase | str | None", default: "Phase | None" = None) -> "Phase":
        """Best-effort map an arbitrary phase string to a valid Phase.

        The PM model frequently invents near-miss phase names (e.g.
        ``infra_scaffolding``, ``devops_scaffolding``) that don't match the
        enum verbatim. Crashing the whole orchestration run on an unknown
        phase string is the wrong failure mode, so we normalise via an alias
        table and fall back to ``default`` (IMPLEMENT) for anything unknown.
        """
        if default is None:
            default = cls.IMPLEMENT
        if isinstance(value, cls):
            return value
        if not value:
            return default
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return cls(key)
        except ValueError:
            return _PHASE_ALIASES.get(key, default)


# Synonyms / near-misses the PM model emits that should map onto a canonical
# Phase rather than crash the routing handoff.
_PHASE_ALIASES: dict[str, "Phase"] = {
    "infra": Phase.INFRASTRUCTURE,
    "infra_scaffolding": Phase.INFRASTRUCTURE,
    "infrastructure_scaffolding": Phase.INFRASTRUCTURE,
    "devops_scaffolding": Phase.INFRASTRUCTURE,
    "devops": Phase.INFRASTRUCTURE,
    "scaffold": Phase.INFRASTRUCTURE,
    "scaffolding": Phase.INFRASTRUCTURE,
    "setup": Phase.INFRASTRUCTURE,
    "backend": Phase.BACKEND_PLANNING,
    "backend_plan": Phase.BACKEND_PLANNING,
    "frontend": Phase.FRONTEND_PLANNING,
    "frontend_plan": Phase.FRONTEND_PLANNING,
    "planning": Phase.BACKEND_PLANNING,
    "implementation": Phase.IMPLEMENT,
    "implementing": Phase.IMPLEMENT,
    "testing": Phase.QA,
    "test": Phase.QA,
}


@dataclass(frozen=True)
class Handoff:
    """Compact structured handoff between agents."""

    target: str
    phase: Phase
    ticket_ids: tuple[str, ...] = ()
    subtask_ids: tuple[str, ...] = ()
    # One-line intent replacing verbose instructions.
    # e.g. "Plan backend subtasks for auth ticket."
    intent: str = ""
    # Cross-agent context: references to prior agent outputs.
    # Points to agent_logs entries, not full text.
    context_refs: tuple[str, ...] = ()
    # Flags for the receiving agent.
    flags: tuple[str, ...] = ()  # e.g. ("requires_research", "has_unanswered_questions")

    def to_message(self) -> str:
        """Serialize to a compact message format for the agent."""
        meta: dict = {
            "t": self.target,
            "p": self.phase.value,
        }
        if self.ticket_ids:
            meta["tik"] = self.ticket_ids
        if self.subtask_ids:
            meta["sid"] = self.subtask_ids
        if self.context_refs:
            meta["c"] = self.context_refs
        if self.flags:
            meta["f"] = self.flags
        header = json.dumps(meta, separators=(",", ":"))
        if self.intent:
            return f"[from project_manager → {self.target}]\n{header}\n{self.intent}"
        return f"[from project_manager → {self.target}]\n{header}"

    @classmethod
    def from_routing_decision(cls, next_agent: str, phase: str, instructions: str, ticket_ids: list[str]) -> "Handoff":
        """Create a Handoff from a RoutingDecision."""
        return cls(
            target=next_agent,
            phase=Phase.coerce(phase),
            ticket_ids=tuple(ticket_ids),
            intent=instructions[:200],  # Cap intent at 200 chars
        )


@dataclass(frozen=True)
class HandoffV2(Handoff):
    """Extended handoff that carries a small ContextStore snapshot.

    Instead of full summaries, the PM drops the ContextStore into state;
    this class just carries the reference keys.
    """

    # Override to_message to also include context refs in the intent line.
    def to_message(self) -> str:
        base = super().to_message()
        if self.context_refs:
            return f"{base}\nContext refs: {', '.join('ctx_{i}' for i in self.context_refs)}"
        return base
