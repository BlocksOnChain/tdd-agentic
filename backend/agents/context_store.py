"""Transient context store for inter-agent communication.

Agents write compact outputs here; other agents can reference them
by ID. Reduces checkpoint size because full summaries don't need
to be in messages.

Usage:
    store = ContextStore()
    ctx_id = store.add("researcher", "research_findings", ("uuid1",), "Auth uses JWT + refresh tokens", output_key="workspace/research.md")
    # Another agent sees: {"context_refs": ("ctx_1",)} in the handoff
    # On lookup: store["ctx_1"].summary == "Auth uses JWT + refresh tokens"
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class ContextEntry:
    agent: str
    kind: str  # "research_findings", "lead_plan", "dev_summary", "qa_report"
    ticket_ids: tuple[str, ...]
    summary: str  # max 500 chars
    timestamp: float = field(default_factory=time.time)
    # Full output is stored separately; this is just the index.
    output_key: str = ""  # key into agent_logs or workspace


class ContextStore(dict):
    """In-memory store for agent context references.

    Each entry is a pointer, not a full summary. Agents that need
    the full context call rag_query or read from workspace.
    """

    MAX_ENTRIES = 20
    MAX_SUMMARY_CHARS = 500

    def add(self, agent: str, kind: str, ticket_ids: tuple[str, ...], summary: str, output_key: str = "") -> str:
        """Add a context entry, return its ID."""
        entry_id = f"ctx_{len(self) + 1}"
        self[entry_id] = ContextEntry(
            agent=agent,
            kind=kind,
            ticket_ids=ticket_ids,
            summary=summary[: self.MAX_SUMMARY_CHARS],
            output_key=output_key,
        )
        if len(self) > self.MAX_ENTRIES:
            oldest = next(iter(self))
            del self[oldest]
        return entry_id

    def lookup(self, ref_id: str) -> ContextEntry | None:
        """Look up a context entry by reference ID."""
        return self.get(ref_id)

    def for_ticket(self, ticket_id: str) -> list[ContextEntry]:
        """Return all entries referencing a specific ticket."""
        return [e for e in self.values() if ticket_id in e.ticket_ids]

    def for_agent(self, agent: str) -> list[ContextEntry]:
        """Return all entries from a specific agent."""
        return [e for e in self.values() if e.agent == agent]
