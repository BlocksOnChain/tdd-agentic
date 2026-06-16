"""Coordinator agent subgraph — handles all DB persistence operations.

The Coordinator is a simple, non-cognitive agent that:
1. Receives execution_plan from Lead (via state)
2. Persists it to the database using persistence tools
3. Returns control to PM with summary

This separation ensures:
- Lead focuses on planning (no tool call accuracy issues)
- Coordinator handles all DB writes (single responsibility)
- Fine-tuning is easier: Lead = Ticket → Plan, Coordinator = Plan → DB
"""
from __future__ import annotations

from backend.agents.llm import coordinator_model
from backend.agents.prompts import COORDINATOR_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.tools.rag_tools import rag_query
from backend.tools.persistence_tools import PERSISTENCE_TOOLS


def build_coordinator_subgraph():
    return build_specialist_subgraph(
        name="coordinator",
        role="coordinator",
        llm_factory=coordinator_model,
        tools=[*PERSISTENCE_TOOLS, rag_query],
        base_system_prompt=COORDINATOR_SYSTEM,
        max_steps=8,
    )
