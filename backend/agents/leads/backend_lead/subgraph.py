"""Backend Lead subgraph — produces ordered subtasks with explicit test cases."""
from __future__ import annotations

from backend.agents.llm import lead_model
from backend.agents.prompts import BACKEND_LEAD_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import LEAD_TICKET_TOOLS


def build_backend_lead_subgraph():
    return build_specialist_subgraph(
        name="backend_lead",
        role="backend_lead",
        llm_factory=lead_model,
        tools=[*LEAD_TICKET_TOOLS, rag_query],
        base_system_prompt=BACKEND_LEAD_SYSTEM,
        max_steps=12,
    )
