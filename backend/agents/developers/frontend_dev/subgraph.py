"""Frontend Dev subgraph — TDD red/green/refactor loop for UI work."""
from __future__ import annotations

from backend.agents.llm import frontend_dev_model
from backend.agents.prompts import FRONTEND_DEV_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.tools.code_tools import CODE_TOOLS
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import DEV_TICKET_TOOLS


def build_frontend_dev_subgraph():
    return build_specialist_subgraph(
        name="frontend_dev",
        role="frontend_dev",
        llm_factory=frontend_dev_model,
        tools=[*DEV_TICKET_TOOLS, *CODE_TOOLS, rag_query],
        base_system_prompt=FRONTEND_DEV_SYSTEM,
        max_steps=20,
    )
