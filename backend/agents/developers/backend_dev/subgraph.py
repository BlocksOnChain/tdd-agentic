"""Backend Dev subgraph — TDD red/green/refactor loop."""
from __future__ import annotations

from backend.agents.llm import backend_dev_model
from backend.agents.prompts import BACKEND_DEV_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.tools.code_tools import CODE_TOOLS
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import DEV_TICKET_TOOLS


def build_backend_dev_subgraph():
    return build_specialist_subgraph(
        name="backend_dev",
        role="backend_dev",
        llm_factory=backend_dev_model,
        tools=[*DEV_TICKET_TOOLS, *CODE_TOOLS, rag_query],
        base_system_prompt=BACKEND_DEV_SYSTEM,
        max_steps=20,
    )
