"""DevOps subgraph — Docker, CI configs, deployment scripts."""
from __future__ import annotations

from backend.agents.llm import dev_model
from backend.agents.prompts import DEVOPS_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.tools.code_tools import CODE_TOOLS
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import DEV_TICKET_TOOLS


def build_devops_subgraph():
    return build_specialist_subgraph(
        name="devops",
        role="devops",
        llm_factory=dev_model,
        tools=[*DEV_TICKET_TOOLS, *CODE_TOOLS, rag_query],
        base_system_prompt=DEVOPS_SYSTEM,
        max_steps=18,
    )
