"""DevOps subgraph — Docker, CI configs, deployment scripts."""
from __future__ import annotations

from backend.agents.llm import devops_model
from backend.agents.prompts import DEVOPS_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.config import get_settings
from backend.tools.code_tools import CODE_TOOLS
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import DEV_TICKET_TOOLS


def build_devops_subgraph():
    return build_specialist_subgraph(
        name="devops",
        role="devops",
        llm_factory=devops_model,
        tools=DEV_TICKET_TOOLS,
        code_tools=[*CODE_TOOLS, rag_query],
        phased_code_tools=True,
        base_system_prompt=DEVOPS_SYSTEM,
        max_steps=get_settings().dev_agent_max_steps,
        verify_completion=True,
        require_subtask_resolution=True,
    )
