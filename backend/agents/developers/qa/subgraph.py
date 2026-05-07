"""QA subgraph — integration / e2e test authoring."""
from __future__ import annotations

from backend.agents.llm import dev_model
from backend.agents.prompts import QA_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.tools.code_tools import CODE_TOOLS
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import DEV_TICKET_TOOLS


def build_qa_subgraph():
    return build_specialist_subgraph(
        name="qa",
        role="qa",
        llm_factory=dev_model,
        tools=[*DEV_TICKET_TOOLS, *CODE_TOOLS, rag_query],
        base_system_prompt=QA_SYSTEM,
        max_steps=18,
    )
