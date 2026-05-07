"""Researcher subgraph — web search, doc writing, RAG ingestion, skill creation."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from backend.agents.llm import researcher_model
from backend.agents.prompts import RESEARCHER_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.agents.skills.registry import upsert_skill
from backend.tools.code_tools import fs_list, fs_read, fs_write
from backend.tools.rag_tools import rag_ingest_text, rag_query
from backend.tools.web_search_tools import web_search


@tool
async def create_skill(
    name: str,
    description: str,
    content: str,
    roles: list[str],
    project_id: str,
) -> str:
    """Create or update a Skill (a focused capability brief) and assign it to roles.

    Roles must be one or more of: project_manager, researcher, backend_lead,
    frontend_lead, backend_dev, frontend_dev, devops, qa.
    """
    info = upsert_skill(
        name=name, description=description, content=content, roles=roles, project_id=project_id
    )
    # Also index the skill into the project's RAG so it's retrievable later
    await rag_ingest_text.ainvoke(
        {
            "project_id": project_id,
            "source": f"skill:{name}",
            "text": f"# {name}\n\n{description}\n\n{content}",
        }
    )
    return json.dumps(info)


RESEARCHER_TOOLS = [
    web_search,
    rag_query,
    rag_ingest_text,
    create_skill,
    fs_write,
    fs_read,
    fs_list,
]


def build_researcher_subgraph():
    return build_specialist_subgraph(
        name="researcher",
        role="researcher",
        llm_factory=researcher_model,
        tools=RESEARCHER_TOOLS,
        base_system_prompt=RESEARCHER_SYSTEM,
        max_steps=14,
    )
