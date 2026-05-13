"""Researcher subgraph — web search, doc writing, RAG ingestion, skill creation."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.agents.deep.adapter import run_deep_specialist
from backend.agents.llm import researcher_model
from backend.agents.prompts import RESEARCHER_SYSTEM
from backend.agents.runner import build_specialist_subgraph
from backend.agents.researcher.tools import RESEARCHER_TOOLS_DEEP, RESEARCHER_TOOLS_LEGACY
from backend.agents.state import SystemState
from backend.config import get_settings


def _build_deep_researcher_subgraph():
    async def researcher(state: SystemState) -> dict[str, Any]:
        return await run_deep_specialist(
            name="researcher",
            role="researcher",
            state=state,
            llm_factory=researcher_model,
            tools=RESEARCHER_TOOLS_DEEP,
            base_system_prompt=RESEARCHER_SYSTEM,
        )

    g = StateGraph(SystemState)
    g.add_node("researcher", researcher)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", END)
    return g.compile()


def build_researcher_subgraph():
    if get_settings().use_deep_agent_researcher:
        return _build_deep_researcher_subgraph()
    return build_specialist_subgraph(
        name="researcher",
        role="researcher",
        llm_factory=researcher_model,
        tools=RESEARCHER_TOOLS_LEGACY,
        base_system_prompt=RESEARCHER_SYSTEM,
        max_steps=14,
    )
