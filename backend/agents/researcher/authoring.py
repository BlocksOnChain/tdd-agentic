"""Legacy tool-loop authoring when the deep researcher does not write docs."""
from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents.prompts import RESEARCHER_AUTHORING_INSTRUCTION, RESEARCHER_SYSTEM
from backend.agents.researcher.tools import RESEARCHER_TOOLS_LEGACY
from backend.agents.runner import run_specialist_tool_loop
from backend.agents.state import SystemState


async def run_researcher_authoring_pass(
    state: SystemState,
    llm_factory: Callable[[], BaseChatModel],
) -> tuple[list[AIMessage], list[ToolMessage]]:
    """Run a bounded legacy tool loop to author docs from research context."""
    extras: list[HumanMessage] = [HumanMessage(content=RESEARCHER_AUTHORING_INSTRUCTION)]
    return await run_specialist_tool_loop(
        name="researcher",
        role="researcher",
        state=state,
        llm_factory=llm_factory,
        tools=RESEARCHER_TOOLS_LEGACY,
        base_system_prompt=RESEARCHER_SYSTEM,
        max_steps=14,
        extra_focused_messages=extras,
    )
