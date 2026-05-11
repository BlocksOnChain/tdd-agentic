"""Run a single specialist turn via LangChain Deep Agents (isolated from checkpoints)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt

from backend.agents.common import emit
from backend.agents.deep.harness import register_harness_exclusions_for_model
from backend.agents.deep.tooling import bind_tools_to_project, invoke_deep_agent_with_tool_telemetry
from backend.agents.deep.workspace import project_filesystem_backend
from backend.agents.researcher.finalize import (
    RESEARCHER_CONTINUATION_NUDGE,
    build_researcher_deep_assignment_messages,
    researcher_turn_incomplete,
)
from backend.agents.researcher.scaffold import discard_placeholder_scaffold_files
from backend.agents.runner import (
    build_specialist_focused_messages,
    build_specialist_instruction_text,
    build_specialist_turn_update,
    last_assistant_summary,
    specialist_closing_instruction,
)
from backend.agents.state import SystemState
from backend.config import get_settings
from backend.tools.web_search_tools import _resolve_backend

# Tickets remain the Kanban source of truth; hide harness planning todo tool for researcher.
_RESEARCHER_HARNESS_EXCLUSIONS: frozenset[str] = frozenset(
    {"write_todos", "execute", "task"}
)
_RESEARCHER_DEEP_MAX_ATTEMPTS = 2


async def run_deep_specialist(
    *,
    name: str,
    role: str,
    state: SystemState,
    llm_factory: Callable[[], BaseChatModel],
    tools: list[BaseTool],
    base_system_prompt: str,
    harness_excluded_tools: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Invoke ``create_deep_agent`` for one turn; return a partial ``SystemState`` update.

    Does not attach the root checkpointer — deep-agent message churn stays inside
    this invocation. Emits ``turn_start`` / ``turn_end`` and streams tool activity.
    """
    turn_start_payload: dict[str, Any] = {}
    if role == "researcher":
        turn_start_payload["web_search_backend"] = _resolve_backend()
    await emit(name, "turn_start", turn_start_payload, state.project_id)
    settings = get_settings()
    pid = state.project_id
    if not pid:
        raise ValueError("run_deep_specialist requires state.project_id")

    # Deep Agents requires a concrete ``BaseChatModel``; ``with_retry`` returns a
    # ``RunnableRetry`` that ``resolve_model`` mis-handles as a model spec.
    model = llm_factory()
    excluded = harness_excluded_tools
    if excluded is None and role == "researcher":
        excluded = _RESEARCHER_HARNESS_EXCLUSIONS
    register_harness_exclusions_for_model(model, excluded_tools=excluded or frozenset())

    system_text = build_specialist_instruction_text(
        role=role,
        base_system_prompt=base_system_prompt,
        state=state,
        closing_instruction=specialist_closing_instruction(role=role, for_deep=True),
        for_deep=True,
    )

    skill_root = Path(settings.workspace_root) / pid / "_skills"
    skills_arg: list[str] | None = ["/_skills"] if skill_root.is_dir() else None

    ag = create_deep_agent(
        model=model,
        tools=bind_tools_to_project(tools, pid),
        system_prompt=system_text,
        backend=project_filesystem_backend(pid),
        skills=skills_arg,
        memory=["/AGENTS.md"],
        name=name,
    )

    if role == "researcher":
        removed = discard_placeholder_scaffold_files(pid)
        if removed:
            await emit(
                name,
                "scaffold_discarded",
                {"paths": removed},
                pid,
            )

    focused = build_specialist_focused_messages(state, agent_name=name)
    if role == "researcher":
        focused = [*focused, *build_researcher_deep_assignment_messages(pid)]

    recursion_limit = max(25, settings.deep_agent_recursion_limit)
    config = {"recursion_limit": recursion_limit}
    input_messages = focused
    out_msgs: list = []
    tool_msgs: list[ToolMessage] = []
    max_attempts = _RESEARCHER_DEEP_MAX_ATTEMPTS if role == "researcher" else 1

    try:
        for attempt in range(max_attempts):
            out, attempt_tools = await invoke_deep_agent_with_tool_telemetry(
                ag,
                agent_name=name,
                project_id=pid,
                input_state={"messages": input_messages},
                config=config,
            )
            out_msgs = list(out.get("messages") or input_messages)
            if attempt_tools:
                tool_msgs = [*tool_msgs, *attempt_tools]
            elif not tool_msgs:
                tool_msgs = [m for m in out_msgs if isinstance(m, ToolMessage)]

            if role != "researcher" or not researcher_turn_incomplete(pid, tool_msgs):
                break
            if attempt + 1 >= max_attempts:
                break

            await emit(
                name,
                "research_continuation",
                {"mode": "deep_agent", "attempt": attempt + 2},
                pid,
            )
            input_messages = [
                *out_msgs,
                HumanMessage(content=f"[research continuation]\n{RESEARCHER_CONTINUATION_NUDGE}"),
            ]
    except GraphInterrupt:
        await emit(name, "turn_end", {"interrupted": True}, state.project_id)
        raise

    turn_end_payload, update = await build_specialist_turn_update(
        name=name,
        role=role,
        state=state,
        ai_text=last_assistant_summary(out_msgs) or "turn complete",
        tool_msgs=tool_msgs,
        emit_researcher_tool_results=False,
    )
    await emit(name, "turn_end", turn_end_payload, state.project_id)
    return update
