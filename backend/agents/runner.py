"""Generic tool-using agent runner shared by specialist subgraphs.

Each specialist (Researcher, Lead, Dev, etc.) is implemented as a small
LangGraph subgraph with two nodes — an LLM step and a tool-execution step
— that loop until the LLM stops calling tools or a max-steps guard fires.
The ``build_specialist_subgraph`` factory builds that loop.

Token-budget design
-------------------
Specialists receive a SMALL, focused input prompt — only the original
project goal and the latest routing instruction from the PM — instead of
the entire shared-state message history. They also return only a
synthesized summary HumanMessage to shared state, never their internal
AI/Tool transcript. This is what keeps Anthropic input-token usage well
below the per-minute rate limits, even on long projects.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from backend.agents.common import emit, run_tool_calls
from backend.agents.llm import with_retry
from backend.agents.llm_audit import (
    log_llm_invoke_exception_context,
    log_llm_invoke_start,
)
from backend.agents.prompts import (
    RESEARCHER_CLOSING_INSTRUCTION,
    RESEARCHER_DEEP_TOOLING_ADDENDUM,
)
from backend.agents.researcher.finalize import (
    build_researcher_handoff_summary,
    finalize_researcher_turn,
)
from backend.agents.session_memory import SessionMemory
from backend.agents.skills.loader import inject_skills
from backend.agents.state import SystemState

# Hard caps to prevent context blow-up. These are characters, not tokens
# (rough 4:1 char-to-token ratio for English).
MAX_TOOL_RESULT_CHARS = 4000
MAX_PROJECT_CONTEXT_CHARS = 1500
MAX_HANDOFF_INSTRUCTION_CHARS = 12000
MAX_SUMMARY_CHARS = 2000


def last_assistant_summary(messages: list[BaseMessage]) -> str:
    """Best-effort textual summary of the final AIMessage in ``messages``."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if isinstance(m.content, str):
                return m.content.strip()
            if isinstance(m.content, list):
                parts: list[str] = []
                for block in m.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif hasattr(block, "text"):
                        parts.append(str(block.text))
                joined = "\n".join(p for p in parts if p)
                if joined.strip():
                    return joined.strip()
    return ""


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated {len(s) - limit} chars]"


def format_pm_assignment_message(message: HumanMessage) -> HumanMessage:
    """Make the latest PM handoff easier for tool models to follow."""
    content = message.content if isinstance(message.content, str) else str(message.content)
    if not content.startswith("[from project_manager →"):
        return message

    lines = content.split("\n", 2)
    if len(lines) < 2:
        return message

    meta_line = lines[1].strip()
    body = lines[2].strip() if len(lines) > 2 else ""
    phase = ""
    if meta_line:
        try:
            parsed = json.loads(meta_line)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            phase = str(parsed.get("phase") or "").strip()

    parts = ["[current PM assignment]"]
    if phase:
        parts.append(f"phase: {phase}")
    if body:
        parts.append(body)
    elif meta_line and not phase:
        parts.append(meta_line)

    return HumanMessage(content="\n".join(parts))


def resolve_project_context(state: SystemState) -> str:
    """Return the best available project brief for relevance checks and prompts."""
    ctx = (state.project_context or "").strip()
    if ctx:
        return ctx
    for message in state.messages:
        if not isinstance(message, HumanMessage):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        if not content or content.startswith("[from "):
            continue
        if content.startswith("[original project goal]\n"):
            content = content[len("[original project goal]\n") :]
        return content.strip()
    return ""


# Product devs fetch ground truth via ticket tools (`next_pending_subtask`, etc.);
# injecting other agents' hand-off summaries only adds latency and stray context.
_AGENT_NAMES_WITHOUT_CROSS_HANDOFFS = frozenset({"backend_dev", "frontend_dev"})


def build_specialist_focused_messages(state: SystemState, agent_name: str) -> list[BaseMessage]:
    """Assemble the minimal message list a specialist needs to act.

    Strategy:
      - Keep the FIRST HumanMessage in shared state (the original project goal).
      - Keep the most recent ``[from project_manager → <agent_name>]``
        HumanMessage (this specialist's actual assignment).
      - For most specialists: keep up to 2 recent specialist→PM hand-off summaries
        for minimal cross-agent context. For backend_dev / frontend_dev, omit these
        so each run relies on DB subtasks + the current PM instruction only.
      - Drop everything else (PM AIMessages with tool calls, prior
        specialists' tool transcripts, etc.).

    This bounds every specialist invocation to a tiny prompt regardless of
    how long the run has gone on.
    """
    msgs = list(state.messages)
    if not msgs:
        return []

    first_human: HumanMessage | None = None
    for m in msgs:
        if isinstance(m, HumanMessage):
            first_human = m
            break

    pm_instruction: HumanMessage | None = None
    handoff_summaries: list[HumanMessage] = []
    collect_handoffs = agent_name not in _AGENT_NAMES_WITHOUT_CROSS_HANDOFFS

    target_tag = f"[from project_manager → {agent_name}]"
    for m in reversed(msgs):
        if not isinstance(m, HumanMessage):
            continue
        content = m.content if isinstance(m.content, str) else ""
        if pm_instruction is None and content.startswith(target_tag):
            pm_instruction = m
            continue
        if collect_handoffs and content.startswith("[from ") and "→ project_manager" in content:
            if agent_name == "researcher" and content.startswith("[from researcher → project_manager]"):
                continue
            if len(handoff_summaries) < 2 and m is not first_human:
                handoff_summaries.append(m)

    out: list[BaseMessage] = []
    include_goal = not (state.project_context or "").strip() or agent_name == "researcher"
    if include_goal:
        initial_goal = first_human
        if initial_goal is not None:
            truncated_goal = (
                initial_goal.content
                if isinstance(initial_goal.content, str)
                else str(initial_goal.content)
            )
            out.append(
                HumanMessage(
                    content=f"[original project goal]\n{_truncate(truncated_goal, 4000)}"
                )
            )
    for s in reversed(handoff_summaries):
        c = s.content if isinstance(s.content, str) else str(s.content)
        out.append(HumanMessage(content=_truncate(c, 2000)))
    if pm_instruction is not None:
        c = format_pm_assignment_message(pm_instruction).content
        c = c if isinstance(c, str) else str(c)
        out.append(HumanMessage(content=_truncate(c, MAX_HANDOFF_INSTRUCTION_CHARS)))
    elif not handoff_summaries:
        out.append(HumanMessage(content="(No explicit instruction. Inspect state and proceed with your role.)"))

    return out


def latest_subtask_id_from_tool_messages(tool_msgs: list[ToolMessage]) -> str | None:
    for tm in reversed(tool_msgs):
        if tm.name not in {"next_pending_subtask", "next_pending_subtask_in_project"}:
            continue
        try:
            data = json.loads(str(tm.content))
        except (json.JSONDecodeError, TypeError):
            continue
        sub = data.get("subtask") or {}
        sid = sub.get("id")
        if sid:
            return str(sid)
    return None


SPECIALIST_CLOSING_INSTRUCTION = (
    "Use tools as needed. When done, respond with a short summary of what you accomplished."
)

_DEEP_KANBAN_CLOSING_NOTE = (
    "Official work tracking uses the ticket system (Kanban). "
    "Do not treat any in-agent todo list as project task state."
)


def specialist_closing_instruction(*, role: str, for_deep: bool = False) -> str:
    if role == "researcher":
        if for_deep:
            from backend.agents.prompts import RESEARCHER_DEEP_CLOSING_INSTRUCTION

            return RESEARCHER_DEEP_CLOSING_INSTRUCTION
        return RESEARCHER_CLOSING_INSTRUCTION
    if for_deep:
        return f"{SPECIALIST_CLOSING_INSTRUCTION}\n\n{_DEEP_KANBAN_CLOSING_NOTE}"
    return SPECIALIST_CLOSING_INSTRUCTION


async def build_specialist_turn_update(
    *,
    name: str,
    role: str,
    state: SystemState,
    ai_text: str,
    tool_msgs: list[ToolMessage],
    emit_researcher_tool_results: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Package specialist handoff, session memory, and optional researcher artifacts."""
    turn_end_payload: dict[str, Any] = {}
    if role == "researcher" and state.project_id:
        turn_end_payload = await finalize_researcher_turn(
            agent=name,
            project_id=state.project_id,
            tool_msgs=tool_msgs,
            emit_tool_results=emit_researcher_tool_results,
            project_context=resolve_project_context(state),
        )

    if role == "researcher":
        summary = build_researcher_handoff_summary(
            ai_text=ai_text,
            project_context=resolve_project_context(state),
            turn_end_payload=turn_end_payload,
        )
    else:
        summary = ai_text or "turn complete"
    summary = _truncate(summary, MAX_SUMMARY_CHARS)
    handoff = HumanMessage(content=f"[from {name} → project_manager]\n{summary}")
    update: dict[str, Any] = {
        "messages": [handoff],
        "session_memory": SessionMemory(
            last_route=name,
            recent_handoffs=[f"{name}: {summary}"],
        ),
    }
    subtask_id = latest_subtask_id_from_tool_messages(tool_msgs)
    if subtask_id:
        update["active_subtask_id"] = subtask_id
    return turn_end_payload, update


def build_specialist_instruction_text(
    *,
    role: str,
    base_system_prompt: str,
    state: SystemState,
    closing_instruction: str = SPECIALIST_CLOSING_INSTRUCTION,
    for_deep: bool = False,
) -> str:
    """Instructions shared by legacy specialist subgraphs and Deep Agents adapters."""
    system = inject_skills(base_system_prompt, role=role, project_id=state.project_id)
    ctx = _truncate(state.project_context or "", MAX_PROJECT_CONTEXT_CHARS)
    pid = state.project_id or "(unknown)"
    parts: list[str] = [
        system,
        "",
        f"PROJECT_ID: {pid}",
        f"PROJECT_CONTEXT:\n{ctx}",
        "",
    ]
    if role == "researcher" and for_deep:
        parts.extend([RESEARCHER_DEEP_TOOLING_ADDENDUM, ""])
    if state.active_ticket_id:
        parts.append(f"ACTIVE_TICKET_ID: {state.active_ticket_id}")
    if state.active_subtask_id:
        parts.append(f"ACTIVE_SUBTASK_ID: {state.active_subtask_id}")
    if parts[-1] != "":
        parts.append("")
    parts.append(closing_instruction)
    return "\n".join(parts)


PromptBuilder = Callable[[SystemState], str]


async def run_specialist_tool_loop(
    *,
    name: str,
    role: str,
    state: SystemState,
    llm_factory: Callable[[], BaseChatModel],
    tools: list[BaseTool],
    base_system_prompt: str,
    max_steps: int = 12,
    extra_focused_messages: list[BaseMessage] | None = None,
    emit_tool_events: bool = True,
) -> tuple[list[AIMessage], list[ToolMessage]]:
    """Run a bounded tool loop for one specialist turn."""
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}
    base = llm_factory()
    bound = base.bind_tools(tools) if tools else base
    llm = with_retry(bound)

    prefix = build_specialist_instruction_text(
        role=role,
        base_system_prompt=base_system_prompt,
        state=state,
        closing_instruction=specialist_closing_instruction(role=role),
    )

    focused_history = build_specialist_focused_messages(state, agent_name=name)
    if extra_focused_messages:
        focused_history = [*focused_history, *extra_focused_messages]
    messages: list[BaseMessage] = [SystemMessage(content=prefix), *focused_history]

    local_ai_msgs: list[AIMessage] = []
    local_tool_msgs: list[ToolMessage] = []

    for step_i in range(max_steps):
        log_llm_invoke_start(
            node_name=name,
            step_index=step_i + 1,
            step_cap=max_steps,
            project_id=state.project_id,
        )
        try:
            ai: AIMessage = await llm.ainvoke(messages)
        except Exception:
            log_llm_invoke_exception_context(
                node_name=name,
                step_index=step_i + 1,
                project_id=state.project_id,
            )
            raise
        messages.append(ai)
        local_ai_msgs.append(ai)

        if not ai.tool_calls:
            break

        tool_msgs = await run_tool_calls(
            ai,
            tools_by_name,
            agent=name,
            project_id=state.project_id,
        )
        for tm in tool_msgs:
            if emit_tool_events:
                await emit(
                    name,
                    "tool_result",
                    {"name": tm.name, "preview": str(tm.content)[:300]},
                    state.project_id,
                )
            truncated = ToolMessage(
                content=_truncate(str(tm.content), MAX_TOOL_RESULT_CHARS),
                name=tm.name,
                tool_call_id=tm.tool_call_id,
            )
            messages.append(truncated)
            local_tool_msgs.append(truncated)

    return local_ai_msgs, local_tool_msgs


def build_specialist_subgraph(
    *,
    name: str,
    role: str,
    llm_factory: Callable[[], BaseChatModel],
    tools: list[BaseTool],
    base_system_prompt: str,
    max_steps: int = 12,
):
    """Build and compile a tool-using specialist subgraph."""
    async def specialist(state: SystemState) -> dict[str, Any]:
        await emit(name, "turn_start", {}, state.project_id)
        local_ai_msgs, local_tool_msgs = await run_specialist_tool_loop(
            name=name,
            role=role,
            state=state,
            llm_factory=llm_factory,
            tools=tools,
            base_system_prompt=base_system_prompt,
            max_steps=max_steps,
        )
        turn_end_payload, update = await build_specialist_turn_update(
            name=name,
            role=role,
            state=state,
            ai_text=last_assistant_summary(local_ai_msgs) or "turn complete",
            tool_msgs=local_tool_msgs,
            emit_researcher_tool_results=False,
        )
        await emit(name, "turn_end", turn_end_payload, state.project_id)
        return update

    g = StateGraph(SystemState)
    g.add_node(name, specialist)
    g.add_edge(START, name)
    g.add_edge(name, END)
    return g.compile()
