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
from backend.agents.llm_audit import (
    log_llm_invoke_exception_context,
    log_llm_invoke_start,
)
from backend.agents.llm import with_retry
from backend.agents.skills.loader import inject_skills
from backend.agents.state import SystemState

# Hard caps to prevent context blow-up. These are characters, not tokens
# (rough 4:1 char-to-token ratio for English).
MAX_TOOL_RESULT_CHARS = 4000
MAX_PROJECT_CONTEXT_CHARS = 1500
MAX_HANDOFF_INSTRUCTION_CHARS = 12000
MAX_SUMMARY_CHARS = 2000


def _last_text(messages: list[BaseMessage]) -> str:
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


# Product devs fetch ground truth via ticket tools (`next_pending_subtask`, etc.);
# injecting other agents' hand-off summaries only adds latency and stray context.
_AGENT_NAMES_WITHOUT_CROSS_HANDOFFS = frozenset({"backend_dev", "frontend_dev"})


def _build_specialist_input(state: SystemState, agent_name: str) -> list[BaseMessage]:
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

    initial_goal: HumanMessage | None = None
    for m in msgs:
        if isinstance(m, HumanMessage):
            initial_goal = m
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
            if len(handoff_summaries) < 2 and m is not initial_goal:
                handoff_summaries.append(m)

    out: list[BaseMessage] = []
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
        c = pm_instruction.content if isinstance(pm_instruction.content, str) else str(pm_instruction.content)
        out.append(HumanMessage(content=_truncate(c, MAX_HANDOFF_INSTRUCTION_CHARS)))
    elif not handoff_summaries:
        out.append(HumanMessage(content="(No explicit instruction. Inspect state and proceed with your role.)"))

    return out


PromptBuilder = Callable[[SystemState], str]


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
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    async def specialist(state: SystemState) -> dict[str, Any]:
        events = [await emit(name, "turn_start", {}, state.project_id)]
        base = llm_factory()
        bound = base.bind_tools(tools) if tools else base
        llm = with_retry(bound)

        system = inject_skills(base_system_prompt, role=role)
        ctx = _truncate(state.project_context or "", MAX_PROJECT_CONTEXT_CHARS)
        pid = state.project_id or "(unknown)"
        prefix = (
            f"{system}\n\nPROJECT_ID: {pid}\nPROJECT_CONTEXT:\n{ctx}\n"
            "Use tools as needed. When done, respond with a short summary of what you accomplished."
        )

        focused_history = _build_specialist_input(state, agent_name=name)
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

            tool_msgs = await run_tool_calls(ai, tools_by_name)
            for tm in tool_msgs:
                events.append(
                    await emit(
                        name,
                        "tool_result",
                        {"name": tm.name, "preview": str(tm.content)[:300]},
                        state.project_id,
                    )
                )
                truncated = ToolMessage(
                    content=_truncate(str(tm.content), MAX_TOOL_RESULT_CHARS),
                    name=tm.name,
                    tool_call_id=tm.tool_call_id,
                )
                messages.append(truncated)
                local_tool_msgs.append(truncated)

        summary = _last_text(local_ai_msgs) or "turn complete"
        summary = _truncate(summary, MAX_SUMMARY_CHARS)
        handoff = HumanMessage(
            content=f"[from {name} → project_manager]\n{summary}"
        )

        events.append(await emit(name, "turn_end", {}, state.project_id))
        return {"messages": [handoff], "events": events}

    g = StateGraph(SystemState)
    g.add_node(name, specialist)
    g.add_edge(START, name)
    g.add_edge(name, END)
    return g.compile()
