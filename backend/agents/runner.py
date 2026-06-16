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
from backend.agents.llm_audit import (
    log_llm_invoke_exception_context,
    log_llm_invoke_start,
)
from backend.agents.llm import with_retry
from backend.agents.runtime_env import get_agent_runtime_prompt_section
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
      - Keep the most recent ``[from project_manager → <agent_name>]``
        HumanMessage (this specialist's actual assignment).
      - For most specialists: keep up to 2 recent specialist→PM hand-off summaries
        for minimal cross-agent context. For backend_dev / frontend_dev, omit these
        so each run relies on DB subtasks + the current PM instruction only.
      - The project goal lives in ``state.project_context`` (set at start_run),
        not in this focused history — it's injected by the system prompt builder.
      - Drop everything else (PM AIMessages with tool calls, prior
        specialists' tool transcripts, etc.).

    This bounds every specialist invocation to a tiny prompt regardless of
    how long the run has gone on.
    """
    msgs = list(state.messages)
    if not msgs:
        return []

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
            if len(handoff_summaries) < 2:
                handoff_summaries.append(m)

    out: list[BaseMessage] = []
    # project_context is always set at start_run (to goal or explicit project_context),
    # so we never need to re-add the goal from first_human here. The goal is read
    # from project_context by the specialist's system prompt builder directly.
    for s in reversed(handoff_summaries):
        c = s.content if isinstance(s.content, str) else str(s.content)
        out.append(HumanMessage(content=_truncate(c, 2000)))
    if pm_instruction is not None:
        c = pm_instruction.content if isinstance(pm_instruction.content, str) else str(pm_instruction.content)
        out.append(HumanMessage(content=_truncate(c, MAX_HANDOFF_INSTRUCTION_CHARS)))
    elif not handoff_summaries:
        out.append(HumanMessage(content="(No explicit instruction. Inspect state and proceed with your role.)"))

    return out


def _latest_subtask_id_from_tools(tool_msgs: list[ToolMessage]) -> str | None:
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


def _verification_outcome(tool_msgs: list[ToolMessage]) -> tuple[str, str]:
    """Classify the turn's verification signal from ``run_tests`` results.

    Returns ``(status, detail)`` where status is ``"passed"`` | ``"failed"`` |
    ``"none"``. Only ``run_tests`` counts as the gate — plain ``shell_run`` is
    used for diagnostics/installs (``node --version`` etc.) and intentionally
    isn't treated as proof the infrastructure works.
    """
    last: tuple[str, str] | None = None
    for tm in tool_msgs:
        if getattr(tm, "name", None) != "run_tests":
            continue
        try:
            data = json.loads(str(tm.content))
        except (json.JSONDecodeError, TypeError):
            last = ("failed", "run_tests returned an unparseable result")
            continue
        if data.get("ok") is True:
            last = ("passed", "")
        else:
            err = str(data.get("stderr") or data.get("error") or "").strip().replace("\n", " ")
            code = data.get("exit_code")
            detail = f"run_tests failed (exit_code={code})"
            if err:
                detail += f": {err[:300]}"
            last = ("failed", detail)
    return last or ("none", "")


def _subtasks_marked_done(tool_msgs: list[ToolMessage]) -> list[str]:
    """IDs of subtasks the agent flipped to ``done`` via update_subtask_status."""
    done_ids: list[str] = []
    for tm in tool_msgs:
        if getattr(tm, "name", None) != "update_subtask_status":
            continue
        try:
            data = json.loads(str(tm.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if data.get("status") == "done" and data.get("id"):
            done_ids.append(str(data["id"]))
    return done_ids


async def _revert_subtasks_to_blocked(subtask_ids: list[str]) -> None:
    """Force subtasks back to ``blocked`` (used when a completion gate fails)."""
    if not subtask_ids:
        return
    from backend.db.session import AsyncSessionLocal
    from backend.ticket_system import schemas, service
    from backend.ticket_system.models import SubtaskStatus

    async with AsyncSessionLocal() as db:
        for sid in subtask_ids:
            try:
                await service.update_subtask(
                    db, sid, schemas.SubtaskUpdate(status=SubtaskStatus.BLOCKED)
                )
            except Exception:  # noqa: BLE001 — best-effort guardrail
                continue


PromptBuilder = Callable[[SystemState], str]


def build_specialist_subgraph(
    *,
    name: str,
    role: str,
    llm_factory: Callable[[], BaseChatModel],
    tools: list[BaseTool],
    base_system_prompt: str,
    max_steps: int = 12,
    verify_completion: bool = False,
):
    """Build and compile a tool-using specialist subgraph.

    When ``verify_completion`` is set, a deterministic gate runs at end-of-turn:
    if the agent engaged a subtask but no ``run_tests`` invocation passed, the
    subtask is forced back to ``blocked`` and the hand-off to the PM is rewritten
    as BLOCKED. This stops roles like devops from declaring infrastructure
    "complete" while the verifying tests/commands are still failing.
    """
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    async def specialist(state: SystemState) -> dict[str, Any]:
        await emit(name, "turn_start", {}, state.project_id)
        base = llm_factory()
        bound = base.bind_tools(tools) if tools else base
        llm = with_retry(bound)

        system = inject_skills(base_system_prompt, role=role)
        runtime = get_agent_runtime_prompt_section()
        ctx = _truncate(state.project_context or "", MAX_PROJECT_CONTEXT_CHARS)
        pid = state.project_id or "(unknown)"
        prefix = (
            f"{system}\n\n{runtime}\n\nPROJECT_ID: {pid}\nPROJECT_CONTEXT:\n{ctx}\n"
        )
        if state.active_ticket_id:
            prefix += f"ACTIVE_TICKET_ID: {state.active_ticket_id}\n"
        if state.active_subtask_id:
            prefix += f"ACTIVE_SUBTASK_ID: {state.active_subtask_id}\n"
        prefix += (
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
                await emit(
                    name,
                    "tool_result",
                    {
                        "name": tm.name,
                        "preview": str(tm.content)[:300],
                        # Store a bounded full tool result for the Logs "Full payload" modal.
                        # (The list view stays fast by reading only payload.preview.)
                        "result": _truncate(str(tm.content), MAX_TOOL_RESULT_CHARS),
                    },
                    state.project_id,
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

        subtask_id = _latest_subtask_id_from_tools(local_tool_msgs)

        # Completion gate: a verify-required role (e.g. devops) may only report a
        # subtask complete when a run_tests invocation actually passed this turn.
        if verify_completion and subtask_id:
            status, detail = _verification_outcome(local_tool_msgs)
            if status != "passed":
                reverted = _subtasks_marked_done(local_tool_msgs)
                await _revert_subtasks_to_blocked(reverted)
                await emit(
                    name,
                    "verification_gate_failed",
                    {
                        "subtask_id": subtask_id,
                        "status": status,
                        "detail": detail,
                        "reverted_done": reverted,
                    },
                    state.project_id,
                )
                if status == "failed":
                    gate = (
                        "[BLOCKED — verification FAILED] The infrastructure is NOT "
                        f"complete. {detail} The subtask was kept/flipped to 'blocked'. "
                        "Resolve the environment/test failure (install the missing "
                        "toolchain or fix the config) and re-run run_tests until it "
                        "exits 0, or report the blocker — do not claim completion."
                    )
                else:  # "none"
                    gate = (
                        "[BLOCKED — UNVERIFIED] No passing test proved the "
                        "infrastructure works this turn. The subtask was kept/flipped "
                        "to 'blocked'. Author an infra smoke/verification test and run "
                        "it with run_tests until it exits 0 before marking the subtask done."
                    )
                summary = _truncate(f"{gate}\n\nOriginal summary:\n{summary}", MAX_SUMMARY_CHARS)

        handoff = HumanMessage(
            content=f"[from {name} → project_manager]\n{summary}"
        )

        await emit(name, "turn_end", {}, state.project_id)
        update: dict[str, Any] = {"messages": [handoff]}
        if subtask_id:
            update["active_subtask_id"] = subtask_id
        return update

    g = StateGraph(SystemState)
    g.add_node(name, specialist)
    g.add_edge(START, name)
    g.add_edge(name, END)
    return g.compile()
