"""Project Manager — supervisor agent.

Acts as the routing decision-maker. Holds tools for ticket management and
emits a structured routing decision (``next_agent``) that the root graph's
conditional edge consumes.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from backend.agents.common import emit, run_tool_calls
from backend.agents.llm_audit import (
    log_llm_invoke_exception_context,
    log_llm_invoke_start,
)
from backend.agents.llm import pm_model, with_retry
from backend.agents.prompts import PROJECT_MANAGER_SYSTEM
from backend.agents.skills.loader import inject_skills
from backend.agents.state import SystemState
from backend.db.session import AsyncSessionLocal
from backend.tools.hitl_tools import ask_human
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import PM_TICKET_TOOLS
from backend.ticket_system import service
from backend.ticket_system.models import AgentRole, Ticket, TicketStatus

MAX_TOOL_RESULT_CHARS = 4000
MAX_PROJECT_CONTEXT_CHARS = 1500
MAX_HUMAN_MSG_CHARS = 2500
MAX_KEPT_HUMAN_MESSAGES = 8

VALID_TARGETS = {
    "researcher",
    "backend_lead",
    "frontend_lead",
    "backend_dev",
    "frontend_dev",
    "devops",
    "qa",
    "project_manager",
    "end",
}


class RoutingDecision(BaseModel):
    next_agent: str = Field(description="Next agent to dispatch")
    rationale: str = Field(default="")
    instructions: str = Field(default="")


PM_TOOLS = [*PM_TICKET_TOOLS, rag_query, ask_human]
TOOLS_BY_NAME = {t.name: t for t in PM_TOOLS}


def _truncate(s: str, limit: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…[truncated {len(s) - limit} chars]"


def _condense_messages_for_supervisor(
    messages: list, keep_last_human: int = MAX_KEPT_HUMAN_MESSAGES
) -> list:
    """Drop ALL specialist AI/tool messages and keep only human turns.

    The supervisor only needs:
      - The original user goal (first HumanMessage)
      - Specialist hand-off summaries (HumanMessage starting with ``[from``)
      - Any human answers from interrupts (HumanMessage)

    Each kept HumanMessage is also truncated. The PM is supposed to rely
    on ticket DB state (via list_tickets / get_ticket) for ground truth,
    so we don't need to retain the specialists' tool transcripts in
    context. This is the single biggest token saver.
    """
    keepers: list = []
    for m in messages:
        if getattr(m, "type", None) == "human":
            keepers.append(m)
    if len(keepers) > keep_last_human:
        keepers = [keepers[0], *keepers[-(keep_last_human - 1):]]

    out: list = []
    for m in keepers:
        c = m.content if isinstance(m.content, str) else str(m.content)
        if len(c) > MAX_HUMAN_MSG_CHARS:
            out.append(HumanMessage(content=_truncate(c, MAX_HUMAN_MSG_CHARS)))
        else:
            out.append(m)
    return out


def _build_system_prompt(state: SystemState) -> str:
    base = inject_skills(PROJECT_MANAGER_SYSTEM, role="project_manager")
    ctx = _truncate(state.project_context or "", MAX_PROJECT_CONTEXT_CHARS)
    pid = state.project_id or "(unknown)"
    return (
        f"{base}\n\nPROJECT_ID: {pid}\nPROJECT_CONTEXT:\n{ctx}\n"
        "Use ticket tools to inspect or mutate state. Then return a routing decision JSON."
    )


# Heuristic for tickets that still need a frontend_lead pass. Used when the PM model
# emits invalid JSON or wrongly chooses "end" while only backend_dev subtasks exist.
_CLIENT_SCOPE_TOKENS: frozenset[str] = frozenset(
    {
        "client",
        "frontend",
        "ui",
        "react",
        "browser",
        "component",
        "components",
        "page",
        "dashboard",
        "spa",
        "tailwind",
        "css",
        "hook",
        "tsx",
        "jsx",
        "dom",
        "canvas",
        "mesh",
        "landing",
        "ssr",
    }
)
_CLIENT_SCOPE_PHRASES: tuple[str, ...] = (
    "next.js",
    "nextjs",
    "webrtc",
    "vite",
    "webpack",
)


def _text_suggests_client_scope(ticket: Ticket) -> bool:
    parts: list[str] = [
        ticket.title or "",
        ticket.description or "",
        *(ticket.business_requirements or []),
        *(ticket.technical_requirements or []),
    ]
    for s in ticket.subtasks or []:
        parts.append(s.title or "")
        parts.append(s.description or "")
        parts.append(s.required_functionality or "")
    raw = " ".join(parts).lower()
    if any(p in raw for p in _CLIENT_SCOPE_PHRASES):
        return True
    blob = re.sub(r"[^a-z0-9]+", " ", raw)
    tokens = set(blob.split())
    return bool(tokens & _CLIENT_SCOPE_TOKENS)


def _ticket_has_unanswered_questions(ticket: Ticket) -> bool:
    for q in ticket.questions or []:
        if not isinstance(q, dict):
            continue
        if q.get("answer") in (None, ""):
            return True
    return False


def _fallback_routing_decision(tickets: list[Ticket]) -> RoutingDecision | None:
    """Pick backend_lead or frontend_lead from DB state when the LLM routing fails.

    This keeps the graph alive after backend planning + PM review when tickets were
    advanced to TODO but no frontend_dev subtasks exist yet for client-scope work.
    """
    if not tickets:
        return None
    if all(t.status == TicketStatus.DONE for t in tickets):
        return None

    active = [t for t in tickets if t.status != TicketStatus.DONE]

    drafts_without_subtasks = [
        t
        for t in active
        if t.status == TicketStatus.DRAFT
        and not (t.subtasks or [])
        and not _ticket_has_unanswered_questions(t)
    ]
    if drafts_without_subtasks:
        ids = ", ".join(t.id for t in drafts_without_subtasks)
        return RoutingDecision(
            next_agent="backend_lead",
            rationale=(
                "Fallback: DRAFT ticket(s) still have no subtasks; routing backend_lead."
            ),
            instructions=(
                f"Break down these DRAFT tickets into backend-domain subtasks first "
                f"(UUIDs): {ids}. Call list_tickets / get_ticket, then create_subtask."
            ),
        )

    fe_candidates: list[Ticket] = []
    for t in active:
        if _ticket_has_unanswered_questions(t):
            continue
        if t.status not in (
            TicketStatus.DRAFT,
            TicketStatus.IN_REVIEW,
            TicketStatus.TODO,
        ):
            continue
        subs = list(t.subtasks or [])
        if not subs:
            continue
        if any(s.assigned_to == AgentRole.FRONTEND_DEV for s in subs):
            continue
        if _text_suggests_client_scope(t):
            fe_candidates.append(t)

    if fe_candidates:
        ids = ", ".join(t.id for t in fe_candidates)
        preview = "; ".join(t.title for t in fe_candidates[:6])
        return RoutingDecision(
            next_agent="frontend_lead",
            rationale=(
                "Fallback: ticket text suggests client/UI work but there is no "
                "frontend_dev subtask yet; routing frontend_lead after backend phase."
            ),
            instructions=(
                f"Plan client-side subtasks for these tickets (UUIDs): {ids}. "
                f"Titles: {preview}. Use list_tickets / get_ticket; assign UI work to "
                f"frontend_dev (or devops for client-only infra). "
                f"Call update_ticket_status(in_review) when frontend planning is complete."
            ),
        )
    return None


async def _infer_fallback_route(project_id: str | None) -> RoutingDecision | None:
    if not project_id:
        return None
    async with AsyncSessionLocal() as db:
        tickets = await service.list_tickets(db, project_id=project_id)
    return _fallback_routing_decision(tickets)


def _parse_routing(text: str) -> RoutingDecision | None:
    """Best-effort JSON extraction from the supervisor's final message."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text[len("json\n"):]
    # Locate the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return RoutingDecision(**data)
    except Exception:
        return None


def build_project_manager_node():
    """Return an async LangGraph node function for the PM supervisor."""

    async def project_manager_node(state: SystemState) -> dict[str, Any]:
        events = [await emit("project_manager", "turn_start", {}, state.project_id)]
        llm = with_retry(pm_model().bind_tools(PM_TOOLS))
        system_prompt = _build_system_prompt(state)

        # Condense state.messages to just the human/handoff turns so we
        # don't pay for every specialist's entire tool transcript on each
        # supervisor invocation.
        condensed = _condense_messages_for_supervisor(list(state.messages))
        messages: list = [SystemMessage(content=system_prompt), *condensed]
        # Anthropic requires the conversation to end with a user message
        # before the model can produce a new assistant turn. If the last
        # message in state is an AIMessage, append a nudge.
        if messages and isinstance(messages[-1], AIMessage):
            messages.append(HumanMessage(content="(continue)"))
        # Also guarantee we have at least one human message to satisfy the
        # API even if the conversation history is empty.
        if not any(getattr(m, "type", None) == "human" for m in messages):
            messages.append(HumanMessage(content="(start)"))
        decision: RoutingDecision | None = None

        for step_i in range(8):  # max tool-calling rounds per supervisor turn
            log_llm_invoke_start(
                node_name="project_manager",
                step_index=step_i + 1,
                step_cap=8,
                project_id=state.project_id,
            )
            try:
                ai: AIMessage = await llm.ainvoke(messages)
            except Exception as exc:
                log_llm_invoke_exception_context(
                    node_name="project_manager",
                    step_index=step_i + 1,
                    project_id=state.project_id,
                )
                # Some OpenAI-compatible local servers (llama.cpp, LM Studio, etc.)
                # can 500 when tool-calling payloads are present or when their chat
                # template parser rejects the prompt. Instead of crashing the whole
                # orchestration run, fall back to DB-based routing and keep going.
                events.append(
                    await emit(
                        "project_manager",
                        "llm_error_fallback",
                        {
                            "step": step_i + 1,
                            "error_preview": _truncate(str(exc), 500),
                        },
                        state.project_id,
                    )
                )
                decision = await _infer_fallback_route(state.project_id)
                break
            messages.append(ai)

            if ai.tool_calls:
                tool_msgs: list[ToolMessage] = await run_tool_calls(ai, TOOLS_BY_NAME)
                for tm in tool_msgs:
                    events.append(
                        await emit(
                            "project_manager",
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
                continue

            decision = _parse_routing(str(ai.content))
            break

        # We do NOT push the PM's intermediate AI/Tool messages back into
        # shared state — those are private to this turn and would only
        # bloat the next agent's context. The PM re-reads ticket DB state
        # at the start of every turn anyway, so nothing important is lost.
        new_msgs: list = []

        if decision is None or decision.next_agent not in VALID_TARGETS:
            fb = await _infer_fallback_route(state.project_id)
            if fb is not None:
                decision = fb
                events.append(
                    await emit(
                        "project_manager",
                        "route_fallback",
                        {
                            "next_agent": decision.next_agent,
                            "rationale": decision.rationale,
                        },
                        state.project_id,
                    )
                )
            else:
                decision = RoutingDecision(
                    next_agent="end",
                    rationale="Supervisor failed to produce a valid routing decision.",
                    instructions="",
                )
        elif decision.next_agent == "end":
            fb = await _infer_fallback_route(state.project_id)
            if fb is not None:
                decision = fb
                events.append(
                    await emit(
                        "project_manager",
                        "route_fallback",
                        {
                            "next_agent": decision.next_agent,
                            "rationale": decision.rationale,
                        },
                        state.project_id,
                    )
                )

        events.append(
            await emit(
                "project_manager",
                "route",
                {"next_agent": decision.next_agent, "rationale": decision.rationale},
                state.project_id,
            )
        )

        # If the PM hands off, append an instruction message visible to the next agent.
        if decision.next_agent not in {"end", "project_manager"} and decision.instructions:
            new_msgs.append(
                HumanMessage(
                    content=f"[from project_manager → {decision.next_agent}]\n{decision.instructions}"
                )
            )

        return {
            "messages": new_msgs,
            "next_agent": decision.next_agent,
            "events": events,
        }

    return project_manager_node
