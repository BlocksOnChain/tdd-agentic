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
from backend.agents.session_memory import SessionMemory
from backend.agents.skills.loader import inject_skills
from backend.agents.state import SystemState
from backend.db.session import AsyncSessionLocal
from backend.agents.researcher.scaffold import is_scaffolded_path
from backend.rag.workspace_sync import list_research_markdown
from backend.tools.hitl_tools import ask_human
from backend.tools.rag_tools import rag_query
from backend.tools.ticket_tools import PM_TICKET_TOOLS
from backend.ticket_system import service
from backend.ticket_system import schemas
from backend.ticket_system.models import AgentRole, SubtaskStatus, Ticket, TicketStatus

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

PLANNING_TARGETS = {
    "backend_lead",
    "frontend_lead",
    "backend_dev",
    "frontend_dev",
    "devops",
    "qa",
}

PM_TICKET_BACKLOG_TOOLS = [*PM_TICKET_TOOLS, rag_query]
PM_TICKET_BACKLOG_TOOLS_BY_NAME = {t.name: t for t in PM_TICKET_BACKLOG_TOOLS}


class RoutingDecision(BaseModel):
    next_agent: str = Field(description="Next agent to dispatch")
    rationale: str = Field(default="")
    instructions: str = Field(default="")
    ticket_ids: list[str] = Field(default_factory=list)
    phase: str = Field(default="")


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
    base = inject_skills(
        PROJECT_MANAGER_SYSTEM,
        role="project_manager",
        project_id=state.project_id,
    )
    ctx = _truncate(state.project_context or "", MAX_PROJECT_CONTEXT_CHARS)
    pid = state.project_id or "(unknown)"
    lines = [
        f"{base}",
        "",
        f"PROJECT_ID: {pid}",
        f"PROJECT_CONTEXT:\n{ctx}",
    ]
    if state.active_ticket_id:
        lines.append(f"ACTIVE_TICKET_ID: {state.active_ticket_id}")
    if state.active_subtask_id:
        lines.append(f"ACTIVE_SUBTASK_ID: {state.active_subtask_id}")
    lines.append(
        "Use ticket tools to inspect or mutate state. Then return a routing decision JSON."
    )
    return "\n".join(lines)


_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _normalise_ticket_ids(decision: RoutingDecision) -> list[str]:
    ids = [str(t).strip() for t in (decision.ticket_ids or []) if str(t).strip()]
    if ids:
        return ids
    return _UUID_RE.findall(decision.instructions or "")


def _format_pm_handoff(target: str, decision: RoutingDecision) -> str:
    ticket_ids = _normalise_ticket_ids(decision)
    meta: dict[str, Any] = {}
    if ticket_ids:
        meta["ticket_ids"] = ticket_ids
    if decision.phase:
        meta["phase"] = decision.phase
    header = json.dumps(meta, separators=(",", ":")) if meta else "{}"
    body = (decision.instructions or "").strip()
    if body:
        return f"[from project_manager → {target}]\n{header}\n{body}"
    return f"[from project_manager → {target}]\n{header}"


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


def _ticket_ready_for_todo(ticket: Ticket) -> bool:
    """Conservative readiness check for moving IN_REVIEW -> TODO.

    User expectation in this app: IN_REVIEW means planning has happened at least
    once and the ticket should be eligible for developer execution unless it
    still has unanswered questions or no subtasks at all.
    """
    if ticket.status != TicketStatus.IN_REVIEW:
        return False
    if _ticket_has_unanswered_questions(ticket):
        return False
    subs = list(ticket.subtasks or [])
    return len(subs) > 0


async def _advance_in_review_to_todo(project_id: str | None) -> list[str]:
    """Auto-advance tickets from IN_REVIEW to TODO when they are ready.

    Returns the list of ticket ids advanced.
    """
    if not project_id:
        return []
    advanced: list[str] = []
    async with AsyncSessionLocal() as db:
        tickets = await service.list_tickets(db, project_id=project_id)
        # list_tickets may not eager-load subtasks depending on the query;
        # fetch details only for candidates to keep this cheap.
        for t in tickets:
            if t.status != TicketStatus.IN_REVIEW:
                continue
            full = await service.get_ticket(db, t.id)
            if not _ticket_ready_for_todo(full):
                continue
            await service.update_ticket(
                db,
                full.id,
                schemas.TicketUpdate(status=TicketStatus.TODO),
            )
            advanced.append(full.id)
    return advanced


async def _infer_next_dev_route(project_id: str | None) -> RoutingDecision | None:
    """Deterministic routing to a dev role based on ticket/subtask state.

    Policy:
      1) Consider tickets in stable execution order (ticket.order_index) with status
         TODO first (then IN_PROGRESS).
      2) If ANY subtask is already IN_PROGRESS on the first eligible ticket, route to
         that subtask's assigned_to role so the same developer can continue.
      3) Otherwise, route to the assigned_to role of the earliest PENDING subtask
         (lowest subtask.order_index).
    """
    if not project_id:
        return None
    async with AsyncSessionLocal() as db:
        tickets = await service.list_tickets(db, project_id=project_id)
        # Stable ticket order already enforced by service.list_tickets().
        eligible = [t for t in tickets if t.status in (TicketStatus.TODO, TicketStatus.IN_PROGRESS)]
        # Prefer TODO over IN_PROGRESS when choosing the "first ticket".
        eligible.sort(key=lambda t: (0 if t.status == TicketStatus.TODO else 1, t.order_index, t.created_at))

        valid_dev_roles = {AgentRole.BACKEND_DEV, AgentRole.FRONTEND_DEV, AgentRole.DEVOPS, AgentRole.QA}

        for t in eligible:
            # list_tickets eager-loads subtasks; only re-fetch if missing for some reason.
            full = t if getattr(t, "subtasks", None) is not None else await service.get_ticket(db, t.id)
            subs = list(full.subtasks or [])
            if not subs:
                continue

            in_progress = [s for s in subs if s.status == SubtaskStatus.IN_PROGRESS and s.assigned_to in valid_dev_roles]
            if in_progress:
                # Continue the earliest in-progress subtask (lowest order_index).
                in_progress.sort(key=lambda s: (s.order_index, s.created_at))
                s0 = in_progress[0]
                return RoutingDecision(
                    next_agent=s0.assigned_to.value,
                    rationale="Auto-route: subtask already in progress.",
                    instructions=(
                        f"Continue in_progress subtask order_index={s0.order_index} "
                        f"({s0.assigned_to.value})."
                    ),
                    ticket_ids=[full.id],
                    phase="implement",
                )

            pending = [s for s in subs if s.status == SubtaskStatus.PENDING and s.assigned_to in valid_dev_roles]
            if not pending:
                continue
            pending.sort(key=lambda s: (s.order_index, s.created_at))
            s0 = pending[0]
            return RoutingDecision(
                next_agent=s0.assigned_to.value,
                rationale="Auto-route: dispatch first pending subtask in order.",
                instructions=(
                    f"Start pending subtask order_index={s0.order_index} "
                    f"({s0.assigned_to.value})."
                ),
                ticket_ids=[full.id],
                phase="implement",
            )
    return None


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
        ids = [t.id for t in drafts_without_subtasks]
        return RoutingDecision(
            next_agent="backend_lead",
            rationale=(
                "Fallback: DRAFT ticket(s) still have no subtasks; routing backend_lead."
            ),
            instructions="Break down DRAFT tickets into backend-domain subtasks.",
            ticket_ids=ids,
            phase="backend_planning",
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
        # If *any* client-side execution subtask exists already, don't
        # force a frontend_lead fallback. Some tickets are legitimately
        # QA-only or DevOps-only even when they relate to client/UI scope.
        if any(
            s.assigned_to in {AgentRole.FRONTEND_DEV, AgentRole.DEVOPS, AgentRole.QA}
            for s in subs
        ):
            continue
        if _text_suggests_client_scope(t):
            fe_candidates.append(t)

    if fe_candidates:
        ids = [t.id for t in fe_candidates]
        return RoutingDecision(
            next_agent="frontend_lead",
            rationale=(
                "Fallback: ticket text suggests client/UI work but there is no "
                "client-side execution subtask yet; routing frontend_lead."
            ),
            instructions=(
                "Plan client-side subtasks; assign frontend_dev, devops, or qa as needed."
            ),
            ticket_ids=ids,
            phase="frontend_planning",
        )
    return None


def _human_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _research_phase_complete(
    messages: list,
    session_memory: SessionMemory | None = None,
) -> bool:
    """True when the researcher already handed control back to the PM."""
    last_dispatch = -1
    last_handback = -1
    for i, message in enumerate(messages):
        if getattr(message, "type", None) != "human":
            continue
        text = _human_message_text(message)
        if text.startswith("[from project_manager → researcher]"):
            last_dispatch = i
        elif text.startswith("[from researcher → project_manager]"):
            last_handback = i
    return last_handback > last_dispatch


def _research_materials_ready(project_id: str | None) -> bool:
    if not project_id:
        return False
    for row in list_research_markdown(project_id):
        path = str(row.get("path") or "")
        if int(row.get("bytes") or 0) <= 0:
            continue
        if not path.startswith("docs/") or not path.endswith(".md"):
            continue
        if path == "docs/README.md":
            continue
        if is_scaffolded_path(project_id, path):
            continue
        return True
    return False


def _pm_research_redispatch_decision() -> RoutingDecision:
    return RoutingDecision(
        next_agent="researcher",
        rationale=(
            "Research docs are missing or empty; researcher must scaffold docs under docs/ "
            "and ingest them into RAG before ticket creation."
        ),
        instructions=(
            "Write or refresh project docs under docs/ (for example docs/tech-stack.md, "
            "docs/architecture.md, docs/conventions.md, docs/api-contracts.md), call "
            "rag_ingest_text for each authored file, then hand back to project_manager."
        ),
        phase="research",
    )


def _pm_ticket_creation_decision() -> RoutingDecision:
    return RoutingDecision(
        next_agent="project_manager",
        rationale=(
            "Fallback: research finished but no tickets exist; PM must create tickets "
            "before lead planning."
        ),
        instructions=(
            "Call list_tickets(project_id). If empty, call rag_query(project_id, query=<project goal>) "
            "and read PROJECT_CONTEXT. Use create_ticket for each major deliverable with "
            "business_requirements and technical_requirements grounded in those sources. "
            "Do not route to leads or devs while the backlog is empty."
        ),
        phase="backend_planning",
    )


def _pm_ticket_creation_nudge() -> str:
    decision = _pm_ticket_creation_decision()
    return _format_pm_handoff("project_manager", decision)


async def _must_create_tickets_before_planning(
    project_id: str | None,
    *,
    messages: list,
    session_memory: SessionMemory | None,
) -> bool:
    if not project_id:
        return False
    if not _research_phase_complete(messages, session_memory):
        return False
    if not _research_materials_ready(project_id):
        return False
    return not await _list_project_tickets(project_id)


async def _list_project_tickets(project_id: str | None) -> list[Ticket]:
    if not project_id:
        return []
    async with AsyncSessionLocal() as db:
        return await service.list_tickets(db, project_id=project_id)


async def _guard_planning_without_tickets(
    decision: RoutingDecision,
    *,
    project_id: str | None,
    messages: list,
    session_memory: SessionMemory | None,
) -> RoutingDecision:
    if decision.next_agent not in PLANNING_TARGETS:
        return decision
    if not project_id:
        return decision
    if await _list_project_tickets(project_id):
        return decision
    if not _research_phase_complete(messages, session_memory):
        return decision
    if not _research_materials_ready(project_id):
        return _pm_research_redispatch_decision()
    return _pm_ticket_creation_decision()


async def _run_pm_ticket_backlog_phase(
    *,
    state: SystemState,
    system_prompt: str,
    condensed: list,
    events: list,
) -> bool:
    """Force PM ticket tools until the backlog is populated or the step budget is exhausted."""
    if not state.project_id:
        return False
    if await _list_project_tickets(state.project_id):
        return True

    tool_llm = with_retry(
        pm_model().bind_tools(PM_TICKET_BACKLOG_TOOLS, tool_choice="any")
    )
    messages: list = [SystemMessage(content=system_prompt), *condensed]
    if not any(getattr(m, "type", None) == "human" for m in messages):
        messages.append(HumanMessage(content="(start)"))

    for step_i in range(8):
        log_llm_invoke_start(
            node_name="project_manager",
            step_index=step_i + 1,
            step_cap=8,
            project_id=state.project_id,
        )
        try:
            ai: AIMessage = await tool_llm.ainvoke(messages)
        except Exception as exc:
            log_llm_invoke_exception_context(
                node_name="project_manager",
                step_index=step_i + 1,
                project_id=state.project_id,
            )
            events.append(
                await emit(
                    "project_manager",
                    "llm_error_fallback",
                    {
                        "step": step_i + 1,
                        "error_preview": _truncate(str(exc), 500),
                        "phase": "ticket_backlog",
                    },
                    state.project_id,
                )
            )
            return False
        messages.append(ai)
        if not ai.tool_calls:
            messages.append(
                HumanMessage(
                    content=(
                        "Call list_tickets(project_id), then rag_query(project_id, query=<project goal>), "
                        "then create_ticket for each major deliverable with business_requirements and "
                        "technical_requirements grounded in the research docs and PROJECT_CONTEXT."
                    )
                )
            )
            continue

        tool_msgs = await run_tool_calls(
            ai,
            PM_TICKET_BACKLOG_TOOLS_BY_NAME,
            agent="project_manager",
            project_id=state.project_id,
        )
        for tm in tool_msgs:
            events.append(
                await emit(
                    "project_manager",
                    "tool_result",
                    {"name": tm.name, "preview": str(tm.content)[:300]},
                    state.project_id,
                )
            )
            messages.append(
                ToolMessage(
                    content=_truncate(str(tm.content), MAX_TOOL_RESULT_CHARS),
                    name=tm.name,
                    tool_call_id=tm.tool_call_id,
                )
            )
        if await _list_project_tickets(state.project_id):
            return True
    return bool(await _list_project_tickets(state.project_id))


async def _infer_fallback_route(
    project_id: str | None,
    *,
    messages: list | None = None,
    session_memory: SessionMemory | None = None,
) -> RoutingDecision | None:
    if not project_id:
        return None
    tickets = await _list_project_tickets(project_id)
    if not tickets:
        if _research_phase_complete(messages or [], session_memory):
            if not _research_materials_ready(project_id):
                return _pm_research_redispatch_decision()
            return _pm_ticket_creation_decision()
        return RoutingDecision(
            next_agent="researcher",
            rationale="Fallback: PM LLM unavailable and no tickets exist yet.",
            instructions=(
                "Gather initial project context and scaffold docs; then ingest into RAG. "
                "After research, hand back to project_manager."
            ),
            phase="research",
        )
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
        decision = RoutingDecision(**data)
        if decision.next_agent.strip().lower() == "pm":
            decision.next_agent = "project_manager"
        return decision
    except Exception:
        return None


def build_project_manager_node():
    """Return an async LangGraph node function for the PM supervisor."""

    async def project_manager_node(state: SystemState) -> dict[str, Any]:
        events = [await emit("project_manager", "turn_start", {}, state.project_id)]

        # Deterministic PM hygiene: tickets IN_REVIEW should be promoted to TODO
        # once planning is complete so dev agents can run.
        advanced = await _advance_in_review_to_todo(state.project_id)
        if advanced:
            events.append(
                await emit(
                    "project_manager",
                    "tickets_advanced_to_todo",
                    {"ticket_ids": advanced},
                    state.project_id,
                )
            )

        # If there is developer work ready, route deterministically even if the
        # LLM later fails or emits invalid routing JSON.
        ready_dev = await _infer_next_dev_route(state.project_id)
        llm = with_retry(pm_model().bind_tools(PM_TOOLS))
        system_prompt = _build_system_prompt(state)

        # Condense state.messages to just the human/handoff turns so we
        # don't pay for every specialist's entire tool transcript on each
        # supervisor invocation.
        condensed = _condense_messages_for_supervisor(list(state.messages))
        research_handback = _research_phase_complete(
            list(state.messages),
            state.session_memory,
        )
        if (
            state.project_id
            and research_handback
            and not await _list_project_tickets(state.project_id)
            and not _research_materials_ready(state.project_id)
        ):
            decision = _pm_research_redispatch_decision()
            events.append(
                await emit(
                    "project_manager",
                    "route",
                    {"next_agent": decision.next_agent, "rationale": decision.rationale},
                    state.project_id,
                )
            )
            return {
                "messages": [
                    HumanMessage(
                        content=_format_pm_handoff(decision.next_agent, decision)
                    )
                ],
                "next_agent": decision.next_agent,
                "active_ticket_id": state.active_ticket_id,
            }

        if await _must_create_tickets_before_planning(
            state.project_id,
            messages=list(state.messages),
            session_memory=state.session_memory,
        ):
            condensed.append(HumanMessage(content=_pm_ticket_creation_nudge()))
            await _run_pm_ticket_backlog_phase(
                state=state,
                system_prompt=system_prompt,
                condensed=condensed,
                events=events,
            )
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
                decision = await _infer_fallback_route(
                    state.project_id,
                    messages=list(state.messages),
                    session_memory=state.session_memory,
                )
                # If DB heuristics can't confidently choose a specialist, keep the
                # run alive (do NOT end) and surface the configuration problem.
                if decision is None:
                    decision = RoutingDecision(
                        next_agent="project_manager",
                        rationale=(
                            "Fallback: PM LLM call failed and no safe DB route was inferred."
                        ),
                        instructions=(
                            "The configured PM model is failing (often llama.cpp chat-template/tool "
                            "compatibility). Fix by routing PM_MODEL to a stable provider/model "
                            "(e.g. anthropic/claude-sonnet-4-6) or adjust llama-server chat template, "
                            "then retry the run."
                        ),
                    )
                break
            messages.append(ai)

            if ai.tool_calls:
                tool_msgs: list[ToolMessage] = await run_tool_calls(
                    ai,
                    TOOLS_BY_NAME,
                    agent="project_manager",
                    project_id=state.project_id,
                )
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
            if await _must_create_tickets_before_planning(
                state.project_id,
                messages=list(state.messages),
                session_memory=state.session_memory,
            ):
                if decision is None:
                    messages.append(
                        HumanMessage(
                            content=(
                                "Research is complete but the ticket backlog is still empty. "
                                "Do not return routing JSON yet. Call list_tickets(project_id), "
                                "then create_ticket for each major deliverable with "
                                "business_requirements and technical_requirements."
                            )
                        )
                    )
                    continue
                messages.append(
                    HumanMessage(
                        content=(
                            "Research is complete but the ticket backlog is still empty. "
                            "Do not return routing JSON yet. Call list_tickets(project_id), "
                            "then create_ticket for each major deliverable with "
                            "business_requirements and technical_requirements."
                        )
                    )
                )
                decision = None
                continue
            break

        # We do NOT push the PM's intermediate AI/Tool messages back into
        # shared state — those are private to this turn and would only
        # bloat the next agent's context. The PM re-reads ticket DB state
        # at the start of every turn anyway, so nothing important is lost.
        new_msgs: list = []

        if decision is None or decision.next_agent not in VALID_TARGETS:
            ready_dev = await _infer_next_dev_route(state.project_id)
            if ready_dev is not None:
                decision = ready_dev
                events.append(
                    await emit(
                        "project_manager",
                        "route_fallback",
                        {"next_agent": decision.next_agent, "rationale": "Auto-route: pending subtasks exist."},
                        state.project_id,
                    )
                )
            else:
                fb = await _infer_fallback_route(
                    state.project_id,
                    messages=list(state.messages),
                    session_memory=state.session_memory,
                )
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
                        next_agent="project_manager",
                        rationale="No valid routing decision and no safe fallback route was inferred.",
                        instructions="Call list_tickets(project_id) and determine next agent from DB state.",
                    )
        elif decision.next_agent == "end":
            fb = await _infer_fallback_route(
                state.project_id,
                messages=list(state.messages),
                session_memory=state.session_memory,
            )
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

        if (
            state.project_id
            and research_handback
            and not await _list_project_tickets(state.project_id)
        ):
            if not _research_materials_ready(state.project_id):
                decision = _pm_research_redispatch_decision()
            else:
                decision = _pm_ticket_creation_decision()

        decision = await _guard_planning_without_tickets(
            decision,
            project_id=state.project_id,
            messages=list(state.messages),
            session_memory=state.session_memory,
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
        if decision.next_agent != "end" and (
            decision.instructions or decision.ticket_ids or decision.phase
        ):
            new_msgs.append(
                HumanMessage(
                    content=_format_pm_handoff(decision.next_agent, decision)
                )
            )

        ticket_ids = _normalise_ticket_ids(decision)
        active_ticket = ticket_ids[0] if ticket_ids else state.active_ticket_id
        if decision.next_agent in {"end", "project_manager"}:
            active_ticket = state.active_ticket_id

        return {
            "messages": new_msgs,
            "next_agent": decision.next_agent,
            "active_ticket_id": active_ticket,
        }

    return project_manager_node
