"""LangChain tools that wrap the ticket-system service for use by agents.

These tools open their own DB session per call so they're safe to invoke
from inside graph nodes without dependency injection. Tool argument
schemas use Pydantic ``Field`` with explicit descriptions so the calling
LLM sees clear, structured "REQUIRED" guidance.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.db.session import AsyncSessionLocal
from backend.ticket_system import schemas, service
from backend.ticket_system.models import (
    AgentRole,
    SubtaskStatus,
    TicketStatus,
    TodoStatus,
)


def _dump(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(mode="json"), default=str, indent=2)
    return json.dumps(obj, default=str, indent=2)


_DEV_ROLES = {
    AgentRole.BACKEND_DEV,
    AgentRole.FRONTEND_DEV,
    AgentRole.DEVOPS,
    AgentRole.QA,
}


def _parse_dev_role(role: str | None) -> AgentRole | None:
    """Parse an optional dev role; treat empty / 'none' / 'null' as unfiltered."""
    if role is None:
        return None
    cleaned = str(role).strip()
    if not cleaned or cleaned.lower() in {"none", "null"}:
        return None
    parsed = AgentRole(cleaned)
    if parsed not in _DEV_ROLES:
        raise ValueError(
            f"role='{role}' is not a valid dev role. "
            "Use one of: backend_dev, frontend_dev, devops, qa."
        )
    return parsed


def _subtask_tool_payload(sub: Any) -> dict[str, Any]:
    return {
        "id": sub.id,
        "ticket_id": sub.ticket_id,
        "title": sub.title,
        "description": sub.description,
        "required_functionality": sub.required_functionality,
        "test_cases": schemas._normalise_test_cases(sub.test_cases or []),
        "assigned_to": sub.assigned_to.value,
        "order_index": sub.order_index,
        "status": sub.status.value,
    }


@tool
async def list_tickets(project_id: str) -> str:
    """List every ticket in a project as compact rows (id, title, status).

    USE WHEN: You need a backlog overview — ticket UUIDs, titles, statuses, subtask counts.
    AVOID WHEN: You need subtask details or RITE test cases — use get_ticket(ticket_id, detail='full') instead.
    AVOID WHEN: You just called this in a previous turn with the same project_id — skip and use cached results.
    Cost: Always returns the full roster (lightweight, ~200-500 bytes).
    RETURNS: Array of {id, title, status, subtask_count} objects.
    """
    async with AsyncSessionLocal() as db:
        tickets = await service.list_tickets(db, project_id=project_id)
        out = [service.to_dict_ticket_list_item(t) for t in tickets]
        return _dump(out)


@tool
async def get_ticket(ticket_id: str, detail: str = "summary") -> str:
    """Fetch one ticket by UUID.

    USE WHEN: You need to read subtasks or RITE test cases.
    AVOID WHEN: You just need status/subtask_count (use list_tickets instead — avoid this tool call entirely).
    AVOID WHEN: You have a UUID from list_tickets — never guess UUIDs.
    AVOID WHEN: You are the PM making a routing decision — use list_tickets + get_ticket_summary instead.

    detail='summary' returns: {id, title, status, subtask_count, business_requirements, technical_requirements} (cheap, ~500 bytes).
    detail='full' returns: complete subtask trees with RITE test case specs (heavy, ~5-10KB per ticket).

    RETURNS: {id, title, status, subtask_count?, subtasks?, business_requirements?, technical_requirements?}
    ON ERROR (invalid UUID): {error: "No ticket with id ...", suggested_next_tool: "list_tickets"}
    """
    from sqlalchemy.exc import NoResultFound

    mode = (detail or "summary").strip().lower()
    if mode not in {"summary", "full"}:
        return _dump(
            {
                "error": "detail must be 'summary' or 'full'.",
                "received": detail,
            }
        )
    async with AsyncSessionLocal() as db:
        try:
            ticket = await service.get_ticket(db, ticket_id)
        except NoResultFound:
            return _dump(
                {
                    "error": (
                        f"No ticket with id '{ticket_id}' exists. UUIDs cannot be "
                        "guessed — call list_tickets(project_id) to get the real "
                        "ids first, then retry get_ticket with one from that list."
                    ),
                    "suggested_next_tool": "list_tickets",
                }
            )
        if mode == "full":
            return _dump(await service.to_dict_ticket(ticket))
        return _dump(await service.to_dict_ticket_summary(ticket))


@tool
async def get_ticket_summary(ticket_id: str) -> str:
    """Fetch one ticket's summary — no subtask trees, no full RITE specs.

    USE WHEN: You need to inspect a ticket's requirements without loading subtasks.
    AVOID WHEN: You need subtask details — use get_ticket(ticket_id, detail='full') instead.
    AVOID WHEN: You have the UUID — never guess UUIDs; call list_tickets first.

    RETURNS: {id, title, status, business_requirements, technical_requirements}
    """
    from sqlalchemy.exc import NoResultFound

    async with AsyncSessionLocal() as db:
        try:
            ticket = await service.get_ticket(db, ticket_id)
        except NoResultFound:
            return _dump(
                {
                    "error": (
                        f"No ticket with id '{ticket_id}' exists. UUIDs cannot be "
                        "guessed — call list_tickets(project_id) to get the real "
                        "ids first, then retry get_ticket_summary with one from that list."
                    ),
                    "suggested_next_tool": "list_tickets",
                }
            )
        return _dump(await service.to_dict_ticket_summary(ticket))


@tool
async def create_ticket(
    project_id: str,
    title: str,
    description: str,
    business_requirements: list[str],
    technical_requirements: list[str],
) -> str:
    """Create a new ticket in DRAFT status owned by the given project.

    USE WHEN: You are creating a new piece of work from the backlog.
    AVOID WHEN: A ticket with the same title already exists — call list_tickets first; create_ticket is idempotent on (project_id, title).

    Common errors to avoid:
    - Missing business_requirements or technical_requirements — both must be non-empty lists.
    - Using vague titles — be specific, e.g. "Implement JWT auth middleware" not "Add auth".

    RETURNS: {id: "<uuid>", status: "draft", title: "<title>"}
    """
    async with AsyncSessionLocal() as db:
        ticket = await service.create_ticket(
            db,
            schemas.TicketCreate(
                project_id=project_id,
                title=title,
                description=description,
                business_requirements=business_requirements,
                technical_requirements=technical_requirements,
            ),
        )
        return _dump({"id": ticket.id, "status": ticket.status.value, "title": ticket.title})


@tool
async def update_ticket_status(ticket_id: str, status: str) -> str:
    """Transition a ticket to a new status. Valid: draft, in_review, questions_pending, todo, in_progress, done, blocked.

    USE WHEN: You need to advance a ticket's lifecycle (e.g. after planning, after all subtasks done).
    AVOID WHEN: You're not the ticket's planner or the ticket is already in the target status.

    Common errors:
    - Skipping in_review → tickets should pass in_review before going to done.
    - Using invalid status values → only the seven listed above.

    RETURNS: {id: "<uuid>", status: "<new_status>"}
    """
    async with AsyncSessionLocal() as db:
        ticket = await service.update_ticket(
            db,
            ticket_id,
            schemas.TicketUpdate(status=TicketStatus(status)),
        )
        return _dump({"id": ticket.id, "status": ticket.status.value})


@tool
async def add_question_to_ticket(ticket_id: str, question: str, asked_by: str = "project_manager") -> str:
    """Append a clarification question to a ticket and mark it questions_pending."""
    async with AsyncSessionLocal() as db:
        ticket = await service.add_question(db, ticket_id, question, asked_by)
        return _dump({"id": ticket.id, "questions": ticket.questions, "status": ticket.status.value})


class TestCaseInput(BaseModel):
    """A single RITE-format test case the lead must specify per subtask."""

    given: str = Field(
        ...,
        description=(
            "REQUIRED. Natural-language precondition / inputs. Avoid literal "
            "values. Example: 'a subtotal and a coupon-percent discount'."
        ),
    )
    should: str = Field(
        ...,
        description=(
            "REQUIRED. Natural-language expected behaviour. Avoid literal "
            "values. Example: 'return the discounted subtotal'."
        ),
    )
    expected: str = Field(
        ...,
        description=(
            "REQUIRED. The EXPLICIT expected output the dev will hard-code "
            "in the assertion. Literal value, error type, or shape. "
            "Example: '1700' or 'throws InvalidCouponError' or "
            "'{ ok: true, id: <string> }'."
        ),
    )
    test_type: str = Field(
        default="unit",
        description=(
            "One of 'unit' | 'integration' | 'functional'. Default 'unit'. "
            "Prefer unit for pure-function business logic; integration for "
            "real-collaborator wiring; functional for end-to-end user flows."
        ),
    )
    notes: str = Field(
        default="",
        description="Optional. Edge cases, fixtures, or determinism hints.",
    )


class CreateSubtaskArgs(BaseModel):
    """Arguments for create_subtask. Every field labelled REQUIRED must be supplied."""

    ticket_id: str = Field(..., description="REQUIRED. UUID of the parent ticket.")
    title: str = Field(..., description="REQUIRED. Short imperative title.")
    description: str = Field(
        default="", description="Optional longer description of what to build."
    )
    required_functionality: str = Field(
        default="",
        description="What capability this subtask must implement.",
    )
    test_cases: list[TestCaseInput] | None = Field(
        default=None,
        description=(
            "RITE-format test specs. REQUIRED for backend_dev/frontend_dev subtasks "
            "(TDD anchor). OPTIONAL for qa/devops subtasks (they may be checklist-driven "
            "or exploratory / operational)."
        ),
    )
    assigned_to: str = Field(
        ...,
        description=(
            "REQUIRED. The agent role that will execute this subtask. Must be exactly "
            "one of: 'backend_dev', 'frontend_dev', 'devops', 'qa'."
        ),
    )
    order_index: int = Field(
        default=0,
        description=(
            "0-based execution order. Lower runs first. Unique per ticket; "
            "reusing an order_index returns the existing subtask."
        ),
    )


@tool(args_schema=CreateSubtaskArgs)
async def create_subtask(
    ticket_id: str,
    title: str,
    test_cases: list[TestCaseInput] | None,
    assigned_to: str,
    description: str = "",
    required_functionality: str = "",
    order_index: int = 0,
) -> str:
    """Create an ordered subtask under a ticket using RITE-format test cases.

    USE WHEN: You are the lead planning subtasks for a ticket.
    AVOID WHEN: A subtask with the same (ticket_id, order_index) already exists — call get_ticket first; this tool is idempotent on that pair.

    REQUIRED fields (must be supplied in every call):
      - ticket_id: UUID from list_tickets (never guess)
      - title: short imperative, e.g. "Implement auth middleware"
      - test_cases: non-empty list of RITE objects for backend_dev/frontend_dev; empty/None OK for qa/devops
      - assigned_to: exactly one of backend_dev, frontend_dev, devops, qa

    Common errors to avoid:
    - Forgetting assigned_to → always include it
    - Using empty test_cases for dev subtasks → include at least one spec
    - Using invalid role names → only backend_dev, frontend_dev, devops, qa
    - Using non-existent ticket_ids → always copy from list_tickets

    RETURNS: {id: "<uuid>", ticket_id: "<uuid>", order_index: 0, assigned_to: "backend_dev", test_case_count: 3}
    ON ERROR: {error: "..."}
    """
    try:
        role = AgentRole(assigned_to)
    except ValueError:
        return _dump(
            {
                "error": (
                    f"assigned_to='{assigned_to}' is not a valid role. "
                    "Use one of: backend_dev, frontend_dev, devops, qa."
                )
            }
        )
    if role in {AgentRole.BACKEND_DEV, AgentRole.FRONTEND_DEV} and not (test_cases or []):
        return _dump(
            {
                "error": (
                    "test_cases must contain at least one entry for backend_dev/frontend_dev "
                    "(TDD anchor required). For qa/devops subtasks, test_cases may be empty."
                ),
            }
        )
    async with AsyncSessionLocal() as db:
        subtask = await service.create_subtask(
            db,
            ticket_id,
            schemas.SubtaskCreate(
                title=title,
                description=description,
                required_functionality=required_functionality,
                test_cases=[t.model_dump() for t in (test_cases or [])],
                assigned_to=role,
                order_index=order_index,
            ),
        )
        return _dump(
            {
                "id": subtask.id,
                "ticket_id": subtask.ticket_id,
                "order_index": subtask.order_index,
                "assigned_to": subtask.assigned_to.value,
                "test_case_count": len(subtask.test_cases or []),
            }
        )


@tool
async def update_subtask_status(subtask_id: str, status: str) -> str:
    """Update a subtask's status. Valid: pending, in_progress, done, blocked."""
    async with AsyncSessionLocal() as db:
        subtask = await service.update_subtask(
            db,
            subtask_id,
            schemas.SubtaskUpdate(status=SubtaskStatus(status)),
        )
        return _dump({"id": subtask.id, "status": subtask.status.value})


class UpdateSubtaskArgs(BaseModel):
    """Lead-only patch tool for fixing an existing subtask in place."""

    subtask_id: str = Field(..., description="REQUIRED. UUID of the subtask to patch.")
    title: str | None = Field(default=None, description="New short imperative title.")
    description: str | None = Field(default=None, description="New longer description.")
    required_functionality: str | None = Field(
        default=None, description="New summary of what to build."
    )
    test_cases: list[TestCaseInput] | None = Field(
        default=None,
        description=(
            "Replace the entire RITE test_cases list. Provide every spec you "
            "want to keep — anything not in the list is dropped. Use only when "
            "the existing specs are wrong; don't rewrite for stylistic tweaks."
        ),
    )
    assigned_to: str | None = Field(
        default=None,
        description="One of: backend_dev, frontend_dev, devops, qa.",
    )
    order_index: int | None = Field(default=None, description="New 0-based execution order.")


@tool(args_schema=UpdateSubtaskArgs)
async def update_subtask(
    subtask_id: str,
    title: str | None = None,
    description: str | None = None,
    required_functionality: str | None = None,
    test_cases: list[TestCaseInput] | None = None,
    assigned_to: str | None = None,
    order_index: int | None = None,
) -> str:
    """Patch fields on an existing subtask. Lead-only tool.

    Use this when an existing subtask is partially wrong and you want to
    fix it in place rather than delete + recreate. Pass only the fields
    you intend to change. To replace test_cases entirely, pass the full
    new list. Cannot be used to flip status — use update_subtask_status.
    """
    patch_kwargs: dict[str, Any] = {}
    if title is not None:
        patch_kwargs["title"] = title
    if description is not None:
        patch_kwargs["description"] = description
    if required_functionality is not None:
        patch_kwargs["required_functionality"] = required_functionality
    if test_cases is not None:
        patch_kwargs["test_cases"] = [t.model_dump() for t in test_cases]
    if assigned_to is not None:
        try:
            patch_kwargs["assigned_to"] = AgentRole(assigned_to)
        except ValueError:
            return _dump(
                {
                    "error": (
                        f"assigned_to='{assigned_to}' is not a valid role. "
                        "Use one of: backend_dev, frontend_dev, devops, qa."
                    )
                }
            )
    if order_index is not None:
        patch_kwargs["order_index"] = order_index
    if not patch_kwargs:
        return _dump({"error": "no fields supplied to update"})
    async with AsyncSessionLocal() as db:
        subtask = await service.update_subtask(
            db,
            subtask_id,
            schemas.SubtaskUpdate(**patch_kwargs),
        )
        return _dump(
            {
                "id": subtask.id,
                "ticket_id": subtask.ticket_id,
                "title": subtask.title,
                "order_index": subtask.order_index,
                "assigned_to": subtask.assigned_to.value,
                "test_case_count": len(subtask.test_cases or []),
            }
        )


@tool
async def delete_subtask(subtask_id: str) -> str:
    """Delete a subtask that's wrong or no longer needed. Lead-only tool.

    Refuses if the subtask is already in_progress or done — those represent
    real work the dev has started; flip the status back to pending or
    blocked first if you genuinely want to discard them.
    """
    async with AsyncSessionLocal() as db:
        try:
            result = await service.delete_subtask(db, subtask_id)
        except service.TicketStateError as exc:
            return _dump({"error": str(exc)})
        return _dump(result)


@tool
async def add_todo_to_subtask(subtask_id: str, title: str, detail: str = "", order_index: int = 0) -> str:
    """Append a todo (low-level step) to a subtask."""
    async with AsyncSessionLocal() as db:
        todo = await service.create_todo(
            db,
            subtask_id,
            schemas.TodoCreate(title=title, detail=detail, order_index=order_index),
        )
        return _dump({"id": todo.id, "subtask_id": todo.subtask_id, "order_index": todo.order_index})


@tool
async def mark_todo_done(todo_id: str) -> str:
    """Mark a todo as completed."""
    async with AsyncSessionLocal() as db:
        todo = await service.update_todo(
            db, todo_id, schemas.TodoUpdate(status=TodoStatus.DONE)
        )
        return _dump({"id": todo.id, "status": todo.status.value})


@tool
async def next_pending_subtask(ticket_id: str, role: str | None = None) -> str:
    """Return the next actionable subtask for a ticket (resume or start).

    USE WHEN: PM routed you to a specific ticket — pass its UUID here.
    AVOID WHEN: You need project-wide dispatch — use next_pending_subtask_in_project.

    Prefers in_progress (resume), then blocked, then pending (lowest order_index).
    Optional role filter: backend_dev, frontend_dev, devops, qa. Omit role to match any assignee.

    RETURNS: {subtask: {...} | null, resume: true|false}
    ON INVALID ROLE: {error: "role='...' is not a valid dev role..."}
    """
    try:
        parsed_role = _parse_dev_role(role)
    except ValueError as exc:
        return _dump({"error": str(exc)})
    async with AsyncSessionLocal() as db:
        if parsed_role is not None:
            sub = await service.next_pending_subtask_for_role(db, ticket_id, parsed_role)
        else:
            sub = await service.next_pending_subtask(db, ticket_id)
        if sub is None:
            return _dump({"subtask": None, "resume": False})
        return _dump(
            {
                "subtask": _subtask_tool_payload(sub),
                "resume": sub.status in {SubtaskStatus.IN_PROGRESS, SubtaskStatus.BLOCKED},
            }
        )


@tool
async def next_pending_subtask_in_project(
    project_id: str,
    role: str,
    ticket_id: str | None = None,
) -> str:
    """Return the next actionable subtask for a role across the project.

    USE WHEN: You are a developer starting your turn — fetch the next work item.
    USE WHEN: PM gave you a ticket UUID — pass ticket_id to scope lookup (fewer tokens).
    AVOID WHEN: You've already fetched a subtask this turn — call it once per turn.

    Prefers in_progress, then blocked, then pending. Ordering uses ticket.order_index,
    then subtask.order_index.

    RETURNS: {ticket: {...}|null, subtask: {...}|null, resume: true|false}
    ON NO MORE WORK: {ticket: null, subtask: null, resume: false}
    ON ERROR (invalid role): {error: "role='...' is not a valid dev role..."}
    """
    try:
        r = _parse_dev_role(role)
        if r is None:
            raise ValueError(
                f"role='{role}' is not a valid dev role. "
                "Use one of: backend_dev, frontend_dev, devops, qa."
            )
    except ValueError as exc:
        return _dump({"error": str(exc)})
    async with AsyncSessionLocal() as db:
        row = await service.next_pending_subtask_in_project(
            db, project_id=project_id, role=r, ticket_id=ticket_id or None
        )
        if row is None:
            return _dump({"ticket": None, "subtask": None, "resume": False})
        ticket, sub = row
        return _dump(
            {
                "ticket": {
                    "id": ticket.id,
                    "title": ticket.title,
                    "status": ticket.status.value,
                    "order_index": ticket.order_index,
                },
                "subtask": _subtask_tool_payload(sub),
                "resume": sub.status in {SubtaskStatus.IN_PROGRESS, SubtaskStatus.BLOCKED},
            }
        )


PM_TICKET_TOOLS = [
    list_tickets,
    get_ticket,
    create_ticket,
    update_ticket_status,
    add_question_to_ticket,
]

LEAD_TICKET_TOOLS = [
    list_tickets,
    get_ticket,
    create_subtask,
    update_subtask,
    delete_subtask,
    update_subtask_status,
    add_todo_to_subtask,
    update_ticket_status,
]

DEV_TICKET_TOOLS = [
    get_ticket,
    next_pending_subtask,
    next_pending_subtask_in_project,
    update_subtask_status,
    mark_todo_done,
    add_todo_to_subtask,
]


@tool
async def list_subtasks(
    project_id: str,
    ticket_id: str | None = None,
    assigned_to: str | None = None,
    status: str | None = None,
) -> str:
    """List subtasks filtered by project_id and optionally ticket_id and/or assigned_to.

    Intended for PM review, dashboards, and automation.
    """
    role: AgentRole | None = None
    if assigned_to is not None:
        try:
            role = AgentRole(assigned_to)
        except ValueError:
            return _dump(
                {
                    "error": (
                        f"assigned_to='{assigned_to}' is not a valid role. "
                        "Use one of: backend_dev, frontend_dev, devops, qa."
                    )
                }
            )
    st: SubtaskStatus | None = None
    if status is not None:
        try:
            st = SubtaskStatus(status)
        except ValueError:
            return _dump(
                {
                    "error": (
                        f"status='{status}' is not a valid subtask status. "
                        "Use one of: pending, in_progress, done, blocked."
                    )
                }
            )
    async with AsyncSessionLocal() as db:
        rows = await service.list_subtasks(
            db,
            project_id=project_id,
            ticket_id=ticket_id,
            assigned_to=role,
            status=st,
        )
        out = [
            {
                "id": s.id,
                "ticket_id": s.ticket_id,
                "title": s.title,
                "assigned_to": s.assigned_to.value,
                "order_index": s.order_index,
                "status": s.status.value,
                "test_case_count": len(s.test_cases or []),
            }
            for s in rows
        ]
        return _dump(out)
