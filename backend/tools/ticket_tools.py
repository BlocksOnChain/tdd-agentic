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


@tool
async def list_tickets(project_id: str) -> str:
    """List every ticket in a project as compact rows (id, title, status).

    This intentionally omits full descriptions, requirements, and subtask
    payloads so the result is not truncated mid-list. Use ``get_ticket``
    for detailed work on a single ticket.
    """
    async with AsyncSessionLocal() as db:
        tickets = await service.list_tickets(db, project_id=project_id)
        out = [service.to_dict_ticket_list_item(t) for t in tickets]
        return _dump(out)


@tool
async def get_ticket(ticket_id: str) -> str:
    """Fetch a single ticket with its subtasks and todos."""
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
                        "ids first, then retry get_ticket with one from that list."
                    ),
                    "suggested_next_tool": "list_tickets",
                }
            )
        return _dump(await service.to_dict_ticket(ticket))


@tool
async def create_ticket(
    project_id: str,
    title: str,
    description: str,
    business_requirements: list[str],
    technical_requirements: list[str],
) -> str:
    """Create a new ticket in DRAFT status owned by the given project."""
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
    """Transition a ticket to a new status. Valid: draft, in_review, questions_pending, todo, in_progress, done, blocked."""
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
        description="0-based execution order. Lower runs first.",
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

    test_cases must be a list of objects with {given, should, expected,
    test_type?, notes?}. The dev will translate each spec into one
    assert({given, should, actual, expected}) call.
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
    """Return the next pending subtask (lowest order_index) for a ticket, optionally filtered by agent role."""
    async with AsyncSessionLocal() as db:
        if role:
            sub = await service.next_pending_subtask_for_role(db, ticket_id, AgentRole(role))
        else:
            sub = await service.next_pending_subtask(db, ticket_id)
        if sub is None:
            return _dump({"subtask": None})
        return _dump(
            {
                "subtask": {
                    "id": sub.id,
                    "title": sub.title,
                    "description": sub.description,
                    "required_functionality": sub.required_functionality,
                    "test_cases": schemas._normalise_test_cases(sub.test_cases or []),
                    "assigned_to": sub.assigned_to.value,
                    "order_index": sub.order_index,
                    "status": sub.status.value,
                }
            }
        )


@tool
async def next_pending_subtask_in_project(project_id: str, role: str) -> str:
    """Return the next pending subtask for a role across the whole project.

    Uses ticket.order_index first, then subtask.order_index.
    """
    try:
        r = AgentRole(role)
    except ValueError:
        return _dump(
            {
                "error": (
                    f"role='{role}' is not a valid role. "
                    "Use one of: backend_dev, frontend_dev, devops, qa."
                )
            }
        )
    async with AsyncSessionLocal() as db:
        row = await service.next_pending_subtask_in_project(db, project_id=project_id, role=r)
        if row is None:
            return _dump({"ticket": None, "subtask": None})
        ticket, sub = row
        return _dump(
            {
                "ticket": {
                    "id": ticket.id,
                    "title": ticket.title,
                    "status": ticket.status.value,
                    "order_index": ticket.order_index,
                },
                "subtask": {
                    "id": sub.id,
                    "ticket_id": sub.ticket_id,
                    "title": sub.title,
                    "description": sub.description,
                    "required_functionality": sub.required_functionality,
                    "test_cases": schemas._normalise_test_cases(sub.test_cases or []),
                    "assigned_to": sub.assigned_to.value,
                    "order_index": sub.order_index,
                    "status": sub.status.value,
                },
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
