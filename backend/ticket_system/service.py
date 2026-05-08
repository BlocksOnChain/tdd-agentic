"""Ticket / subtask state-machine service layer.

Encapsulates all reads, writes, and transitions; the API routes and the
LangChain tools call into this module so behaviour stays consistent.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from backend.config import get_settings

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.events import Event, bus
from backend.ticket_system import schemas
from backend.ticket_system.models import (
    AgentRole,
    Project,
    Subtask,
    SubtaskStatus,
    Ticket,
    TicketStatus,
    Todo,
    TodoStatus,
)

# ===== state-machine transitions =====
ALLOWED_TICKET_TRANSITIONS: dict[TicketStatus, set[TicketStatus]] = {
    TicketStatus.DRAFT: {TicketStatus.IN_REVIEW, TicketStatus.QUESTIONS_PENDING},
    TicketStatus.IN_REVIEW: {
        TicketStatus.QUESTIONS_PENDING,
        TicketStatus.TODO,
        TicketStatus.DRAFT,
    },
    TicketStatus.QUESTIONS_PENDING: {TicketStatus.IN_REVIEW, TicketStatus.DRAFT},
    TicketStatus.TODO: {TicketStatus.IN_PROGRESS, TicketStatus.IN_REVIEW},
    TicketStatus.IN_PROGRESS: {TicketStatus.DONE, TicketStatus.BLOCKED},
    TicketStatus.BLOCKED: {TicketStatus.IN_PROGRESS, TicketStatus.IN_REVIEW},
    TicketStatus.DONE: set(),
}


class TicketStateError(ValueError):
    """Raised when a status transition is not permitted."""


# ===== Workspace seed (AGENTS.md, docs/) =====
_WORKSPACE_SEED_DIR = Path(__file__).resolve().parent.parent / "workspace_seed"


def seed_project_workspace(project_id: str) -> None:
    """Lay down AGENTS.md, docs/, and conventions so agents never cite node_modules."""
    settings = get_settings()
    dest = settings.workspace_root / project_id
    try:
        if dest.exists():
            return
        if _WORKSPACE_SEED_DIR.is_dir():
            shutil.copytree(_WORKSPACE_SEED_DIR, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Workspace is optional at create time; agents can mkdir via tools later.
        pass


# ===== Project =====
async def create_project(db: AsyncSession, payload: schemas.ProjectCreate) -> Project:
    project = Project(name=payload.name, description=payload.description, goal=payload.goal)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    seed_project_workspace(project.id)
    await bus.publish(Event(type="project", payload={"action": "created", "id": project.id}))
    return project


async def list_projects(db: AsyncSession) -> list[Project]:
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    return list(result.scalars().all())


async def get_project(db: AsyncSession, project_id: str) -> Project:
    project = await db.get(Project, project_id)
    if project is None:
        raise NoResultFound(f"Project {project_id} not found")
    return project


# ===== Ticket =====
async def create_ticket(db: AsyncSession, payload: schemas.TicketCreate) -> Ticket:
    """Create a ticket, or return the existing one if a same-title ticket
    already exists in this project (case-insensitive). Idempotent so an
    agent re-running after a crash doesn't produce duplicates.
    """
    await get_project(db, payload.project_id)

    existing_stmt = (
        select(Ticket)
        .options(selectinload(Ticket.subtasks).selectinload(Subtask.todos))
        .where(
            Ticket.project_id == payload.project_id,
            Ticket.title.ilike(payload.title.strip()),
        )
        .limit(1)
    )
    existing = (await db.execute(existing_stmt)).scalar_one_or_none()
    if existing is not None:
        return existing

    ticket = Ticket(
        project_id=payload.project_id,
        title=payload.title,
        description=payload.description,
        business_requirements=list(payload.business_requirements),
        technical_requirements=list(payload.technical_requirements),
        order_index=payload.order_index,
        status=TicketStatus.DRAFT,
        questions=[],
    )
    db.add(ticket)
    await db.commit()
    ticket = await get_ticket(db, ticket.id)
    await bus.publish(
        Event(
            type="ticket",
            project_id=ticket.project_id,
            payload={"action": "created", "id": ticket.id, "status": ticket.status.value},
        )
    )
    return ticket


async def list_tickets(db: AsyncSession, project_id: str | None = None) -> list[Ticket]:
    stmt = (
        select(Ticket)
        .options(selectinload(Ticket.subtasks).selectinload(Subtask.todos))
        .order_by(Ticket.order_index, Ticket.created_at)
    )
    if project_id is not None:
        stmt = stmt.where(Ticket.project_id == project_id)
    result = await db.execute(stmt)
    return list(result.scalars().unique().all())


async def get_ticket(db: AsyncSession, ticket_id: str) -> Ticket:
    stmt = (
        select(Ticket)
        .options(selectinload(Ticket.subtasks).selectinload(Subtask.todos))
        .where(Ticket.id == ticket_id)
    )
    result = await db.execute(stmt)
    ticket = result.scalar_one_or_none()
    if ticket is None:
        raise NoResultFound(f"Ticket {ticket_id} not found")
    return ticket


async def update_ticket(
    db: AsyncSession, ticket_id: str, patch: schemas.TicketUpdate
) -> Ticket:
    ticket = await get_ticket(db, ticket_id)

    if patch.status is not None and patch.status != ticket.status:
        if patch.status not in ALLOWED_TICKET_TRANSITIONS[ticket.status]:
            raise TicketStateError(
                f"Cannot transition ticket from {ticket.status.value} to {patch.status.value}"
            )

    data = patch.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(ticket, k, v)
    await db.commit()
    await db.refresh(ticket)
    await bus.publish(
        Event(
            type="ticket",
            project_id=ticket.project_id,
            payload={"action": "updated", "id": ticket.id, "status": ticket.status.value},
        )
    )
    return ticket


async def add_question(db: AsyncSession, ticket_id: str, question: str, asked_by: str) -> Ticket:
    ticket = await get_ticket(db, ticket_id)
    qlist = list(ticket.questions or [])
    qlist.append(
        {"question": question, "answer": None, "asked_by": asked_by, "answered_by": None,
         "ts": time.time()}
    )
    ticket.questions = qlist
    if ticket.status in ALLOWED_TICKET_TRANSITIONS and TicketStatus.QUESTIONS_PENDING in (
        ALLOWED_TICKET_TRANSITIONS.get(ticket.status, set())
    ):
        ticket.status = TicketStatus.QUESTIONS_PENDING
    await db.commit()
    await bus.publish(
        Event(
            type="interrupt",
            project_id=ticket.project_id,
            payload={
                "kind": "question",
                "ticket_id": ticket.id,
                "question": question,
                "asked_by": asked_by,
            },
        )
    )
    return await get_ticket(db, ticket_id)


async def answer_question(
    db: AsyncSession, ticket_id: str, payload: schemas.AnswerQuestion
) -> Ticket:
    ticket = await get_ticket(db, ticket_id)
    qlist = list(ticket.questions or [])
    if payload.question_index < 0 or payload.question_index >= len(qlist):
        raise IndexError("question_index out of range")
    q = dict(qlist[payload.question_index])
    q["answer"] = payload.answer
    q["answered_by"] = payload.answered_by
    qlist[payload.question_index] = q
    ticket.questions = qlist

    if all(q.get("answer") for q in qlist) and ticket.status == TicketStatus.QUESTIONS_PENDING:
        ticket.status = TicketStatus.IN_REVIEW

    await db.commit()
    await bus.publish(
        Event(
            type="ticket",
            project_id=ticket.project_id,
            payload={"action": "answered", "id": ticket.id, "index": payload.question_index},
        )
    )
    return await get_ticket(db, ticket_id)


# ===== Subtask =====
async def create_subtask(
    db: AsyncSession, ticket_id: str, payload: schemas.SubtaskCreate
) -> Subtask:
    """Create a subtask, or return the existing one if a same-title
    subtask already exists on this ticket (case-insensitive). Idempotent.
    """
    ticket = await get_ticket(db, ticket_id)

    existing_stmt = (
        select(Subtask)
        .options(selectinload(Subtask.todos))
        .where(
            Subtask.ticket_id == ticket.id,
            Subtask.title.ilike(payload.title.strip()),
        )
        .limit(1)
    )
    existing = (await db.execute(existing_stmt)).scalar_one_or_none()
    if existing is not None:
        return existing

    subtask = Subtask(
        ticket_id=ticket.id,
        title=payload.title,
        description=payload.description,
        required_functionality=payload.required_functionality,
        test_cases=list(payload.test_cases),
        assigned_to=payload.assigned_to,
        order_index=payload.order_index,
        status=SubtaskStatus.PENDING,
    )
    db.add(subtask)
    await db.flush()
    for todo in payload.todos:
        db.add(
            Todo(
                subtask_id=subtask.id,
                title=todo.title,
                detail=todo.detail,
                order_index=todo.order_index,
                status=TodoStatus.PENDING,
            )
        )
    await db.commit()
    await db.refresh(subtask)
    await bus.publish(
        Event(
            type="ticket",
            project_id=ticket.project_id,
            payload={
                "action": "subtask_created",
                "ticket_id": ticket.id,
                "subtask_id": subtask.id,
            },
        )
    )
    return subtask


async def get_subtask(db: AsyncSession, subtask_id: str) -> Subtask:
    stmt = (
        select(Subtask)
        .options(selectinload(Subtask.todos))
        .where(Subtask.id == subtask_id)
    )
    result = await db.execute(stmt)
    sub = result.scalar_one_or_none()
    if sub is None:
        raise NoResultFound(f"Subtask {subtask_id} not found")
    return sub


async def update_subtask(
    db: AsyncSession, subtask_id: str, patch: schemas.SubtaskUpdate
) -> Subtask:
    subtask = await get_subtask(db, subtask_id)
    data = patch.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(subtask, k, v)
    await db.commit()
    await db.refresh(subtask)
    await bus.publish(
        Event(
            type="ticket",
            payload={
                "action": "subtask_updated",
                "subtask_id": subtask.id,
                "status": subtask.status.value,
            },
        )
    )
    return subtask


async def delete_subtask(db: AsyncSession, subtask_id: str) -> dict[str, Any]:
    """Delete a subtask (and its todos via cascade).

    Refuses to delete subtasks that are already in_progress or done so we
    don't silently throw away completed work; the caller must explicitly
    transition the subtask back to pending or blocked first.
    """
    subtask = await get_subtask(db, subtask_id)
    if subtask.status in {SubtaskStatus.IN_PROGRESS, SubtaskStatus.DONE}:
        raise TicketStateError(
            f"Cannot delete subtask {subtask_id} in status {subtask.status.value}; "
            "set status back to 'pending' or 'blocked' first if you really intend to remove it."
        )
    ticket_id = subtask.ticket_id
    title = subtask.title
    parent_ticket = await db.get(Ticket, ticket_id)
    project_id = parent_ticket.project_id if parent_ticket is not None else None
    await db.delete(subtask)
    await db.commit()
    await bus.publish(
        Event(
            type="ticket",
            project_id=project_id,
            payload={
                "action": "subtask_deleted",
                "ticket_id": ticket_id,
                "subtask_id": subtask_id,
                "title": title,
            },
        )
    )
    return {"id": subtask_id, "ticket_id": ticket_id, "deleted": True}


async def next_pending_subtask(db: AsyncSession, ticket_id: str) -> Subtask | None:
    """Return the lowest-order pending subtask for a ticket, or None."""
    stmt = (
        select(Subtask)
        .where(Subtask.ticket_id == ticket_id, Subtask.status == SubtaskStatus.PENDING)
        .order_by(Subtask.order_index)
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def next_pending_subtask_for_role(
    db: AsyncSession, ticket_id: str, role: AgentRole
) -> Subtask | None:
    stmt = (
        select(Subtask)
        .where(
            Subtask.ticket_id == ticket_id,
            Subtask.status == SubtaskStatus.PENDING,
            Subtask.assigned_to == role,
        )
        .order_by(Subtask.order_index)
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


# ===== Todo =====
async def create_todo(
    db: AsyncSession, subtask_id: str, payload: schemas.TodoCreate
) -> Todo:
    await get_subtask(db, subtask_id)
    todo = Todo(
        subtask_id=subtask_id,
        title=payload.title,
        detail=payload.detail,
        order_index=payload.order_index,
        status=TodoStatus.PENDING,
    )
    db.add(todo)
    await db.commit()
    await db.refresh(todo)
    return todo


async def update_todo(db: AsyncSession, todo_id: str, patch: schemas.TodoUpdate) -> Todo:
    todo = await db.get(Todo, todo_id)
    if todo is None:
        raise NoResultFound(f"Todo {todo_id} not found")
    data = patch.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(todo, k, v)
    await db.commit()
    await db.refresh(todo)
    return todo


# ===== Helpers used by agents =====
async def all_subtasks_done(db: AsyncSession, ticket_id: str) -> bool:
    stmt = select(Subtask.status).where(Subtask.ticket_id == ticket_id)
    result = await db.execute(stmt)
    statuses = [r[0] for r in result.all()]
    return bool(statuses) and all(s == SubtaskStatus.DONE for s in statuses)


async def to_dict_ticket(ticket: Ticket) -> dict[str, Any]:
    return {
        "id": ticket.id,
        "project_id": ticket.project_id,
        "title": ticket.title,
        "description": ticket.description,
        "business_requirements": list(ticket.business_requirements or []),
        "technical_requirements": list(ticket.technical_requirements or []),
        "status": ticket.status.value,
        "questions": list(ticket.questions or []),
        "subtasks": [
            {
                "id": s.id,
                "title": s.title,
                "description": s.description,
                "required_functionality": s.required_functionality,
                "test_cases": schemas._normalise_test_cases(s.test_cases or []),
                "assigned_to": s.assigned_to.value,
                "order_index": s.order_index,
                "status": s.status.value,
                "todos": [
                    {
                        "id": t.id,
                        "title": t.title,
                        "detail": t.detail,
                        "order_index": t.order_index,
                        "status": t.status.value,
                    }
                    for t in s.todos
                ],
            }
            for s in ticket.subtasks
        ],
    }


def to_dict_ticket_list_item(ticket: Ticket) -> dict[str, Any]:
    """Lightweight row for bulk listing — keeps tool outputs under truncation limits."""

    subs = list(ticket.subtasks or [])
    return {
        "id": ticket.id,
        "title": ticket.title,
        "status": ticket.status.value,
        "order_index": ticket.order_index,
        "subtask_count": len(subs),
        "description_preview": (ticket.description or "")[:260],
        "hint": (
            "Call get_ticket(<id>) for requirements and full subtask trees."
        ),
    }
