"""Ticket / subtask / todo CRUD routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import get_db
from backend.ticket_system import schemas, service
from backend.ticket_system.models import AgentRole, SubtaskStatus

router = APIRouter()


# ===== tickets =====
@router.post("", response_model=schemas.TicketOut)
async def create_ticket(payload: schemas.TicketCreate, db: AsyncSession = Depends(get_db)):
    return await service.create_ticket(db, payload)


@router.get("", response_model=list[schemas.TicketOut])
async def list_tickets(project_id: str | None = None, db: AsyncSession = Depends(get_db)):
    return await service.list_tickets(db, project_id)


@router.get("/{ticket_id}", response_model=schemas.TicketOut)
async def get_ticket(ticket_id: str, db: AsyncSession = Depends(get_db)):
    try:
        return await service.get_ticket(db, ticket_id)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{ticket_id}", response_model=schemas.TicketOut)
async def update_ticket(
    ticket_id: str,
    patch: schemas.TicketUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.update_ticket(db, ticket_id, patch)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.TicketStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{ticket_id}/answer", response_model=schemas.TicketOut)
async def answer(
    ticket_id: str,
    payload: schemas.AnswerQuestion,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.answer_question(db, ticket_id, payload)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ===== subtasks =====
@router.get("/subtasks", response_model=list[schemas.SubtaskOut])
async def list_subtasks(
    project_id: str,
    ticket_id: str | None = None,
    assigned_to: AgentRole | None = None,
    status: SubtaskStatus | None = None,
    db: AsyncSession = Depends(get_db),
):
    return await service.list_subtasks(
        db,
        project_id=project_id,
        ticket_id=ticket_id,
        assigned_to=assigned_to,
        status=status,
    )


@router.post("/{ticket_id}/subtasks", response_model=schemas.SubtaskOut)
async def create_subtask(
    ticket_id: str,
    payload: schemas.SubtaskCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.create_subtask(db, ticket_id, payload)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/subtasks/{subtask_id}", response_model=schemas.SubtaskOut)
async def update_subtask(
    subtask_id: str,
    patch: schemas.SubtaskUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.update_subtask(db, subtask_id, patch)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ===== todos =====
@router.post("/subtasks/{subtask_id}/todos", response_model=schemas.TodoOut)
async def create_todo(
    subtask_id: str,
    payload: schemas.TodoCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.create_todo(db, subtask_id, payload)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/todos/{todo_id}", response_model=schemas.TodoOut)
async def update_todo(
    todo_id: str,
    patch: schemas.TodoUpdate,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await service.update_todo(db, todo_id, patch)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
