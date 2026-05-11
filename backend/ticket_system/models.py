"""SQLAlchemy ORM models for the ticket platform.

Hierarchy: Project → Ticket → Subtask → Todo
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TicketStatus(str, enum.Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    QUESTIONS_PENDING = "questions_pending"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class SubtaskStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class TodoStatus(str, enum.Enum):
    PENDING = "pending"
    DONE = "done"


class AgentRole(str, enum.Enum):
    PROJECT_MANAGER = "project_manager"
    RESEARCHER = "researcher"
    BACKEND_LEAD = "backend_lead"
    FRONTEND_LEAD = "frontend_lead"
    BACKEND_DEV = "backend_dev"
    FRONTEND_DEV = "frontend_dev"
    DEVOPS = "devops"
    QA = "qa"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    goal: Mapped[str] = mapped_column(Text, default="")
    workspace_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    tickets: Mapped[list["Ticket"]] = relationship(
        "Ticket", back_populates="project", cascade="all, delete-orphan"
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    business_requirements: Mapped[list[str]] = mapped_column(JSON, default=list)
    technical_requirements: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[TicketStatus] = mapped_column(
        Enum(TicketStatus, name="ticket_status"), default=TicketStatus.DRAFT, index=True
    )
    questions: Mapped[list[dict]] = mapped_column(JSON, default=list)
    """List of {question, answer|null, asked_by, answered_by, ts}."""
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    project: Mapped[Project] = relationship("Project", back_populates="tickets")
    subtasks: Mapped[list["Subtask"]] = relationship(
        "Subtask",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="Subtask.order_index",
    )


class Subtask(Base):
    __tablename__ = "subtasks"
    __table_args__ = (
        UniqueConstraint("ticket_id", "order_index", name="uq_subtasks_ticket_order_index"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    ticket_id: Mapped[str] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    required_functionality: Mapped[str] = mapped_column(Text, default="")
    test_cases: Mapped[list[dict]] = mapped_column(JSON, default=list)
    """RITE test specs: ``[{given, should, expected, test_type, notes}, ...]``.

    Plain strings from older rows are still accepted on read; the API layer
    normalises them into the structured form via ``_normalise_test_cases``.
    """
    assigned_to: Mapped[AgentRole] = mapped_column(
        Enum(AgentRole, name="agent_role"), default=AgentRole.BACKEND_DEV
    )
    order_index: Mapped[int] = mapped_column(Integer, default=0, index=True)
    status: Mapped[SubtaskStatus] = mapped_column(
        Enum(SubtaskStatus, name="subtask_status"), default=SubtaskStatus.PENDING, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    ticket: Mapped[Ticket] = relationship("Ticket", back_populates="subtasks")
    todos: Mapped[list["Todo"]] = relationship(
        "Todo",
        back_populates="subtask",
        cascade="all, delete-orphan",
        order_by="Todo.order_index",
    )


class Todo(Base):
    __tablename__ = "todos"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    subtask_id: Mapped[str] = mapped_column(
        ForeignKey("subtasks.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    detail: Mapped[str] = mapped_column(Text, default="")
    order_index: Mapped[int] = mapped_column(Integer, default=0, index=True)
    status: Mapped[TodoStatus] = mapped_column(
        Enum(TodoStatus, name="todo_status"), default=TodoStatus.PENDING, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    subtask: Mapped[Subtask] = relationship("Subtask", back_populates="todos")


class AgentLog(Base):
    """Persisted log of every agent action — duplicates EventBus for replay."""

    __tablename__ = "agent_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    project_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    agent: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    ticket_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    subtask_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


__all__ = [
    "Project",
    "Ticket",
    "Subtask",
    "Todo",
    "AgentLog",
    "TicketStatus",
    "SubtaskStatus",
    "TodoStatus",
    "AgentRole",
]
