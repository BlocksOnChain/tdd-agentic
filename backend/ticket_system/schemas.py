"""Pydantic schemas for ticket-system API requests and responses."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.ticket_system.models import (
    AgentRole,
    SubtaskStatus,
    TicketStatus,
    TodoStatus,
)


class _ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


# ===== RITE test-case spec =====
TestType = Literal["unit", "integration", "functional"]


class TestCaseSpec(BaseModel):
    """A single test case in RITE format.

    Encodes the 5 questions every test must answer:
      - unit_under_test (subtask context)
      - given: input/precondition (natural language, no literals)
      - should: expected behaviour (natural language, no literals)
      - expected: the EXPLICIT, hard-coded expected value or shape
      - test_type: unit | integration | functional
      - notes: optional free text (edge cases, fixtures, etc.)
    """

    model_config = ConfigDict(extra="forbid")

    given: str = Field(..., description="Precondition / inputs in natural language.")
    should: str = Field(..., description="Expected observable behaviour in natural language.")
    expected: str = Field(
        ...,
        description=(
            "The EXPLICIT expected output (literal value, error type, or shape)."
            " The dev hard-codes this in the assertion."
        ),
    )
    test_type: TestType = Field(default="unit")
    notes: str = Field(default="")


def _normalise_test_cases(value: object) -> list[dict]:
    """Accept either plain strings (legacy) or structured dicts and
    return a list of normalised dict specs.
    Plain strings become unit tests with the string as ``should``.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("test_cases must be a list")
    out: list[dict] = []
    for item in value:
        if isinstance(item, str):
            out.append(
                TestCaseSpec(
                    given="(see subtask description)",
                    should=item.strip(),
                    expected="(define explicit expected value when writing the test)",
                    test_type="unit",
                ).model_dump()
            )
        elif isinstance(item, dict):
            out.append(TestCaseSpec(**item).model_dump())
        elif isinstance(item, TestCaseSpec):
            out.append(item.model_dump())
        else:
            raise TypeError(f"unsupported test-case item: {type(item)!r}")
    return out


# ===== Project =====
class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    goal: str = ""


class ProjectOut(_ORM):
    id: str
    name: str
    description: str
    goal: str
    workspace_path: str | None
    created_at: datetime
    updated_at: datetime


# ===== Todo =====
class TodoCreate(BaseModel):
    title: str
    detail: str = ""
    order_index: int = 0


class TodoOut(_ORM):
    id: str
    subtask_id: str
    title: str
    detail: str
    order_index: int
    status: TodoStatus


class TodoUpdate(BaseModel):
    title: str | None = None
    detail: str | None = None
    order_index: int | None = None
    status: TodoStatus | None = None


# ===== Subtask =====
class SubtaskCreate(BaseModel):
    title: str
    description: str = ""
    required_functionality: str = ""
    # Note: the field is typed `list[dict]` so Pydantic preserves whatever
    # the ``before`` validator returns (plain JSON-serialisable dicts) —
    # otherwise it would try to coerce dicts back into ``TestCaseSpec``
    # model instances, which SQLAlchemy's JSON column can't serialise.
    test_cases: list[dict] = Field(default_factory=list)
    assigned_to: AgentRole = AgentRole.BACKEND_DEV
    order_index: int = 0
    todos: list[TodoCreate] = Field(default_factory=list)

    @field_validator("test_cases", mode="before")
    @classmethod
    def _coerce_test_cases(cls, v: object) -> list[dict]:
        return _normalise_test_cases(v)


class SubtaskOut(_ORM):
    id: str
    ticket_id: str
    title: str
    description: str
    required_functionality: str
    test_cases: list[dict]
    assigned_to: AgentRole
    order_index: int
    status: SubtaskStatus
    todos: list[TodoOut] = Field(default_factory=list)

    @field_validator("test_cases", mode="before")
    @classmethod
    def _coerce_test_cases(cls, v: object) -> list[dict]:
        return _normalise_test_cases(v)


class SubtaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    required_functionality: str | None = None
    test_cases: list[dict] | None = None
    assigned_to: AgentRole | None = None
    order_index: int | None = None
    status: SubtaskStatus | None = None

    @field_validator("test_cases", mode="before")
    @classmethod
    def _coerce_test_cases(cls, v: object) -> list[dict] | None:
        if v is None:
            return None
        return _normalise_test_cases(v)


# ===== Ticket =====
class TicketCreate(BaseModel):
    project_id: str
    title: str
    description: str = ""
    business_requirements: list[str] = Field(default_factory=list)
    technical_requirements: list[str] = Field(default_factory=list)
    order_index: int = 0


class TicketQuestion(BaseModel):
    question: str
    answer: str | None = None
    asked_by: str = "project_manager"
    answered_by: str | None = None
    ts: float | None = None


class TicketOut(_ORM):
    id: str
    project_id: str
    title: str
    description: str
    business_requirements: list[str]
    technical_requirements: list[str]
    status: TicketStatus
    questions: list[TicketQuestion]
    order_index: int
    created_at: datetime
    updated_at: datetime
    subtasks: list[SubtaskOut] = Field(default_factory=list)


class TicketUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    business_requirements: list[str] | None = None
    technical_requirements: list[str] | None = None
    status: TicketStatus | None = None
    order_index: int | None = None


class AnswerQuestion(BaseModel):
    question_index: int
    answer: str
    answered_by: str = "human"
