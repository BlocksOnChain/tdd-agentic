"""Persistence tools for Coordinator agent.

These tools handle all database writes. The Coordinator is a non-cognitive agent
that simply executes these operations based on execution_plan from Lead.
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
)


def _dump(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(mode="json"), default=str, indent=2)
    return json.dumps(obj, default=str, indent=2)


class CreateTicketArgs(BaseModel):
    project_id: str = Field(..., description="UUID of the parent project.")
    title: str = Field(..., description="Short imperative title.")
    description: str = Field(default="", description="Optional longer description.")
    business_requirements: list[str] = Field(default_factory=list, description="Business requirements.")
    technical_requirements: list[str] = Field(default_factory=list, description="Technical requirements.")


@tool(args_schema=CreateTicketArgs)
async def save_ticket(
    project_id: str,
    title: str,
    description: str = "",
    business_requirements: list[str] = [],
    technical_requirements: list[str] = [],
) -> str:
    """Create or update a ticket in the database.

    USE WHEN: Persisting a new ticket or updating existing ticket metadata.
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


class TransitionTicketArgs(BaseModel):
    ticket_id: str = Field(..., description="UUID of the ticket to transition.")
    status: str = Field(..., description="Target status: draft, in_review, questions_pending, todo, in_progress, done, blocked.")


@tool(args_schema=TransitionTicketArgs)
async def transition_ticket(ticket_id: str, status: str) -> str:
    """Transition a ticket to a new status.

    USE WHEN: Moving tickets through the workflow (e.g., draft → in_review → todo).
    RETURNS: {id: "<uuid>", status: "<new_status>"}
    """
    async with AsyncSessionLocal() as db:
        from backend.ticket_system.models import TicketStatus

        ticket = await service.update_ticket(
            db,
            ticket_id,
            schemas.TicketUpdate(status=TicketStatus(status)),
        )
        return _dump({"id": ticket.id, "status": ticket.status.value})


class SaveExecutionPlanArgs(BaseModel):
    project_id: str = Field(..., description="UUID of the parent project.")
    ticket_id: str | None = Field(default=None, description="UUID of existing ticket, or None to create new.")
    subtasks: list[dict] = Field(default_factory=list, description="List of subtask plans.")


@tool(args_schema=SaveExecutionPlanArgs)
async def save_execution_plan(
    project_id: str,
    subtasks: list[dict],
    ticket_id: str | None = None,
) -> str:
    """Persist an execution plan to the database.

    If ticket_id is provided, creates subtasks under that ticket.
    If ticket_id is None, creates a new ticket first, then adds subtasks.

    USE WHEN: Coordinator receives execution_plan from Lead agent.
    RETURNS: {ticket_id: "<uuid>", subtask_count: N, subtask_ids: [...]}
    """
    async with AsyncSessionLocal() as db:
        # Determine target ticket
        if ticket_id is None:
            # Create new ticket first
            # Extract title from first subtask or generate one
            title = subtasks[0].get("title", "New Ticket") if subtasks else "New Ticket"
            ticket = await service.create_ticket(
                db,
                schemas.TicketCreate(
                    project_id=project_id,
                    title=title,
                    description="",
                    business_requirements=[],
                    technical_requirements=[],
                ),
            )
            ticket_id = ticket.id
        else:
            # Verify ticket exists
            try:
                ticket = await service.get_ticket(db, ticket_id)
            except Exception:
                return _dump({"error": f"No ticket with id '{ticket_id}'"})

        # Create subtasks
        created_subtask_ids: list[str] = []
        for idx, subtask_data in enumerate(subtasks):
            assigned_to_str = subtask_data.get("assigned_to", "backend_dev")
            try:
                assigned_to = AgentRole(assigned_to_str)
            except ValueError:
                assigned_to = AgentRole.BACKEND_DEV

            test_cases_raw = subtask_data.get("test_cases") or []
            # Normalize test cases - extract fields from dict if needed
            test_cases: list[dict] = []
            for tc in test_cases_raw:
                if isinstance(tc, dict):
                    test_cases.append({
                        "given": tc.get("given", ""),
                        "should": tc.get("should", ""),
                        "expected": tc.get("expected", ""),
                        "test_type": tc.get("test_type", "unit"),
                        "notes": tc.get("notes", ""),
                    })

            subtask = await service.create_subtask(
                db,
                ticket_id,
                schemas.SubtaskCreate(
                    title=subtask_data.get("title", f"Subtask {idx}"),
                    description=subtask_data.get("description", ""),
                    required_functionality=subtask_data.get("required_functionality", ""),
                    test_cases=test_cases,
                    assigned_to=assigned_to,
                    order_index=subtask_data.get("order_index", idx),
                ),
            )
            created_subtask_ids.append(subtask.id)

        return _dump({
            "ticket_id": ticket_id,
            "subtask_count": len(created_subtask_ids),
            "subtask_ids": created_subtask_ids,
        })


class CompleteAssignmentArgs(BaseModel):
    subtask_id: str = Field(..., description="UUID of the completed subtask.")


@tool(args_schema=CompleteAssignmentArgs)
async def complete_assignment(subtask_id: str) -> str:
    """Mark a subtask as done and transition related ticket if ready.

    USE WHEN: Developer finishes a subtask and wants to mark it complete.
    RETURNS: {id: "<uuid>", status: "done"}
    """
    async with AsyncSessionLocal() as db:
        from backend.ticket_system.models import SubtaskStatus

        subtask = await service.update_subtask(
            db,
            subtask_id,
            schemas.SubtaskUpdate(status=SubtaskStatus.DONE),
        )
        return _dump({"id": subtask.id, "status": subtask.status.value})


# Persistence tools available to Coordinator
PERSISTENCE_TOOLS = [
    save_ticket,
    transition_ticket,
    save_execution_plan,
    complete_assignment,
]

__all__ = ["save_ticket", "transition_ticket", "save_execution_plan", "complete_assignment", "PERSISTENCE_TOOLS"]
