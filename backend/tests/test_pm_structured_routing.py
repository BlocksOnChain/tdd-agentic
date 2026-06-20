"""PM structured routing and ticket-id validation."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agents.project_manager.supervisor import (
    RoutingDecision,
    _resolve_routing_decision,
    _validate_ticket_ids,
)


@pytest.mark.asyncio
async def test_resolve_routing_decision_uses_structured_output() -> None:
    expected = RoutingDecision(
        next_agent="backend_dev",
        rationale="implement",
        ticket_ids=["7c4f842e-e29b-41d4-a716-446655440000"],
    )
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=expected)
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    with (
        patch(
            "backend.agents.project_manager.supervisor.pm_model",
            return_value=mock_model,
        ),
        patch(
            "backend.agents.project_manager.supervisor.with_retry",
            side_effect=lambda r: r,
        ),
    ):
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="go"),
            AIMessage(content='{"next_agent": "ignored"}'),
        ]
        decision = await _resolve_routing_decision(messages, '{"next_agent": "ignored"}')

    assert decision == expected
    mock_structured.ainvoke.assert_awaited_once_with(messages)
    mock_model.with_structured_output.assert_called_once_with(RoutingDecision)


@pytest.mark.asyncio
async def test_resolve_routing_decision_falls_back_to_parse() -> None:
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("structured output failed"))
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    with (
        patch(
            "backend.agents.project_manager.supervisor.pm_model",
            return_value=mock_model,
        ),
        patch(
            "backend.agents.project_manager.supervisor.with_retry",
            side_effect=lambda r: r,
        ),
    ):
        text = '{"next_agent": "qa", "rationale": "verify", "instructions": "run tests"}'
        decision = await _resolve_routing_decision([], text)

    assert decision is not None
    assert decision.next_agent == "qa"


@pytest.mark.asyncio
async def test_validate_ticket_ids_strips_hallucinated_uuids() -> None:
    real = "7c4f842e-e29b-41d4-a716-446655440000"
    fake = "00000000-0000-0000-0000-000000000001"
    mock_ticket = SimpleNamespace(id=real)

    mock_db = AsyncMock()
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "backend.agents.project_manager.supervisor.AsyncSessionLocal",
            return_value=mock_session,
        ),
        patch(
            "backend.agents.project_manager.supervisor.service.list_tickets",
            new_callable=AsyncMock,
            return_value=[mock_ticket],
        ),
        patch(
            "backend.agents.project_manager.supervisor.emit",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(kind="ticket_ids_stripped"),
        ) as mock_emit,
    ):
        decision = RoutingDecision(
            next_agent="backend_dev",
            ticket_ids=[fake, real],
        )
        validated, event = await _validate_ticket_ids(decision, "proj-1")

    assert validated.ticket_ids == [real]
    mock_emit.assert_awaited_once()
    assert event is not None
    assert event.kind == "ticket_ids_stripped"


@pytest.mark.asyncio
async def test_validate_ticket_ids_falls_back_to_instructions_uuids() -> None:
    real = "7c4f842e-e29b-41d4-a716-446655440000"
    fake = "00000000-0000-0000-0000-000000000002"
    mock_ticket = SimpleNamespace(id=real)

    mock_db = AsyncMock()
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "backend.agents.project_manager.supervisor.AsyncSessionLocal",
            return_value=mock_session,
        ),
        patch(
            "backend.agents.project_manager.supervisor.service.list_tickets",
            new_callable=AsyncMock,
            return_value=[mock_ticket],
        ),
        patch(
            "backend.agents.project_manager.supervisor.emit",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(kind="ticket_ids_stripped"),
        ),
    ):
        decision = RoutingDecision(
            next_agent="lead",
            ticket_ids=[fake],
            instructions=f"Plan ticket {real}.",
        )
        validated, event = await _validate_ticket_ids(decision, "proj-1")

    assert validated.ticket_ids == [real]
    assert event is not None


@pytest.mark.asyncio
async def test_validate_ticket_ids_noop_without_project_id() -> None:
    decision = RoutingDecision(
        next_agent="researcher",
        ticket_ids=["any-id"],
    )
    validated, event = await _validate_ticket_ids(decision, None)
    assert validated.ticket_ids == ["any-id"]
    assert event is None
