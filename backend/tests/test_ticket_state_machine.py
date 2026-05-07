"""Smoke tests for the ticket-system state-machine logic.

These exercise the pure-Python transition table in
``backend/ticket_system/service.py`` without needing a live database.
"""
from __future__ import annotations

import pytest

from backend.ticket_system.service import ALLOWED_TICKET_TRANSITIONS
from backend.ticket_system.models import TicketStatus


@pytest.mark.parametrize(
    "src,dst,allowed",
    [
        (TicketStatus.DRAFT, TicketStatus.IN_REVIEW, True),
        (TicketStatus.DRAFT, TicketStatus.QUESTIONS_PENDING, True),
        (TicketStatus.DRAFT, TicketStatus.DONE, False),
        (TicketStatus.IN_REVIEW, TicketStatus.TODO, True),
        (TicketStatus.TODO, TicketStatus.IN_PROGRESS, True),
        (TicketStatus.IN_PROGRESS, TicketStatus.DONE, True),
        (TicketStatus.IN_PROGRESS, TicketStatus.DRAFT, False),
        (TicketStatus.DONE, TicketStatus.IN_PROGRESS, False),
    ],
)
def test_transition_allowed(src: TicketStatus, dst: TicketStatus, allowed: bool) -> None:
    is_allowed = dst in ALLOWED_TICKET_TRANSITIONS.get(src, set())
    assert is_allowed is allowed
