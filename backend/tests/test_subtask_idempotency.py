"""Subtask order_index idempotency helpers."""
from __future__ import annotations

from backend.ticket_system.service import _normalise_title


def test_normalise_title_strips_edges() -> None:
    assert _normalise_title("  Implement auth  ") == "Implement auth"
