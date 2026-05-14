
"""Verify the PM supervisor's JSON routing parser tolerates common LLM outputs."""
from __future__ import annotations

from backend.agents.project_manager.supervisor import _parse_routing


def test_parses_plain_json() -> None:
    text = '{"next_agent": "researcher", "rationale": "scope tech", "instructions": "look up FastAPI"}'
    decision = _parse_routing(text)
    assert decision is not None
    assert decision.next_agent == "researcher"


def test_parses_json_in_markdown_fence() -> None:
    text = """```json
    {"next_agent": "backend_lead", "rationale": "decompose", "instructions": "split TICKET-1"}
    ```"""
    decision = _parse_routing(text)
    assert decision is not None
    assert decision.next_agent == "backend_lead"


def test_returns_none_on_garbage() -> None:
    assert _parse_routing("just some prose with no JSON") is None


def test_parses_optional_ticket_ids() -> None:
    text = (
        '{"next_agent": "qa", "rationale": "verify", '
        '"ticket_ids": ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"], "phase": "qa"}'
    )
    decision = _parse_routing(text)
    assert decision is not None
    assert decision.ticket_ids == ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"]
