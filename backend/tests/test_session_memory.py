"""Session memory merge and schema."""
from __future__ import annotations

from backend.agents.session_memory import SessionMemory, merge_session_memory


def test_merge_session_memory_scalars_and_lists() -> None:
    left = SessionMemory(
        goal="ship api",
        last_route="backend_lead",
        recent_handoffs=["backend_lead: done"],
    )
    right = SessionMemory(
        goal="",
        last_route="researcher",
        recent_handoffs=["researcher: sources found"],
        blockers=["need api key"],
    )
    m = merge_session_memory(left, right)
    assert m.goal == "ship api"
    assert m.last_route == "researcher"
    assert m.recent_handoffs[-1].startswith("researcher:")
    assert "need api key" in m.blockers


def test_merge_session_memory_none_right() -> None:
    left = SessionMemory(goal="x")
    assert merge_session_memory(left, None).goal == "x"


def test_merge_session_memory_none_left() -> None:
    right = SessionMemory(last_route="qa")
    m = merge_session_memory(None, right)
    assert m.last_route == "qa"
