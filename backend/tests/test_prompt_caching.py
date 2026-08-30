"""Tests for prompt caching utility."""
from __future__ import annotations

from backend.agents.prompts import get_cached_role_base, LEAD_SYSTEM


def test_cached_role_base_returns_correct_prompt() -> None:
    """get_cached_role_base returns the correct prompt for each role."""
    assert get_cached_role_base("project_manager") is not None
    assert get_cached_role_base("researcher") is not None
    assert get_cached_role_base("lead") is LEAD_SYSTEM
    assert get_cached_role_base("backend_dev") is not None
    assert get_cached_role_base("frontend_dev") is not None
    assert get_cached_role_base("devops") is not None
    assert get_cached_role_base("qa") is not None


def test_cached_role_base_unknown_role_raises() -> None:
    """get_cached_role_base raises ValueError for unknown roles."""
    try:
        get_cached_role_base("unknown_role_xyz")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "unknown_role_xyz" in str(e)


def test_lead_prompt_carries_the_rite_contract() -> None:
    """The Lead is cognitive-only, so its prompt must be self-contained."""
    assert "RITE" in LEAD_SYSTEM
    assert "execution_plan" in LEAD_SYSTEM


def test_lead_prompt_does_not_describe_tools_it_lacks() -> None:
    """The Lead has tools=[]. Instructing it to call create_subtask/list_tickets
    wasted prompt budget and contradicted its actual instructions."""
    for phantom in ("create_subtask(", "list_tickets(", "delete_subtask("):
        assert phantom not in LEAD_SYSTEM, f"LEAD_SYSTEM still references {phantom}"


def test_cached_returns_same_object() -> None:
    """Calling get_cached_role_base multiple times returns the same object."""
    result1 = get_cached_role_base("researcher")
    result2 = get_cached_role_base("researcher")
    assert result1 is result2


def test_project_manager_prompt_has_routing_protocol() -> None:
    """The PM prompt includes a routing protocol section."""
    pm = get_cached_role_base("project_manager")
    assert "routing protocol" in pm.lower() or "routing json" in pm.lower()


def test_project_manager_prompt_has_constraints() -> None:
    """The PM prompt includes a constraints section."""
    pm = get_cached_role_base("project_manager")
    assert "=== CONSTRAINTS ===" in pm


def test_project_manager_prompt_has_tool_selection_guide() -> None:
    """The PM prompt includes a tool selection guide."""
    pm = get_cached_role_base("project_manager")
    assert "tool selection guide" in pm.lower() or "TOOL SELECTION GUIDE" in pm
