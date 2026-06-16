"""Tests for prompt caching utility."""
from __future__ import annotations

from backend.agents.prompts import get_cached_role_base, get_cached_lead_appendix, LEAD_SYSTEM


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


def test_cached_lead_appendix_not_empty() -> None:
    """The lead appendix contains RITE contract and tool contract."""
    appendix = get_cached_lead_appendix()
    assert "RITE" in appendix
    assert "tools you control" in appendix.lower() or "list_tickets" in appendix


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
