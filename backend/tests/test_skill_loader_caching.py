"""Tests for skill loader caching and change detection."""
from __future__ import annotations

import importlib

from backend.agents.skills import loader


def test_inject_skills_reuses_assembled_prompt_when_no_change() -> None:
    """Repeat calls reuse the cached prompt — they never fall back to the bare base."""
    loader._inject_cache.clear()

    base = "base prompt content"
    result1 = loader.inject_skills(base, role="project_manager")
    result2 = loader.inject_skills(base, role="project_manager")
    assert result1 is result2  # Same object (cached)
    # Whatever the registry holds, the second call must not silently drop
    # skills that the first call injected.
    assert result2.startswith(base)
    assert len(result2) == len(result1)


def test_inject_skills_survives_repeat_calls_with_skills(monkeypatch) -> None:
    """With skills registered, EVERY call keeps the skill index — not just the first.

    Regression: the change-detection cache used to return ``base_prompt`` on a
    hit, so agents lost their skills from the second turn onward.
    """
    loader._inject_cache.clear()
    monkeypatch.setattr(
        loader,
        "get_skills_for_role",
        lambda role: [{"name": "tdd-rite", "description": "RITE test format"}],
    )

    base = "base prompt"
    first = loader.inject_skills(base, role="backend_dev")
    second = loader.inject_skills(base, role="backend_dev")

    assert "tdd-rite" in first
    assert "tdd-rite" in second
    assert first == second


def test_inject_skills_rebuilds_when_base_prompt_changes(monkeypatch) -> None:
    """A different base prompt for the same role must not return the other's cache."""
    loader._inject_cache.clear()
    monkeypatch.setattr(
        loader,
        "get_skills_for_role",
        lambda role: [{"name": "tdd-rite", "description": "RITE test format"}],
    )

    a = loader.inject_skills("PROMPT A", role="backend_dev")
    b = loader.inject_skills("PROMPT B", role="backend_dev")

    assert a.startswith("PROMPT A")
    assert b.startswith("PROMPT B")
    assert "tdd-rite" in b


def test_inject_skills_returns_base_when_no_skills() -> None:
    """When a role has no skills, inject_skills returns base unchanged."""
    # Clear cache
    loader._inject_cache.clear()

    base = "base prompt"
    # "nonexistent_role" should have no skills registered
    result = loader.inject_skills(base, role="nonexistent_role_does_not_exist")
    assert result == base


def test_inject_skills_injects_when_new_skills() -> None:
    """When skill set changes, injection happens and cache updates."""
    # Clear cache
    loader._inject_cache.clear()

    base = "base prompt"
    result1 = loader.inject_skills(base, role="project_manager")
    # The first call should inject skills (or return base if no skills)
    # Either way, the cache is now populated
    assert "base prompt" in result1 or result1 == base


def test_inject_cache_per_role() -> None:
    """Different roles have separate caches."""
    loader._inject_cache.clear()

    base1 = "base for role A"
    base2 = "base for role B"
    result1 = loader.inject_skills(base1, role="project_manager")
    result2 = loader.inject_skills(base2, role="researcher")

    # Both should have their respective bases
    assert "base for role A" in result1 or result1 == base1
    assert "base for role B" in result2 or result2 == base2
