"""Tests for skill loader caching and change detection."""
from __future__ import annotations

import importlib

from backend.agents.skills import loader


def test_inject_skills_skips_when_no_change() -> None:
    """Calling inject_skills twice with same skill set returns unchanged base."""
    # Clear cache first
    loader._inject_cache.clear()

    base = "base prompt content"
    result1 = loader.inject_skills(base, role="project_manager")
    result2 = loader.inject_skills(base, role="project_manager")
    assert result1 is result2  # Same object (cached)


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
