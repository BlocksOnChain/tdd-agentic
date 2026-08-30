"""Every agent role must resolve to a usable model slug.

Regression: ``devops_model()`` / ``qa_model()`` passed ``None`` straight through
to ``get_chat_model`` when the env var was unset (the default), and
``backend_dev_model`` / ``frontend_dev_model`` were read from ``Settings``
without ever being declared there. Both crashed the node — and the missing
fields also took down ``log_resolved_llm_routing()``, which runs in the FastAPI
lifespan, so the app failed to boot.
"""
from __future__ import annotations

import pytest

from backend.agents import llm as llm_mod
from backend.config import Settings

# (settings field, factory) for every role whose slug is optional and falls back.
OPTIONAL_ROLE_FIELDS = [
    "backend_dev_model",
    "frontend_dev_model",
    "coordinator_model",
    "devops_model",
    "qa_model",
]

ROLE_FACTORIES = [
    llm_mod.pm_model,
    llm_mod.researcher_model,
    llm_mod.lead_model,
    llm_mod.dev_model,
    llm_mod.backend_dev_model,
    llm_mod.frontend_dev_model,
    llm_mod.coordinator_model,
    llm_mod.devops_model,
    llm_mod.qa_model,
    llm_mod.grader_model,
]


@pytest.fixture
def bare_settings(monkeypatch) -> Settings:
    """Settings with only the required slugs set — every optional override unset."""
    settings = Settings(
        _env_file=None,
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        openrouter_api_key="",
        openai_base_url="",
    )
    monkeypatch.setattr(llm_mod, "get_settings", lambda: settings)
    return settings


def test_optional_role_fields_exist_on_settings(bare_settings: Settings) -> None:
    """The fields the factories read must actually be declared."""
    for field in OPTIONAL_ROLE_FIELDS:
        assert field in Settings.model_fields, f"Settings is missing '{field}'"
        # Unset is either None (no env var) or "" (env var present but blank).
        # Both mean "not configured" and both must fall back to dev_model.
        assert not getattr(bare_settings, field)


@pytest.mark.parametrize("unset_value", [None, ""])
@pytest.mark.parametrize("field", OPTIONAL_ROLE_FIELDS)
def test_blank_and_missing_overrides_both_fall_back(
    field, unset_value, bare_settings
) -> None:
    setattr(bare_settings, field, unset_value)
    bare_settings.dev_model = "anthropic/claude-sonnet-4-6"

    factory = {
        "backend_dev_model": llm_mod.backend_dev_model,
        "frontend_dev_model": llm_mod.frontend_dev_model,
        "coordinator_model": llm_mod.coordinator_model,
        "devops_model": llm_mod.devops_model,
        "qa_model": llm_mod.qa_model,
    }[field]

    model = factory()
    assert getattr(model, "model", None) == "claude-sonnet-4-6"


@pytest.mark.parametrize("factory", ROLE_FACTORIES, ids=lambda f: f.__name__)
def test_every_role_builds_a_model_with_defaults(factory, bare_settings) -> None:
    """No role may crash when its optional override is unset."""
    model = factory()
    assert model is not None


@pytest.mark.parametrize("factory", ROLE_FACTORIES, ids=lambda f: f.__name__)
def test_every_role_resolves_to_a_non_empty_slug(factory, bare_settings) -> None:
    """A role must never resolve to None/'' — _split_slug would raise on it."""
    captured: list[str] = []
    real = llm_mod.get_chat_model

    def _spy(slug, **kwargs):
        captured.append(slug)
        return real(slug, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(llm_mod, "get_chat_model", _spy):
        factory()

    assert captured, f"{factory.__name__} never called get_chat_model"
    slug = captured[0]
    assert isinstance(slug, str) and slug.strip(), f"{factory.__name__} -> {slug!r}"


def test_startup_routing_log_does_not_raise(bare_settings, caplog) -> None:
    """log_resolved_llm_routing runs in the FastAPI lifespan — it must not raise."""
    llm_mod.log_resolved_llm_routing()


def test_routing_log_covers_every_role(bare_settings, caplog) -> None:
    """Roles absent from the log are roles whose misconfiguration is invisible."""
    import logging

    with caplog.at_level(logging.INFO, logger="backend.agents.llm"):
        llm_mod.log_resolved_llm_routing()

    logged = "\n".join(r.getMessage() for r in caplog.records)
    for field in ("devops_model", "qa_model", "backend_dev_model", "frontend_dev_model"):
        assert field in logged, f"{field} missing from startup routing log"


def test_audit_slug_map_matches_factory_fallbacks(bare_settings) -> None:
    """llm_audit must report the same slug the factory actually uses."""
    from backend.agents.llm_audit import resolve_model_slug_for_node

    bare_settings.dev_model = "anthropic/claude-sonnet-4-6"
    bare_settings.devops_model = None
    bare_settings.qa_model = None

    import backend.agents.llm_audit as audit_mod
    import unittest.mock

    with unittest.mock.patch.object(audit_mod, "get_settings", lambda: bare_settings):
        assert resolve_model_slug_for_node("devops") == "anthropic/claude-sonnet-4-6"
        assert resolve_model_slug_for_node("qa") == "anthropic/claude-sonnet-4-6"

        bare_settings.devops_model = "openai/local-devops"
        assert resolve_model_slug_for_node("devops") == "openai/local-devops"
