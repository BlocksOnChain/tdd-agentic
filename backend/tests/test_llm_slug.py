from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_split_slug_explicit_openrouter() -> None:
    from backend.agents.llm import _split_slug

    assert _split_slug("openrouter/nex-agi/some-model") == (
        "openrouter",
        "nex-agi/some-model",
    )


def test_split_slug_known_direct_providers() -> None:
    from backend.agents.llm import _split_slug

    assert _split_slug("anthropic/claude-sonnet-4-6") == (
        "anthropic",
        "claude-sonnet-4-6",
    )
    assert _split_slug("openai/gpt-4o") == ("openai", "gpt-4o")


def test_split_slug_openrouter_vendor_model_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.agents import llm as llm_mod

    monkeypatch.setattr(
        llm_mod,
        "get_settings",
        lambda: SimpleNamespace(openrouter_api_key="sk-or-test"),
    )

    assert llm_mod._split_slug("nex-agi/some-model") == (
        "openrouter",
        "nex-agi/some-model",
    )


def test_split_slug_unknown_vendor_without_openrouter_key() -> None:
    from backend.agents import llm as llm_mod

    # Default test env has no OPENROUTER_API_KEY — treat first segment as provider.
    assert llm_mod._split_slug("nex-agi/some-model") == ("nex-agi", "some-model")
