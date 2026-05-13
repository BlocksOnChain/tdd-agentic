"""Tests for researcher deep-agent OpenAI tool-choice middleware."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from backend.agents.deep import middleware as mw


@pytest.fixture
def openai_model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", api_key="sk-test", rate_limiter=None, max_retries=0)


def test_openai_first_call_tool_choice_requires_when_clean(monkeypatch, openai_model) -> None:
    monkeypatch.setattr(
        mw,
        "get_settings",
        lambda: SimpleNamespace(researcher_is_local=False),
    )
    req = SimpleNamespace(
        model=openai_model,
        messages=[HumanMessage(content="hi")],
        tools=[{"name": "web_search"}],
    )
    assert mw.openai_first_model_call_tool_choice(req) == "required"


def test_openai_first_call_tool_choice_skips_after_tool_message(monkeypatch, openai_model) -> None:
    monkeypatch.setattr(
        mw,
        "get_settings",
        lambda: SimpleNamespace(researcher_is_local=False),
    )
    req = SimpleNamespace(
        model=openai_model,
        messages=[
            HumanMessage(content="hi"),
            ToolMessage(content="{}", name="web_search", tool_call_id="1"),
        ],
        tools=[{"name": "web_search"}],
    )
    assert mw.openai_first_model_call_tool_choice(req) is None


def test_openai_first_call_tool_choice_skips_when_researcher_is_local(monkeypatch, openai_model) -> None:
    monkeypatch.setattr(
        mw,
        "get_settings",
        lambda: SimpleNamespace(researcher_is_local=True),
    )
    req = SimpleNamespace(
        model=openai_model,
        messages=[HumanMessage(content="hi")],
        tools=[{"name": "web_search"}],
    )
    assert mw.openai_first_model_call_tool_choice(req) is None


def test_openai_first_call_tool_choice_still_required_with_base_url_if_not_local(
    monkeypatch, openai_model
) -> None:
    """OPENAI_BASE_URL may point at the real OpenAI API; only RESEARCHER_IS_LOCAL opts out."""
    monkeypatch.setattr(
        mw,
        "get_settings",
        lambda: SimpleNamespace(researcher_is_local=False),
    )
    req = SimpleNamespace(
        model=openai_model,
        messages=[HumanMessage(content="hi")],
        tools=[{"name": "web_search"}],
    )
    assert mw.openai_first_model_call_tool_choice(req) == "required"


def test_openai_first_call_tool_choice_skips_without_tools(monkeypatch, openai_model) -> None:
    monkeypatch.setattr(
        mw,
        "get_settings",
        lambda: SimpleNamespace(researcher_is_local=False),
    )
    req = SimpleNamespace(model=openai_model, messages=[HumanMessage(content="hi")], tools=[])
    assert mw.openai_first_model_call_tool_choice(req) is None
