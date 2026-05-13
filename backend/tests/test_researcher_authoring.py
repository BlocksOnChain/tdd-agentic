"""Researcher legacy authoring pass after deep agent."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents.researcher.authoring import run_researcher_authoring_pass
from backend.agents.state import SystemState


@pytest.mark.asyncio
async def test_authoring_pass_includes_authoring_instruction(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_loop(**kwargs):
        captured["extra_focused_messages"] = kwargs.get("extra_focused_messages")
        return [AIMessage(content="authored docs")], [
            ToolMessage(content="ok", name="fs_write", tool_call_id="1")
        ]

    monkeypatch.setattr(
        "backend.agents.researcher.authoring.run_specialist_tool_loop",
        fake_loop,
    )

    state = SystemState(
        project_context="Todo app with shadcn animations",
        messages=[
            HumanMessage(content="Todo app with shadcn animations"),
            HumanMessage(content="[from project_manager → researcher]\nWrite docs/."),
        ],
    )
    await run_researcher_authoring_pass(state, AsyncMock())
    extras = captured["extra_focused_messages"]
    assert isinstance(extras, list)
    assert any("[research authoring]" in str(msg.content) for msg in extras)
