"""Lightweight orchestration smoke tests (mock PM + one specialist)."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agents.session_memory import SessionMemory
from backend.agents.state import SystemState


@pytest.mark.asyncio
async def test_pm_researcher_pm_checkpoint_round_trip() -> None:
    """Two supervisor turns with researcher in the middle; state merges cleanly."""
    calls: dict[str, int] = {"pm": 0}

    async def fake_pm(state: SystemState):
        calls["pm"] += 1
        if calls["pm"] == 1:
            return {
                "next_agent": "researcher",
                "messages": [
                    HumanMessage(
                        content="[from project_manager → researcher]\nInvestigate topic.",
                    )
                ],
            }
        return {"next_agent": "end", "messages": []}

    async def fake_researcher(state: SystemState):
        return {
            "messages": [
                HumanMessage(
                    content="[from researcher → project_manager]\nSummary here.",
                )
            ],
            "session_memory": SessionMemory(
                last_route="researcher",
                recent_handoffs=["researcher: Summary here."],
            ),
        }

    def route_pm(state: SystemState) -> str:
        if (state.next_agent or "").lower() == "end":
            return END
        if (state.next_agent or "").lower() == "researcher":
            return "researcher"
        return "project_manager"

    g = StateGraph(SystemState)
    g.add_node("project_manager", fake_pm)
    g.add_node("researcher", fake_researcher)
    g.add_edge(START, "project_manager")
    g.add_conditional_edges(
        "project_manager",
        route_pm,
        {"researcher": "researcher", "project_manager": "project_manager", END: END},
    )
    g.add_edge("researcher", "project_manager")
    compiled = g.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "proj-test-1"}}
    await compiled.ainvoke(
        SystemState(
            project_id="proj-test-1",
            project_context="ctx",
            messages=[HumanMessage(content="Build a thing")],
        ),
        cfg,
    )
    st = await compiled.aget_state(cfg)
    assert st.values is not None
    values = st.values
    assert str(values.get("next_agent") or "").lower() == "end"
    mem = values.get("session_memory")
    assert mem is not None
    assert getattr(mem, "last_route", None) == "researcher"
