"""Checkpoint message trimming."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents.message_reducer import add_messages_trimmed, trim_checkpoint_messages


def test_trim_keeps_first_human_and_recent_tail() -> None:
    msgs = [HumanMessage(content=f"h{i}") for i in range(20)]
    trimmed = trim_checkpoint_messages(msgs, max_human=5)
    assert len(trimmed) == 5
    assert trimmed[0].content == "h0"
    assert trimmed[-1].content == "h19"


def test_add_messages_trimmed_merges_then_trims() -> None:
    left = [HumanMessage(content="goal")]
    right = [HumanMessage(content="handoff-1")]
    merged = add_messages_trimmed(left, right)
    assert len(merged) == 2
    assert merged[0].content == "goal"
    assert merged[1].content == "handoff-1"


def _interleaved(n: int) -> list:
    """n human/AI pairs in strict alternation: H0, A0, H1, A1, ..."""
    msgs: list = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i}"))
    return msgs


def test_trim_preserves_chronological_order() -> None:
    """Survivors keep their original positions — humans are not hoisted above AI turns.

    Regression: the reducer used to return ``keep_humans + keep_ai``, which
    scrambled every checkpoint into all-humans-then-all-AI.
    """
    msgs = _interleaved(4)
    trimmed = trim_checkpoint_messages(msgs, max_human=3, max_ai=3)

    contents = [m.content for m in trimmed]
    assert contents == ["h0", "a0", "h2", "a2", "h3", "a3"]

    # Stronger invariant: the output is a subsequence of the input.
    positions = [msgs.index(m) for m in trimmed]
    assert positions == sorted(positions)


def test_trim_under_budget_is_identity() -> None:
    msgs = _interleaved(3)
    trimmed = trim_checkpoint_messages(msgs, max_human=99, max_ai=99)
    assert [m.content for m in trimmed] == [m.content for m in msgs]


def test_trim_keeps_tool_call_adjacent_to_its_result() -> None:
    """An AIMessage with tool_calls must never be separated from its ToolMessage."""
    msgs = [
        HumanMessage(content="goal"),
        AIMessage(content="", tool_calls=[{"name": "list_tickets", "args": {}, "id": "c1"}]),
        ToolMessage(content="[]", name="list_tickets", tool_call_id="c1"),
        HumanMessage(content="[from lead → project_manager] done"),
    ]
    trimmed = trim_checkpoint_messages(msgs, max_human=5, max_ai=5)
    types = [type(m).__name__ for m in trimmed]
    assert types == ["HumanMessage", "AIMessage", "ToolMessage", "HumanMessage"]


def test_trim_handles_degenerate_budgets() -> None:
    msgs = _interleaved(4)
    assert [m.content for m in trim_checkpoint_messages(msgs, max_human=1, max_ai=1)] == [
        "h0",
        "a0",
    ]
    assert trim_checkpoint_messages(msgs, max_human=0, max_ai=5) == []
    assert trim_checkpoint_messages([], max_human=3, max_ai=3) == []
