"""Checkpoint message trimming."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

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
