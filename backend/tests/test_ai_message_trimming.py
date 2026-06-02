"""Tests for AI message trimming in checkpoints."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agents.message_reducer import trim_checkpoint_messages


def test_trims_ai_messages_when_exceeding_max_ai() -> None:
    """When AI messages exceed max_ai, keep first + tail."""
    humans = [HumanMessage(content=f"h{i}") for i in range(3)]
    ai_msgs = [AIMessage(content=f"ai{i}") for i in range(15)]
    mixed: list = []
    for h in humans:
        mixed.append(h)
    for a in ai_msgs:
        mixed.append(a)

    trimmed = trim_checkpoint_messages(mixed, max_human=10, max_ai=5)
    # Humans all kept (under max_human)
    # AI: first + 4 tail = 5
    ai_kept = [m for m in trimmed if getattr(m, "type", None) != "human"]
    assert len(ai_kept) == 5
    # First AI message should be present
    assert ai_kept[0].content == "ai0"
    # Last 4 should be present
    assert ai_kept[-1].content == "ai14"


def test_no_trim_when_ai_within_limit() -> None:
    """When AI messages are within max_ai, all are kept."""
    humans = [HumanMessage(content="goal")]
    ai_msgs = [AIMessage(content=f"ai{i}") for i in range(5)]
    mixed = list(humans) + ai_msgs

    trimmed = trim_checkpoint_messages(mixed, max_human=10, max_ai=10)
    assert len(trimmed) == 6


def test_trims_ai_even_when_humans_within_limit() -> None:
    """AI trimming works even when human count is within limit."""
    humans = [HumanMessage(content="goal")]
    ai_msgs = [AIMessage(content=f"ai{i}") for i in range(20)]
    mixed = list(humans) + ai_msgs

    trimmed = trim_checkpoint_messages(mixed, max_human=10, max_ai=3)
    ai_kept = [m for m in trimmed if getattr(m, "type", None) != "human"]
    assert len(ai_kept) == 3


def test_ai_trim_keeps_first_and_tail_without_dupe() -> None:
    """First AI msg appears only once when it's also in the tail."""
    humans = [HumanMessage(content="goal")]
    ai_msgs = [AIMessage(content=f"ai{i}") for i in range(5)]
    mixed = list(humans) + ai_msgs

    # max_ai=3 means first + 2 tail
    trimmed = trim_checkpoint_messages(mixed, max_human=10, max_ai=3)
    ai_kept = [m for m in trimmed if getattr(m, "type", None) != "human"]
    # ai0 is both first and in tail[-(3-1):] = ai3, ai4 — no overlap
    # first=ai0, tail=[ai3, ai4] = 3 total, no dupes
    assert len(ai_kept) == 3
    # Check no duplicate content
    contents = [m.content for m in ai_kept]
    assert len(contents) == len(set(contents))


def test_all_ai_is_first() -> None:
    """When there's exactly one AI message, it should be kept."""
    humans = [HumanMessage(content="goal")]
    ai_msgs = [AIMessage(content="only_ai")]
    mixed = list(humans) + ai_msgs

    trimmed = trim_checkpoint_messages(mixed, max_human=10, max_ai=1)
    ai_kept = [m for m in trimmed if getattr(m, "type", None) != "human"]
    assert len(ai_kept) == 1
    assert ai_kept[0].content == "only_ai"


def test_empty_messages() -> None:
    """Empty messages return empty."""
    assert trim_checkpoint_messages([], max_human=5, max_ai=5) == []


def test_no_human_returns_all() -> None:
    """When there are no human messages, all messages are returned as-is."""
    ai_msgs = [AIMessage(content=f"ai{i}") for i in range(5)]
    trimmed = trim_checkpoint_messages(ai_msgs, max_human=10, max_ai=10)
    assert len(trimmed) == 5


def test_mixed_types_preserved() -> None:
    """SystemMessage and ToolMessage count as AI messages."""
    humans = [HumanMessage(content="goal")]
    mixed_ai = [
        SystemMessage(content="sys"),
        AIMessage(content="ai"),
        ToolMessage(content="tool", tool_call_id="1"),
    ]
    for _ in range(8):
        mixed_ai.append(AIMessage(content="extra_ai"))
    mixed = list(humans) + mixed_ai

    trimmed = trim_checkpoint_messages(mixed, max_human=10, max_ai=3)
    ai_kept = [m for m in trimmed if getattr(m, "type", None) != "human"]
    assert len(ai_kept) == 3
    # SystemMessage (first AI) should be kept
    assert isinstance(ai_kept[0], SystemMessage)
