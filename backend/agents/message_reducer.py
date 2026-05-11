"""Checkpoint message reducer — merge then trim human handoffs."""
from __future__ import annotations

from langgraph.graph.message import add_messages


def trim_checkpoint_messages(
    messages: list,
    *,
    max_human: int | None = None,
) -> list:
    if max_human is None:
        from backend.config import get_settings

        max_human = get_settings().checkpoint_max_human_messages
    """Keep the first human turn and the most recent handoffs."""
    if not messages or max_human < 1:
        return []

    humans = [m for m in messages if getattr(m, "type", None) == "human"]
    if not humans:
        return list(messages)

    if len(humans) <= max_human:
        return list(messages)

    first = humans[0]
    tail = humans[-(max_human - 1) :]
    if tail and tail[0] is first:
        keep = tail
    else:
        keep = [first, *[m for m in tail if m is not first]]
    return keep


def add_messages_trimmed(existing: list | None, new: list | None) -> list:
    merged = add_messages(existing, new)
    return trim_checkpoint_messages(merged)
