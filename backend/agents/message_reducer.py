"""Checkpoint message reducer — merge then trim human handoffs."""
from __future__ import annotations

from langgraph.graph.message import add_messages


def trim_checkpoint_messages(
    messages: list,
    *,
    max_human: int | None = None,
    max_ai: int | None = None,
) -> list:
    """Keep first human, recent humans, bounded AI messages — in original order.

    This is the primary checkpoint-size guard: without it, AI/tool messages
    accumulate unbounded across hundreds of turns, bloating Postgres and
    making resume-from-checkpoint fragile.

    Chronological order is part of the contract. Human and AI turns are budgeted
    separately, but the survivors are re-emitted in their original positions —
    returning all humans followed by all AI messages would scramble the
    transcript and can separate an ``AIMessage``'s ``tool_calls`` from the
    ``ToolMessage`` that answers them, which providers reject outright.
    """
    if max_human is None:
        from backend.config import get_settings

        max_human = get_settings().checkpoint_max_human_messages
    if max_ai is None:
        from backend.config import get_settings

        max_ai = get_settings().checkpoint_max_ai_messages
    if not messages or max_human < 1:
        return []

    humans = [m for m in messages if getattr(m, "type", None) == "human"]
    ai_msgs = [m for m in messages if getattr(m, "type", None) != "human"]

    if not humans:
        return list(messages)

    keep_humans = _budget(humans, max_human)
    keep_ai = _budget(ai_msgs, max_ai)

    # Re-emit in the original order rather than concatenating the two buckets.
    # Identity (not equality) — messages can compare equal without being the
    # same turn, and BaseMessage is not reliably hashable.
    keep_ids = {id(m) for m in keep_humans}
    keep_ids.update(id(m) for m in keep_ai)
    return [m for m in messages if id(m) in keep_ids]


def _budget(msgs: list, limit: int | None) -> list:
    """Keep the first message plus the most recent ``limit - 1``, preserving order."""
    if limit is None or len(msgs) <= limit:
        return list(msgs)
    if limit < 1:
        return []
    first = msgs[0]
    tail = msgs[-(limit - 1):] if limit > 1 else []
    return [first, *[m for m in tail if m is not first]]


def add_messages_trimmed(existing: list | None, new: list | None) -> list:
    merged = add_messages(existing, new)
    return trim_checkpoint_messages(merged)
