"""Checkpoint message reducer — merge then trim human handoffs."""
from __future__ import annotations

from langgraph.graph.message import add_messages


def trim_checkpoint_messages(
    messages: list,
    *,
    max_human: int | None = None,
    max_ai: int | None = None,
) -> list:
    """Keep first human, recent humans, bounded AI messages.

    This is the primary checkpoint-size guard: without it, AI/tool messages
    accumulate unbounded across hundreds of turns, bloating Postgres and
    making resume-from-checkpoint fragile.
    """
    if max_human is None:
        from backend.config import get_settings

        max_human = get_settings().checkpoint_max_human_messages
    if max_ai is None:
        from backend.config import get_settings

        max_ai = get_settings().checkpoint_max_ai_messages
    """Keep the first human turn and the most recent handoffs."""
    if not messages or max_human < 1:
        return []

    humans = [m for m in messages if getattr(m, "type", None) == "human"]
    ai_msgs = [m for m in messages if getattr(m, "type", None) != "human"]

    if not humans:
        return list(messages)

    if len(humans) <= max_human:
        keep_humans = humans
    else:
        first = humans[0]
        tail = humans[-(max_human - 1):]
        keep_humans = [first, *[m for m in tail if m is not first]]

    # Trim AI messages: keep first AI msg + recent tail
    if max_ai is not None and len(ai_msgs) > max_ai:
        first_ai = ai_msgs[0]
        tail_ai = ai_msgs[-(max_ai - 1):]
        keep_ai = [first_ai, *[m for m in tail_ai if m is not first_ai]]
    else:
        keep_ai = ai_msgs

    return keep_humans + keep_ai


def add_messages_trimmed(existing: list | None, new: list | None) -> list:
    merged = add_messages(existing, new)
    return trim_checkpoint_messages(merged)
