"""Human-in-the-loop tool that pauses graph execution via ``interrupt()``.

Calling this tool inside a graph node throws a ``GraphInterrupt`` that
LangGraph catches and surfaces to the API layer. The client resumes by
sending ``Command(resume=<answer>)`` to the same thread_id.
"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def ask_human(question: str, ticket_id: str | None = None) -> str:
    """Ask the human supervisor a question and wait for the answer.

    Use sparingly — only when ticket_id-level questions aren't sufficient
    (e.g. you need an immediate yes/no without persisting a question).

    IMPORTANT (callers): ``interrupt()`` resumes the *whole node* from the top,
    so every tool call made earlier in the same turn runs a second time. Only
    call this before any state-mutating tool in a turn, or make the node's
    replay idempotent. See ``supervisor.project_manager_node``, which records
    the mutations it has already applied and skips them on replay.
    """
    answer = interrupt(
        {
            "kind": "ask_human",
            "question": question,
            "ticket_id": ticket_id,
        }
    )
    return str(answer)
