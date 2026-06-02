"""Tests for the Handoff protocol and compact serialization."""
from __future__ import annotations

from backend.agents.handoff import Handoff, Phase


def test_handoff_minimal_serialization() -> None:
    """Handoff with only target and phase produces minimal JSON."""
    h = Handoff(target="backend_dev", phase=Phase.IMPLEMENT)
    msg = h.to_message()
    assert "[from project_manager → backend_dev]" in msg
    import json

    # Parse the header line (second line)
    lines = msg.strip().split("\n")
    header = json.loads(lines[1])
    assert header["t"] == "backend_dev"
    assert header["p"] == "implement"


def test_handoff_with_ticket_ids() -> None:
    """Handoff with ticket_ids serializes them in the compact key."""
    h = Handoff(
        target="backend_dev",
        phase=Phase.IMPLEMENT,
        ticket_ids=("abc123", "def456"),
        intent="Implement auth subtask.",
    )
    msg = h.to_message()
    import json

    lines = msg.strip().split("\n")
    header = json.loads(lines[1])
    assert header["tik"] == ["abc123", "def456"]
    assert "Implement auth subtask." in msg


def test_handoff_with_context_refs() -> None:
    """Handoff with context_refs includes them in the compact format."""
    h = Handoff(
        target="backend_lead",
        phase=Phase.BACKEND_PLANNING,
        ticket_ids=("uuid1",),
        context_refs=("ctx_1", "ctx_2"),
        intent="Audit and plan backend subtasks.",
    )
    msg = h.to_message()
    import json

    lines = msg.strip().split("\n")
    header = json.loads(lines[1])
    assert header["c"] == ["ctx_1", "ctx_2"]
    assert header["tik"] == ["uuid1"]
    assert header["t"] == "backend_lead"


def test_handoff_from_routing_decision() -> None:
    """Creating a Handoff from a RoutingDecision-like input."""
    h = Handoff.from_routing_decision(
        next_agent="frontend_dev",
        phase="frontend_planning",
        instructions="Plan client-side subtasks for the dashboard ticket.",
        ticket_ids=["dash-uuid"],
    )
    assert h.target == "frontend_dev"
    assert h.phase == Phase.FRONTEND_PLANNING
    assert h.intent == "Plan client-side subtasks for the dashboard ticket."
    assert h.ticket_ids == ("dash-uuid",)


def test_handoff_intent_capped_at_200_chars() -> None:
    """Intent longer than 200 chars is truncated."""
    long_instructions = "x" * 500
    h = Handoff.from_routing_decision(
        next_agent="devops",
        phase="implement",
        instructions=long_instructions,
        ticket_ids=["uuid1"],
    )
    assert len(h.intent) == 200
    assert h.intent == "x" * 200


def test_phase_coerce_known_value() -> None:
    """A verbatim phase value round-trips to the matching Phase."""
    assert Phase.coerce("research") is Phase.RESEARCH
    assert Phase.coerce("infrastructure") is Phase.INFRASTRUCTURE
    assert Phase.coerce(Phase.QA) is Phase.QA


def test_phase_coerce_infra_aliases() -> None:
    """Near-miss infra phase names map onto the canonical INFRASTRUCTURE phase.

    Regression: the PM model emitted ``infra_scaffolding`` which previously
    crashed routing with ``'infra_scaffolding' is not a valid Phase``.
    """
    for alias in (
        "infra_scaffolding",
        "infrastructure_scaffolding",
        "devops_scaffolding",
        "infra",
        "scaffolding",
        "DevOps-Scaffolding",
    ):
        assert Phase.coerce(alias) is Phase.INFRASTRUCTURE


def test_phase_coerce_unknown_falls_back() -> None:
    """Unknown / empty phase strings degrade to the default instead of raising."""
    assert Phase.coerce("totally_made_up") is Phase.IMPLEMENT
    assert Phase.coerce("") is Phase.IMPLEMENT
    assert Phase.coerce(None) is Phase.IMPLEMENT
    assert Phase.coerce("totally_made_up", default=Phase.REVIEW) is Phase.REVIEW


def test_handoff_from_routing_decision_with_invalid_phase() -> None:
    """Building a Handoff from an invalid phase string no longer crashes."""
    h = Handoff.from_routing_decision(
        next_agent="devops",
        phase="infra_scaffolding",
        instructions="Scaffold project infrastructure.",
        ticket_ids=["uuid1"],
    )
    assert h.phase is Phase.INFRASTRUCTURE
    assert h.to_message().splitlines()[1].count("infrastructure") == 1


def test_handoff_flags_in_message() -> None:
    """Flags are included in the serialized message."""
    h = Handoff(
        target="qa",
        phase=Phase.QA,
        ticket_ids=("uuid1",),
        flags=("requires_research", "has_unanswered_questions"),
    )
    msg = h.to_message()
    import json

    lines = msg.strip().split("\n")
    header = json.loads(lines[1])
    assert "f" in header
    assert "requires_research" in header["f"]
