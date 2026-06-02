"""Tests for the ContextStore inter-agent communication system."""
from __future__ import annotations

from backend.agents.context_store import ContextStore, ContextEntry


def test_add_returns_incrementing_id() -> None:
    """Each add call returns a sequentially incrementing ID."""
    store = ContextStore()
    id1 = store.add("agent_a", "research_findings", ("t1",), "First entry")
    id2 = store.add("agent_b", "lead_plan", ("t1",), "Second entry")
    assert id1 == "ctx_1"
    assert id2 == "ctx_2"


def test_add_max_entries_evicts_oldest() -> None:
    """Store evicts oldest entry when exceeding MAX_ENTRIES."""
    store = ContextStore()
    for i in range(ContextStore.MAX_ENTRIES + 5):
        store.add("agent", "kind", (), f"Entry {i}")
    assert len(store) == ContextStore.MAX_ENTRIES
    # Oldest entries should be gone
    assert "ctx_1" not in store
    assert "ctx_6" in store  # First entry after eviction


def test_summary_truncated_to_max_chars() -> None:
    """Summaries longer than MAX_SUMMARY_CHARS are truncated."""
    store = ContextStore()
    long_text = "x" * 1000
    store.add("agent", "kind", (), long_text)
    entry = store["ctx_1"]
    assert len(entry.summary) == ContextStore.MAX_SUMMARY_CHARS
    assert entry.summary == "x" * ContextStore.MAX_SUMMARY_CHARS


def test_lookup_returns_none_for_missing() -> None:
    """lookup returns None for non-existent IDs."""
    store = ContextStore()
    assert store.lookup("ctx_999") is None


def test_lookup_for_ticket() -> None:
    """for_ticket returns entries referencing a specific ticket."""
    store = ContextStore()
    store.add("researcher", "research_findings", ("uuid1", "uuid2"), "Auth findings")
    store.add("lead", "lead_plan", ("uuid1",), "Backend plan")
    store.add("other", "kind", ("uuid3",), "Unrelated")
    results = store.for_ticket("uuid1")
    assert len(results) == 2
    agents = {e.agent for e in results}
    assert agents == {"researcher", "lead"}


def test_for_agent_filters_by_agent() -> None:
    """for_agent returns only entries from a specific agent."""
    store = ContextStore()
    store.add("researcher", "research_findings", ("uuid1",), "Research")
    store.add("lead", "lead_plan", ("uuid1",), "Plan")
    results = store.for_agent("researcher")
    assert len(results) == 1
    assert results[0].agent == "researcher"


def test_entry_is_frozen_dataclass() -> None:
    """ContextEntry is a frozen dataclass (immutable)."""
    entry = ContextEntry(
        agent="test",
        kind="test_kind",
        ticket_ids=("uuid1",),
        summary="test summary",
    )
    # Should raise because it's frozen
    try:
        entry.agent = "modified"
        assert False, "Expected FrozenError"
    except Exception:
        pass
