"""Lead subgraph — plans all subtasks (backend + frontend) in a single execution plan."""
from __future__ import annotations

from backend.agents.llm import lead_model
from backend.agents.prompts import LEAD_SYSTEM
from backend.agents.runner import build_specialist_subgraph


def build_lead_subgraph():
    """Build the merged Lead agent subgraph (handles both backend and frontend planning)."""
    return build_specialist_subgraph(
        name="lead",
        role="lead",
        llm_factory=lead_model,
        tools=[],  # Lead is cognitive-only; Coordinator handles DB persistence
        base_system_prompt=LEAD_SYSTEM,
        max_steps=8,
    )
