"""Root LangGraph orchestration graph.

The Project Manager acts as the supervisor; specialist subgraphs are mounted
as nodes. Routing happens via the supervisor's ``next_agent`` decision plus
conditional edges from this root graph.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from backend.agents.coordinator.subgraph import build_coordinator_subgraph
from backend.agents.developers.backend_dev.subgraph import build_backend_dev_subgraph
from backend.agents.developers.devops.subgraph import build_devops_subgraph
from backend.agents.developers.frontend_dev.subgraph import build_frontend_dev_subgraph
from backend.agents.developers.qa.subgraph import build_qa_subgraph
from backend.agents.leads.lead.subgraph import build_lead_subgraph
from backend.agents.project_manager.supervisor import build_project_manager_node
from backend.agents.researcher.subgraph import build_researcher_subgraph
from backend.agents.state import SystemState

AGENT_NODES = (
    "researcher",
    "lead",           # Merged backend/frontend lead
    "coordinator",    # Persists plans to DB
    "backend_dev",
    "frontend_dev",
    "devops",
    "qa",
)


def _route_from_pm(state: SystemState) -> str:
    """Conditional edge: dispatch from the project manager based on its decision.

    The PM updates ``state.next_agent`` to one of the registered agent
    names, ``"end"`` to terminate, or ``"pm"`` to loop back for another
    supervisor turn after a tool call.
    """
    target = (state.next_agent or "").lower()
    if target == "end":
        return END
    if target in AGENT_NODES:
        return target
    return "project_manager"  # default: keep planning


# Compiled graphs, keyed by the identity of the checkpointer they were built
# with. Building one compiles seven specialist subgraphs and re-reads settings;
# the API layer did that on every /state, /interrupts and /checkpoints poll.
#
# Caching is only safe because specialist subgraphs hold no per-run mutable
# state — see the note in ``runner.build_specialist_subgraph``. If you add
# closure state there, this cache will leak it across projects.
_GRAPH_CACHE: dict[int, Any] = {}


def build_root_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the full multi-agent orchestration graph (memoized)."""
    key = id(checkpointer)
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    compiled = _compile_root_graph(checkpointer)
    # Bound: one entry per distinct saver, and there is normally exactly one.
    if len(_GRAPH_CACHE) > 8:
        _GRAPH_CACHE.clear()
    _GRAPH_CACHE[key] = compiled
    return compiled


def clear_graph_cache() -> None:
    _GRAPH_CACHE.clear()


def _compile_root_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph = StateGraph(SystemState)

    # Supervisor
    graph.add_node("project_manager", build_project_manager_node())

    # Specialist subgraphs (each compiled independently)
    graph.add_node("researcher", build_researcher_subgraph())
    graph.add_node("lead", build_lead_subgraph())  # Merged lead
    graph.add_node("coordinator", build_coordinator_subgraph())
    graph.add_node("backend_dev", build_backend_dev_subgraph())
    graph.add_node("frontend_dev", build_frontend_dev_subgraph())
    graph.add_node("devops", build_devops_subgraph())
    graph.add_node("qa", build_qa_subgraph())

    graph.add_edge(START, "project_manager")
    graph.add_conditional_edges(
        "project_manager",
        _route_from_pm,
        {
            "researcher": "researcher",
            "lead": "lead",
            "coordinator": "coordinator",
            "backend_dev": "backend_dev",
            "frontend_dev": "frontend_dev",
            "devops": "devops",
            "qa": "qa",
            "project_manager": "project_manager",
            END: END,
        },
    )

    # All specialists return control to the PM
    for node in AGENT_NODES:
        graph.add_edge(node, "project_manager")

    return graph.compile(checkpointer=checkpointer)


__all__ = ["build_root_graph", "clear_graph_cache"]
