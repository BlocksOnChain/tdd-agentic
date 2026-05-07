"""Root LangGraph orchestration graph.

The Project Manager acts as the supervisor; specialist subgraphs are mounted
as nodes. Routing happens via the supervisor's ``next_agent`` decision plus
conditional edges from this root graph.
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from backend.agents.developers.backend_dev.subgraph import build_backend_dev_subgraph
from backend.agents.developers.devops.subgraph import build_devops_subgraph
from backend.agents.developers.frontend_dev.subgraph import build_frontend_dev_subgraph
from backend.agents.developers.qa.subgraph import build_qa_subgraph
from backend.agents.leads.backend_lead.subgraph import build_backend_lead_subgraph
from backend.agents.leads.frontend_lead.subgraph import build_frontend_lead_subgraph
from backend.agents.project_manager.supervisor import build_project_manager_node
from backend.agents.researcher.subgraph import build_researcher_subgraph
from backend.agents.state import SystemState

AGENT_NODES = (
    "researcher",
    "backend_lead",
    "frontend_lead",
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


def build_root_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the full multi-agent orchestration graph."""
    graph = StateGraph(SystemState)

    # Supervisor
    graph.add_node("project_manager", build_project_manager_node())

    # Specialist subgraphs (each compiled independently)
    graph.add_node("researcher", build_researcher_subgraph())
    graph.add_node("backend_lead", build_backend_lead_subgraph())
    graph.add_node("frontend_lead", build_frontend_lead_subgraph())
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
            "backend_lead": "backend_lead",
            "frontend_lead": "frontend_lead",
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


__all__ = ["build_root_graph"]
