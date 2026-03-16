"""
LangGraph graph for Phase 1: Discovery + Recommendation.

Graph:
  START → discover_catalog → analyze_estate → rank_opportunities → human_checkpoint → END
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.state import AgentState
from agent.nodes import discover_catalog, analyze_estate, rank_opportunities, human_checkpoint
from tools.workspace_context import WorkspaceContext, validate_workspace

logger = logging.getLogger(__name__)

# Module-level graph + checkpointer (MemorySaver for local dev)
_checkpointer = MemorySaver()
_graph = None


def build_graph(checkpointer=None):
    """
    Build and compile the Phase 1 agent graph.
    Pass a Delta-based checkpointer for production Databricks Apps deployment.
    """
    builder = StateGraph(AgentState)

    builder.add_node("discover_catalog", discover_catalog)
    builder.add_node("analyze_estate", analyze_estate)
    builder.add_node("rank_opportunities", rank_opportunities)
    builder.add_node("human_checkpoint", human_checkpoint)

    builder.add_edge(START, "discover_catalog")
    builder.add_edge("discover_catalog", "analyze_estate")
    builder.add_edge("analyze_estate", "rank_opportunities")
    builder.add_edge("rank_opportunities", "human_checkpoint")
    builder.add_edge("human_checkpoint", END)

    return builder.compile(checkpointer=checkpointer or _checkpointer)


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public API ───────────────────────────────────────────────────────────────

def run_discovery(
    run_id: str,
    host: str = "",
    token: str = "",
    catalog: str = "",
    schema: str = "",
    cluster_id: str = "",
) -> dict:
    """
    Start a discovery run with explicit workspace credentials.
    Empty strings fall back to ~/.databrickscfg DEFAULT profile.

    Returns:
      {"status": "awaiting_approval", "opportunities": [...], "run_id": "..."}
      {"status": "error", "error": "...", "run_id": "..."}
    """
    ctx = WorkspaceContext(
        host=host, token=token, catalog=catalog,
        schema=schema, cluster_id=cluster_id,
    )

    # Pre-flight: validate catalog + schema exist before running the graph
    validation = validate_workspace(ctx)
    if not validation["valid"]:
        return {"status": "error", "error": validation["error"], "field": validation.get("field"), "run_id": run_id}

    initial_state: AgentState = {
        "workspace": ctx,
        "tables": [],
        "estate_summary": "",
        "opportunities": [],
        "approved_opportunity": None,
        "error": None,
    }

    config = {"configurable": {"thread_id": run_id}}
    graph = _get_graph()

    try:
        graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error("Graph execution failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    snapshot = graph.get_state(config)
    is_interrupted = bool(snapshot.next)

    if is_interrupted:
        return {
            "status": "awaiting_approval",
            "run_id": run_id,
            "opportunities": snapshot.values.get("opportunities", []),
        }

    return {
        "status": "completed",
        "run_id": run_id,
        "approved_opportunity": snapshot.values.get("approved_opportunity"),
        "error": snapshot.values.get("error"),
    }


def approve_opportunity(run_id: str, selected_rank: int) -> dict:
    """Resume a paused run with the user's approved opportunity."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        result = graph.invoke(Command(resume={"selected_rank": selected_rank}), config=config)
    except Exception as e:
        logger.error("Graph resume failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    return {
        "status": "completed",
        "run_id": run_id,
        "approved_opportunity": result.get("approved_opportunity"),
        "error": result.get("error"),
    }


def get_run_state(run_id: str) -> dict:
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return {"status": "not_found", "run_id": run_id}

    return {
        "run_id": run_id,
        "status": "awaiting_approval" if bool(snapshot.next) else "completed",
        "values": snapshot.values,
    }
