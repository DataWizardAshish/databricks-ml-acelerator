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

from agent.state import AgentState, MLOpportunity
from agent.nodes import discover_catalog, analyze_estate, rank_opportunities, human_checkpoint
from config.settings import get_settings

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None):
    """
    Build and compile the Phase 1 agent graph.
    checkpointer: MemorySaver for local dev, pass a Delta-based checkpointer for prod.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

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

    return builder.compile(checkpointer=checkpointer)


# ── Convenience runner ───────────────────────────────────────────────────────

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_discovery(
    run_id: str,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
) -> dict:
    """
    Start a discovery run. Returns the state snapshot after the human_checkpoint
    interrupt is hit. Caller must then call approve_opportunity() to resume.

    Returns: {"status": "awaiting_approval", "opportunities": [...], "run_id": ...}
             or {"status": "error", "error": "..."}
    """
    settings = get_settings()
    graph = _get_graph()

    initial_state: AgentState = {
        "catalog": catalog or settings.uc_catalog,
        "schema": schema or settings.uc_discovery_schema,
        "tables": [],
        "estate_summary": "",
        "opportunities": [],
        "approved_opportunity": None,
        "error": None,
    }

    config = {"configurable": {"thread_id": run_id}}

    # Run until interrupt
    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error("Graph execution failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    # Check current graph state — if interrupted, result is the state at interrupt
    snapshot = graph.get_state(config)
    is_interrupted = bool(snapshot.next)  # non-empty means graph is paused

    if is_interrupted:
        return {
            "status": "awaiting_approval",
            "run_id": run_id,
            "opportunities": snapshot.values.get("opportunities", []),
        }

    return {
        "status": "completed",
        "run_id": run_id,
        "approved_opportunity": result.get("approved_opportunity"),
        "error": result.get("error"),
    }


def approve_opportunity(run_id: str, selected_rank: int) -> dict:
    """
    Resume a paused graph run with the user's approved opportunity.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        result = graph.invoke(
            Command(resume={"selected_rank": selected_rank}),
            config=config,
        )
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
    """Get current state of a run by ID."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if not snapshot:
        return {"status": "not_found", "run_id": run_id}

    is_interrupted = bool(snapshot.next)
    return {
        "run_id": run_id,
        "status": "awaiting_approval" if is_interrupted else "completed",
        "values": snapshot.values,
    }
