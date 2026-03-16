"""
LangGraph graph — Phase 1 + Phase 2.

Full graph:
  START
    → discover_catalog
    → analyze_estate
    → rank_opportunities
    → human_checkpoint          ⏸ approve opportunity
    → plan_features
    → generate_code
    → human_checkpoint_code     ⏸ review generated notebooks
    → write_bundle
  END
"""

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.state import AgentState
from agent.nodes import discover_catalog, analyze_estate, rank_opportunities, human_checkpoint
from agent.code_gen_nodes import plan_features, generate_code, human_checkpoint_code, write_bundle
from tools.workspace_context import WorkspaceContext, validate_workspace

logger = logging.getLogger(__name__)

_checkpointer = MemorySaver()
_graph = None


def build_graph(checkpointer=None):
    """
    Build and compile the full agent graph (Phase 1 + Phase 2).
    Swap MemorySaver for a Delta-backed checkpointer for Databricks Apps production.
    """
    builder = StateGraph(AgentState)

    # Phase 1 nodes
    builder.add_node("discover_catalog", discover_catalog)
    builder.add_node("analyze_estate", analyze_estate)
    builder.add_node("rank_opportunities", rank_opportunities)
    builder.add_node("human_checkpoint", human_checkpoint)

    # Phase 2 nodes
    builder.add_node("plan_features", plan_features)
    builder.add_node("generate_code", generate_code)
    builder.add_node("human_checkpoint_code", human_checkpoint_code)
    builder.add_node("write_bundle", write_bundle)

    # Edges — Phase 1
    builder.add_edge(START, "discover_catalog")
    builder.add_edge("discover_catalog", "analyze_estate")
    builder.add_edge("analyze_estate", "rank_opportunities")
    builder.add_edge("rank_opportunities", "human_checkpoint")

    # Edges — Phase 1 → Phase 2
    builder.add_edge("human_checkpoint", "plan_features")
    builder.add_edge("plan_features", "generate_code")
    builder.add_edge("generate_code", "human_checkpoint_code")
    builder.add_edge("human_checkpoint_code", "write_bundle")
    builder.add_edge("write_bundle", END)

    return builder.compile(checkpointer=checkpointer or _checkpointer)


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Graph state snapshot helper ──────────────────────────────────────────────

def _snapshot_to_status(snapshot) -> str:
    """Map the current graph position to a user-facing status string."""
    if not snapshot or not snapshot.values:
        return "not_found"
    if not snapshot.next:
        return "completed"
    next_node = snapshot.next[0] if snapshot.next else ""
    if next_node == "plan_features":
        return "awaiting_approval"         # paused at human_checkpoint (opp approval)
    if next_node == "write_bundle":
        return "awaiting_code_review"      # paused at human_checkpoint_code
    return "running"


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
    Start a full run (Phase 1 + Phase 2).
    Returns after the first human_checkpoint (opportunity approval).
    """
    ctx = WorkspaceContext(host=host, token=token, catalog=catalog, schema=schema, cluster_id=cluster_id)

    validation = validate_workspace(ctx)
    if not validation["valid"]:
        return {"status": "error", "error": validation["error"], "field": validation.get("field"), "run_id": run_id}

    initial_state: AgentState = {
        "workspace": ctx,
        "tables": [],
        "estate_summary": "",
        "opportunities": [],
        "approved_opportunity": None,
        "feature_plan": None,
        "generated_artifacts": [],
        "bundle_written": False,
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
    status = _snapshot_to_status(snapshot)

    return {
        "status": status,
        "run_id": run_id,
        "opportunities": snapshot.values.get("opportunities", []),
    }


def approve_opportunity(run_id: str, selected_rank: int) -> dict:
    """
    Resume from opportunity approval checkpoint.
    Graph then runs: plan_features → generate_code → human_checkpoint_code (pauses again).
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        graph.invoke(Command(resume={"selected_rank": selected_rank}), config=config)
    except Exception as e:
        logger.error("Graph resume (approve_opportunity) failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    snapshot = graph.get_state(config)
    status = _snapshot_to_status(snapshot)
    values = snapshot.values

    result = {
        "status": status,
        "run_id": run_id,
        "approved_opportunity": values.get("approved_opportunity"),
        "error": values.get("error"),
    }

    if status == "awaiting_code_review":
        artifacts = values.get("generated_artifacts", [])
        result["notebooks"] = [
            {
                "filename": a["filename"],
                "filepath": a["filepath"],
                "content": a["content"],
            }
            for a in artifacts if a["filename"].endswith(".py")
        ]
        result["job_yamls"] = [
            {"filename": a["filename"], "filepath": a["filepath"]}
            for a in artifacts if a["filename"].endswith(".yml")
        ]

    return result


def approve_code(run_id: str, action: str = "approve", instructions: str = "") -> dict:
    """
    Resume from code review checkpoint.
    action="approve" → write bundle to disk.
    action="regenerate" → (Phase 4) provide instructions to regenerate.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        result = graph.invoke(
            Command(resume={"action": action, "instructions": instructions}),
            config=config,
        )
    except Exception as e:
        logger.error("Graph resume (approve_code) failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    snapshot = graph.get_state(config)
    values = snapshot.values

    return {
        "status": "completed" if values.get("bundle_written") else "error",
        "run_id": run_id,
        "bundle_written": values.get("bundle_written", False),
        "error": values.get("error"),
        "artifacts_written": [
            a["filepath"] for a in values.get("generated_artifacts", [])
        ],
    }


def get_run_state(run_id: str) -> dict:
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return {"status": "not_found", "run_id": run_id}

    status = _snapshot_to_status(snapshot)
    return {"run_id": run_id, "status": status, "values": snapshot.values}
