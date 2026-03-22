"""
LangGraph graph — Phase 1 + Trust Layer + Phase 2.

Full graph:
  START
    → discover_catalog → analyze_estate → rank_opportunities
    → human_checkpoint              ⏸ approve opportunity
    → dry_run_explain
    → generate_business_brief       (CTO pre-code brief — no LLM, deterministic)
    → dry_run_checkpoint            ⏸ confirm dry run before code gen
    → plan_features → generate_code → compute_risk_scorecard
    → human_checkpoint_code         ⏸ review notebooks + risk scorecard
    → write_bundle → generate_exec_summary  (full post-bundle SUMMARY.md)
  END
"""

import logging
import os
import sqlite3
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from agent.state import AgentState
from agent.nodes import discover_catalog, analyze_estate, rank_opportunities, human_checkpoint
from agent.code_gen_nodes import plan_features, generate_code, human_checkpoint_code, write_bundle
from agent.trust_nodes import (
    dry_run_explain, generate_business_brief, dry_run_checkpoint,
    compute_risk_scorecard, generate_exec_summary,
)
from tools.workspace_context import WorkspaceContext, validate_workspace

logger = logging.getLogger(__name__)

_checkpointer = None
_graph = None


def _get_checkpointer():
    """
    Return a durable checkpointer.
    SQLite (file-backed) for local dev — survives API restarts.
    Falls back to MemorySaver if langgraph-checkpoint-sqlite is not installed.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = os.getenv("CHECKPOINT_DB_PATH", "./data/checkpoints.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
        logger.info("Checkpointer: SQLite at %s", db_path)
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        logger.warning("langgraph-checkpoint-sqlite not installed — using in-memory checkpointer")

    return _checkpointer


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    # Phase 1
    builder.add_node("discover_catalog", discover_catalog)
    builder.add_node("analyze_estate", analyze_estate)
    builder.add_node("rank_opportunities", rank_opportunities)
    builder.add_node("human_checkpoint", human_checkpoint)

    # Trust layer
    builder.add_node("dry_run_explain", dry_run_explain)
    builder.add_node("generate_business_brief", generate_business_brief)
    builder.add_node("dry_run_checkpoint", dry_run_checkpoint)
    builder.add_node("compute_risk_scorecard", compute_risk_scorecard)
    builder.add_node("generate_exec_summary", generate_exec_summary)

    # Phase 2
    builder.add_node("plan_features", plan_features)
    builder.add_node("generate_code", generate_code)
    builder.add_node("human_checkpoint_code", human_checkpoint_code)
    builder.add_node("write_bundle", write_bundle)

    # ── Edges ────────────────────────────────────────────────────────────────
    builder.add_edge(START, "discover_catalog")
    builder.add_edge("discover_catalog", "analyze_estate")
    builder.add_edge("analyze_estate", "rank_opportunities")
    builder.add_edge("rank_opportunities", "human_checkpoint")

    # human_checkpoint ⏸ → dry run → business brief → confirmation
    builder.add_edge("human_checkpoint", "dry_run_explain")
    builder.add_edge("dry_run_explain", "generate_business_brief")
    builder.add_edge("generate_business_brief", "dry_run_checkpoint")

    # dry_run_checkpoint ⏸ → code gen
    builder.add_edge("dry_run_checkpoint", "plan_features")
    builder.add_edge("plan_features", "generate_code")
    builder.add_edge("generate_code", "compute_risk_scorecard")
    builder.add_edge("compute_risk_scorecard", "human_checkpoint_code")

    # human_checkpoint_code ⏸ → write + summarize
    builder.add_edge("human_checkpoint_code", "write_bundle")
    builder.add_edge("write_bundle", "generate_exec_summary")
    builder.add_edge("generate_exec_summary", END)

    return builder.compile(checkpointer=checkpointer or _get_checkpointer())


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Status mapping ────────────────────────────────────────────────────────────

def _snapshot_to_status(snapshot) -> str:
    """
    snapshot.next == the interrupted node name when interrupt() was called inside it.
    """
    if not snapshot or not snapshot.values:
        return "not_found"
    if not snapshot.next:
        return "completed"
    next_node = snapshot.next[0] if snapshot.next else ""
    if next_node == "human_checkpoint":
        return "awaiting_approval"
    if next_node == "dry_run_checkpoint":
        return "awaiting_dry_run_confirmation"
    if next_node == "human_checkpoint_code":
        return "awaiting_code_review"
    return "running"


# ── Public API ────────────────────────────────────────────────────────────────

def run_discovery(
    run_id: str,
    host: str = "",
    token: str = "",
    catalog: str = "",
    schema: str = "",
    cluster_id: str = "",
) -> dict:
    """Start a full run. Returns after first human_checkpoint (opportunity approval)."""
    ctx = WorkspaceContext(host=host, token=token, catalog=catalog, schema=schema, cluster_id=cluster_id)

    validation = validate_workspace(ctx)
    if not validation["valid"]:
        return {"status": "error", "error": validation["error"], "field": validation.get("field"), "run_id": run_id}

    initial_state: AgentState = {
        # Store workspace as plain dict so state is JSON-serializable for SQLite checkpointer
        "workspace": ctx.model_dump(),
        "tables": [], "estate_summary": "",
        "opportunities": [], "approved_opportunity": None,
        "dry_run_plan": None, "risk_scorecard": None, "exec_summary": "",
        "feature_plan": None, "generated_artifacts": [],
        "bundle_written": False, "error": None,
    }

    config = {"configurable": {"thread_id": run_id}}
    graph = _get_graph()

    try:
        graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error("Graph failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    snapshot = graph.get_state(config)

    # tables are already plain dicts (converted in discover_catalog node)
    return {
        "status": _snapshot_to_status(snapshot),
        "run_id": run_id,
        "opportunities": snapshot.values.get("opportunities", []),
        "tables": snapshot.values.get("tables", []),
        "estate_summary": snapshot.values.get("estate_summary", ""),
    }


def approve_opportunity(run_id: str, selected_rank: int) -> dict:
    """
    Resume from opportunity approval.
    Graph runs dry_run_explain then pauses at dry_run_checkpoint.
    Returns the dry run plan for user review.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        graph.invoke(Command(resume={"selected_rank": selected_rank}), config=config)
    except Exception as e:
        logger.error("approve_opportunity failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    snapshot = graph.get_state(config)
    values = snapshot.values
    return {
        "status": _snapshot_to_status(snapshot),
        "run_id": run_id,
        "approved_opportunity": values.get("approved_opportunity"),
        "dry_run_plan": values.get("dry_run_plan"),
        "exec_summary": values.get("exec_summary", ""),   # CTO business brief
        "error": values.get("error"),
    }


def confirm_dry_run(run_id: str) -> dict:
    """
    Resume from dry run confirmation.
    Graph runs plan_features → generate_code → compute_risk_scorecard
    then pauses at human_checkpoint_code with notebooks + scorecard.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        graph.invoke(Command(resume={"confirmed": True}), config=config)
    except Exception as e:
        logger.error("confirm_dry_run failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    snapshot = graph.get_state(config)
    values = snapshot.values

    result = {
        "status": _snapshot_to_status(snapshot),
        "run_id": run_id,
        "approved_opportunity": values.get("approved_opportunity"),
        "risk_scorecard": values.get("risk_scorecard"),
        "error": values.get("error"),
    }

    if result["status"] == "awaiting_code_review":
        artifacts = values.get("generated_artifacts", [])
        result["notebooks"] = [
            {"filename": a["filename"], "filepath": a["filepath"], "content": a["content"]}
            for a in artifacts if a["filename"].endswith(".py")
        ]
        result["job_yamls"] = [
            {"filename": a["filename"], "filepath": a["filepath"]}
            for a in artifacts if a["filename"].endswith(".yml")
        ]

    return result


def approve_code(run_id: str, action: str = "approve", instructions: str = "") -> dict:
    """Resume from code review. Writes bundle + generates exec summary."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        graph.invoke(Command(resume={"action": action, "instructions": instructions}), config=config)
    except Exception as e:
        logger.error("approve_code failed: %s", e)
        return {"status": "error", "error": str(e), "run_id": run_id}

    snapshot = graph.get_state(config)
    values = snapshot.values
    return {
        "status": "completed" if values.get("bundle_written") else "error",
        "run_id": run_id,
        "bundle_written": values.get("bundle_written", False),
        "exec_summary": values.get("exec_summary", ""),
        "error": values.get("error"),
        "artifacts_written": [a["filepath"] for a in values.get("generated_artifacts", [])],
    }


def get_run_state(run_id: str) -> dict:
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return {"status": "not_found", "run_id": run_id}
    return {"run_id": run_id, "status": _snapshot_to_status(snapshot), "values": snapshot.values}


def get_run_rehydrate(run_id: str) -> dict:
    """
    Return a clean UI-ready payload from a persisted checkpoint.
    Called by GET /runs/{id}/rehydrate to restore Streamlit session state after a page refresh.
    """
    graph = _get_graph()
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return {"status": "not_found", "run_id": run_id}

    values = snapshot.values
    status = _snapshot_to_status(snapshot)
    bundle_written = values.get("bundle_written", False)
    workspace = values.get("workspace", {})

    # exec_summary holds the CTO brief before bundle write, full summary after
    raw_exec_summary = values.get("exec_summary", "")
    business_brief = raw_exec_summary if not bundle_written else ""
    exec_summary = raw_exec_summary if bundle_written else ""

    # Notebooks from generated_artifacts (.py files only)
    artifacts = values.get("generated_artifacts", [])
    notebooks = [
        {"filename": a["filename"], "filepath": a["filepath"], "content": a["content"]}
        for a in artifacts if a["filename"].endswith(".py")
    ]

    return {
        "status": status,
        "run_id": run_id,
        "catalog": workspace.get("catalog", ""),
        "schema": workspace.get("schema", ""),
        "tables": values.get("tables", []),
        "estate_summary": values.get("estate_summary", ""),
        "opportunities": values.get("opportunities", []),
        "approved_opportunity": values.get("approved_opportunity"),
        "dry_run_plan": values.get("dry_run_plan"),
        "business_brief": business_brief,
        "risk_scorecard": values.get("risk_scorecard"),
        "notebooks": notebooks,
        "exec_summary": exec_summary,
        "bundle_written": bundle_written,
    }
