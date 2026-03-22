"""
FastAPI backend for the ML Accelerator.

Endpoints:
  GET  /health
  GET  /browse/catalogs
  GET  /browse/schemas
  POST /runs                              — start discovery run
  GET  /runs/{run_id}                     — get run status
  GET  /runs/{run_id}/rehydrate           — restore full UI state from checkpoint (page refresh)
  POST /runs/{run_id}/approve             — approve ML opportunity → dry run plan
  POST /runs/{run_id}/confirm-dry-run     — confirm dry run → code generation
  POST /runs/{run_id}/approve-code        — approve generated code → write bundle
  POST /runs/{run_id}/ask                 — contextual Q&A scoped to current run state
"""

import logging
import uuid

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from agent.graph import (
    run_discovery, approve_opportunity, confirm_dry_run,
    approve_code, get_run_state, get_run_rehydrate,
)
from tools.workspace_context import WorkspaceContext, list_catalogs, list_schemas
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Databricks ML Accelerator",
    description="Agentic ML acceleration — UC discovery → recommendations → code generation.",
    version="0.4.0",
)


# ── Request / Response models ────────────────────────────────────────────────

class WorkspaceParams(BaseModel):
    host: str = ""
    token: str = ""


class StartRunRequest(BaseModel):
    workspace: WorkspaceParams = WorkspaceParams()
    catalog: str = ""
    schema: str = ""


class ApproveOpportunityRequest(BaseModel):
    selected_rank: int = 1


class ConfirmDryRunRequest(BaseModel):
    pass  # No body needed — user just clicks Confirm


class ApproveCodeRequest(BaseModel):
    action: str = "approve"          # "approve" | "regenerate"
    instructions: str = ""           # only used when action="regenerate"


class AskRequest(BaseModel):
    question: str
    step: str = "general"            # approve_opportunity | dry_run | code_review | done


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "version": "0.4.0",
        "default_workspace": settings.databricks_host or "~/.databrickscfg DEFAULT",
        "llm_endpoint": settings.llm_endpoint_name,
        "default_catalog": settings.uc_catalog,
        "default_schema": settings.uc_discovery_schema,
    }


# ── Browse ───────────────────────────────────────────────────────────────────

@app.get("/browse/catalogs")
def browse_catalogs(host: str = Query(default=""), token: str = Query(default="")):
    ctx = WorkspaceContext(host=host, token=token)
    try:
        return {"catalogs": list_catalogs(ctx)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/browse/schemas")
def browse_schemas(
    catalog: str = Query(...),
    host: str = Query(default=""),
    token: str = Query(default=""),
):
    ctx = WorkspaceContext(host=host, token=token)
    try:
        return {"catalog": catalog, "schemas": list_schemas(ctx, catalog=catalog)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Runs ─────────────────────────────────────────────────────────────────────

@app.post("/runs", status_code=202)
def start_run(body: StartRunRequest = StartRunRequest()):
    """Start a new discovery + recommendation run (Phase 1)."""
    run_id = str(uuid.uuid4())
    logger.info("Starting run %s | catalog=%s schema=%s", run_id, body.catalog, body.schema)
    return run_discovery(
        run_id=run_id,
        host=body.workspace.host,
        token=body.workspace.token,
        catalog=body.catalog,
        schema=body.schema,
    )


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    state = get_run_state(run_id)
    if state.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return state


@app.get("/runs/{run_id}/rehydrate")
def rehydrate_run(run_id: str):
    """
    Return a clean UI-ready payload from a persisted checkpoint.
    Called by the Streamlit UI on page refresh to restore all session state.
    Includes tables, opportunities, approved_opportunity, dry_run_plan,
    generated notebooks, risk_scorecard, exec_summary, bundle_written.
    """
    result = get_run_rehydrate(run_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return result


@app.post("/runs/{run_id}/approve")
def approve_run(run_id: str, body: ApproveOpportunityRequest = ApproveOpportunityRequest()):
    """Approve an ML opportunity → triggers Phase 2 code generation."""
    state = get_run_state(run_id)
    if state.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if state.get("status") != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} status is '{state.get('status')}', expected 'awaiting_approval'",
        )
    return approve_opportunity(run_id=run_id, selected_rank=body.selected_rank)


@app.post("/runs/{run_id}/confirm-dry-run")
def confirm_dry_run_endpoint(run_id: str):
    """Confirm the dry run plan → triggers code generation."""
    state = get_run_state(run_id)
    if state.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if state.get("status") != "awaiting_dry_run_confirmation":
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} status is '{state.get('status')}', expected 'awaiting_dry_run_confirmation'",
        )
    return confirm_dry_run(run_id=run_id)


@app.post("/runs/{run_id}/approve-code")
def approve_run_code(run_id: str, body: ApproveCodeRequest = ApproveCodeRequest()):
    """Approve generated notebooks → writes artifacts to bundles/ directory."""
    state = get_run_state(run_id)
    if state.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if state.get("status") != "awaiting_code_review":
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} status is '{state.get('status')}', expected 'awaiting_code_review'",
        )
    return approve_code(run_id=run_id, action=body.action, instructions=body.instructions)


@app.post("/runs/{run_id}/ask")
def ask_about_run(run_id: str, body: AskRequest):
    """
    Answer a question about the current run using run state as context.
    Scoped to what is known at the current step — not a general Databricks assistant.
    """
    from agent.chat import ask_about_run as _ask

    run_state = get_run_state(run_id)
    if run_state.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    values = run_state.get("values", {})
    workspace_dict = values.get("workspace")
    if not workspace_dict:
        raise HTTPException(status_code=400, detail="Run workspace context not available")
    # workspace stored as plain dict in checkpoint — reconstruct WorkspaceContext for LLM access
    workspace_ctx = WorkspaceContext(**workspace_dict) if isinstance(workspace_dict, dict) else workspace_dict

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = _ask(
        workspace_ctx=workspace_ctx,
        step=body.step,
        question=body.question,
        values=values,
    )
    return {"run_id": run_id, "step": body.step, "question": body.question, "answer": answer}
