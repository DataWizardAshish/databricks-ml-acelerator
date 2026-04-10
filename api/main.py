"""
FastAPI backend for the ML Accelerator.

Endpoints:
  GET  /health
  GET  /browse/catalogs
  GET  /browse/schemas
  GET  /runs/history                      — recent runs list (episodic memory)
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

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from agent.graph import (
    run_discovery, approve_opportunity, confirm_dry_run,
    approve_code, get_run_state, get_run_rehydrate,
    record_run_history, get_run_history,
)
from tools.workspace_context import WorkspaceContext, list_catalogs, list_schemas
from config.settings import get_settings
from audit.store import get_audit_store
from audit.verifier import ChainVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_current_user(request: Request) -> dict:
    """
    Extracts the end-user's identity from Databricks Apps OBO headers.
    In Databricks Apps, these are injected by the platform for every request.
    Falls back to empty strings in local dev (no headers present).
    """
    return {
        "email": request.headers.get("X-Forwarded-Email", ""),
        "token": request.headers.get("X-Forwarded-Access-Token", ""),
    }


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


class AuditTrailResponse(BaseModel):
    run_id: str
    event_count: int
    chain_valid: bool
    events: list[dict]


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health(request: Request):
    settings = get_settings()
    user = get_current_user(request)
    return {
        "status": "ok",
        "version": "0.4.0",
        "default_workspace": settings.databricks_host or "~/.databrickscfg DEFAULT",
        "llm_endpoint": settings.llm_endpoint_name,
        "default_catalog": settings.uc_catalog,
        "default_schema": settings.uc_discovery_schema,
        "authenticated_user": user["email"] or "local-dev",
    }


# ── Browse ───────────────────────────────────────────────────────────────────

@app.get("/browse/catalogs")
def browse_catalogs(host: str = Query(default=""), token: str = Query(default="")):
    # Browse uses the App's own identity (service principal in Apps, ~/.databrickscfg
    # in local dev). OBO token is intentionally NOT used here — listing catalog
    # metadata does not require acting as the end user, and OAuth JWTs must not be
    # passed as query params (URL length + auth_type mismatch).
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

@app.get("/runs/history")
def list_run_history(limit: int = Query(default=20, ge=1, le=100)):
    """Return recent runs ordered by last update, newest first (episodic memory)."""
    return {"runs": get_run_history(limit=limit)}


@app.post("/runs", status_code=202)
def start_run(request: Request, body: StartRunRequest = StartRunRequest()):
    """Start a new discovery + recommendation run (Phase 1)."""
    user = get_current_user(request)
    run_id = str(uuid.uuid4())
    # OBO token flows into WorkspaceContext.token and is used ONLY by get_sql_connection()
    # (databricks.sql connector — Path A). It is never passed to get_workspace_client()
    # (WorkspaceClient — Path B), which avoids the M2M + PAT auth conflict entirely.
    token = body.workspace.token or user["token"]
    user_email = user["email"]
    logger.info("Starting run %s | user=%s catalog=%s schema=%s", run_id, user_email or "local-dev", body.catalog, body.schema)
    return run_discovery(
        run_id=run_id,
        host=body.workspace.host,
        token=token,
        catalog=body.catalog,
        schema=body.schema,
        user_email=user_email,
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
    result = approve_opportunity(run_id=run_id, selected_rank=body.selected_rank)
    if result.get("approved_opportunity"):
        use_case = result["approved_opportunity"].get("use_case", "")
        record_run_history(run_id, use_case=use_case, status=result.get("status", ""))
    return result


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
    result = confirm_dry_run(run_id=run_id)
    record_run_history(run_id, status=result.get("status", ""))
    return result


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
    result = approve_code(run_id=run_id, action=body.action, instructions=body.instructions)
    record_run_history(run_id, status=result.get("status", ""))
    return result


@app.get("/runs/{run_id}/audit", response_model=AuditTrailResponse)
def get_audit_trail(run_id: str):
    """Return the complete ordered audit trail for a run, with hash chain integrity check."""
    store = get_audit_store()
    events = store.get_events(run_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No audit trail found for run {run_id}")
    result = ChainVerifier(store).verify(run_id)
    return AuditTrailResponse(
        run_id=run_id,
        event_count=len(events),
        chain_valid=result["valid"],
        events=events,
    )


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
