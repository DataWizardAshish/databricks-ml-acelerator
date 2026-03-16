"""
FastAPI backend for the ML Accelerator.

Endpoints:
  GET  /health
  GET  /browse/catalogs          — list UC catalogs (for UI dropdowns)
  GET  /browse/schemas           — list UC schemas in a catalog
  POST /runs                     — start a discovery + recommendation run
  GET  /runs/{run_id}            — get run status
  POST /runs/{run_id}/approve    — approve an opportunity and resume the graph
"""

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from agent.graph import run_discovery, approve_opportunity, get_run_state
from tools.workspace_context import WorkspaceContext, list_catalogs, list_schemas
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Databricks ML Accelerator",
    description="Agentic ML acceleration — connects to Unity Catalog and recommends ML opportunities.",
    version="0.2.0",
)


# ── Request / Response models ────────────────────────────────────────────────

class WorkspaceParams(BaseModel):
    """Workspace connection overrides. Empty = use ~/.databrickscfg DEFAULT profile."""
    host: str = ""
    token: str = ""


class StartRunRequest(BaseModel):
    workspace: WorkspaceParams = WorkspaceParams()
    catalog: str = ""
    schema: str = ""


class ApproveRequest(BaseModel):
    selected_rank: int = 1


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "default_workspace": settings.databricks_host or "~/.databrickscfg DEFAULT",
        "llm_endpoint": settings.llm_endpoint_name,
        "default_catalog": settings.uc_catalog,
        "default_schema": settings.uc_discovery_schema,
    }


# ── Browse (catalog + schema dropdowns) ─────────────────────────────────────

@app.get("/browse/catalogs")
def browse_catalogs(
    host: str = Query(default=""),
    token: str = Query(default=""),
):
    """List all UC catalogs accessible to the given credentials."""
    ctx = WorkspaceContext(host=host, token=token)
    try:
        catalogs = list_catalogs(ctx)
        return {"catalogs": catalogs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/browse/schemas")
def browse_schemas(
    catalog: str = Query(...),
    host: str = Query(default=""),
    token: str = Query(default=""),
):
    """List all UC schemas in a catalog."""
    ctx = WorkspaceContext(host=host, token=token)
    try:
        schemas = list_schemas(ctx, catalog=catalog)
        return {"catalog": catalog, "schemas": schemas}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Runs ─────────────────────────────────────────────────────────────────────

@app.post("/runs", status_code=202)
def start_run(body: StartRunRequest = StartRunRequest()):
    """Start a new discovery + recommendation run."""
    run_id = str(uuid.uuid4())
    logger.info("Starting run %s for catalog=%s schema=%s", run_id, body.catalog, body.schema)

    result = run_discovery(
        run_id=run_id,
        host=body.workspace.host,
        token=body.workspace.token,
        catalog=body.catalog,
        schema=body.schema,
    )
    return result


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    state = get_run_state(run_id)
    if state.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return state


@app.post("/runs/{run_id}/approve")
def approve_run(run_id: str, body: ApproveRequest = ApproveRequest()):
    """Approve an ML opportunity and resume the agent graph."""
    state = get_run_state(run_id)
    if state.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if state.get("status") != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} is not awaiting approval (status: {state.get('status')})",
        )
    return approve_opportunity(run_id=run_id, selected_rank=body.selected_rank)
