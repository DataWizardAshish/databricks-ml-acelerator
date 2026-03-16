"""
FastAPI backend for the ML Accelerator.

Endpoints:
  POST /runs            — start a discovery run
  GET  /runs/{run_id}   — get run status + state
  POST /runs/{run_id}/approve  — approve an opportunity and resume the graph
"""

import logging
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import run_discovery, approve_opportunity, get_run_state
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Databricks ML Accelerator",
    description="Agentic ML acceleration — connects to Unity Catalog and recommends ML opportunities.",
    version="0.1.0",
)


# ── Request / Response models ────────────────────────────────────────────────

class StartRunRequest(BaseModel):
    catalog: Optional[str] = None   # defaults to settings.uc_catalog
    schema: Optional[str] = None    # defaults to settings.uc_discovery_schema


class ApproveRequest(BaseModel):
    selected_rank: int = 1


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "workspace": settings.databricks_host or "~/.databrickscfg DEFAULT",
        "llm_endpoint": settings.llm_endpoint_name,
        "uc_catalog": settings.uc_catalog,
        "uc_schema": settings.uc_discovery_schema,
    }


@app.post("/runs", status_code=202)
def start_run(body: StartRunRequest = StartRunRequest()):
    """Start a new discovery + recommendation run."""
    run_id = str(uuid.uuid4())
    logger.info("Starting run %s", run_id)
    result = run_discovery(
        run_id=run_id,
        catalog=body.catalog,
        schema=body.schema,
    )
    return result


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """Get the current state of a run."""
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

    result = approve_opportunity(run_id=run_id, selected_rank=body.selected_rank)
    return result
