# Deployment Plan — Databricks ML Accelerator

> This document is the authoritative reference for Phase 3 (Databricks Apps packaging and deployment).
> Read this before implementing any deployment-related code.

---

## Core Principle

**There is no external server.** The product deploys entirely inside the customer's Databricks workspace as a **Databricks App**. FastAPI and Streamlit both run in the same managed app container. Streamlit calls FastAPI on `localhost:8000`. No cloud hosting. No VMs. No Docker registry. Databricks manages the runtime.

---

## What Is a Databricks App

- A managed serverless runtime for web applications, running inside the workspace
- Accessible at `https://{workspace}.cloud.databricks.com/apps/{app-name}`
- SSO inherited automatically — users authenticate with their existing Databricks identity
- Environment variables `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are injected automatically
- Has native access to Unity Catalog, MLflow, and serving endpoints via the user's identity
- Deployed and managed via Databricks CLI: `databricks apps deploy`

---

## Current State (Before Phase 3)

| Component | Local Dev | Missing for App |
|---|---|---|
| FastAPI backend | `uvicorn api.main:app --port 8000` | Startup script |
| Streamlit UI | `streamlit run ui/app.py --port 8501` | Startup script |
| Auth | `~/.databrickscfg` DEFAULT profile | Env var detection |
| `API_BASE` in UI | Hardcoded `http://localhost:8000` | Make configurable |
| LangGraph checkpointer | `MemorySaver` (in-process memory) | Delta table for prod |
| State persistence | In-memory only | UC Delta tables |

---

## Files to Create (Phase 3 Deliverables)

### 1. `app.yaml` (root of repo)

Databricks Apps reads this to know how to start the application.

```yaml
command: ["bash", "start_app.sh"]

env:
  - name: LLM_ENDPOINT_NAME
    value: "ds-brand-funnel-ai_claude-sonnet-4-5_chat_dev"
  - name: UC_CATALOG
    value: "lakehouse_dev"
  - name: UC_DISCOVERY_SCHEMA
    value: "ds_brand_funnel_ai_lhdev"
  - name: UC_OUTPUT_SCHEMA
    value: "ml_accelerator"
  - name: API_BASE
    value: "http://localhost:8000"
```

**Note:** `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are injected automatically by Databricks Apps — do NOT set them in `app.yaml`.

---

### 2. `start_app.sh` (root of repo)

```bash
#!/bin/bash
set -e

# Start FastAPI in background on port 8000
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info &

# Give FastAPI a moment to bind
sleep 2

# Start Streamlit as foreground process on port 8501
# Databricks Apps proxies the exposed port — typically 8501 for Streamlit
exec streamlit run ui/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

Make executable: `chmod +x start_app.sh`

---

### 3. Auth adapter in `config/settings.py`

**Problem:** `~/.databrickscfg` does not exist in the Databricks App container.

**Current code** reads from `.databrickscfg` DEFAULT profile as fallback.
**Required:** Detect if running as Databricks App and use injected env vars instead.

```python
import os

class Settings(BaseSettings):
    # Auth — two sources:
    # 1. Databricks App: DATABRICKS_HOST + DATABRICKS_TOKEN injected automatically
    # 2. Local dev: read from ~/.databrickscfg DEFAULT profile
    databricks_host: str = Field(default="")
    databricks_token: str = Field(default="")

    @validator("databricks_host", pre=True, always=True)
    def resolve_host(cls, v):
        # Databricks Apps injects DATABRICKS_HOST
        return v or os.getenv("DATABRICKS_HOST", "")

    @validator("databricks_token", pre=True, always=True)
    def resolve_token(cls, v):
        # Databricks Apps injects DATABRICKS_TOKEN
        return v or os.getenv("DATABRICKS_TOKEN", "")
```

**Also update `tools/workspace_context.py`:** When `host` and `token` are empty strings,
`WorkspaceContext` currently falls back to `~/.databrickscfg`. In the app container,
it should fall back to `DATABRICKS_HOST` / `DATABRICKS_TOKEN` env vars instead.

```python
# In WorkspaceContext.__post_init__ or get_workspace_client():
if not self.host:
    self.host = os.getenv("DATABRICKS_HOST", "")
if not self.token:
    self.token = os.getenv("DATABRICKS_TOKEN", "")
```

---

### 4. `API_BASE` — make configurable in `ui/app.py`

```python
import os
# Databricks App: both processes in same container → localhost always works
# Local dev: same. This env var exists for future flexibility only.
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
```

---

### 5. LangGraph checkpointer — swap MemorySaver for DeltaCheckpointer (prod)

**Current:** `MemorySaver` — stored in process memory, lost on restart.

**Required for prod:** Delta table checkpointer so runs survive app restarts.

```python
# agent/graph.py — controlled by environment
import os

def build_graph(checkpointer=None):
    if checkpointer is None:
        if os.getenv("DATABRICKS_APPS_ENV"):
            # Production: use Delta-backed checkpointer
            checkpointer = _build_delta_checkpointer()
        else:
            # Local dev: in-memory is fine
            checkpointer = MemorySaver()
    ...

def _build_delta_checkpointer():
    # Uses langgraph-checkpoint-databricks or custom Delta writer
    # Writes to: {UC_CATALOG}.{UC_OUTPUT_SCHEMA}.langgraph_checkpoints
    from langgraph.checkpoint.sqlite import SqliteSaver  # placeholder
    # TODO: implement DeltaCheckpointer using Delta Lake
    # For initial Phase 3, MemorySaver is acceptable — runs are short-lived
    return MemorySaver()
```

**Decision:** For Phase 3 initial deployment, `MemorySaver` is acceptable.
Users complete a full run in a single session (< 30 minutes). Delta checkpointer is Phase 4+ work.

---

## Deployment Steps

### Step 1 — One-time workspace setup (per customer)

```bash
# 1. Customer must have a Claude Sonnet external model endpoint
# This is the ONLY manual setup step. Takes ~10 minutes.
# In Databricks UI: Serving → External Models → Create endpoint
# Provider: Anthropic, Model: claude-sonnet-4-5 (or 4-6 when available)

# 2. Create the output UC schema (agent creates tables on first run)
# SQL:  CREATE SCHEMA IF NOT EXISTS {catalog}.ml_accelerator;
# GRANT ALL PRIVILEGES ON SCHEMA {catalog}.ml_accelerator TO `app_service_principal`;
```

### Step 2 — Deploy the app

```bash
# Install Databricks CLI (>= 0.200)
pip install databricks-cli

# Authenticate (uses ~/.databrickscfg DEFAULT or env vars)
databricks auth login

# Create the app (first time only)
databricks apps create ml-accelerator \
  --description "Agentic ML acceleration — UC discovery to production code"

# Deploy current code
databricks apps deploy ml-accelerator \
  --source-code-path /path/to/databricks-ml-accelerator

# Check deployment status
databricks apps get ml-accelerator
```

### Step 3 — Verify

```bash
# Health check
curl https://{workspace}.cloud.databricks.com/apps/ml-accelerator/health

# Or open in browser — SSO handles auth automatically
# https://{workspace}.cloud.databricks.com/apps/ml-accelerator
```

### Step 4 — Update (redeploy after code changes)

```bash
# Re-deploy (same command, replaces running app)
databricks apps deploy ml-accelerator --source-code-path .

# No restart needed — Databricks handles it
```

---

## Customer Onboarding Runbook (First 3–5 Design Partners)

For each new customer workspace:

| Step | Owner | Action |
|---|---|---|
| 1 | Customer | Provide workspace URL + admin PAT (temporary) |
| 2 | Builder | Create Claude Sonnet external model endpoint in their workspace |
| 3 | Builder | Create `ml_accelerator` UC schema with correct grants |
| 4 | Builder | Deploy app: `databricks apps deploy ml-accelerator` |
| 5 | Builder | Update `app.yaml` with their endpoint name, redeploy |
| 6 | Customer | Access app at their workspace URL — SSO works automatically |
| 7 | Builder | Revoke temporary admin PAT — app uses its own service principal |

**Time per customer: ~45 minutes** (mostly the external model endpoint setup)

---

## Scale Path: Databricks Marketplace

When ready to self-serve (after 3–5 validated design partners):

1. Register as Databricks Technology Partner
2. Package as a Databricks App in Marketplace
3. Customer clicks "Get" in Databricks Marketplace
4. App deploys to their workspace automatically
5. On first launch, app guides customer through the external model endpoint setup
6. Billing flows through Databricks (customer pays on existing Databricks invoice)

**Key requirement for Marketplace:** The app cannot hardcode any endpoint name. The UI must let the user enter or select their Claude endpoint name at setup. This is already partially implemented (the sidebar accepts host/token per session). The endpoint name needs the same treatment.

---

## Environment Variable Reference

| Variable | Set by | Value (example) | Notes |
|---|---|---|---|
| `DATABRICKS_HOST` | Databricks Apps (auto) | `https://workspace.cloud.databricks.com` | Do not set manually |
| `DATABRICKS_TOKEN` | Databricks Apps (auto) | `dapi...` | Do not set manually |
| `LLM_ENDPOINT_NAME` | `app.yaml` | `ds-brand-funnel-ai_claude-sonnet-4-5_chat_dev` | Customer-specific |
| `UC_CATALOG` | `app.yaml` | `lakehouse_dev` | Customer-specific |
| `UC_DISCOVERY_SCHEMA` | `app.yaml` | `ds_brand_funnel_ai_lhdev` | Customer-specific |
| `UC_OUTPUT_SCHEMA` | `app.yaml` | `ml_accelerator` | Created by agent on first run |
| `API_BASE` | `app.yaml` | `http://localhost:8000` | Same container — don't change |
| `DATABRICKS_APPS_ENV` | `app.yaml` | `true` | Used to detect app environment |

---

## What Does NOT Change for Deployment

- `agent/graph.py`, `agent/nodes.py`, `agent/trust_nodes.py`, `agent/code_gen_nodes.py` — no changes
- `api/main.py` — no changes
- `tools/uc_reader.py`, `tools/bundle_writer.py` — no changes
- LangGraph graph structure — no changes
- All API endpoints — no changes

The only code changes are:
1. `config/settings.py` — env var fallback
2. `tools/workspace_context.py` — env var fallback when host/token are empty
3. `ui/app.py` — `API_BASE` from env var (1 line change)

---

## Known Risks

| Risk | Mitigation |
|---|---|
| `MemorySaver` loses runs on app restart | Runs complete in < 30 min; acceptable for Phase 3. Delta checkpointer in Phase 4. |
| External model endpoint name varies per customer | Make `LLM_ENDPOINT_NAME` configurable in UI settings (not just `app.yaml`) |
| App service principal may lack UC permissions | Document required grants in onboarding runbook (step 3) |
| Streamlit port (8501) vs Databricks Apps exposed port | Test with actual Databricks Apps — may need to use port 8080 instead |
| `bundles/` write path inside app container | App container is ephemeral. Bundle write must target UC volume or workspace files API, not local disk. See note below. |

### Critical: Bundle Write Path in App Container

**Current:** `bundle_writer.py` writes to `bundles/` on the local filesystem.

**Problem:** The Databricks App container's local filesystem is ephemeral — files are lost on restart.

**Required fix for Phase 3:**
- Option A (simpler): Write bundles to a Databricks Workspace path via the Files API
  (`/api/2.0/workspace/import` or workspace files)
- Option B (cleaner): Write to a UC Volume (`/Volumes/{catalog}/{schema}/bundles/`)
- Option C (Phase 3 only): Expose a download endpoint that streams the generated files as a ZIP

**Recommendation:** Implement Option B (UC Volume) — it fits the governance model and gives the customer a permanent, versioned artifact location. Path: `/Volumes/{catalog}/ml_accelerator/generated/{use_case}/`.

This is the single most important code change needed before the app can work in production.

---

## Phase 3 Implementation Order

1. Fix `bundle_writer.py` to write to UC Volume (not local disk) — **blocking**
2. Update `tools/workspace_context.py` for env var auth fallback
3. Update `config/settings.py` for env var auth fallback
4. Create `start_app.sh`
5. Create `app.yaml`
6. Make `API_BASE` configurable in `ui/app.py`
7. Test locally by setting `DATABRICKS_HOST` and `DATABRICKS_TOKEN` env vars manually
8. Deploy to own workspace (`adidas-dwhm-lh-wrk-gazelle-dev`)
9. Validate full 6-step flow end-to-end in the app
10. Document external model endpoint setup steps for first customer
