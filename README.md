# ⚡ Databricks ML Accelerator

> **Connect your Unity Catalog. Get ML in production in days, not months. With governance already handled.**

A **Databricks-native agentic ML system** that connects to a customer's Unity Catalog, understands their entire data estate, recommends ML opportunities, generates production-ready code, deploys it, and monitors it continuously.

Not a report generator. Not a notebook template. An autonomous ML engineer that lives inside your Databricks workspace.

---

## The Problem

Databricks customers have data in Unity Catalog but the gap between *"data is ready"* and *"model is in production"* is still 3–6 months. The bottleneck is not infrastructure — it's the intelligence layer. This product closes that gap.

| Traditional Approach | This Product |
|---|---|
| Hire ML engineer ($180–250K/year) | Agentic system included |
| 3–6 months to first model | Days to first model |
| Manual RBAC/governance setup | Auto-generated GRANT statements |
| Notebooks that break as jobs | Parameterized, job-safe code |
| No drift monitoring | Continuous monitoring loop |

---

## Architecture

```
Your Databricks Workspace
│
├── Databricks App (this product)
│   ├── Streamlit UI
│   └── FastAPI backend
│       └── LangGraph orchestration engine
│
├── External Model Endpoint
│   └── Claude Sonnet (proxied through Databricks — no data egress)
│
├── Unity Catalog
│   ├── your existing data  ← agent reads metadata only, never raw data
│   └── ml_accelerator schema  ← agent writes outputs here
│       ├── discovered_opportunities
│       ├── generated_code_artifacts
│       └── model_health_scores
│
├── MLflow
│   └── experiments, runs, model registry
│
└── Databricks Jobs (generated + deployed by agent)
    ├── {use_case}_feature_pipeline_job
    ├── {use_case}_training_job
    ├── {use_case}_batch_inference_job
    └── nightly_drift_check_job
```

### Agent Pipeline

```
[Unity Catalog Discovery]   ← REST API only, zero compute
        ↓
[Data Estate Analysis]      ← LLM analysis of table/column metadata
        ↓
[ML Opportunity Ranker]     ← Top 3 use cases ranked by value × readiness
        ↓
⏸ HUMAN CHECKPOINT — approve opportunity
        ↓
[Feature Engineering Planner]   ← Phase 2
        ↓
[Code Generator]                ← Phase 2
        ↓
⏸ HUMAN CHECKPOINT — review generated code
        ↓
[Deployment Executor]           ← Phase 2
        ↓
[Monitoring Setup]              ← Phase 5
        ↓
[Drift Detection Loop] ←────────── runs continuously
```

---

## Project Structure

```
databricks-ml-accelerator/
│
├── agent/                   # LangGraph orchestration
│   ├── graph.py             # Graph definition + public run_discovery() API
│   ├── nodes.py             # 4 nodes: discover → analyze → rank → checkpoint
│   └── state.py             # AgentState TypedDict
│
├── tools/                   # Reusable tools called by agent nodes
│   ├── uc_reader.py         # Unity Catalog metadata reader (no Spark)
│   └── workspace_context.py # Per-request auth context + UC validation
│
├── api/                     # FastAPI backend
│   └── main.py              # /health, /browse/*, /runs/* endpoints
│
├── ui/                      # Streamlit frontend
│   └── app.py               # Connect → browse → discover → approve flow
│
├── config/
│   └── settings.py          # Pydantic settings (PAT now, OAuth-ready)
│
├── bundles/                 # Databricks Asset Bundle (Phase 2 output lands here)
│   ├── databricks.yml       # Bundle config: dev + prod targets
│   ├── resources/jobs/      # Generated job YAML definitions
│   └── src/                 # Generated notebooks per use case
│
├── .env.example             # Config template (copy to .env, never commit .env)
├── requirements.txt
└── CLAUDE.md                # Project context, decisions, and build phases
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| Databricks workspace | Unity Catalog enabled |
| Python | 3.11+ |
| Databricks CLI | For Asset Bundle deployments (`databricks bundle deploy`) |
| Auth | PAT in `~/.databrickscfg` or provided per-session in UI |
| Serving endpoint | Claude Sonnet via Databricks External Models |

---

## Local Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/DataWizardAshish/databricks-ml-acelerator.git
cd databricks-ml-accelerator
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `databricks-connect` is intentionally excluded from the base install — Phase 1 uses only REST APIs (zero compute). Add it when Spark is needed in Phase 2.

### 3. Configure authentication

**Option A — Use `~/.databrickscfg` DEFAULT profile (recommended for local dev):**

Your `~/.databrickscfg` should have:
```ini
[DEFAULT]
host  = https://your-workspace.cloud.databricks.com
token = dapi...
```

**Option B — Per-session via UI:** Enter your workspace URL and PAT directly in the Streamlit sidebar. Useful for connecting to multiple workspaces.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env — only need to set LLM_ENDPOINT_NAME and cluster ID if different from defaults
```

### 5. Start the backend and UI

Terminal 1 — FastAPI backend:
```bash
python run_api.py
# API runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

Terminal 2 — Streamlit UI:
```bash
python run_ui.py
# UI runs at http://localhost:8501
```

---

## Usage

1. Open `http://localhost:8501`
2. Click **Connect & Browse Catalogs** in the sidebar (or leave blank to use `~/.databrickscfg`)
3. Select your **Catalog** and **Schema** from the dropdowns
4. Click **Discover ML Opportunities**
5. Review the top 3 ranked ML use cases
6. Click **Approve** on the one you want to build
7. Phase 2 will generate the notebooks and jobs automatically

---

## Multi-Team / Multi-Workspace

Each team provides their own credentials at runtime — no shared config file needed:

| Team | Workspace | Catalog | Schema |
|---|---|---|---|
| Brand Funnel AI | `adidas-dev.cloud.databricks.com` | `lakehouse_dev` | `ds_brand_funnel_ai_lhdev` |
| Supply Chain | `adidas-dev.cloud.databricks.com` | `lakehouse_dev` | `supply_chain_lhdev` |
| Another Org | `other-org.cloud.databricks.com` | `main` | `analytics` |

Teams enter their workspace URL + PAT in the UI sidebar. The backend creates a scoped `WorkspaceClient` per request — no shared global credentials.

---

## Databricks Asset Bundle Deployment (Phase 2+)

Generated code lands in `bundles/` and is deployed via the Databricks CLI:

```bash
# Install Databricks CLI
pip install databricks-cli

# Deploy to dev
databricks bundle deploy --target dev

# Run the feature pipeline job
databricks bundle run feature_pipeline_job --target dev

# Deploy to prod (requires workspace_host variable)
databricks bundle deploy --target prod --var workspace_host=https://prod-workspace.cloud.databricks.com
```

---

## Build Phases

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | UC Discovery + ML Opportunity Recommendation |
| **Step A** | ✅ Complete | Multi-workspace auth, UC validation, browse mode, DAB scaffold |
| **Phase 2** | 🚧 In Progress | Feature pipeline + training job + inference job code generation |
| **Phase 3** | ⏳ Planned | Databricks Apps packaging + deployment |
| **Phase 4** | ⏳ Planned | Human-in-the-loop code review + editing |
| **Phase 5** | ⏳ Planned | Drift detection + auto-retraining loop |

---

## Key Design Decisions

- **Governance-first code generation** — all generated code uses OBO patterns, emits correct `GRANT` statements, never bypasses UC lineage
- **Zero compute for discovery** — Phase 1 uses only UC REST APIs. No cluster spin-up, no SQL warehouse cost
- **Serverless SQL for profiling** — data quality checks in Phase 2 use serverless SQL warehouses (fast startup, no idle cost)
- **Request-scoped auth** — `WorkspaceClient` created per API request, never shared globally — thread-safe, multi-workspace
- **DAB from day one** — generated code is always a deployable bundle, not loose notebooks

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| Primary LLM | Claude Sonnet via Databricks External Models |
| UC Metadata | Databricks SDK (`WorkspaceClient`) |
| LLM Integration | `databricks-langchain` (`ChatDatabricks`) |
| API | FastAPI |
| UI | Streamlit → React (later) |
| Deployment | Databricks Apps |
| CI/CD | Databricks Asset Bundles |
| ML Tracking | MLflow |
| State Persistence | Delta tables in Unity Catalog |

---

## Contributing

This is a solo build. For questions or collaboration:
- Open an issue on GitHub
- Reference `CLAUDE.md` for all architectural decisions — decisions listed there are final unless there's a strong technical reason to revisit

---

*Built by a senior Databricks data engineer. Deep expertise in Unity Catalog, Mosaic AI, Governance (RBAC + ABAC + OBO), MLOps, and Databricks Apps.*
