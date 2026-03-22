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
[Unity Catalog Discovery]       ← REST API only, zero compute
        ↓
[Data Estate Analysis]          ← LLM enrichment: relationships, quality signals
        ↓
[ML Opportunity Ranker]         ← Top 3 use cases + financial impact ($) + confidence level
        ↓
⏸ HUMAN CHECKPOINT — approve opportunity
        ↓
[Dry Run / Explain]             ← LLM plan: tables read/written, grants, DBU cost,
        ↓                         feature_columns, join_keys, row_count estimate
[Business Brief Generator]      ← Deterministic CTO brief (no LLM) stored pre-code-gen
        ↓
⏸ HUMAN CHECKPOINT — confirm dry run plan
        ↓
[Feature Engineering Planner]
        ↓
[Code Generator]                ← 3 notebooks: feature pipeline, training, batch inference
        ↓
[Risk Scorecard]                ← Automated checks: leakage, temporal split, governance, MLflow,
        ↓                         dbutils.widgets, Champion alias, class imbalance
⏸ HUMAN CHECKPOINT — review code + risk scorecard
        ↓
[Bundle Writer]                 ← Writes to bundles/ (notebooks + job YAMLs)
        ↓
[Executive Summary]             ← Full technical SUMMARY.md: scorecard + artifacts + ROI
        ↓
[END — ready to deploy]

Future: Deployment Executor (Phase 3) → Monitoring + Drift Detection (Phase 5)
```

---

## Project Structure

```
databricks-ml-accelerator/
│
├── agent/                      # LangGraph orchestration
│   ├── graph.py                # Graph definition + public API (run_discovery, approve_opportunity, …)
│   ├── nodes.py                # Phase 1: discover_catalog → analyze_estate → rank_opportunities
│   ├── trust_nodes.py          # Trust layer: dry_run_explain, generate_business_brief,
│   │                           #   dry_run_checkpoint, compute_risk_scorecard, generate_exec_summary
│   ├── code_gen_nodes.py       # Phase 2: plan_features → generate_code → human_checkpoint_code → write_bundle
│   ├── chat.py                 # Contextual Q&A: step-scoped run context → LLM answer
│   └── state.py                # AgentState + DryRunPlan TypedDicts
│
├── tools/                      # Reusable tools called by agent nodes
│   ├── uc_reader.py            # Unity Catalog metadata reader (no Spark, REST only)
│   ├── bundle_writer.py        # Writes generated notebooks + job YAMLs to bundles/
│   └── workspace_context.py   # Per-request auth context, UC validation, browse helpers
│
├── api/                        # FastAPI backend
│   └── main.py                 # /health, /browse/*, /runs/*, /runs/{id}/ask endpoints
│
├── ui/                         # Streamlit frontend
│   └── app.py                  # 6-step flow: Discover → Data Estate → Approve Opportunity
│                               #   → Confirm Dry Run → Code Review → Bundle Written
│                               #   Q&A panel on steps 3–6, client-side off-topic guardrail
│
├── config/
│   └── settings.py             # Pydantic settings (PAT now, OAuth-ready)
│
├── bundles/                    # Databricks Asset Bundle (generated code lands here)
│   ├── databricks.yml          # Bundle config: dev + prod targets
│   ├── SUMMARY.md              # Auto-generated CTO-facing executive summary
│   ├── resources/jobs/         # Generated job YAML definitions (3 per use case)
│   └── src/{use_case}/         # Generated notebooks (feature, training, inference)
│
├── .env.example                # Config template (copy to .env, never commit .env)
├── requirements.txt
└── CLAUDE.md                   # Project context, decisions, and build phases
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
2. Click **Connect & Browse** in the sidebar (or leave blank to use `~/.databrickscfg`)
3. Select your **Catalog** and **Schema** from the dropdowns
4. Click **🔍 Discover** — agent reads UC metadata, analyzes the estate, ranks top 3 ML use cases

**Step 2 — Data Estate Overview:**

5. Review the discovered tables — type (MANAGED / EXTERNAL), column count, descriptions
6. Expand any table to see its columns with types and nullability
7. Read the AI analysis of entity relationships and data quality signals
8. Click **→ See ML Recommendations** to advance

**Step 3 — Approve Opportunity:**

9. Each opportunity shows **financial impact estimate** and **confidence level**
10. Ask questions using the **💬 Ask about this step** panel — scoped to this run
11. Click **✅ Approve** on the use case you want to build

**Step 4 — Confirm Dry Run:**

12. Review the **Business Brief** tab (CTO view) — business case, execution plan, governance summary
13. Review the **Technical Plan** tab — feature columns, join keys, tables to read/write, GRANT statements
14. Ask questions in the Q&A panel, then click **✅ Confirm & Generate Code**

**Step 5 — Code Review:**

15. Review the **Risk Scorecard** — pass/warn/fail checks for leakage, temporal split, governance, MLflow
16. Review the 3 generated notebooks — feature pipeline, training, batch inference
17. Ask questions in the Q&A panel, then click **✅ Approve & Write Bundle** (blocked if any check is `fail`)

**Step 6 — Bundle Written:**

18. Review the **Executive Summary** tab and the **Generated Files** tab
19. Use the **Deploy** tab to run `databricks bundle deploy`
20. Ask post-deployment questions (monitoring, drift, etc.) in the Q&A panel

> **Schema change guard:** Changing catalog/schema mid-run shows a confirmation dialog to prevent accidental loss of run progress. Use the **Back** button on any step to return without losing upstream state.

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

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + version |
| `GET` | `/browse/catalogs` | List UC catalogs in workspace |
| `GET` | `/browse/schemas` | List schemas in a catalog |
| `POST` | `/runs` | Start discovery run → returns `tables`, `estate_summary`, `opportunities` |
| `GET` | `/runs/{id}` | Get run status + full state values |
| `POST` | `/runs/{id}/approve` | Approve ML opportunity → returns dry run plan + CTO business brief |
| `POST` | `/runs/{id}/confirm-dry-run` | Confirm plan → triggers code generation → returns notebooks + risk scorecard |
| `POST` | `/runs/{id}/approve-code` | Approve code → writes bundle → returns exec summary + artifact paths |
| `POST` | `/runs/{id}/ask` | Answer a question scoped to the current run state and step |

Swagger UI available at `http://localhost:8000/docs`

---

## Build Phases

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | UC Discovery + ML Opportunity Recommendation (financial impact, confidence) |
| **Step A** | ✅ Complete | Multi-workspace auth, UC validation, browse mode, DAB scaffold |
| **Phase 2** | ✅ Complete | Code generation + Trust Layer: dry run, business brief, risk scorecard, exec summary |
| **Phase 2 UX** | ✅ Complete | 6-step UI flow, Data Estate view, Q&A panel, schema change guard, back buttons |
| **Phase 3** | ⏳ Planned | Databricks Apps packaging + deployment |
| **Phase 4** | ⏳ Planned | Advanced human-in-loop (edit generated code inline, re-generate specific notebook) |
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
