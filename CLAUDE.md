# Databricks ML Accelerator — Project Context

## What This Product Is

A **Databricks-native agentic ML acceleration system** that connects to a customer's Unity Catalog, understands their entire data estate, recommends ML opportunities, generates production-ready Databricks code, deploys it, and monitors it continuously.

Not a CSV upload tool. Not a report generator. An autonomous ML engineer that lives inside the customer's Databricks workspace.

---

## The Problem We Solve

Databricks customers have invested in data infrastructure but the gap between "data is ready in Unity Catalog" and "ML model is in production" is still 3–6 months. The bottleneck is not infrastructure — it's the intelligence layer. Hiring ML engineers costs $180K–$250K/year. Consultants cost $15–40K per engagement. This product closes that gap.

---

## Primary Persona

**Senior data engineers / ML engineers at mid-market companies (100–2000 employees) already on Databricks.**
- They know pipelines and SQL
- They don't have dedicated ML engineers
- They have data in Unity Catalog they haven't extracted value from
- They have budget pain around hiring

Secondary: Freelance data consultants who need to deliver ML scoping faster.

**Not targeting:** Individual data scientists, cloud-agnostic companies, startups without Databricks.

---

## Core Value Proposition

> "Connect your Unity Catalog. Get ML in production in days, not months. With governance already handled."

---

## Real Example (Reference This When Building)

RetailCo (mid-market retailer, Databricks on AWS):
- 4-person data engineering team, no ML engineers
- CTO asked for churn prediction 6 months ago, nothing shipped
- Connected workspace → agent discovered catalog → recommended churn prediction with 78-84% AUC estimate
- Approved → agent generated feature pipeline + training job + batch inference job + MLflow tracking + UC governance grants
- Day 5: model in production, scores in Unity Catalog, marketing team using them via existing Tableau
- Week 6: drift detected → agent retrained automatically → promoted new model
- Time saved vs traditional approach: 3–4 months

---

## Tech Stack (DECIDED — do not revisit without strong reason)

| Layer | Technology | Reason |
|---|---|---|
| Agent Orchestration | **LangGraph** | Stateful graphs, human-in-loop interrupts, tool calling |
| Structured Extraction | **DSPy** (selective use only) | Typed schema/quality extraction where structured output matters |
| Primary LLM | **Claude Sonnet 4.6** (`claude-sonnet-4-6`) | Best for Databricks code gen, complex reasoning, large context |
| Secondary LLM | **Llama 3.3 70B** via Mosaic AI | Lighter classification tasks, cost optimization |
| LLM Routing | **Databricks External Models** | All API calls proxied through Databricks — audit trail, no data egress, enterprise compliant |
| UI | **Streamlit** (start) → React (later) | Fast to build, native Databricks Apps support |
| API | **FastAPI** | Async, clean, testable |
| Deployment | **Databricks Apps** | Runs INSIDE customer workspace — no separate infra, SSO inherited, data never leaves |
| Distribution | **Databricks Marketplace** | Discovery + install channel + billing handled |
| Data Layer | **Unity Catalog** | Source of truth, customer's own data |
| ML Tracking | **MLflow** | Already in every Databricks workspace |
| Job Scheduling | **Databricks Jobs** | Native scheduling, no extra infrastructure |
| State Persistence | **Delta tables in Unity Catalog** | Durable, queryable, governed |
| Agent Observability | **LangSmith** | Full trace debugging for agent steps |
| Cloud Storage | **AWS S3** | Raw data layer |

---

## Architecture (Inside Customer's Workspace)

```
Customer Databricks Workspace
│
├── Databricks App (this product)
│   ├── Streamlit UI
│   └── FastAPI backend
│       └── LangGraph orchestration engine
│
├── Mosaic AI Serving Endpoints
│   ├── schema-discovery-agent
│   ├── ml-recommendation-agent
│   ├── code-generation-agent
│   └── drift-monitor-agent
│
├── External Model Endpoint
│   └── claude-sonnet-4-6 (proxied through Databricks)
│
├── Unity Catalog
│   ├── customer's existing data
│   └── ml_accelerator schema (product writes here)
│       ├── discovered_opportunities
│       ├── generated_code_artifacts
│       └── model_health_scores
│
├── MLflow
│   └── experiments, runs, model registry
│
└── Databricks Jobs
    ├── weekly-retraining-job
    └── nightly-drift-check-job
```

---

## Agent Pipeline (LangGraph Graph)

Current implemented flow (Phase 1 + Phase 2 + Trust Layer):

```
[Unity Catalog Discovery]       ← REST API only, zero compute
        ↓
[Data Estate Analysis]          ← LLM enrichment: inferred relationships, quality signals
        ↓
[ML Opportunity Ranker]         ← Top 3 use cases ranked by value × readiness
        ↓                         Includes: financial_impact ($), confidence level
⏸ HUMAN CHECKPOINT — approve opportunity
        ↓
[Dry Run / Explain]             ← LLM generates plain-English plan before any code runs
        ↓                         Fields: tables_to_read/write, grant_statements, DBU cost,
        ↓                         runtime, feature_columns, join_keys, row_count, target_detail
[Business Brief Generator]      ← Deterministic (no LLM) — CTO-facing markdown brief
        ↓                         Sections: Business Case, What Will Be Built, Execution Plan,
        ↓                         Governance. Stored in exec_summary before code gen.
⏸ HUMAN CHECKPOINT — confirm dry run plan
        ↓
[Feature Engineering Planner]  ← Schema-aware feature plan
        ↓
[Code Generator]                ← 3 notebooks: feature pipeline, training, batch inference
        ↓                         Rules: mandatory aliasing before every join, no self-joins,
        ↓                         groupBy/agg over window functions, temporal train/test split
[Risk Scorecard]                ← Rule-based checks: temporal split, leakage, GRANT, MLflow,
        ↓                         dbutils.widgets, Champion alias, class imbalance
⏸ HUMAN CHECKPOINT — review generated code + scorecard
        ↓
[Bundle Writer]                 ← Writes notebooks + job YAMLs to bundles/
        ↓
[Executive Summary]             ← Full technical SUMMARY.md: risk scorecard + artifacts +
        ↓                         business case. Overwrites exec_summary in state.
[END]

Future:
[Deployment Executor]           ← Phase 3
        ↓
[Monitoring Setup]              ← Phase 5
        ↓
[Drift Detection Loop] ←──────── runs continuously
```

### Human Checkpoints and API Status Mapping

| Interrupt node | API status | API endpoint to resume |
|---|---|---|
| `human_checkpoint` | `awaiting_approval` | `POST /runs/{id}/approve` |
| `dry_run_checkpoint` | `awaiting_dry_run_confirmation` | `POST /runs/{id}/confirm-dry-run` |
| `human_checkpoint_code` | `awaiting_code_review` | `POST /runs/{id}/approve-code` |

### Key node files

| File | Nodes |
|---|---|
| `agent/nodes.py` | `discover_catalog`, `analyze_estate`, `rank_opportunities`, `human_checkpoint` |
| `agent/trust_nodes.py` | `dry_run_explain`, `generate_business_brief`, `dry_run_checkpoint`, `compute_risk_scorecard`, `generate_exec_summary` |
| `agent/code_gen_nodes.py` | `plan_features`, `generate_code`, `human_checkpoint_code`, `write_bundle` |
| `agent/chat.py` | `ask_about_run()` — scoped Q&A per run/step, called by `/runs/{id}/ask` |

---

## Governance Model (Builder's Key Differentiator)

The builder has deep Databricks governance expertise:
- RBAC + ABAC + OBO (On-Behalf-Of) identity patterns
- Unity Catalog grants and lineage
- Service principal vs user token patterns

All generated code must:
- Use OBO pattern where applicable (not hardcoded service principals)
- Emit correct `GRANT` statements respecting existing RBAC
- Write outputs to `ml_accelerator` schema with appropriate access controls
- Never bypass Unity Catalog lineage tracking

This governance-aware code generation is the primary technical differentiator from generic AI code tools.

---

## Pre-Phase 2 Improvements (All Completed)

Architectural gaps addressed before Phase 2:

1. ✅ **Request-scoped workspace connection** — `WorkspaceContext` pattern: host, PAT, catalog, schema passed per API request. Thread-safe, multi-workspace. Lives in `tools/workspace_context.py`.
2. ✅ **UC validation before agent runs** — `validate_workspace(ctx)` calls `catalogs.get()` + `schemas.get()` at startup. Returns structured error with `field` key for clean UI display.
3. ✅ **Catalog + schema browse mode** — `/browse/catalogs` + `/browse/schemas` endpoints. UI renders dropdowns, no free-text typos.
4. ✅ **Databricks Asset Bundle (DAB) scaffold** — `bundles/` created with `databricks.yml`, `resources/jobs/`, `src/`. All generated code lands here.

---

## Common ML Engineering Bottlenecks This Agent Must Solve

These are the recurring, everyday blockers — not rare edge cases. The agent should detect and handle these automatically:

### Data Quality (before training)
- **Schema drift** — columns renamed/dropped/type-changed between discovery and training; pipelines break silently
- **Type surprises** — numeric columns stored as strings, dates with mixed formats, nulls encoded as "-" or "N/A"
- **Class imbalance** — 99% accuracy on a 1% minority class; engineer thinks model is great
- **Missing value patterns** — not just count but which rows/time periods; imputation choices made wrong early

### Feature Engineering
- **Target leakage** — feature derived from or correlated with the target in a way that doesn't hold in production (status columns, post-event timestamps)
- **Training-serving skew** — feature transformation in training notebook differs from inference notebook; #1 cause of silent model degradation
- **Temporal split done wrong** — random split on time-series data inflates test metrics; should always be time-based
- **High-cardinality categoricals** — user IDs, SKUs, free-text passed as raw features; memory explosion, bad models

### MLflow / Experiment Tracking
- **Unnamed/unstructured runs** — "Run 1", "Run 2", can't find what worked 2 weeks later
- **Model registry chaos** — multiple versions, no clear designation of which is production
- **Missing params/metrics logged** — can't reproduce a run, can't compare experiments

### Deployment & Governance
- **UC permissions not set for the inference identity** — job service principal can't read feature table; fails at deployment, not at dev time
- **Notebook → Job translation breaks** — notebook runs interactively, fails as scheduled job due to widget vs. parameter differences
- **No baseline comparison** — model deployed with no simple baseline (mean predictor, last-value) to confirm it's actually better
- **Cluster library mismatch** — trained on cluster with specific library version, deployed to different cluster; silent bugs

### Operations
- **Stale feature tables** — inference reads feature table not updated in days; scores are silently wrong
- **No job retry logic** — transient cluster failure, no retry, on-call alert at 2am
- **Drift detected, no action path** — alert fires but no automated or guided retraining flow; alert fatigue

---

## Build Phases

### Phase 1 (Weeks 1–3): Core Discovery + Recommendation
- LangGraph setup
- Unity Catalog reader tool (reads metadata, not raw data)
- Claude via Databricks External Model
- Can discover catalog and rank ML opportunities

**Test:** Given a real Unity Catalog, does it return credible ML opportunity recommendations?

### Phase 2 (Weeks 4–6): Code Generation + Trust Layer ✅ Complete

**Code generation:**
- Feature engineering notebook (PySpark, mandatory join aliasing, no self-joins, groupBy/agg only)
- Training notebook (MLflow integrated, UC model registry, Champion alias, temporal split)
- Batch inference notebook (dbutils.widgets, GRANT statements, lineage-safe)
- 3 Databricks Job YAMLs via DAB scaffold → `bundles/`

**Trust layer:**
- **Business impact estimator** — financial_impact ($range) + confidence on every opportunity
- **Dry run / explain mode** — LLM plan before any code runs: tables, grants, DBU cost, runtime, feature_columns, join_keys, row_count estimate
- **Business Brief Generator** — deterministic (no LLM) CTO brief between dry run and code gen
- **Risk scorecard** — rule-based checks: temporal split, leakage, GRANT completeness, MLflow tracking, dbutils.widgets usage, Champion alias, class imbalance
- **Executive summary** — full technical SUMMARY.md after bundle write (risk scorecard + artifacts)

**UI (6-step flow):**
1. **Discover** — connect workspace, select catalog/schema, run discovery
2. **Data Estate** — table/column inventory with type badges, column explorer, AI analysis
3. **Approve Opportunity** — ranked use cases with financial impact + confidence
4. **Confirm Dry Run** — two-tab view: CTO business brief + technical plan (features, joins, grants)
5. **Code Review** — risk scorecard banner + notebook viewer + approve/regenerate
6. **Bundle Written** — exec summary + generated files + deploy commands

**UX guardrails:**
- Schema change guard — warning modal when changing catalog/schema mid-run
- Back buttons on all steps (each step resets only its downstream state)
- Contextual Q&A panel on steps 3–6: scoped to current run state, client-side off-topic filter before any LLM call

**Test:** Does the generated notebook execute without errors on a real cluster?

### Phase 3 (Weeks 7–8): Databricks App Packaging
- Streamlit UI
- FastAPI backend
- Package as Databricks App
- Deploy to own workspace for demo

**Test:** Can a non-technical person use it inside Databricks?

### Phase 4 (Weeks 9–10): Human-in-the-Loop
- LangGraph interrupt/approval checkpoints
- User can modify plans before code is generated
- User can edit generated code before deployment

### Phase 5 (Month 3+): Monitoring + Autonomous Loop
- Drift detection agent
- Auto-retraining with approval gate
- Model health dashboard

---

## Go-To-Market

- **First 3–5 customers:** Direct outreach to Databricks users (LinkedIn, Databricks community, user groups). Charge $500–1500/month. Be part consultant, part product.
- **Scale:** Databricks Marketplace listing — appears in customer's existing workflow
- **Billing:** Through Databricks Marketplace (customer pays via existing Databricks relationship)
- **Pricing model:** Per workspace per month

---

## What NOT to Build

- Multi-cloud (Azure ML, SageMaker) — Databricks only to start
- CSV upload interface — Unity Catalog connection only
- Generic PRD/document output — actionable code only
- Competing with Databricks features — extend them, don't replace them
- Enterprise sales infrastructure — self-serve first

---

## Builder Context

- Solo senior data engineer
- Deep Databricks expertise: Unity Catalog, Jobs, Pipelines, Serving Endpoints, Mosaic AI, Genie, SQL Warehouse, Databricks Apps, Governance (RBAC + ABAC + OBO)
- MLOps, CI/CD, Docker, Jenkins
- AWS S3
- Building solo — scope discipline is critical
- Previous project: `c:\agentic-data-analyst` (DSPy-based prototype, has agents for schema analysis, profiling, quality, ML advice, deployment planning, code generation — reference for patterns but do not port directly)

---

## Decisions Already Made (Do Not Re-litigate)

1. Databricks-native only (not multi-cloud)
2. LangGraph for orchestration (not DSPy, not CrewAI, not AutoGen)
3. Claude Sonnet 4.6 as primary LLM
4. Databricks Apps as deployment target (not VSCode extension, not standalone SaaS)
5. Unity Catalog as entry point (not CSV upload)
6. Governance-aware code generation is the core differentiator
7. Solo build → design partners first → Marketplace second
