# Backend Architecture Design

## Overview

The Databricks ML Accelerator backend is a **FastAPI + LangGraph** system deployed as a Databricks App. It connects to a customer's Unity Catalog, analyzes their data estate, recommends ML opportunities, and generates production-ready Databricks code — all with human-in-the-loop approval gates at every critical decision point.

This document covers the full backend architecture: layers, data flows, storage tables, and design rationale.

---

## High-Level Architecture

```
Databricks App (single process via start.sh)
│
├── Streamlit UI        → public port ($DATABRICKS_APP_PORT)
│   └── reads OBO headers from st.context.headers
│   └── forwards X-Forwarded-Email + X-Forwarded-Access-Token on every API call
│
└── FastAPI API         → internal port (127.0.0.1:8080)
    └── LangGraph Agent Graph
        ├── Phase 1 nodes   (discover_catalog, analyze_estate, rank_opportunities)
        ├── Trust nodes     (dry_run_explain, generate_business_brief, dry_run_checkpoint, compute_risk_scorecard, generate_exec_summary)
        └── Phase 2 nodes   (plan_features, generate_code, human_checkpoint_code, write_bundle)
```

All Streamlit → FastAPI calls are **server-side to localhost** — no CORS, no external network hop.

---

## Entry Points

### `start.sh`
Starts both processes from a single Databricks App entry point:
- **gunicorn** (FastAPI, 2 workers, `127.0.0.1:8080`) — internal only
- **streamlit** (`$DATABRICKS_APP_PORT`) — public, receives Databricks platform OBO headers

### `app.yaml`
Databricks App manifest. Declares the single command (`bash start.sh`) and injects `API_BASE=http://127.0.0.1:8080` as an environment variable so Streamlit knows where to reach FastAPI.

---

## Layer Breakdown

### 1. API Layer — `api/main.py`

FastAPI application. Routes all requests, handles identity extraction, validates run state before resuming the graph.

**OBO Identity Pattern:**
Every request carries `X-Forwarded-Email` and `X-Forwarded-Access-Token` headers injected by the Databricks Apps platform. The `get_current_user(request)` dependency extracts these and passes the token downstream as `WorkspaceContext.token` — so every Databricks SDK call runs as the end user, not a service principal.

```
Databricks Apps platform
  → injects X-Forwarded-Email, X-Forwarded-Access-Token per HTTP request
    → FastAPI reads them via Request.headers
      → token passed into WorkspaceContext
        → WorkspaceClient(token=obo_token)
          → all SDK calls run as end user
```

**Key endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + config check. Returns authenticated user email. |
| `GET` | `/browse/catalogs` | Lists UC catalogs for the dropdown. |
| `GET` | `/browse/schemas` | Lists UC schemas within a catalog. |
| `GET` | `/runs/history` | Episodic memory — 20 most recent runs. |
| `POST` | `/runs` | Start discovery. Returns after first interrupt. |
| `GET` | `/runs/{id}` | Raw run state from LangGraph checkpoint. |
| `GET` | `/runs/{id}/rehydrate` | UI-ready state payload for page refresh restore. |
| `POST` | `/runs/{id}/approve` | Resume from opportunity approval. |
| `POST` | `/runs/{id}/confirm-dry-run` | Resume from dry run confirmation. |
| `POST` | `/runs/{id}/approve-code` | Resume from code review. |
| `GET` | `/runs/{id}/audit` | Full audit trail with hash chain integrity check. |
| `POST` | `/runs/{id}/ask` | Scoped Q&A against current run state. |

**Status guard pattern:**
All resume endpoints check current run status before calling graph.invoke. If a run is in `awaiting_approval` and you call `/confirm-dry-run`, you get a 400 with a clear error — no silent state corruption.

---

### 2. Config Layer — `config/settings.py`

Pydantic `BaseSettings` loaded from `.env` with `@lru_cache`. All config read once at startup, injected nowhere — callers call `get_settings()`.

**Key fields:**

| Setting | Purpose | Default |
|---------|---------|---------|
| `databricks_host` | Workspace URL | From `~/.databrickscfg` |
| `databricks_token` | PAT for local dev | From `~/.databrickscfg` |
| `auth_type` | `pat` or `oauth` | `pat` |
| `databricks_cluster_id` | Compute for jobs | `0402-042550-3eto1bj6` |
| `llm_endpoint_name` | Mosaic AI endpoint | `databricks-meta-llama-3-3-70b-instruct` |
| `llm_max_tokens` | LLM response limit | `4096` |
| `llm_temperature` | LLM creativity | `0.0` |
| `uc_catalog` | Unity Catalog name | `dbw_vectoflow_dev` |
| `uc_discovery_schema` | Schema to analyze | `vectoflow` |
| `uc_output_schema` | Where agent writes | `ml_accelerator` |

Auth type swap: setting `AUTH_TYPE=oauth` in `.env` switches to OAuth U2M — no code changes needed anywhere else.

---

### 3. Workspace Context — `tools/workspace_context.py`

`WorkspaceContext` is a Pydantic model passed with every API request. It carries the connection identity for a single agent run.

```python
class WorkspaceContext(BaseModel):
    host: str = ""           # workspace URL — falls back to ~/.databrickscfg
    token: str = ""          # PAT or OBO token — falls back to ~/.databrickscfg
    catalog: str = ""        # falls back to settings.uc_catalog
    schema: str = ""         # falls back to settings.uc_discovery_schema
    cluster_id: str = ""     # falls back to settings.databricks_cluster_id
    user_email: str = ""     # from X-Forwarded-Email (OBO identity)
```

**Why this exists:** Makes the system multi-workspace safe. Two concurrent API requests can analyze different Databricks workspaces simultaneously. No global connection state.

**`validate_workspace(ctx)`** — called before starting any run. Calls `catalogs.get()` + `schemas.get()` on the UC REST API. Returns a structured error with a `field` key so the UI can highlight exactly which input is wrong.

**`get_llm()`** — creates a `ChatDatabricks` instance bound to this workspace's LLM endpoint. Passes `databricks_host` and `databricks_token` explicitly — safe for concurrent requests.

---

### 4. Agent Graph — `agent/graph.py`

The LangGraph `StateGraph` is the core orchestration engine. It is compiled once (`_get_graph()` singleton) and reused across all requests.

**Graph topology:**

```
START
  → discover_catalog → analyze_estate → rank_opportunities
  → human_checkpoint              ⏸  awaiting_approval
  → dry_run_explain → generate_business_brief
  → dry_run_checkpoint            ⏸  awaiting_dry_run_confirmation
  → plan_features → generate_code → compute_risk_scorecard
  → human_checkpoint_code         ⏸  awaiting_code_review
  → write_bundle → generate_exec_summary
END
```

**Interrupt → Resume pattern:**
Each `human_checkpoint_*` node calls LangGraph's `interrupt()` internally. The graph suspends, serializes full state to the SQLite checkpointer, and returns. When the user approves, the API calls `graph.invoke(Command(resume=...), config=...)` — the graph resumes from exactly where it left off with the same `thread_id`.

**Status mapping** (`_snapshot_to_status`):

| `snapshot.next` | Returned status |
|-----------------|----------------|
| `human_checkpoint` | `awaiting_approval` |
| `dry_run_checkpoint` | `awaiting_dry_run_confirmation` |
| `human_checkpoint_code` | `awaiting_code_review` |
| `[]` (empty) | `completed` |

**Thread ID = Run ID:** Every run gets a UUID as its `run_id`. This UUID is used as the LangGraph `thread_id`, the SQLite checkpoint key, the `run_history` primary key, and the audit trail `run_id`. One identifier traces everything.

---

### 5. Agent State — `agent/state.py`

`AgentState` is a `TypedDict` — the single source of truth passed between all graph nodes and persisted by the checkpointer.

**Key fields:**

| Field | Type | Phase | Purpose |
|-------|------|-------|---------|
| `workspace` | `dict` | 1 | Serialized `WorkspaceContext` — plain dict for JSON checkpoint compatibility |
| `tables` | `list[dict]` | 1 | Discovered UC tables with column metadata |
| `estate_summary` | `str` | 1 | LLM-generated narrative of the data estate |
| `opportunities` | `list[MLOpportunity]` | 1 | Ranked ML use cases with financial impact + confidence |
| `approved_opportunity` | `MLOpportunity` | 1 | Selected opportunity after human approval |
| `dry_run_plan` | `DryRunPlan` | Trust | Tables, grants, DBU cost, runtime, feature columns, join keys, row counts |
| `risk_scorecard` | `RiskScorecard` | Trust | Rule-based code quality checks (temporal split, leakage, GRANT, MLflow, etc.) |
| `exec_summary` | `str` | Trust/2 | CTO brief before bundle write; full SUMMARY.md after |
| `feature_plan` | `FeaturePlan` | 2 | Column transforms, split strategy, model suggestion, MLflow experiment name |
| `generated_artifacts` | `list[GeneratedArtifact]` | 2 | Feature pipeline + training + inference notebooks + job YAMLs |
| `bundle_written` | `bool` | 2 | True after `write_bundle` completes |
| `error` | `str` | All | Node-level error message for UI display |

**Serialization contract:** All complex objects (WorkspaceContext, TableInfo dataclasses) are converted to plain dicts before entering state. This is required for the SQLite checkpointer — it serializes state as JSON. Every node that reads `workspace` must reconstruct: `WorkspaceContext(**state["workspace"])`.

---

### 6. Node Files

#### `agent/nodes.py` — Phase 1
- **`discover_catalog`** — calls `databricks-sdk` UC REST API (no Spark). Lists all tables + columns in the target schema. Converts `TableInfo` dataclasses to plain dicts. Emits `catalog_discovered` audit event.
- **`analyze_estate`** — sends table metadata to LLM. Generates `estate_summary`: inferred relationships, data quality signals, readiness indicators.
- **`rank_opportunities`** — LLM generates top 3 ML opportunities. Each includes `financial_impact` (dollar range), `confidence` (high/medium/low), `estimated_auc_range`, `complexity`. Uses DSPy-style typed output.
- **`human_checkpoint`** — calls `interrupt()`. Graph suspends. Waits for `/approve` endpoint with `selected_rank`.

#### `agent/trust_nodes.py` — Trust Layer
- **`dry_run_explain`** — LLM generates `DryRunPlan`: what tables will be read/written, what GRANT statements are needed, estimated DBU cost, runtime, feature columns, join keys, row count estimate. No code runs. Plain English summary for non-technical stakeholders.
- **`generate_business_brief`** — **deterministic, no LLM**. Takes `DryRunPlan` + `approved_opportunity` and generates a structured markdown CTO brief. Sections: Business Case, What Will Be Built, Execution Plan, Governance. Written to `exec_summary` in state before code gen.
- **`dry_run_checkpoint`** — calls `interrupt()`. Waits for `/confirm-dry-run`.
- **`compute_risk_scorecard`** — rule-based checks against generated code. Checks: temporal split present, no target leakage patterns, GRANT statements included, MLflow tracking calls present, `dbutils.widgets` used (not hardcoded params), Champion alias set in model registry, class imbalance handling. Returns `RiskScorecard` with `pass/warn/fail` per check and overall `ready/review_needed/blocked`.
- **`generate_exec_summary`** — LLM generates full technical SUMMARY.md after bundle write. Includes risk scorecard results, artifact list, business case. Overwrites `exec_summary` in state.

#### `agent/code_gen_nodes.py` — Phase 2
- **`plan_features`** — LLM generates `FeaturePlan`: column-level transform decisions (keep, log-transform, one-hot, hash-encode, drop), split strategy (temporal vs random), class balance strategy, suggested model, MLflow experiment name, output table names.
- **`generate_code`** — LLM generates 3 PySpark notebooks: (1) feature pipeline, (2) training with MLflow + UC model registry + Champion alias, (3) batch inference with `dbutils.widgets` + GRANT statements. Code rules enforced by prompt: mandatory alias before every join, no self-joins, `groupBy/agg` over window functions, temporal train/test split. Also generates 3 Databricks Asset Bundle YAMLs.
- **`human_checkpoint_code`** — calls `interrupt()`. Waits for `/approve-code`. Supports `action=regenerate` with free-text instructions — graph re-runs `generate_code` with the feedback.
- **`write_bundle`** — writes all artifacts to `bundles/` directory. Sets `bundle_written=True`.

#### `agent/chat.py` — Scoped Q&A
`ask_about_run()` — called by `/runs/{id}/ask`. Takes current run state values and a step name, builds a scoped system prompt from the relevant state fields (only what's available at that step), and calls the LLM. Off-topic questions are filtered client-side in the UI before reaching this function.

---

### 7. Storage Architecture

All storage currently in a single SQLite file: `data/checkpoints.db`.

#### Table 1: `checkpoints` (LangGraph internal)

Managed entirely by `SqliteSaver` from `langgraph-checkpoint-sqlite`. Not queried directly.

**Purpose:** Durable run state. Survives API restarts. Enables the interrupt/resume pattern — without this, a server restart would lose all in-flight runs. Each row stores a serialized snapshot of `AgentState` keyed by `thread_id` (= `run_id`) and `checkpoint_id`.

**Why SQLite (now):** Zero-dependency, file-backed, works in Databricks Apps. No external service required.

**Phase 3 upgrade path:** Swap `SqliteSaver` → `PostgresSaver` (Databricks Lakebase). Lakebase is managed Postgres inside the Databricks workspace. OLTP workload — random reads/writes, high concurrency, low latency. No code changes in nodes or graph topology — only `_get_checkpointer()` in `graph.py` changes.

#### Table 2: `run_history`

**Schema:**

| Column | Type | Purpose |
|--------|------|---------|
| `run_id` | TEXT PK | UUID linking to LangGraph checkpoint |
| `catalog` | TEXT | Which UC catalog was analyzed |
| `schema_name` | TEXT | Which schema was analyzed |
| `use_case` | TEXT | Approved ML use case name (e.g. "churn prediction") |
| `status` | TEXT | Last known status: `awaiting_approval`, `completed`, `error`, etc. |
| `created_at` | TEXT | ISO-8601 UTC — when run started |
| `updated_at` | TEXT | ISO-8601 UTC — last state change |

**Purpose:** Episodic memory. The `/runs/history` endpoint returns this table. Users can return to in-progress runs after page refresh. Status is updated at every human checkpoint and completion. Upsert logic (`ON CONFLICT DO UPDATE`) preserves `catalog/schema/use_case` if a later update passes empty strings — prevents accidental field erasure.

**Why this exists separately from checkpoints:** LangGraph checkpoints are opaque blobs. Run history is a clean queryable summary. You can list recent runs without deserializing checkpoint state.

**Phase 3 upgrade path:** Delta table `ml_accelerator.run_history`. Queryable from Databricks SQL, dashboards, Genie.

#### Table 3: `audit_trail`

**Schema:**

| Column | Type | Purpose |
|--------|------|---------|
| `event_id` | TEXT PK | UUID4 — globally unique event identifier |
| `run_id` | TEXT | Links to `run_history.run_id` |
| `sequence_number` | INTEGER | 1-based, monotonically increasing per run |
| `event_type` | TEXT | Controlled vocabulary (see below) |
| `actor` | TEXT | `"agent"` or `"user"` |
| `node_name` | TEXT | Which graph node emitted this event |
| `timestamp_utc` | TEXT | ISO-8601 UTC |
| `payload` | TEXT | JSON blob — event-specific structured data |
| `prev_hash` | TEXT | SHA-256 hex of previous event, or `"GENESIS"` |
| `event_hash` | TEXT | SHA-256 hex of this event's canonical serialization |

**Unique constraint:** `(run_id, sequence_number)` — prevents duplicate sequence numbers.

**Index:** `idx_audit_run_seq ON (run_id, sequence_number)` — fast retrieval of a run's full event chain.

**Controlled event types:**

| Event type | Actor | When emitted |
|------------|-------|--------------|
| `catalog_discovered` | agent | After `discover_catalog` |
| `opportunity_ranked` | agent | After `rank_opportunities` |
| `opportunity_approved` | user | On `POST /runs/{id}/approve` — before graph resumes |
| `dry_run_generated` | agent | After `dry_run_explain` |
| `dry_run_confirmed` | user | On `POST /runs/{id}/confirm-dry-run` — before graph resumes |
| `code_generated` | agent | After `generate_code` |
| `code_approved` | user | On `POST /runs/{id}/approve-code` with `action=approve` |
| `code_regeneration_requested` | user | On `POST /runs/{id}/approve-code` with `action=regenerate` |
| `bundle_written` | agent | After `write_bundle` |

**Hash chain integrity:**
- Every event stores its own `event_hash` (SHA-256 of all fields except `event_hash` itself, keys sorted, deterministic JSON).
- Every event stores `prev_hash` (the `event_hash` of the previous event, or `"GENESIS"`).
- `ChainVerifier.verify(run_id)` recomputes every hash and checks every prev_hash link. If any event was tampered with or deleted, the chain breaks and verification returns `tampered_at_sequence`.
- **Why this matters for enterprise:** Databricks customers in regulated industries (finance, healthcare) need immutable evidence that a human approved AI-generated code before it was deployed. The hash chain provides cryptographic proof of the approval sequence — tamper-evident without a blockchain.

**AuditWriter pattern:**
- `AuditWriter` is a process-level singleton. `emit()` returns immediately — write happens in a background daemon thread.
- Sequence numbers and `prev_hash` tracking are serialized per `run_id` (per-run lock), not globally — concurrent runs don't block each other.
- On process restart, `_next_sequence()` seeds from `store.get_max_sequence(run_id)` and `prev_hash` from the last stored event — chain continuity survives restarts.
- Failed writes are buffered (max 100 events) with exponential backoff retry (1s → 60s cap) — transient SQLite/Delta errors don't lose audit events.

**Phase 3 upgrade path:** Delta table `ml_accelerator.audit_trail` with `TBLPROPERTIES ('delta.appendOnly'='true')`. Append-only enforced at the storage layer — no UPDATE/DELETE possible, not just at the application layer. Queryable from Databricks SQL for compliance reporting.

---

### 8. Audit Module — `audit/`

| File | Responsibility |
|------|---------------|
| `models.py` | `AuditEvent` TypedDict, `VerificationResult` TypedDict |
| `hashing.py` | `compute_event_hash()` — SHA-256 of canonical JSON (sorted keys, ASCII-safe) |
| `store.py` | `AuditStore` Protocol + `SqliteAuditStore` implementation |
| `writer.py` | `AuditWriter` — threading, sequencing, retry buffer |
| `verifier.py` | `ChainVerifier` — full chain replay and integrity check |

**Protocol pattern (`AuditStore`):**
```python
@runtime_checkable
class AuditStore(Protocol):
    def write(self, event: AuditEvent) -> None: ...
    def get_events(self, run_id: str) -> list[AuditEvent]: ...
    def get_max_sequence(self, run_id: str) -> int: ...
    def ensure_table(self) -> None: ...
```

Any class implementing these four methods is a valid `AuditStore`. Swap `SqliteAuditStore` → `DeltaAuditStore` in `get_audit_store()` — nothing else changes. The Protocol is `runtime_checkable` so the swap can be validated at startup.

---

## Storage: Current vs Phase 3

| Workload | Current (Phase 1–2) | Phase 3 Upgrade | Reason for upgrade |
|----------|--------------------|-----------------|--------------------|
| LangGraph checkpoints | SQLite (`checkpoints` table) | Databricks Lakebase (PostgresSaver) | OLTP, concurrent runs, no file lock contention |
| Run history | SQLite (`run_history` table) | Delta `ml_accelerator.run_history` | SQL queryable, Genie dashboards, lineage |
| Audit trail | SQLite (`audit_trail` table) | Delta `ml_accelerator.audit_trail` | Append-only enforcement, compliance, SQL access |

All three upgrades are independent. Each has an isolated swap point:
- Checkpoints: `_get_checkpointer()` in `agent/graph.py`
- Run history: `_ensure_history_table()` / `record_run_history()` / `get_run_history()` in `agent/graph.py`
- Audit: `get_audit_store()` in `audit/store.py`

---

## Data Flow: Full Run

```
POST /runs
  ├── validate_workspace() — UC catalog + schema exist?
  ├── run_discovery(run_id, ...)
  │   ├── graph.invoke(initial_state)
  │   │   ├── discover_catalog     → audit: catalog_discovered
  │   │   ├── analyze_estate       → LLM enrichment
  │   │   ├── rank_opportunities   → audit: opportunity_ranked
  │   │   └── human_checkpoint     → interrupt() ⏸ SUSPEND
  │   └── record_run_history(status=awaiting_approval)
  └── return {status, opportunities, tables, estate_summary}

POST /runs/{id}/approve
  ├── audit: opportunity_approved (before graph.invoke — captured even if graph fails)
  ├── graph.invoke(Command(resume={selected_rank}))
  │   ├── dry_run_explain          → audit: dry_run_generated
  │   ├── generate_business_brief  → CTO brief → exec_summary
  │   └── dry_run_checkpoint       → interrupt() ⏸ SUSPEND
  ├── record_run_history(use_case, status=awaiting_dry_run_confirmation)
  └── return {dry_run_plan, exec_summary (business brief), approved_opportunity}

POST /runs/{id}/confirm-dry-run
  ├── audit: dry_run_confirmed
  ├── graph.invoke(Command(resume={confirmed: True}))
  │   ├── plan_features            → LLM feature plan
  │   ├── generate_code            → audit: code_generated
  │   ├── compute_risk_scorecard   → rule-based checks
  │   └── human_checkpoint_code   → interrupt() ⏸ SUSPEND
  └── return {risk_scorecard, notebooks, job_yamls}

POST /runs/{id}/approve-code
  ├── audit: code_approved OR code_regeneration_requested
  ├── graph.invoke(Command(resume={action, instructions}))
  │   ├── write_bundle             → audit: bundle_written
  │   └── generate_exec_summary   → full SUMMARY.md → exec_summary
  └── return {bundle_written, exec_summary, artifacts_written}
```

---

## Key Design Decisions and Why

### 1. Run ID as universal key
A single UUID traces: LangGraph checkpoint (`thread_id`), `run_history.run_id`, `audit_trail.run_id`. No join complexity. Any piece of state — checkpoint, history row, audit events — can be retrieved with one ID.

### 2. Audit events emitted BEFORE graph.invoke
In `approve_opportunity`, `confirm_dry_run`, `approve_code` — the audit event is written **before** calling `graph.invoke(Command(resume=...))`. If the graph fails, we still have a record that the user made a decision. This prevents audit gaps when LLM calls fail or nodes throw exceptions.

### 3. Deterministic business brief (no LLM)
`generate_business_brief` uses no LLM. It is a string-formatting function over `DryRunPlan` + `MLOpportunity`. Reason: a CTO brief shown between dry run and code approval must be 100% reproducible — the same inputs produce the exact same output, always. LLM non-determinism is unacceptable in a legal/compliance-adjacent context.

### 4. WorkspaceContext stored as plain dict in state
`AgentState.workspace` is `dict`, not `WorkspaceContext`. The LangGraph SQLite checkpointer serializes state as JSON. Pydantic models are not directly JSON-serializable. Plain dicts are. Every node that needs a `WorkspaceContext` calls `WorkspaceContext(**state["workspace"])` to reconstruct it.

### 5. `exec_summary` is reused
Before bundle write, `exec_summary` holds the CTO business brief. After `write_bundle`, `generate_exec_summary` overwrites it with the full technical SUMMARY.md. The `get_run_rehydrate` endpoint separates these: `business_brief = exec_summary if not bundle_written else ""`.

### 6. AuditWriter never raises
`AuditWriter.emit()` wraps everything in `try/except`. Failed writes go to a retry buffer. The pipeline never fails because audit logging failed. Audit is observability infrastructure — it must be fire-and-forget.

### 7. Per-run locks in AuditWriter
Sequence number increment and `prev_hash` chaining are protected by a `threading.Lock` per `run_id`. This means two concurrent runs can emit events simultaneously without blocking each other, while two events in the same run are serialized correctly.

---

## File Structure

```
databricks-ml-accelerator/
├── api/
│   └── main.py              FastAPI app, routes, OBO identity extraction
├── agent/
│   ├── state.py             AgentState TypedDict + sub-TypedDicts
│   ├── graph.py             LangGraph graph, checkpointer, run_history, public API
│   ├── nodes.py             Phase 1: discover, analyze, rank, human_checkpoint
│   ├── trust_nodes.py       Trust layer: dry_run, business_brief, risk_scorecard, exec_summary
│   ├── code_gen_nodes.py    Phase 2: plan_features, generate_code, write_bundle
│   └── chat.py              Scoped Q&A against run state
├── audit/
│   ├── models.py            AuditEvent, VerificationResult TypedDicts
│   ├── hashing.py           SHA-256 canonical hash
│   ├── store.py             AuditStore Protocol + SqliteAuditStore
│   ├── writer.py            AuditWriter (threading, retry, chain tracking)
│   └── verifier.py          ChainVerifier (hash chain replay)
├── tools/
│   └── workspace_context.py WorkspaceContext, validate_workspace, list_catalogs/schemas
├── config/
│   └── settings.py          Pydantic BaseSettings from .env
├── bundles/
│   ├── databricks.yml       Databricks Asset Bundle root
│   └── resources/jobs/      Generated job YAML files land here
├── data/
│   └── checkpoints.db       SQLite: checkpoints + run_history + audit_trail
├── app.yaml                 Databricks App manifest
├── start.sh                 Process launcher (FastAPI + Streamlit)
└── requirements.txt         Python dependencies
```

---

## Phase 3 Storage Migration Plan

When moving from local SQLite to Databricks-native storage:

### Step 1 — Databricks Lakebase for LangGraph checkpoints
```python
# agent/graph.py — _get_checkpointer()
from langgraph.checkpoint.postgres import PostgresSaver
conn_str = get_lakebase_connection_string()  # from secret scope
_checkpointer = PostgresSaver.from_conn_string(conn_str)
```
Why Lakebase over Delta for checkpoints: Checkpointing is an OLTP workload — random reads and writes by thread_id, not batch analytics. Postgres handles this efficiently. Delta is optimized for batch append and scan, not point lookups.

### Step 2 — Delta table for audit trail
```python
# audit/store.py — DeltaAuditStore
class DeltaAuditStore:
    def ensure_table(self):
        spark.sql("""
            CREATE TABLE IF NOT EXISTS ml_accelerator.audit_trail (
                event_id STRING NOT NULL,
                run_id STRING NOT NULL,
                sequence_number BIGINT NOT NULL,
                event_type STRING NOT NULL,
                actor STRING NOT NULL,
                node_name STRING NOT NULL,
                timestamp_utc STRING NOT NULL,
                payload STRING NOT NULL,
                prev_hash STRING NOT NULL,
                event_hash STRING NOT NULL
            )
            TBLPROPERTIES ('delta.appendOnly'='true')
        """)
```
Why append-only Delta: The `delta.appendOnly=true` property makes UPDATE/DELETE raise an error at the storage layer, not just the application layer. Compliance teams can verify this via `SHOW TBLPROPERTIES`. SQLite has no equivalent enforcement.

### Step 3 — Delta table for run history
```python
spark.sql("""
    CREATE TABLE IF NOT EXISTS ml_accelerator.run_history (
        run_id STRING NOT NULL,
        catalog STRING,
        schema_name STRING,
        use_case STRING,
        status STRING,
        created_at STRING,
        updated_at STRING
    )
""")
```
Why Delta for run history (not Lakebase): Run history is read infrequently, written infrequently, but needs to be queryable from Databricks SQL for dashboards and Genie. Delta integrates naturally with the UC governance model already in place.
