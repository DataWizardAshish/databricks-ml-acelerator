# Lakebase & Databricks Apps Auth — Connection Guide

## Overview

This document covers the authentication architecture for the ML Accelerator Databricks App, including:
- How Databricks Apps inject credentials and why they conflict
- The two-path auth split (Path A: OBO, Path B: App SP)
- Lakebase (PostgresSaver) connection via App SP M2M
- Every error encountered, its root cause, and the resolution

---

## Databricks Apps Auth Injection (What the Platform Does)

When a Databricks App starts, the platform automatically injects credentials via **two separate mechanisms simultaneously**:

| Mechanism | What it sets | Purpose |
|---|---|---|
| Environment variables | `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` | App Service Principal M2M OAuth |
| Config file | `~/.databrickscfg` (written by platform) | Legacy PAT for the App SP |

The Databricks Python SDK (`WorkspaceClient`, `Config`) reads **both** on startup. When it finds M2M OAuth credentials in env vars AND a PAT in the config file simultaneously, it raises:

```
ValueError: validate: more than one authorization method configured: oauth and pat
```

This is not a bug — it is intentional SDK validation. The SDK refuses to guess which credential to use.

---

## Auth Architecture: Two Separate Paths

The solution is to split auth into two completely independent paths that never touch each other.

```
┌─────────────────────────────────────────────────────────┐
│  Path A — OBO (end user identity)                       │
│  databricks.sql connector ← access_token=obo_token      │
│  Used for: UC metadata, SQL Warehouse queries            │
│  Token source: X-Forwarded-Access-Token header           │
│  No WorkspaceClient, No Config() — zero conflict risk    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Path B — App SP (M2M OAuth)                            │
│  WorkspaceClient(host=host, config_file="/dev/null")     │
│  Used for: Lakebase credential generation, browse APIs   │
│  Token source: DATABRICKS_CLIENT_ID/SECRET (env only)   │
│  config_file="/dev/null" prevents reading ~/.databrickscfg│
└─────────────────────────────────────────────────────────┘
```

### Key Rule
- `DATABRICKS_TOKEN` must **never** appear as an env var in `app.yaml` — it would conflict with M2M OAuth
- The PAT goes nowhere in App mode — it is neither passed to WorkspaceClient nor set as an env var
- `Config()` (bare, no args) must **never** be called — it validates all auth sources and raises the conflict

---

## Path A: OBO Token for UC / SQL Warehouse

### How OBO Works in Databricks Apps

Databricks Apps inject the **end user's access token** into every request via:
```
X-Forwarded-Access-Token: eyJraWQiOiJj...
```

This is a scoped JWT representing the logged-in user. It is forwarded to the backend on every request.

### Required App Scopes (User Authorization)

The OBO token only carries scopes that the app has declared. Default scopes are insufficient for SQL. Configure in:

**Databricks Workspace UI → Compute → Apps → [app-name] → Edit → Configure Resources → User Authorization**

Required scopes for this app:

| Scope | Purpose |
|---|---|
| `sql` | Execute SQL and connect to SQL Warehouses |
| `catalog.catalogs:read` | Read Unity Catalog catalog list |
| `catalog.schemas:read` | Read schemas within a catalog |
| `catalog.tables:read` | Read table metadata |

### Activating Scopes (Critical Step)

After adding scopes in the UI, a **redeploy alone is not sufficient**. The app must be:
1. **Fully stopped** (not just redeployed)
2. **Started** again
3. User must go through the **consent screen** on first access after restart

Without the consent flow, the OBO token is issued without the new scopes and SQL connections return 403.

### Python Implementation

```python
# api/main.py — extract OBO token per request
user_token = request.headers.get("x-forwarded-access-token", "")

# tools/workspace_context.py — Path A, zero SDK auth involvement
def get_sql_connection(self):
    from databricks import sql
    host = self.host or os.environ.get("DATABRICKS_HOST", "")
    token = self.token or settings.databricks_token  # OBO in App, PAT in local dev
    return sql.connect(
        server_hostname=host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{settings.sql_warehouse_id}",
        access_token=token,
    )
```

**Rules:**
- Create a new connection per request — never reuse connections (OBO tokens expire)
- Never cache the token
- Never log the full token value

---

## Path B: App SP M2M for Lakebase

### How App SP M2M Works

When `config_file="/dev/null"` is passed, the SDK ignores `~/.databrickscfg` entirely. Only `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` from env vars are used — pure M2M OAuth. No conflict.

```python
# tools/workspace_context.py — Path B
def get_workspace_client(self) -> WorkspaceClient:
    host = self.host or os.environ.get("DATABRICKS_HOST") or None
    return WorkspaceClient(host=host, config_file="/dev/null")
```

### Lakebase Connection Flow

```
App SP M2M (DATABRICKS_CLIENT_ID/SECRET)
  → WorkspaceClient(config_file="/dev/null")
  → w.postgres.generate_database_credential(endpoint=endpoint_name)
  → returns short-lived OAuth token
  → passed as `password` to psycopg3 ConnectionPool
  → connects to ep-xxx.database.eastus.azuredatabricks.net:5432
```

Token rotation is handled automatically — `OAuthConnection.connect()` is called by the pool on every new connection, generating a fresh token each time.

### Lakebase Grants (One-time Setup)

Run these in a Databricks SQL Editor against the Lakebase endpoint. Use the App SP's **UUID** (not the display name, not the personal email):

```sql
-- Create a Postgres role tied to the App SP identity
SELECT databricks_create_role('98732ce4-1f2d-4768-bbb2-caa6bd246b37', 'service_principal');

-- Grant database access
GRANT CONNECT ON DATABASE databricks_postgres TO "98732ce4-1f2d-4768-bbb2-caa6bd246b37";

-- Grant schema access
GRANT CREATE, USAGE ON SCHEMA public TO "98732ce4-1f2d-4768-bbb2-caa6bd246b37";

-- Grant table access
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "98732ce4-1f2d-4768-bbb2-caa6bd246b37";
```

Find the App SP UUID in: Workspace UI → Apps → [app-name] → service_principal_client_id field.

### `LAKEBASE_USER` Must Be the App SP UUID

The `user` field in the Postgres connection string must match the identity that generated the credential. For App SP M2M, that is the **UUID**, not the email address.

```yaml
# app.yaml — CORRECT
- name: LAKEBASE_USER
  value: "98732ce4-1f2d-4768-bbb2-caa6bd246b37"

# WRONG — personal email is not a registered Postgres role
- name: LAKEBASE_USER
  value: "ashish.devforge@gmail.com"
```

### Checkpointer Fallback Chain

```
1. Lakebase (Postgres)   — when LAKEBASE_ENDPOINT_NAME + LAKEBASE_HOST are set
2. SQLite (file-backed)  — ./data/checkpoints.db — local dev fallback
3. MemorySaver           — last resort if langgraph-checkpoint-sqlite not installed
```

---

## Errors Encountered — Root Causes and Resolutions

### Error 1: `more than one authorization method configured: oauth and pat`

**Where:** Any `WorkspaceClient()` or `Config()` call inside the Databricks App  
**Root cause:** Platform injects M2M OAuth via env vars AND writes a PAT to `~/.databrickscfg`. SDK reads both and refuses to proceed.  
**What didn't work:**
- Passing `client_id=""`, `client_secret=""` — empty string is falsy, SDK still reads env vars
- Removing the OBO token from the request — error persisted from a different call site
- Setting `DATABRICKS_CONFIG_FILE=/dev/null` in `app.yaml` alone — `Config()` calls in code still triggered before the env var was evaluated

**Resolution:**
1. Pass `config_file="/dev/null"` to every `WorkspaceClient()` call in code
2. Never call `Config()` bare — replace with `os.environ.get("DATABRICKS_HOST")` wherever only the host URL is needed
3. Never set `DATABRICKS_TOKEN` as an env var in `app.yaml`

```python
# Before — triggers conflict
cfg = Config()
host = cfg.host

# After — no SDK auth involved
host = os.environ.get("DATABRICKS_HOST", "")
```

---

### Error 2: `Received 403 - FORBIDDEN` on SQL Warehouse (OBO token)

**Where:** `validate_workspace()` → `get_sql_connection()` → `databricks.sql.connect()`  
**Root cause:** OBO token was present but lacked `sql` scope. App only had `iam.current-user:read` and `iam.access-control:read` scopes by default.  
**What didn't work:** Redeploying the app after adding scopes — consent flow not triggered without full stop + start.  
**Resolution:**
1. Add `sql`, `catalog.catalogs:read`, `catalog.schemas:read`, `catalog.tables:read` scopes via workspace UI
2. **Fully stop** the app, then **start** it (not just redeploy)
3. User must go through the consent screen on first access

---

### Error 3: `PRINCIPAL_DOES_NOT_EXIST` when granting UC permissions

**Where:** Databricks SQL Editor when running GRANT statements  
**Root cause:** Used the App SP's display name (`app-2imtvs ml-accelerator`) which is not a valid principal identifier for GRANT statements.  
**Resolution:** Use the App SP's UUID in backticks:

```sql
-- WRONG
GRANT USE CATALOG ON CATALOG dbw_vectoflow_dev TO `app-2imtvs ml-accelerator`;

-- CORRECT
GRANT USE CATALOG ON CATALOG dbw_vectoflow_dev TO `98732ce4-1f2d-4768-bbb2-caa6bd246b37`;
```

---

### Error 4: `password authentication failed for user 'ashish.devforge@gmail.com'` (Lakebase)

**Where:** `OAuthConnection.connect()` → psycopg3 Lakebase connection  
**Root cause:** `LAKEBASE_USER` was set to the personal email address. The Lakebase grants were executed with the App SP UUID, so no Postgres role exists for the email. `generate_database_credential` returns a token for the App SP, but the connection attempts to log in as a different user.  
**Resolution:** Set `LAKEBASE_USER` to the App SP UUID in `app.yaml`:

```yaml
- name: LAKEBASE_USER
  value: "98732ce4-1f2d-4768-bbb2-caa6bd246b37"
```

---

## app.yaml Auth-Related Settings Reference

```yaml
env:
  # Prevents SDK from reading ~/.databrickscfg — forces M2M-only for Path B
  - name: DATABRICKS_CONFIG_FILE
    value: "/dev/null"

  # SQL Warehouse for Path A (OBO queries)
  - name: SQL_WAREHOUSE_ID
    value: "cd22865cf1b63262"

  # Lakebase — App SP UUID as the Postgres user (matches generate_database_credential identity)
  - name: LAKEBASE_USER
    value: "98732ce4-1f2d-4768-bbb2-caa6bd246b37"

  # DO NOT set DATABRICKS_TOKEN here — conflicts with M2M OAuth
```

---

## Local Dev vs App Mode Comparison

| | Local Dev | Databricks App |
|---|---|---|
| Auth method | PAT from `~/.databrickscfg` | M2M OAuth via `DATABRICKS_CLIENT_ID/SECRET` |
| SQL token | PAT from `.env` → `settings.databricks_token` | OBO token from `X-Forwarded-Access-Token` |
| Lakebase token | PAT-based (or SQLite fallback) | `generate_database_credential` via App SP |
| `DATABRICKS_TOKEN` env | Set in `.env` (blank — do not populate) | Never set |
| Checkpointer | SQLite (`./data/checkpoints.db`) | Lakebase Postgres (when configured) |
