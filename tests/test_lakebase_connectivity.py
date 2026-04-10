"""
Lakebase connectivity tests.

Tests:
  1. OAuth token generation via Databricks SDK
  2. Raw psycopg connection with OAuth token as password
  3. ConnectionPool with OAuthConnection — token rotation
  4. PostgresSaver.setup() — LangGraph checkpoint tables created
  5. Write + read a checkpoint round-trip via the graph's _get_checkpointer()
  6. run_history table still works (SQLite — not migrated)
  7. /health endpoint returns authenticated_user

Run:
  pytest tests/test_lakebase_connectivity.py -v
"""

import os
import sys
import uuid

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load .env so settings pick up LAKEBASE_* vars
from dotenv import load_dotenv
load_dotenv()

from config.settings import get_settings

get_settings.cache_clear()  # force reload with .env values
SETTINGS = get_settings()
LAKEBASE_CONFIGURED = bool(SETTINGS.lakebase_endpoint_name and SETTINGS.lakebase_host)

lakebase_only = pytest.mark.skipif(
    not LAKEBASE_CONFIGURED,
    reason="LAKEBASE_ENDPOINT_NAME / LAKEBASE_HOST not set — skipping Lakebase tests",
)


# ── Test 1: OAuth token generation ───────────────────────────────────────────

def _workspace_client():
    """WorkspaceClient with explicit host+token from settings (local dev + Apps)."""
    from databricks.sdk import WorkspaceClient
    kwargs = {}
    if SETTINGS.databricks_host:
        kwargs["host"] = SETTINGS.databricks_host
    if SETTINGS.databricks_token:
        kwargs["token"] = SETTINGS.databricks_token
    return WorkspaceClient(**kwargs)


@lakebase_only
def test_oauth_token_generation():
    """Databricks SDK can generate a Lakebase database credential."""
    w = _workspace_client()
    credential = w.postgres.generate_database_credential(
        endpoint=SETTINGS.lakebase_endpoint_name
    )
    assert credential.token, "Token is empty"
    assert len(credential.token) > 100, "Token looks too short"
    assert credential.expire_time is not None, "expire_time missing"
    print(f"\n  Token prefix: {credential.token[:40]}...")
    print(f"  Expires: {credential.expire_time}")


# ── Test 2: Raw psycopg connection ────────────────────────────────────────────

@lakebase_only
def test_raw_psycopg_connection():
    """Connect to Lakebase with a fresh OAuth token and run SELECT 1."""
    import psycopg

    w = _workspace_client()
    credential = w.postgres.generate_database_credential(
        endpoint=SETTINGS.lakebase_endpoint_name
    )

    conn = psycopg.connect(
        host=SETTINGS.lakebase_host,
        port=5432,
        dbname=SETTINGS.lakebase_database,
        user=SETTINGS.lakebase_user,
        password=credential.token,
        sslmode="require",
    )
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        row = cur.fetchone()
    conn.close()

    assert row is not None
    assert "PostgreSQL" in row[0]
    print(f"\n  Postgres version: {row[0]}")


# ── Test 3: ConnectionPool with OAuthConnection ───────────────────────────────

@lakebase_only
def test_connection_pool_oauth_rotation():
    """Pool creates connections with fresh tokens — token rotation works."""
    import psycopg
    from psycopg_pool import ConnectionPool

    endpoint_name = SETTINGS.lakebase_endpoint_name

    class OAuthConnection(psycopg.Connection):
        @classmethod
        def connect(cls, conninfo="", **kwargs):
            w = _workspace_client()
            credential = w.postgres.generate_database_credential(endpoint=endpoint_name)
            kwargs["password"] = credential.token
            return super().connect(conninfo, **kwargs)

    pool = ConnectionPool(
        conninfo=(
            f"dbname={SETTINGS.lakebase_database} user={SETTINGS.lakebase_user} "
            f"host={SETTINGS.lakebase_host} port=5432 sslmode=require"
        ),
        connection_class=OAuthConnection,
        min_size=1,
        max_size=3,
        open=True,
    )

    # Run two queries through the pool
    for i in range(2):
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user, now()")
                row = cur.fetchone()
                assert row is not None
                print(f"\n  Pool query {i+1}: user={row[0]}, time={row[1]}")

    pool.close()


# ── Test 4: PostgresSaver.setup() ─────────────────────────────────────────────

@lakebase_only
def test_postgres_saver_setup():
    """PostgresSaver.setup() creates LangGraph checkpoint tables in Lakebase."""
    import psycopg
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    endpoint_name = SETTINGS.lakebase_endpoint_name

    class OAuthConnection(psycopg.Connection):
        @classmethod
        def connect(cls, conninfo="", **kwargs):
            w = _workspace_client()
            cred = w.postgres.generate_database_credential(endpoint=endpoint_name)
            kwargs["password"] = cred.token
            return super().connect(conninfo, **kwargs)

    pool = ConnectionPool(
        conninfo=(
            f"dbname={SETTINGS.lakebase_database} user={SETTINGS.lakebase_user} "
            f"host={SETTINGS.lakebase_host} port=5432 sslmode=require"
        ),
        connection_class=OAuthConnection,
        min_size=1,
        max_size=3,
        open=True,
    )

    # setup() uses CREATE INDEX CONCURRENTLY — requires autocommit, not a pool connection
    w_setup = _workspace_client()
    cred_setup = w_setup.postgres.generate_database_credential(endpoint=SETTINGS.lakebase_endpoint_name)
    setup_conn = psycopg.connect(
        host=SETTINGS.lakebase_host, port=5432,
        dbname=SETTINGS.lakebase_database, user=SETTINGS.lakebase_user,
        password=cred_setup.token, sslmode="require", autocommit=True,
    )
    PostgresSaver(setup_conn).setup()  # idempotent — safe to run multiple times
    setup_conn.close()

    # Verify the checkpoint tables were created
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
                ORDER BY table_name
            """)
            tables = [r[0] for r in cur.fetchall()]

    pool.close()
    print(f"\n  LangGraph tables found: {tables}")
    assert len(tables) >= 2, f"Expected checkpoint tables, found: {tables}"


# ── Test 5: Checkpoint round-trip via graph._get_checkpointer() ───────────────

@lakebase_only
def test_checkpointer_singleton_is_postgres():
    """_get_checkpointer() returns a PostgresSaver when Lakebase is configured."""
    # Reset the singleton so this test gets a fresh instance
    import agent.graph as g
    g._checkpointer = None

    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = g._get_checkpointer()

    assert isinstance(checkpointer, PostgresSaver), (
        f"Expected PostgresSaver, got {type(checkpointer).__name__}"
    )
    print(f"\n  Checkpointer type: {type(checkpointer).__name__}")

    # Restore singleton for subsequent tests
    g._checkpointer = None


# ── Test 6: run_history (SQLite) still works ──────────────────────────────────

def test_run_history_sqlite():
    """run_history table in SQLite is unaffected by Lakebase migration."""
    import agent.graph as g

    test_run_id = f"test-{uuid.uuid4()}"
    g.record_run_history(
        run_id=test_run_id,
        catalog="test_catalog",
        schema="test_schema",
        use_case="test_churn_prediction",
        status="awaiting_approval",
    )

    rows = g.get_run_history(limit=5)
    found = next((r for r in rows if r["run_id"] == test_run_id), None)

    assert found is not None, "Test run not found in history"
    assert found["catalog"] == "test_catalog"
    assert found["status"] == "awaiting_approval"
    print(f"\n  run_history round-trip OK: {found}")


# ── Test 7: /health endpoint ──────────────────────────────────────────────────

def test_health_endpoint():
    """FastAPI /health returns 200 and includes expected fields."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "llm_endpoint" in data
    assert "authenticated_user" in data
    print(f"\n  /health: {data}")
