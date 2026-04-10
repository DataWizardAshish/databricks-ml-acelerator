"""
Per-request workspace connection context.
Carries host/token/catalog/schema for a single agent run.

Auth split (two separate paths — never mixed):
  Path A — OBO (end user):   get_sql_connection() → databricks.sql with OBO/PAT token
  Path B — App SP (M2M):     get_workspace_client() → WorkspaceClient, no token ever

Path A is used for all UC metadata queries (discover, validate, list tables/columns).
Path B is used for SDK operations only (Lakebase credential generation, audit trail).
"""

import os

from pydantic import BaseModel, model_validator
from databricks.sdk import WorkspaceClient

from config.settings import get_settings


class WorkspaceContext(BaseModel):
    """
    Passed with every API request.
    token = OBO JWT in Databricks Apps (X-Forwarded-Access-Token)
    token = PAT in local dev (from request body or ~/.databrickscfg)
    token is used ONLY in get_sql_connection() — never in get_workspace_client().
    """
    host: str = ""
    token: str = ""
    catalog: str = ""
    schema: str = ""
    cluster_id: str = ""
    user_email: str = ""  # populated from X-Forwarded-Email in Databricks Apps

    @model_validator(mode="after")
    def apply_defaults(self) -> "WorkspaceContext":
        settings = get_settings()
        if not self.catalog:
            self.catalog = settings.uc_catalog
        if not self.schema:
            self.schema = settings.uc_discovery_schema
        if not self.cluster_id:
            self.cluster_id = settings.databricks_cluster_id
        return self

    def get_workspace_client(self) -> WorkspaceClient:
        """
        Path B — App service principal (M2M OAuth) only.
        In Databricks Apps, the platform injects DATABRICKS_CLIENT_ID/SECRET (M2M) via env
        AND also writes a ~/.databrickscfg with a PAT for the service principal.
        The SDK reads BOTH and raises "multiple auth methods" conflict.
        Fix: config_file="/dev/null" tells the SDK to ignore ~/.databrickscfg entirely,
        leaving only M2M from env — no conflict, no token ever passed explicitly.
        """
        host = self.host or os.environ.get("DATABRICKS_HOST") or None
        return WorkspaceClient(host=host, config_file="/dev/null")

    def get_sql_connection(self):
        """
        Path A — end user identity via OBO token (or PAT in local dev).
        Uses databricks-sql-connector directly — bypasses WorkspaceClient auth entirely.
        No Config() / WorkspaceClient involved — zero SDK auth conflict risk.
        UC row-level policies and column masking are enforced as the end user.
        Used for: all UC metadata queries (SHOW TABLES, information_schema).
        """
        import logging
        _log = logging.getLogger(__name__)
        from databricks import sql
        settings = get_settings()
        # Resolve host from explicit context, env var, or settings — no SDK Config() call
        host = self.host or os.environ.get("DATABRICKS_HOST", "") or settings.databricks_host
        # OBO JWT in Apps, PAT from settings in local dev (settings reads .env / env vars)
        token = self.token or settings.databricks_token
        _log.info(
            "get_sql_connection: host=%s warehouse=%s token_source=%s token_prefix=%s",
            host,
            settings.sql_warehouse_id,
            "obo" if self.token else ("settings" if settings.databricks_token else "EMPTY"),
            (token[:12] + "...") if token else "NONE",
        )
        return sql.connect(
            server_hostname=host.replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{settings.sql_warehouse_id}",
            access_token=token,
        )

    def get_llm(self, max_tokens: int | None = None, temperature: float | None = None):
        """
        Path B — App service principal handles LLM endpoint auth via M2M.
        No token passed — SDK auto-discovers M2M from env.
        No Config() call — host resolved directly from env to avoid auth conflict.
        """
        from databricks_langchain import ChatDatabricks
        settings = get_settings()
        kwargs: dict = {
            "endpoint": settings.llm_endpoint_name,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
        }
        # Resolve host without triggering SDK auth validation
        host = self.host or os.environ.get("DATABRICKS_HOST", "") or settings.databricks_host
        if host:
            kwargs["databricks_host"] = host
        return ChatDatabricks(**kwargs)


def validate_workspace(ctx: WorkspaceContext) -> dict:
    """
    Pre-flight check via SQL connector (Path A — runs as end user).
    Returns {"valid": True} or {"valid": False, "error": "...", "field": "..."}
    """
    try:
        conn = ctx.get_sql_connection()
    except Exception as e:
        return {"valid": False, "error": f"SQL connection failed: {e}", "field": ""}

    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW CATALOGS LIKE '{ctx.catalog}'")
            if not cur.fetchone():
                return {
                    "valid": False,
                    "error": f"Catalog '{ctx.catalog}' not found or you don't have access to it.",
                    "field": "catalog",
                }
            cur.execute(f"SHOW SCHEMAS IN `{ctx.catalog}` LIKE '{ctx.schema}'")
            if not cur.fetchone():
                return {
                    "valid": False,
                    "error": f"Schema '{ctx.schema}' not found in catalog '{ctx.catalog}'.",
                    "field": "schema",
                }
    except Exception as e:
        return {"valid": False, "error": str(e), "field": ""}
    finally:
        conn.close()

    return {"valid": True}


def list_catalogs(ctx: WorkspaceContext) -> list[str]:
    """Browse — uses App SP (Path B). No OBO token needed for catalog listing."""
    client = ctx.get_workspace_client()
    return sorted([c.name for c in client.catalogs.list() if c.name])


def list_schemas(ctx: WorkspaceContext, catalog: str) -> list[str]:
    """Browse — uses App SP (Path B). No OBO token needed for schema listing."""
    client = ctx.get_workspace_client()
    return sorted([s.name for s in client.schemas.list(catalog_name=catalog) if s.name])
