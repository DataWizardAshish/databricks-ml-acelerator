"""
Per-request workspace connection context.
Carries host/token/catalog/schema for a single agent run.
Falls back to ~/.databrickscfg DEFAULT profile when host/token are empty.
This makes the system multi-workspace safe — different teams pass their own creds.
"""

from pydantic import BaseModel, model_validator
from databricks.sdk import WorkspaceClient

from config.settings import get_settings


class WorkspaceContext(BaseModel):
    """
    Passed with every API request. Empty fields fall through to
    ~/.databrickscfg DEFAULT profile (PAT today, OAuth when AUTH_TYPE=oauth).
    """
    host: str = ""
    token: str = ""
    catalog: str = ""
    schema: str = ""
    cluster_id: str = ""

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
        """Thread-safe — each call creates a new client with its own credentials."""
        kwargs: dict = {}
        if self.host:
            kwargs["host"] = self.host
        if self.token:
            kwargs["token"] = self.token
        return WorkspaceClient(**kwargs)

    def get_llm(self, max_tokens: int | None = None, temperature: float | None = None):
        """
        Returns ChatDatabricks bound to this workspace's endpoint.
        Passes host/token explicitly — safe for concurrent requests to different workspaces.
        """
        from databricks_langchain import ChatDatabricks
        settings = get_settings()

        kwargs: dict = {
            "endpoint": settings.llm_endpoint_name,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
        }
        if self.host:
            kwargs["databricks_host"] = self.host
        if self.token:
            kwargs["databricks_token"] = self.token

        return ChatDatabricks(**kwargs)


def validate_workspace(ctx: WorkspaceContext) -> dict:
    """
    Pre-flight check before starting the agent.
    Returns {"valid": True} or {"valid": False, "error": "...", "field": "..."}
    """
    client = ctx.get_workspace_client()

    # Check catalog exists
    try:
        client.catalogs.get(name=ctx.catalog)
    except Exception:
        return {
            "valid": False,
            "error": f"Catalog '{ctx.catalog}' not found or you don't have access to it.",
            "field": "catalog",
        }

    # Check schema exists
    try:
        client.schemas.get(full_name=f"{ctx.catalog}.{ctx.schema}")
    except Exception:
        return {
            "valid": False,
            "error": f"Schema '{ctx.schema}' not found in catalog '{ctx.catalog}'.",
            "field": "schema",
        }

    return {"valid": True}


def list_catalogs(ctx: WorkspaceContext) -> list[str]:
    client = ctx.get_workspace_client()
    return sorted([c.name for c in client.catalogs.list() if c.name])


def list_schemas(ctx: WorkspaceContext, catalog: str) -> list[str]:
    client = ctx.get_workspace_client()
    return sorted([s.name for s in client.schemas.list(catalog_name=catalog) if s.name])
