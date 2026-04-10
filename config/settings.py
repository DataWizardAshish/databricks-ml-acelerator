"""
Central config. Auth defaults to ~/.databrickscfg DEFAULT profile (PAT).
Set AUTH_TYPE=oauth when ready to switch — no other code changes needed.
"""

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Databricks auth ──────────────────────────────────────────────────────
    # Leave blank to use ~/.databrickscfg DEFAULT profile automatically
    databricks_host: str = ""
    databricks_token: str = ""

    # Auth type — swap to "oauth" later, nothing else changes
    auth_type: Literal["pat", "oauth"] = "pat"

    # ── Compute ──────────────────────────────────────────────────────────────
    databricks_cluster_id: str = "0402-042550-3eto1bj6"

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_endpoint_name: str = "databricks-meta-llama-3-3-70b-instruct"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # ── Unity Catalog ────────────────────────────────────────────────────────
    uc_catalog: str = "dbw_vectoflow_dev"
    uc_discovery_schema: str = "vectoflow"
    uc_output_schema: str = "ml_accelerator"

    # ── Lakebase (Postgres checkpointer) ─────────────────────────────────────
    # Leave blank to fall back to SQLite (local dev without Lakebase)
    # ── SQL Warehouse (OBO path — UC metadata queries run as end user) ───────────
    sql_warehouse_id: str = ""         # e.g. cd22865cf1b63262

    lakebase_endpoint_name: str = ""   # projects/ml-accelerator/branches/production/endpoints/primary
    lakebase_host: str = ""            # ep-xxx.database.eastus.azuredatabricks.net
    lakebase_user: str = ""            # your.email@domain.com or service-principal UUID
    lakebase_database: str = "databricks_postgres"

    @model_validator(mode="after")
    def validate_auth(self) -> "Settings":
        if self.auth_type == "oauth" and (self.databricks_host or self.databricks_token):
            # OAuth will use the Databricks SDK's OAuth flow — token not needed
            self.databricks_token = ""
        return self

    @property
    def workspace_url(self) -> str:
        return self.databricks_host.rstrip("/") if self.databricks_host else ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
