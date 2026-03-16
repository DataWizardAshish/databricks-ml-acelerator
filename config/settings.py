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
    databricks_cluster_id: str = "1217-094102-xswsfzua"

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_endpoint_name: str = "ds-brand-funnel-ai_claude-sonnet-4-5_chat_dev"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # ── Unity Catalog ────────────────────────────────────────────────────────
    uc_catalog: str = "lakehouse_dev"
    uc_discovery_schema: str = "ds_brand_funnel_ai_lhdev"
    uc_output_schema: str = "ml_accelerator"

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
