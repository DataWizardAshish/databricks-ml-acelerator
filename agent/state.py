"""
LangGraph state for the ML Accelerator agent pipeline.
WorkspaceContext is carried through every node so all tools use the
same per-request credentials. Never read from global settings inside nodes.
"""

from typing import Optional
from typing_extensions import TypedDict

from tools.workspace_context import WorkspaceContext
from tools.uc_reader import TableInfo


class MLOpportunity(TypedDict):
    rank: int
    use_case: str
    target_table: str
    target_column: str
    feature_tables: list[str]
    ml_type: str                  # classification | regression | clustering | forecasting
    estimated_auc_range: str
    business_value: str
    complexity: str               # low | medium | high
    rationale: str


class ColumnTransform(TypedDict):
    column: str
    transform: str   # keep_numeric | log_transform | extract_days_since | one_hot | drop | hash_encode
    note: str


class FeaturePlan(TypedDict):
    feature_decisions: list[ColumnTransform]
    target_encoding: str            # e.g. "binary_0_1" or "keep_as_is"
    split_strategy: str             # "temporal" | "random"
    split_column: Optional[str]     # date/timestamp column used for temporal split
    class_balance: dict             # {"type": "balanced|imbalanced", "ratio": "...", "strategy": "class_weight|none"}
    high_cardinality_cols: list[str]
    suggested_model: str            # "XGBClassifier" | "XGBRegressor" | "XGBClassifier_multiclass"
    mlflow_experiment_name: str
    feature_table_name: str         # written to ml_accelerator schema
    scores_table_name: str          # written to ml_accelerator schema


class GeneratedArtifact(TypedDict):
    filename: str       # e.g. "01_feature_engineering.py"
    filepath: str       # relative path inside bundles/
    content: str        # full file content


class AgentState(TypedDict):
    # ── Phase 1 ──────────────────────────────────────────────────────────────
    workspace: WorkspaceContext
    tables: list[TableInfo]
    estate_summary: str
    opportunities: list[MLOpportunity]
    approved_opportunity: Optional[MLOpportunity]

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    feature_plan: Optional[FeaturePlan]
    generated_artifacts: list[GeneratedArtifact]  # notebooks + job YAMLs
    bundle_written: bool

    # ── Error tracking ────────────────────────────────────────────────────────
    error: Optional[str]
