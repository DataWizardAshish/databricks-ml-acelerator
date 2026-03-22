"""
LangGraph state for the ML Accelerator agent pipeline.

workspace is stored as a plain dict (WorkspaceContext.model_dump()) so state
is fully JSON-serializable for SQLite/Delta checkpointers.
Reconstruct: WorkspaceContext(**state["workspace"]) inside nodes.

tables is stored as list of plain dicts (dataclasses.asdict(TableInfo)).
"""

from typing import Optional
from typing_extensions import TypedDict


class MLOpportunity(TypedDict):
    rank: int
    use_case: str
    target_table: str
    target_column: str
    feature_tables: list[str]
    ml_type: str                  # classification | regression | clustering | forecasting
    estimated_auc_range: str
    business_value: str
    financial_impact: str         # e.g. "$1.2-2.4M retained ARR annually"
    confidence: str               # high | medium | low
    complexity: str               # low | medium | high
    rationale: str


class ColumnTransform(TypedDict):
    column: str
    transform: str   # keep_numeric | log_transform | extract_days_since | one_hot | drop | hash_encode
    note: str


class FeaturePlan(TypedDict):
    feature_decisions: list[ColumnTransform]
    target_encoding: str
    split_strategy: str             # "temporal" | "random"
    split_column: Optional[str]
    class_balance: dict             # {"type": "balanced|imbalanced", "ratio": "...", "strategy": "class_weight|none"}
    high_cardinality_cols: list[str]
    suggested_model: str
    mlflow_experiment_name: str
    feature_table_name: str
    scores_table_name: str


class DryRunPlan(TypedDict):
    tables_to_read: list[str]
    tables_to_write: list[str]
    grant_statements: list[str]
    estimated_dbu_cost: str         # e.g. "$12-18 total for feature + training + inference"
    estimated_run_time: str         # e.g. "8-12 minutes"
    plain_english_summary: str      # 3-5 sentence plain English for non-technical stakeholders
    # Extended detail fields
    feature_columns: list[str]      # expected feature columns in the output feature table
    join_keys: list[str]            # columns used to join feature tables together
    estimated_row_count: str        # estimated rows in the feature table (e.g. "~50K rows")
    target_column_detail: str       # what is being predicted and how it is defined


class RiskScorecardItem(TypedDict):
    check: str
    status: str     # "pass" | "warn" | "fail"
    detail: str


class RiskScorecard(TypedDict):
    items: list[RiskScorecardItem]
    overall: str    # "ready" | "review_needed" | "blocked"
    summary: str


class GeneratedArtifact(TypedDict):
    filename: str
    filepath: str
    content: str


class AgentState(TypedDict):
    # ── Phase 1 ──────────────────────────────────────────────────────────────
    # workspace: plain dict — reconstruct with WorkspaceContext(**state["workspace"])
    workspace: dict
    # tables: list of plain dicts — keys: full_name, name, table_type, comment, row_count, columns
    tables: list[dict]
    estate_summary: str
    opportunities: list[MLOpportunity]
    approved_opportunity: Optional[MLOpportunity]

    # ── Trust layer ───────────────────────────────────────────────────────────
    dry_run_plan: Optional[DryRunPlan]
    risk_scorecard: Optional[RiskScorecard]
    exec_summary: str

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    feature_plan: Optional[FeaturePlan]
    generated_artifacts: list[GeneratedArtifact]
    bundle_written: bool

    # ── Error tracking ────────────────────────────────────────────────────────
    error: Optional[str]
