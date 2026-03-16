"""
LangGraph state for the ML Accelerator agent pipeline.
"""

from typing import Optional
from typing_extensions import TypedDict

from tools.uc_reader import TableInfo


class MLOpportunity(TypedDict):
    rank: int
    use_case: str                # e.g. "Customer Churn Prediction"
    target_table: str            # full UC table name
    target_column: str           # predicted column
    feature_tables: list[str]    # supporting tables
    ml_type: str                 # classification | regression | clustering | forecasting
    estimated_auc_range: str     # e.g. "75–82%"
    business_value: str          # 1–2 sentence business impact
    complexity: str              # low | medium | high
    rationale: str               # why this opportunity exists in this data


class AgentState(TypedDict):
    # Inputs
    catalog: str
    schema: str

    # Discovery results
    tables: list[TableInfo]
    estate_summary: str

    # Recommendation results
    opportunities: list[MLOpportunity]

    # Human checkpoint result
    approved_opportunity: Optional[MLOpportunity]

    # Error tracking
    error: Optional[str]
