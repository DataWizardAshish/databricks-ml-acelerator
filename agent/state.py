"""
LangGraph state for the ML Accelerator agent pipeline.
WorkspaceContext is carried through every node so all tools use the
same per-request credentials. Never read from global settings inside nodes.
"""

from typing import Optional, TYPE_CHECKING
from typing_extensions import TypedDict

if TYPE_CHECKING:
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


class AgentState(TypedDict):
    # Per-request workspace connection (host, token, catalog, schema)
    workspace: "WorkspaceContext"

    # Discovery results
    tables: list[TableInfo]
    estate_summary: str

    # Recommendation results
    opportunities: list[MLOpportunity]

    # Human checkpoint result
    approved_opportunity: Optional[MLOpportunity]

    # Error tracking
    error: Optional[str]
