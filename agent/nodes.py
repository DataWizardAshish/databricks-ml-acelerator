"""
LangGraph nodes for Phase 1: Discovery + Recommendation.

Each node receives WorkspaceContext from state — no global settings used.

Node flow:
  discover_catalog → analyze_estate → rank_opportunities → human_checkpoint
"""

import dataclasses
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from tools.uc_reader import UCReader
from tools.workspace_context import WorkspaceContext
from agent.state import AgentState, MLOpportunity

logger = logging.getLogger(__name__)


# ── Node 1: Discover Unity Catalog ──────────────────────────────────────────

def discover_catalog(state: AgentState) -> AgentState:
    """Read all table metadata from UC. No raw data, no compute."""
    ctx = WorkspaceContext(**state["workspace"])
    logger.info("Discovering %s.%s", ctx.catalog, ctx.schema)
    try:
        reader = UCReader(ctx)
        tables = reader.list_tables()
        estate_summary = reader.build_estate_summary(tables)
        # Convert TableInfo dataclasses to plain dicts for JSON-serializable state
        tables_dicts = [dataclasses.asdict(t) for t in tables]
        logger.info("Discovered %d tables", len(tables_dicts))
        return {**state, "tables": tables_dicts, "estate_summary": estate_summary, "error": None}
    except Exception as e:
        logger.error("Discovery failed: %s", e)
        return {**state, "tables": [], "estate_summary": "", "error": str(e)}


# ── Node 2: Analyze Data Estate ──────────────────────────────────────────────

def analyze_estate(state: AgentState) -> AgentState:
    """Ask Claude to enrich the estate summary with inferred relationships and data quality signals."""
    if state.get("error") or not state.get("tables"):
        return state

    logger.info("Analyzing data estate")
    llm = WorkspaceContext(**state["workspace"]).get_llm()

    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert ML engineer specializing in Databricks and Unity Catalog. "
            "Analyze data estates to identify ML opportunities. Be concise, 300 words max."
        )),
        HumanMessage(content=(
            f"Analyze this Unity Catalog data estate and provide:\n"
            f"1. Likely entity relationships between tables (FK/PK patterns from column names)\n"
            f"2. Data quality signals (labeled data, timestamps for time-series, entity IDs)\n"
            f"3. Any data gaps relevant for ML\n\n"
            f"DATA ESTATE:\n{state['estate_summary']}"
        )),
    ])
    return {**state, "estate_summary": state["estate_summary"] + "\n\nANALYSIS:\n" + response.content}


# ── Node 3: Rank ML Opportunities ────────────────────────────────────────────

_RANK_SYSTEM = """You are a senior ML engineer and product strategist at a Databricks consultancy.
Identify the top 3 most valuable, most achievable ML use cases from the customer's Unity Catalog data estate.

Rules:
- Only recommend use cases directly supported by the observed tables/columns
- Estimate AUC/RMSE ranges conservatively based on data signals
- Rank by: business value × data readiness × inverse complexity
- For financial_impact: translate the ML outcome into a business dollar range using realistic industry benchmarks.
  Examples: churn reduction → retained ARR, demand forecasting → inventory cost reduction,
  fraud detection → prevented losses, propensity → conversion uplift on revenue
- Be conservative. Use ranges. Never invent data not present in the estate.
- Output MUST be valid JSON only — no prose, no markdown fences"""

_RANK_PROMPT = """Given this data estate, identify the top 3 ML opportunities.

{estate_summary}

Return a JSON array of exactly 3 objects:
[
  {{
    "rank": 1,
    "use_case": "string",
    "target_table": "catalog.schema.table",
    "target_column": "column_to_predict",
    "feature_tables": ["catalog.schema.table1"],
    "ml_type": "classification|regression|clustering|forecasting",
    "estimated_auc_range": "e.g. 75-83%",
    "business_value": "1-2 sentence operational impact statement",
    "financial_impact": "Estimated $X-Y [saved/retained/recovered] annually based on [specific mechanism]. Confidence: medium.",
    "confidence": "high|medium|low",
    "complexity": "low|medium|high",
    "rationale": "why this data supports this use case"
  }}
]"""


def rank_opportunities(state: AgentState) -> AgentState:
    """Ask Claude to rank the top 3 ML opportunities from the estate analysis."""
    if state.get("error") or not state.get("tables"):
        return state

    logger.info("Ranking ML opportunities")
    llm = WorkspaceContext(**state["workspace"]).get_llm()

    response = llm.invoke([
        SystemMessage(content=_RANK_SYSTEM),
        HumanMessage(content=_RANK_PROMPT.format(estate_summary=state["estate_summary"])),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        opportunities: list[MLOpportunity] = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse opportunities JSON: %s", e)
        return {**state, "error": f"LLM returned invalid JSON: {e}"}

    logger.info("Ranked %d opportunities", len(opportunities))
    return {**state, "opportunities": opportunities}


# ── Node 4: Human Checkpoint ─────────────────────────────────────────────────

def human_checkpoint(state: AgentState) -> AgentState:
    """
    Pause the graph — surfaces opportunities to the user.
    Resumes when the API receives POST /runs/{id}/approve with selected_rank.
    """
    if state.get("error"):
        return state

    user_response: dict = interrupt({
        "message": "Review the ML opportunities and approve one to proceed.",
        "opportunities": state["opportunities"],
    })

    selected_rank = user_response.get("selected_rank", 1)
    approved = next(
        (o for o in state["opportunities"] if o["rank"] == selected_rank),
        state["opportunities"][0] if state["opportunities"] else None,
    )

    logger.info("Human approved: %s", approved.get("use_case") if approved else "none")
    return {**state, "approved_opportunity": approved}
