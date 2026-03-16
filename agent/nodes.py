"""
LangGraph nodes for Phase 1: Discovery + Recommendation.

Node flow:
  discover_catalog → analyze_estate → rank_opportunities → human_checkpoint
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from config.settings import get_settings
from tools.uc_reader import UCReader
from agent.state import AgentState, MLOpportunity

logger = logging.getLogger(__name__)


def _get_llm():
    """Lazy import to avoid import errors if databricks-langchain not yet installed."""
    from databricks_langchain import ChatDatabricks
    settings = get_settings()
    return ChatDatabricks(
        endpoint=settings.llm_endpoint_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )


# ── Node 1: Discover Unity Catalog ──────────────────────────────────────────

def discover_catalog(state: AgentState) -> AgentState:
    """
    Read all table metadata from UC. No raw data is read.
    """
    logger.info("Discovering catalog: %s.%s", state["catalog"], state["schema"])
    try:
        reader = UCReader()
        tables = reader.list_tables(
            catalog=state["catalog"],
            schema=state["schema"],
        )
        estate_summary = reader.build_estate_summary(tables)
        logger.info("Discovered %d tables", len(tables))
        return {
            **state,
            "tables": tables,
            "estate_summary": estate_summary,
            "error": None,
        }
    except Exception as e:
        logger.error("Discovery failed: %s", e)
        return {**state, "tables": [], "estate_summary": "", "error": str(e)}


# ── Node 2: Analyze Data Estate ──────────────────────────────────────────────

def analyze_estate(state: AgentState) -> AgentState:
    """
    Ask Claude to enrich the estate summary with inferred relationships
    and data quality signals. Prepares context for opportunity ranking.
    """
    if state.get("error") or not state.get("tables"):
        return state

    logger.info("Analyzing data estate with LLM")
    llm = _get_llm()

    system = SystemMessage(content="""You are an expert ML engineer specializing in Databricks and Unity Catalog.
You are analyzing a customer's data estate to identify ML opportunities.
Respond concisely. Focus only on what's useful for ML — don't describe obvious things.""")

    prompt = f"""Analyze this Unity Catalog data estate and provide:
1. Likely entity relationships between tables (FK/PK patterns from column names)
2. Data quality signals (columns that suggest labeled data, timestamps for time-series, customer IDs)
3. Any data gaps or limitations relevant for ML

DATA ESTATE:
{state['estate_summary']}

Be concise. 300 words max."""

    response = llm.invoke([system, HumanMessage(content=prompt)])
    enriched_summary = state["estate_summary"] + "\n\nANALYSIS:\n" + response.content

    return {**state, "estate_summary": enriched_summary}


# ── Node 3: Rank ML Opportunities ────────────────────────────────────────────

_RANK_SYSTEM = """You are a senior ML engineer at a Databricks consultancy.
Your job is to identify the top 3 most valuable, most achievable ML use cases
from a customer's Unity Catalog data estate.

Rules:
- Only recommend use cases directly supported by the observed tables/columns
- Estimate AUC/RMSE ranges conservatively based on data signals
- Rank by: business value × data readiness × ML complexity (prefer high value, high readiness, lower complexity)
- Output MUST be valid JSON only — no prose, no markdown fences"""

_RANK_PROMPT = """Given this data estate, identify the top 3 ML opportunities.

{estate_summary}

Return a JSON array of exactly 3 objects with this schema:
[
  {{
    "rank": 1,
    "use_case": "string",
    "target_table": "catalog.schema.table",
    "target_column": "column_to_predict",
    "feature_tables": ["catalog.schema.table1", ...],
    "ml_type": "classification|regression|clustering|forecasting",
    "estimated_auc_range": "e.g. 75-83%",
    "business_value": "1-2 sentence impact statement",
    "complexity": "low|medium|high",
    "rationale": "why this data supports this use case"
  }}
]"""


def rank_opportunities(state: AgentState) -> AgentState:
    """
    Ask Claude to rank the top 3 ML opportunities from the estate analysis.
    """
    if state.get("error") or not state.get("tables"):
        return state

    logger.info("Ranking ML opportunities with LLM")
    llm = _get_llm()

    prompt = _RANK_PROMPT.format(estate_summary=state["estate_summary"])
    response = llm.invoke([
        SystemMessage(content=_RANK_SYSTEM),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()
    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        opportunities: list[MLOpportunity] = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse opportunities JSON: %s\nRaw: %s", e, raw)
        return {**state, "error": f"LLM returned invalid JSON: {e}"}

    logger.info("Ranked %d opportunities", len(opportunities))
    return {**state, "opportunities": opportunities}


# ── Node 4: Human Checkpoint ─────────────────────────────────────────────────

def human_checkpoint(state: AgentState) -> AgentState:
    """
    Pause the graph and surface opportunities to the user.
    Resumes when the user approves one opportunity (by rank index).
    The interrupt payload is what the UI/API will display to the user.
    """
    if state.get("error"):
        return state

    # LangGraph interrupt — execution pauses here until graph.invoke() is called
    # again with a Command(resume=<user_response>)
    user_response: dict = interrupt({
        "message": "Review the ML opportunities and approve one to proceed.",
        "opportunities": state["opportunities"],
    })

    # user_response expected: {"selected_rank": 1}
    selected_rank = user_response.get("selected_rank", 1)
    approved = next(
        (o for o in state["opportunities"] if o["rank"] == selected_rank),
        state["opportunities"][0] if state["opportunities"] else None,
    )

    logger.info("Human approved opportunity: %s", approved.get("use_case") if approved else "none")
    return {**state, "approved_opportunity": approved}
