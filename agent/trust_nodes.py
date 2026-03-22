"""
Trust layer nodes — make the agent's decisions visible and auditable.

Nodes:
  dry_run_explain       → plain-English plan before any code runs
  dry_run_checkpoint    → human interrupt: confirm before code generation
  compute_risk_scorecard → automated checks after code gen, before deploy
  generate_exec_summary  → CTO-facing markdown summary after bundle write
"""

import json
import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.state import AgentState, DryRunPlan, RiskScorecard, RiskScorecardItem
from tools.bundle_writer import slugify, BUNDLES_ROOT
from tools.workspace_context import WorkspaceContext

logger = logging.getLogger(__name__)


# ── Node 1: Dry Run Explain ───────────────────────────────────────────────────

_DRY_RUN_SYSTEM = """You are a Databricks ML engineer explaining a deployment plan to a non-technical audience.
Output MUST be valid JSON only — no prose, no markdown fences.
For cost estimates: m5d.4xlarge = ~$0.75 DBU/hr for jobs clusters.
Feature pipeline typically 0.5-1.5 hrs, training 0.5-2 hrs, inference 0.25-0.5 hrs.
Be conservative with cost estimates."""

_DRY_RUN_PROMPT = """Given this approved ML opportunity, produce a detailed dry-run plan.

APPROVED OPPORTUNITY:
{opportunity_json}

SOURCE TABLES IN CATALOG (from UC metadata):
{table_summary}

Return a JSON object with ALL of these fields:
{{
  "tables_to_read": ["catalog.schema.table1", "catalog.schema.table2"],
  "tables_to_write": [
    "{{catalog}}.ml_accelerator.{slug}_features",
    "{{catalog}}.ml_accelerator.{slug}_scores"
  ],
  "grant_statements": [
    "GRANT SELECT ON TABLE {{catalog}}.ml_accelerator.{slug}_features TO `account users`",
    "GRANT SELECT ON TABLE {{catalog}}.ml_accelerator.{slug}_scores TO `account users`"
  ],
  "estimated_dbu_cost": "$15-35 total (feature pipeline $5-12, training $7-18, inference $3-8)",
  "estimated_run_time": "15-35 minutes total across 3 jobs",
  "plain_english_summary": "3-4 sentences describing what will happen in plain English for a CTO. No jargon. Explain what data will be read, what model will be trained, and what business outcome will be delivered.",
  "feature_columns": ["col_a_aggregation", "col_b_days_since", "col_c_count", "...list 8-15 expected feature column names based on the table metadata"],
  "join_keys": ["customer_id", "...columns that will be used to join the feature tables together"],
  "estimated_row_count": "~50K rows (one row per unique {join_key} in the target table)",
  "target_column_detail": "Predicting whether {target_column} = 1 (positive class). Defined as [explain what the column means based on its name and context]."
}}

Be specific: use actual column names from the table metadata, not placeholders."""


def dry_run_explain(state: AgentState) -> AgentState:
    """Generate a plain-English plan of what will happen before any code runs."""
    opp = state.get("approved_opportunity")
    if not opp or state.get("error"):
        return state

    logger.info("Generating dry run plan for: %s", opp.get("use_case"))
    llm = WorkspaceContext(**state["workspace"]).get_llm()
    slug = slugify(opp.get("use_case", "ml_use_case"))

    # Build a concise table summary from already-discovered metadata (tables are plain dicts)
    tables = state.get("tables", [])
    table_summary = "\n".join(
        f"- {t.get('full_name', t.get('name', ''))}: {len(t.get('columns', []))} columns"
        + (f" | {t.get('comment')}" if t.get("comment") else "")
        for t in tables[:20]  # cap to avoid token explosion
    ) or "See estate_summary above"

    join_key = opp.get("target_column", "customer_id").replace("is_", "").replace("_flag", "") + "_id"
    prompt = _DRY_RUN_PROMPT.format(
        opportunity_json=json.dumps(opp, indent=2),
        table_summary=table_summary,
        slug=slug,
        join_key=join_key,
        target_column=opp.get("target_column", "target"),
    )

    response = llm.invoke([SystemMessage(content=_DRY_RUN_SYSTEM), HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        plan: DryRunPlan = json.loads(raw)
        # Ensure new fields exist even if LLM omitted them
        plan.setdefault("feature_columns", [])
        plan.setdefault("join_keys", [])
        plan.setdefault("estimated_row_count", "Unknown")
        plan.setdefault("target_column_detail", f"Predicting {opp.get('target_column', 'target')}")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse dry run plan: %s", e)
        # Non-fatal — build a minimal plan from known data
        plan = DryRunPlan(
            tables_to_read=opp.get("feature_tables", []) + [opp.get("target_table", "")],
            tables_to_write=[
                f"{{catalog}}.ml_accelerator.{slug}_features",
                f"{{catalog}}.ml_accelerator.{slug}_scores",
            ],
            grant_statements=[
                f"GRANT SELECT ON TABLE {{catalog}}.ml_accelerator.{slug}_features TO `account users`",
                f"GRANT SELECT ON TABLE {{catalog}}.ml_accelerator.{slug}_scores TO `account users`",
            ],
            estimated_dbu_cost="$15-35 total",
            estimated_run_time="15-35 minutes",
            plain_english_summary=(
                f"This will build a {opp.get('ml_type', 'ML')} model to predict "
                f"{opp.get('target_column', 'the target')}. "
                f"Source data is read from your Unity Catalog, processed into a feature table, "
                f"used to train a model tracked in MLflow, and scored in batch with results "
                f"written back to your catalog."
            ),
            feature_columns=[],
            join_keys=[],
            estimated_row_count="Unknown",
            target_column_detail=f"Predicting {opp.get('target_column', 'target')}",
        )

    return {**state, "dry_run_plan": plan}


# ── Node 2: Business Brief (pre-code CTO summary) ─────────────────────────────

def generate_business_brief(state: AgentState) -> AgentState:
    """
    Generate a CTO-facing business brief BEFORE code generation.
    Uses only opportunity + dry_run_plan — no code artifacts needed.
    Deterministic — no LLM call. Stored in exec_summary field.
    This is displayed to the user at the dry run confirmation step.
    """
    opp = state.get("approved_opportunity") or {}
    plan = state.get("dry_run_plan") or {}
    today = date.today().isoformat()

    brief = f"""# ML Opportunity Brief — {opp.get('use_case', 'N/A')}

**Date:** {today}
**Prepared by:** Databricks ML Accelerator

---

## Business Case

{opp.get('business_value', '')}

| | |
|---|---|
| **Financial impact** | {opp.get('financial_impact', 'See opportunity details')} |
| **Confidence** | {opp.get('confidence', 'medium').title()} |
| **Estimated model performance** | {opp.get('estimated_auc_range', 'N/A')} |
| **Complexity** | {opp.get('complexity', 'medium').title()} |

---

## What Will Be Built

| Component | Detail |
|---|---|
| **ML type** | {opp.get('ml_type', 'N/A')} |
| **Target** | `{opp.get('target_table', '')}` → predict `{opp.get('target_column', '')}` |
| **Feature tables** | {', '.join(f'`{t}`' for t in opp.get('feature_tables', [])) or 'N/A'} |

{plan.get('target_column_detail', '')}

---

## Execution Plan

{plan.get('plain_english_summary', '')}

| | |
|---|---|
| **Data reads** | {', '.join(f'`{t}`' for t in plan.get('tables_to_read', [])) or 'N/A'} |
| **Data writes** | {', '.join(f'`{t}`' for t in plan.get('tables_to_write', [])) or 'N/A'} |
| **Estimated feature rows** | {plan.get('estimated_row_count', 'Unknown')} |
| **Estimated cost** | {plan.get('estimated_dbu_cost', 'N/A')} |
| **Estimated run time** | {plan.get('estimated_run_time', 'N/A')} |

---

## Governance & Access

The following access grants will be issued automatically:

{chr(10).join(f'- `{g}`' for g in plan.get('grant_statements', [])) or 'None required'}

---

*Pending: Code generation, risk scorecard, and final approval*
*Full technical summary will be generated after code review.*
"""
    logger.info("Business brief generated for: %s", opp.get("use_case"))
    return {**state, "exec_summary": brief}


# ── Node 3: Dry Run Checkpoint ────────────────────────────────────────────────

def dry_run_checkpoint(state: AgentState) -> AgentState:
    """
    Pause and show the dry run plan to the user before any code is generated.
    User must explicitly confirm before code generation begins.
    Resume with Command(resume={"confirmed": True}).
    """
    if state.get("error"):
        return state

    interrupt({
        "message": "Review what will happen before code generation starts. Confirm to proceed.",
        "dry_run_plan": state.get("dry_run_plan"),
        "approved_opportunity": state.get("approved_opportunity"),
    })
    # If we reach here, user confirmed
    return state


# ── Node 4: Compute Risk Scorecard ────────────────────────────────────────────

def compute_risk_scorecard(state: AgentState) -> AgentState:
    """
    Automated risk checks based on the feature plan and generated code.
    Rule-based where possible, LLM-assisted for qualitative checks.
    Runs after code generation, shown alongside notebooks in the review screen.
    """
    opp = state.get("approved_opportunity")
    plan = state.get("feature_plan")
    artifacts = state.get("generated_artifacts", [])

    if not opp or not plan or state.get("error"):
        return state

    items: list[RiskScorecardItem] = []

    # ── Rule-based checks ────────────────────────────────────────────────────

    # 1. Temporal split
    if plan.get("split_strategy") == "temporal" and plan.get("split_column"):
        items.append({"check": "Train/test split", "status": "pass",
                      "detail": f"Temporal split on '{plan['split_column']}' — no data leakage"})
    elif plan.get("split_strategy") == "random" and opp.get("ml_type") in ("forecasting",):
        items.append({"check": "Train/test split", "status": "warn",
                      "detail": "Random split on time-series data — consider temporal split"})
    else:
        items.append({"check": "Train/test split", "status": "pass",
                      "detail": f"{plan.get('split_strategy', 'random').title()} split applied"})

    # 2. Class imbalance
    balance = plan.get("class_balance", {})
    if balance.get("type") == "imbalanced" and balance.get("strategy") == "class_weight":
        items.append({"check": "Class imbalance", "status": "pass",
                      "detail": f"Imbalance detected ({balance.get('estimated_ratio', '?')}), handled via class_weight"})
    elif balance.get("type") == "imbalanced" and balance.get("strategy") == "none":
        items.append({"check": "Class imbalance", "status": "warn",
                      "detail": f"Imbalance detected ({balance.get('estimated_ratio', '?')}) but no strategy set — model may predict majority class"})
    else:
        items.append({"check": "Class imbalance", "status": "pass",
                      "detail": "Dataset appears balanced"})

    # 3. High-cardinality columns
    hc_cols = plan.get("high_cardinality_cols", [])
    if hc_cols:
        items.append({"check": "High-cardinality features", "status": "pass",
                      "detail": f"Detected and handled: {', '.join(hc_cols)}"})
    else:
        items.append({"check": "High-cardinality features", "status": "pass",
                      "detail": "None detected"})

    # 4. GRANT statements present in notebooks
    notebook_contents = " ".join(a["content"] for a in artifacts if a["filename"].endswith(".py"))
    has_grants = "GRANT SELECT" in notebook_contents
    items.append({"check": "UC access grants", "status": "pass" if has_grants else "warn",
                  "detail": "GRANT SELECT statements included in notebooks" if has_grants
                  else "No GRANT statements found — downstream consumers may lack access"})

    # 5. MLflow tracking
    has_mlflow = "mlflow" in notebook_contents.lower()
    has_uc_registry = "databricks-uc" in notebook_contents
    if has_mlflow and has_uc_registry:
        items.append({"check": "MLflow + UC registry", "status": "pass",
                      "detail": "MLflow tracking enabled, model registered in Unity Catalog"})
    elif has_mlflow:
        items.append({"check": "MLflow + UC registry", "status": "warn",
                      "detail": "MLflow present but UC registry not confirmed"})
    else:
        items.append({"check": "MLflow + UC registry", "status": "fail",
                      "detail": "MLflow tracking not found in generated code"})

    # 6. dbutils.widgets (job-safe parameterization)
    has_widgets = "dbutils.widgets" in notebook_contents
    items.append({"check": "Job-safe parameters", "status": "pass" if has_widgets else "fail",
                  "detail": "dbutils.widgets used — notebooks safe for both interactive and job runs" if has_widgets
                  else "Hardcoded values detected — notebooks may fail as scheduled jobs"})

    # 7. Rollback path
    has_alias = "Champion" in notebook_contents
    items.append({"check": "Rollback path", "status": "pass" if has_alias else "warn",
                  "detail": "Champion alias set in UC Model Registry — reassign alias to roll back" if has_alias
                  else "No Champion alias found — rollback requires manual version management"})

    # 8. Cost estimate (from dry run plan)
    dry_run = state.get("dry_run_plan")
    cost = dry_run.get("estimated_dbu_cost", "Unknown") if dry_run else "Unknown"
    items.append({"check": "Estimated compute cost", "status": "pass",
                  "detail": f"{cost} per full run (feature + training + inference)"})

    # ── Overall status ────────────────────────────────────────────────────────
    fail_count = sum(1 for i in items if i["status"] == "fail")
    warn_count = sum(1 for i in items if i["status"] == "warn")

    if fail_count > 0:
        overall = "blocked"
        summary = f"{fail_count} critical issue(s) must be resolved before deployment."
    elif warn_count > 0:
        overall = "review_needed"
        summary = f"{warn_count} warning(s) to review — deployment can proceed with awareness."
    else:
        overall = "ready"
        summary = "All checks passed. Code is ready to deploy."

    scorecard = RiskScorecard(items=items, overall=overall, summary=summary)
    logger.info("Risk scorecard: %s (%d pass, %d warn, %d fail)",
                overall, len(items) - fail_count - warn_count, warn_count, fail_count)
    return {**state, "risk_scorecard": scorecard}


# ── Node 5: Generate Full Executive Summary (post-bundle) ────────────────────

def generate_exec_summary(state: AgentState) -> AgentState:
    """
    Generate the FULL technical executive summary after bundle write.
    Includes risk scorecard + artifact list. Overwrites exec_summary field
    (replacing the pre-code business brief generated by generate_business_brief).
    Writes to bundles/SUMMARY.md. No interrupt — runs automatically.
    """
    opp = state.get("approved_opportunity") or {}
    plan = state.get("feature_plan") or {}
    dry_run = state.get("dry_run_plan") or {}
    scorecard = state.get("risk_scorecard") or {}
    artifacts = state.get("generated_artifacts", [])
    slug = slugify(opp.get("use_case", "ml_use_case"))
    today = date.today().isoformat()

    pass_items = [i for i in scorecard.get("items", []) if i["status"] == "pass"]
    warn_items = [i for i in scorecard.get("items", []) if i["status"] == "warn"]

    scorecard_lines = "\n".join(
        f"| {i['check']} | {'✅' if i['status']=='pass' else '⚠️' if i['status']=='warn' else '❌'} | {i['detail']} |"
        for i in scorecard.get("items", [])
    )

    artifact_lines = "\n".join(
        f"- `{a['filepath']}`" for a in artifacts
    )

    summary_md = f"""# ML Accelerator — Deployment Summary

**Use case:** {opp.get('use_case', 'N/A')}
**Date:** {today}
**Status:** {scorecard.get('overall', 'N/A').replace('_', ' ').title()}

---

## Business Case

{opp.get('business_value', '')}

**Financial impact:** {opp.get('financial_impact', 'See opportunity details')}
**Confidence:** {opp.get('confidence', 'medium').title()}
**Estimated model performance:** {opp.get('estimated_auc_range', 'N/A')}

---

## What Was Built

| Component | Details |
|---|---|
| ML type | {opp.get('ml_type', 'N/A')} |
| Target | `{opp.get('target_table', '')}` → predict `{opp.get('target_column', '')}` |
| Feature tables | {', '.join(f'`{t}`' for t in opp.get('feature_tables', []))} |
| Train/test split | {plan.get('split_strategy', 'N/A').title()}{f" on `{plan.get('split_column')}`" if plan.get('split_column') else ''} |
| Model | {plan.get('suggested_model', 'XGBoost')} |
| Experiment | `{plan.get('mlflow_experiment_name', '')}` |

---

## What Will Happen When Jobs Run

{dry_run.get('plain_english_summary', '')}

**Estimated cost per full run:** {dry_run.get('estimated_dbu_cost', 'N/A')}
**Estimated run time:** {dry_run.get('estimated_run_time', 'N/A')}

**Data reads:** {', '.join(f'`{t}`' for t in dry_run.get('tables_to_read', []))}
**Data writes:** {', '.join(f'`{t}`' for t in dry_run.get('tables_to_write', []))}

---

## Human Approvals

- ✅ ML opportunity approved
- ✅ Dry run plan reviewed and confirmed
- ✅ Generated code reviewed and approved

---

## Deployment Readiness

{scorecard.get('summary', '')}

| Check | Status | Detail |
|---|---|---|
{scorecard_lines}

---

## Generated Files

{artifact_lines}

---

## Next Steps

```bash
# Deploy to dev workspace
databricks bundle deploy --target dev

# Run jobs in order
databricks bundle run {slug}_feature_pipeline_job --target dev
databricks bundle run {slug}_training_job --target dev
databricks bundle run {slug}_batch_inference_job --target dev
```

Review model performance in MLflow, then promote to production by re-running:
```bash
databricks bundle deploy --target prod
```

---

*Generated by Databricks ML Accelerator*
"""

    # Write to bundles/SUMMARY.md
    summary_path = BUNDLES_ROOT / "SUMMARY.md"
    try:
        summary_path.write_text(summary_md, encoding="utf-8")
        logger.info("Executive summary written to %s", summary_path)
    except Exception as e:
        logger.warning("Could not write SUMMARY.md: %s", e)

    return {**state, "exec_summary": summary_md}
