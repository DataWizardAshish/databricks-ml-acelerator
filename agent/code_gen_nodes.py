"""
Phase 2 LangGraph nodes: Feature Planning → Code Generation → Human Review → Bundle Write.

Nodes:
  plan_features          → structured JSON plan (transforms, split, imbalance, model)
  generate_code          → 3 Databricks notebooks + 3 job YAMLs
  human_checkpoint_code  → LangGraph interrupt: user reviews code before writing
  write_bundle           → writes artifacts to bundles/ directory
"""

import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.state import AgentState, FeaturePlan
from tools.bundle_writer import slugify, prepare_artifacts_from_generation, write_artifacts

logger = logging.getLogger(__name__)


# ── Node 1: Plan Features ────────────────────────────────────────────────────

_PLAN_SYSTEM = """You are a senior ML engineer. Given an approved ML opportunity and its source table metadata,
produce a structured feature engineering and training plan.

Output MUST be valid JSON only — no prose, no markdown fences.
Be conservative: prefer simple, safe transforms over complex ones.
Always flag high-cardinality columns (>500 unique values likely) for hash encoding or dropping."""

_PLAN_PROMPT = """Approved ML opportunity:
{opportunity_json}

Source table metadata (columns and types):
{estate_summary}

Produce a JSON object with this exact schema:
{{
  "feature_decisions": [
    {{"column": "col_name", "transform": "keep_numeric|log_transform|extract_days_since|one_hot|drop|hash_encode|binary_encode", "note": "reason"}}
  ],
  "target_encoding": "binary_0_1 | keep_as_is | label_encode",
  "split_strategy": "temporal | random",
  "split_column": "date_column_name or null",
  "class_balance": {{
    "type": "balanced | imbalanced",
    "estimated_ratio": "e.g. 90:10",
    "strategy": "class_weight | none"
  }},
  "high_cardinality_cols": ["col1", "col2"],
  "suggested_model": "XGBClassifier | XGBRegressor",
  "mlflow_experiment_name": "/ml_accelerator/{use_case_slug}",
  "feature_table_name": "{use_case_slug}_features",
  "scores_table_name": "{use_case_slug}_scores"
}}

Rules:
- Use temporal split if any date/timestamp column is present in feature tables
- Mark class imbalance if target column name suggests binary flag (churn, is_*, flag_*, active)
- Drop columns that are primary keys (id, uuid, *_id) from features — keep only as join keys
- hash_encode for high cardinality (>500 likely unique), one_hot for low cardinality (<20)"""


def plan_features(state: AgentState) -> AgentState:
    """Ask Claude to produce a structured feature engineering plan for the approved opportunity."""
    opp = state.get("approved_opportunity")
    if not opp or state.get("error"):
        return state

    logger.info("Planning features for: %s", opp.get("use_case"))
    llm = state["workspace"].get_llm()
    slug = slugify(opp.get("use_case", "ml_use_case"))

    prompt = _PLAN_PROMPT.format(
        opportunity_json=json.dumps(opp, indent=2),
        estate_summary=state.get("estate_summary", ""),
        use_case_slug=slug,
    )

    response = llm.invoke([SystemMessage(content=_PLAN_SYSTEM), HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        plan: FeaturePlan = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse feature plan: %s", e)
        return {**state, "error": f"Feature planning returned invalid JSON: {e}"}

    logger.info("Feature plan ready. Model: %s, Split: %s", plan.get("suggested_model"), plan.get("split_strategy"))
    return {**state, "feature_plan": plan}


# ── Node 2: Generate Code ────────────────────────────────────────────────────

_NB_SYSTEM = """You are an expert Databricks engineer. Generate production-ready Databricks Python notebooks.

STRICT RULES — violating any of these will break the notebook:
1. Start every notebook with exactly: # Databricks notebook source
2. Separate every cell with exactly: # COMMAND ----------
3. Use spark (already available) and dbutils (already available) — never import them
4. ALL table names come from dbutils.widgets — never hardcode catalog/schema/table names
5. Use spark.table(f"{{catalog}}.{{schema}}.{{table}}") to read UC tables
6. Use df.write.format("delta").mode("overwrite").saveAsTable(f"{{catalog}}.{{schema}}.{{table}}") to write
7. For MLflow UC registry: mlflow.set_registry_uri("databricks-uc") BEFORE any mlflow calls
8. Register model as: mlflow.register_model(f"runs:/{{run_id}}/model", f"{{catalog}}.{{schema}}.{{model_name}}")
9. Set Champion alias using MlflowClient().set_registered_model_alias(...)
10. Load model in inference as: mlflow.pyfunc.load_model(f"models:/{{catalog}}.{{schema}}.{{model_name}}@Champion")
11. GRANT statement must use spark.sql(), NOT %sql magic
12. Every widget must have a default value in dbutils.widgets.text("name", "default")
13. Never use pandas on large Spark DataFrames — convert with .toPandas() only after filtering to model input size
14. Output ONLY the notebook code — no explanation, no markdown"""


def _feature_engineering_prompt(opp: dict, plan: FeaturePlan, catalog: str, output_schema: str) -> str:
    return f"""Generate a complete Databricks feature engineering notebook.

APPROVED OPPORTUNITY:
{json.dumps(opp, indent=2)}

FEATURE PLAN:
{json.dumps(plan, indent=2)}

NOTEBOOK REQUIREMENTS:
- Widgets: catalog (default "{catalog}"), discovery_schema (default "{opp.get('target_table', '').split('.')[1] if '.' in opp.get('target_table','') else 'schema'}"), output_schema (default "{output_schema}"), run_date (default today)
- Read source tables using spark.table() with full catalog.schema.table path from widgets
- Apply EVERY transform in feature_decisions exactly as specified
- Drop all high_cardinality_cols: {plan.get('high_cardinality_cols', [])} OR hash_encode them per plan
- Apply target encoding: {plan.get('target_encoding')}
- Drop rows where target column is null
- Add a run_date partition column
- Write feature table to: {{catalog}}.{output_schema}.{plan.get('feature_table_name')}
- After writing, emit: spark.sql(f"GRANT SELECT ON TABLE {{catalog}}.{output_schema}.{plan.get('feature_table_name')} TO `account users`")
- Print row count before and after each major step"""


def _training_prompt(opp: dict, plan: FeaturePlan, catalog: str, output_schema: str) -> str:
    split_instructions = (
        f"Sort by {plan.get('split_column')} ascending. Use the last 20% of rows (by date) as test set."
        if plan.get("split_strategy") == "temporal"
        else "Use train_test_split(test_size=0.2, random_state=42, stratify=y) for classification."
    )
    class_weight = (
        'scale_pos_weight=<compute from y_train: (y_train==0).sum()/(y_train==1).sum()>'
        if plan.get("class_balance", {}).get("strategy") == "class_weight"
        else "None"
    )
    return f"""Generate a complete Databricks training notebook using XGBoost + sklearn + MLflow.

APPROVED OPPORTUNITY:
{json.dumps(opp, indent=2)}

FEATURE PLAN:
{json.dumps(plan, indent=2)}

NOTEBOOK REQUIREMENTS:
- Widgets: catalog, output_schema (default "{output_schema}"), model_name (default "{plan.get('feature_table_name','model').replace('_features','_model')}"), experiment_name (default "{plan.get('mlflow_experiment_name')}")
- Read feature table: {{catalog}}.{output_schema}.{plan.get('feature_table_name')}
- Drop run_date column before training (it's a partition key, not a feature)
- Train/test split: {split_instructions}
- Class imbalance strategy: {class_weight}
- Use sklearn Pipeline: [SimpleImputer(strategy='median') for numeric, XGBClassifier/XGBRegressor]
- Model: {plan.get('suggested_model')} with eval_metric='auc' for classification, 'rmse' for regression
- MLflow: set_registry_uri("databricks-uc"), set experiment to widget value, log all params + metrics
- Log model with mlflow.sklearn.log_model(pipeline, "model", input_example=X_test.head(5))
- Register to UC: {{catalog}}.{output_schema}.{{model_name}}
- Set alias "Champion" on the newly registered version
- Print final AUC/RMSE and feature importances (top 15)"""


def _inference_prompt(opp: dict, plan: FeaturePlan, catalog: str, output_schema: str) -> str:
    return f"""Generate a complete Databricks batch inference notebook.

APPROVED OPPORTUNITY:
{json.dumps(opp, indent=2)}

FEATURE PLAN:
{json.dumps(plan, indent=2)}

NOTEBOOK REQUIREMENTS:
- Widgets: catalog, output_schema (default "{output_schema}"), model_name (default "{plan.get('feature_table_name','model').replace('_features','_model')}"), run_date (default today)
- Load model: mlflow.pyfunc.load_model(f"models:/{{catalog}}.{output_schema}.{{model_name}}@Champion")
- Read feature table: {{catalog}}.{output_schema}.{plan.get('feature_table_name')} filtered to run_date = widget value
- Score as pandas batch (convert to pandas, predict, convert back to Spark)
- Add columns: score_date = run_date, model_name = widget value
- Write scores to: {{catalog}}.{output_schema}.{plan.get('scores_table_name')} in append mode with partitionBy("score_date")
- After writing emit: spark.sql(f"GRANT SELECT ON TABLE {{catalog}}.{output_schema}.{plan.get('scores_table_name')} TO `account users`")
- Print count of scored records"""


def generate_code(state: AgentState) -> AgentState:
    """
    Generate all 3 Databricks notebooks using Claude.
    Uses focused per-notebook prompts for reliable, runnable output.
    """
    opp = state.get("approved_opportunity")
    plan = state.get("feature_plan")
    if not opp or not plan or state.get("error"):
        return state

    ctx = state["workspace"]
    llm = ctx.get_llm(max_tokens=8192)
    output_schema = "ml_accelerator"
    slug = slugify(opp.get("use_case", "ml_use_case"))

    logger.info("Generating notebooks for: %s", opp.get("use_case"))

    notebooks: dict[str, str] = {}
    notebook_prompts = {
        "01_feature_engineering": _feature_engineering_prompt(opp, plan, ctx.catalog, output_schema),
        "02_training": _training_prompt(opp, plan, ctx.catalog, output_schema),
        "03_batch_inference": _inference_prompt(opp, plan, ctx.catalog, output_schema),
    }

    for nb_key, prompt in notebook_prompts.items():
        logger.info("Generating: %s", nb_key)
        try:
            response = llm.invoke([SystemMessage(content=_NB_SYSTEM), HumanMessage(content=prompt)])
            content = response.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            # Ensure notebook source header
            if not content.startswith("# Databricks notebook source"):
                content = "# Databricks notebook source\n\n" + content
            notebooks[nb_key] = content
        except Exception as e:
            logger.error("Failed to generate %s: %s", nb_key, e)
            return {**state, "error": f"Code generation failed for {nb_key}: {e}"}

    artifacts = prepare_artifacts_from_generation(
        use_case_slug=slug,
        notebook_contents=notebooks,
        cluster_id=ctx.cluster_id,
        catalog=ctx.catalog,
        output_schema=output_schema,
        feature_table_name=plan.get("feature_table_name", f"{slug}_features"),
        model_name=plan.get("feature_table_name", f"{slug}_features").replace("_features", "_model"),
        scores_table_name=plan.get("scores_table_name", f"{slug}_scores"),
    )

    logger.info("Generated %d artifacts", len(artifacts))
    return {**state, "generated_artifacts": artifacts}


# ── Node 3: Human Checkpoint — Code Review ───────────────────────────────────

def human_checkpoint_code(state: AgentState) -> AgentState:
    """
    Pause and show generated notebooks to the user for review.
    User can approve as-is or provide edit instructions.
    Resumes with Command(resume={"action": "approve"}) or {"action": "regenerate", "instructions": "..."}.
    """
    if state.get("error"):
        return state

    artifacts = state.get("generated_artifacts", [])
    notebook_previews = [
        {"filename": a["filename"], "filepath": a["filepath"], "preview": a["content"][:800] + "..."}
        for a in artifacts if a["filename"].endswith(".py")
    ]

    user_response: dict = interrupt({
        "message": "Review the generated notebooks. Approve to write to bundles/, or provide edit instructions.",
        "notebooks": notebook_previews,
        "total_artifacts": len(artifacts),
    })

    action = user_response.get("action", "approve")
    if action == "approve":
        return state
    # "regenerate" — pass instructions back; graph would re-enter generate_code (Phase 4 feature)
    return {**state, "error": f"Regeneration requested: {user_response.get('instructions', '')}"}


# ── Node 4: Write Bundle ─────────────────────────────────────────────────────

def write_bundle(state: AgentState) -> AgentState:
    """Write all generated artifacts to the bundles/ directory."""
    if state.get("error"):
        return state

    artifacts = state.get("generated_artifacts", [])
    if not artifacts:
        return {**state, "error": "No artifacts to write"}

    try:
        written = write_artifacts(artifacts)
        logger.info("Bundle written: %d files", len(written))
        for path in written:
            logger.info("  %s", path)
        return {**state, "bundle_written": True}
    except Exception as e:
        logger.error("Bundle write failed: %s", e)
        return {**state, "error": f"Bundle write failed: {e}", "bundle_written": False}
