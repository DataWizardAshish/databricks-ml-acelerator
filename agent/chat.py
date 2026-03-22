"""
Contextual Q&A for a specific run.

Answers questions scoped to the current run state — opportunity, dry run plan,
feature plan, generated code, risk scorecard. NOT a general Databricks assistant.

Usage:
  answer = ask_about_run(workspace_ctx, step="dry_run", question="...", values=snapshot.values)
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


_CHAT_SYSTEM = """You are an assistant embedded in Databricks ML Accelerator.
Your job is to answer questions about ONE specific ML run that is in progress.

The full context for this run is provided below.

RULES:
1. Answer ONLY based on the provided run context — do not speculate beyond what is given.
2. If asked about something not available in this context, say clearly:
   "That information isn't available at this stage of the run."
3. Do not answer general Databricks questions unrelated to this run.
   Example of out-of-scope: "How do I set up Azure AD?" — decline politely.
   Example of in-scope: "Why was customer_id excluded from the features?" — answer from context.
4. Be concise and direct. Your audience is senior data engineers and CTOs.
5. When referencing specific columns, tables, or code, use backticks.
6. Do not proactively suggest changes to the plan unless the user explicitly asks "should I change..."
7. If the user asks whether to approve or proceed, give a balanced view based on the risk scorecard
   and dry run data — never just say "yes go ahead".

---

## RUN CONTEXT

{context}
"""


# ── Context assemblers (one per step) ─────────────────────────────────────────

def _fmt(obj) -> str:
    """Compact JSON — no nulls, trimmed."""
    return json.dumps(obj, indent=2, default=str)


def _build_context(step: str, values: dict) -> str:
    """
    Assemble step-appropriate context from run state.
    Only include data that exists at the given step — avoids confusing the LLM
    with empty fields.
    """
    opp = values.get("approved_opportunity") or {}
    opportunities = values.get("opportunities", [])
    estate_summary = values.get("estate_summary", "")
    dry_run = values.get("dry_run_plan") or {}
    feature_plan = values.get("feature_plan") or {}
    scorecard = values.get("risk_scorecard") or {}
    artifacts = values.get("generated_artifacts", [])

    parts: list[str] = []

    if step == "approve_opportunity":
        if estate_summary:
            parts.append(f"### Data Estate Summary\n{estate_summary[:1000]}")
        if opportunities:
            parts.append(f"### ML Opportunities Ranked\n{_fmt(opportunities)}")

    elif step == "dry_run":
        if opp:
            parts.append(f"### Approved ML Opportunity\n{_fmt(opp)}")
        if dry_run:
            parts.append(f"### Dry Run Plan\n{_fmt(dry_run)}")

    elif step == "code_review":
        if opp:
            parts.append(f"### Approved ML Opportunity\n{_fmt(opp)}")
        if dry_run:
            # Only the summary fields — skip large lists to save tokens
            dry_run_summary = {
                k: dry_run.get(k)
                for k in [
                    "tables_to_read", "tables_to_write", "grant_statements",
                    "estimated_dbu_cost", "estimated_run_time",
                    "plain_english_summary", "feature_columns", "join_keys",
                    "estimated_row_count", "target_column_detail",
                ]
                if dry_run.get(k)
            }
            parts.append(f"### Dry Run Plan\n{_fmt(dry_run_summary)}")
        if feature_plan:
            parts.append(f"### Feature Engineering Plan\n{_fmt(feature_plan)}")
        if scorecard:
            parts.append(f"### Risk Scorecard\n{_fmt(scorecard)}")
        # Include notebook structure — first 500 chars of each notebook (structure, not full code)
        nb_previews = []
        for a in artifacts:
            if a["filename"].endswith(".py"):
                preview = a["content"][:500].rstrip()
                nb_previews.append(f"#### {a['filename']}\n```python\n{preview}\n...\n```")
        if nb_previews:
            parts.append("### Generated Notebook Previews (first 500 chars each)\n" + "\n\n".join(nb_previews))

    elif step == "done":
        if opp:
            parts.append(f"### Approved ML Opportunity\n{_fmt(opp)}")
        if feature_plan:
            parts.append(f"### Feature Engineering Plan\n{_fmt(feature_plan)}")
        if dry_run:
            parts.append(f"### Dry Run Plan\n{_fmt(dry_run)}")
        if scorecard:
            parts.append(f"### Risk Scorecard\n{_fmt(scorecard)}")
        artifact_paths = [a["filepath"] for a in artifacts]
        if artifact_paths:
            parts.append("### Generated Files\n" + "\n".join(f"- `{p}`" for p in artifact_paths))

    return "\n\n---\n\n".join(parts) if parts else "No run context available for this step."


# ── Public API ─────────────────────────────────────────────────────────────────

def ask_about_run(workspace_ctx, step: str, question: str, values: dict) -> str:
    """
    Answer a question about the current run using scoped run state as context.

    Args:
        workspace_ctx: WorkspaceContext instance (provides LLM access)
        step: current UI step — "approve_opportunity" | "dry_run" | "code_review" | "done"
        question: user's question string
        values: snapshot.values dict from the LangGraph checkpoint

    Returns:
        Answer string (plain text or markdown)
    """
    context = _build_context(step, values)
    system_prompt = _CHAT_SYSTEM.format(context=context)

    llm = workspace_ctx.get_llm(max_tokens=1024)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])
        return response.content.strip()
    except Exception as e:
        logger.error("ask_about_run failed: %s", e)
        return f"Unable to answer right now: {e}"
