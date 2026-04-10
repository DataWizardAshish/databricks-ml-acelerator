"""
Streamlit UI — Databricks ML Accelerator (Phase 1 + Phase 2 + Trust Layer).

Flow:
  Connect workspace → Browse catalog/schema → Discover
  → Approve Opportunity         (shows financial impact + confidence)
  → Confirm Dry Run             (shows CTO business brief + detailed plan)
  → Code Review + Risk Scorecard (shows notebooks + pass/warn/fail checks)
  → Bundle Written + Full Exec Summary
"""

import os
import re

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── OBO identity (Databricks Apps injects these headers per request) ──────────
# st.context.headers is per-session; read once and cache in session_state.
if "obo_email" not in st.session_state:
    headers = getattr(st.context, "headers", {}) or {}
    st.session_state.obo_email = headers.get("X-Forwarded-Email", "")
    st.session_state.obo_token = headers.get("X-Forwarded-Access-Token", "")

_OBO_HEADERS = {}
if st.session_state.obo_email:
    _OBO_HEADERS["X-Forwarded-Email"] = st.session_state.obo_email
if st.session_state.obo_token:
    _OBO_HEADERS["X-Forwarded-Access-Token"] = st.session_state.obo_token

# ── Q&A guardrails ────────────────────────────────────────────────────────────
# Checked client-side BEFORE any API/LLM call.
# Only blocks clearly off-topic questions; nuanced cases fall through to the
# LLM which has its own system-prompt rules.

_OFF_TOPIC_PATTERNS = [
    # Unrelated domains
    r"\b(weather|temperature|forecast|earthquake|hurricane)\b",
    r"\b(stock\s*price|cryptocurrency|bitcoin|ethereum|nft)\b",
    r"\b(recipe|cooking|ingredient|restaurant)\b",
    r"\b(sports?|football|soccer|cricket|basketball|tennis)\b",
    r"\b(movie|tv\s*show|music|song|lyrics|celebrity|actor)\b",
    r"\b(joke|poem|story|rhyme|write\s+me\s+a)\b",
    # General setup / admin unrelated to this run
    r"\b(how\s+(do\s+i|to)\s+(install|set\s*up|sign\s*up|create\s+an?\s+account|configure)\b)",
    r"\b(azure\s+ad|active\s+directory|vpn|firewall|dns|ldap|oauth\s+setup)\b",
    r"\b(aws\s+console|gcp\s+console|azure\s+portal)\b",
    # General AI / ML definitions not tied to this run
    r"^(what\s+is|define|explain|tell\s+me\s+about)\s+(machine\s+learning|artificial\s+intelligence|deep\s+learning|neural\s+network|llm|gpt|chatgpt|openai)\b",
    # Pure trivia / chit-chat
    r"\b(who\s+is\s+(the\s+)?(president|prime\s+minister|ceo\s+of\s+(?!databricks)))\b",
    r"\b(what\s+time\s+is\s+it|what\s+day\s+is\s+(it|today))\b",
]

_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)


def _is_off_topic(question: str) -> bool:
    """Return True if the question is clearly unrelated to the current ML run."""
    q = question.strip()
    if len(q) < 8:
        return True  # too vague / empty-ish
    return bool(_OFF_TOPIC_RE.search(q))

st.set_page_config(page_title="Databricks ML Accelerator", page_icon="⚡", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "run_id": None,
    # Data estate (Step 2)
    "tables": [],
    "estate_summary": "",
    "estate_confirmed": False,    # user clicked "Continue to ML Recommendations"
    # Downstream
    "opportunities": [],
    "approved_opportunity": None,
    "dry_run_plan": None,
    "business_brief": "",         # pre-code CTO summary (from generate_business_brief)
    "risk_scorecard": None,
    "generated_notebooks": [],
    "exec_summary": "",           # full post-bundle summary
    "bundle_written": False,
    "catalogs": [],
    "schemas": [],
    "selected_catalog": "",
    "selected_schema": "",
    # Schema change guard
    "confirm_reset_pending": False,
    "pending_catalog": "",
    "pending_schema": "",
    # Contextual Q&A: dict of step_key → list of {question, answer}
    "qa_history": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def api_get(path, params=None):
    r = httpx.get(f"{API_BASE}{path}", params=params or {}, headers=_OBO_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def api_post(path, body=None, timeout=300):
    r = httpx.post(f"{API_BASE}{path}", json=body or {}, headers=_OBO_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _try_rehydrate(run_id: str) -> bool:
    """
    Restore session state from a persisted checkpoint.
    Called on page load when run_id is in URL query params but session_state is empty.
    Returns True if rehydration succeeded.
    """
    try:
        data = api_get(f"/runs/{run_id}/rehydrate")
    except Exception:
        return False

    if data.get("status") == "not_found":
        return False

    st.session_state.run_id = run_id
    st.session_state.tables = data.get("tables", [])
    st.session_state.estate_summary = data.get("estate_summary", "")
    st.session_state.opportunities = data.get("opportunities", [])
    st.session_state.approved_opportunity = data.get("approved_opportunity")
    st.session_state.dry_run_plan = data.get("dry_run_plan")
    st.session_state.business_brief = data.get("business_brief", "")
    st.session_state.risk_scorecard = data.get("risk_scorecard")
    st.session_state.generated_notebooks = data.get("notebooks", [])
    st.session_state.exec_summary = data.get("exec_summary", "")
    st.session_state.bundle_written = data.get("bundle_written", False)
    # estate_confirmed is derived: True if user got past the opportunities step
    st.session_state.estate_confirmed = bool(data.get("approved_opportunity"))
    # Catalog/schema from the saved workspace context
    if data.get("catalog") and not st.session_state.selected_catalog:
        st.session_state.selected_catalog = data["catalog"]
    if data.get("schema") and not st.session_state.selected_schema:
        st.session_state.selected_schema = data["schema"]
    return True


def _has_active_run() -> bool:
    """True if the user has started a run and has progress to lose."""
    return bool(
        st.session_state.run_id
        or st.session_state.opportunities
        or st.session_state.approved_opportunity
    )


def _reset_all():
    """Reset all run state but keep workspace connection."""
    for k, v in _defaults.items():
        if k not in ("catalogs", "schemas", "selected_catalog", "selected_schema"):
            st.session_state[k] = v
    # Clear run_id from URL so the next discovery starts fresh
    st.query_params.clear()


def _render_qa_panel(step_key: str, placeholder: str = "e.g. Why was this opportunity ranked #1?"):
    """
    Render the contextual Q&A panel for a given step.
    step_key maps to the backend step names used in agent/chat.py.
    Q&A history is persisted in session state per step.
    """
    if not st.session_state.run_id:
        return

    with st.expander("💬 Ask about this step", expanded=False):
        # Show conversation history for this step
        history = st.session_state.qa_history.get(step_key, [])
        for qa in history:
            st.markdown(f"**Q:** {qa['question']}")
            st.info(qa["answer"])
            st.divider()

        q = st.text_input(
            "Your question",
            key=f"q_input_{step_key}",
            placeholder=placeholder,
            label_visibility="collapsed",
        )
        if st.button("Ask", key=f"ask_btn_{step_key}", disabled=not bool(q and q.strip())):
            if _is_off_topic(q):
                st.warning(
                    "That question doesn't appear to be about this run. "
                    "Ask about the data, features, model choice, risk checks, or generated code."
                )
            else:
                with st.spinner("Thinking…"):
                    try:
                        data = api_post(
                            f"/runs/{st.session_state.run_id}/ask",
                            body={"question": q.strip(), "step": step_key},
                            timeout=30,
                        )
                        if step_key not in st.session_state.qa_history:
                            st.session_state.qa_history[step_key] = []
                        st.session_state.qa_history[step_key].append({
                            "question": q.strip(),
                            "answer": data.get("answer", "No answer returned."),
                        })
                        # Keep last 5 Q&As per step
                        st.session_state.qa_history[step_key] = \
                            st.session_state.qa_history[step_key][-5:]
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        if history:
            if st.button("Clear history", key=f"clear_qa_{step_key}"):
                st.session_state.qa_history[step_key] = []
                st.rerun()


def _restore_discovery_from_checkpoint(run_id: str) -> bool:
    """
    Restore only tables/estate_summary/opportunities from checkpoint.
    Used when going Back to Discover then clicking Discover again — skips LLM re-run.
    Does NOT restore approved_opportunity or downstream state (those were cleared by _go_back_to).
    """
    try:
        data = api_get(f"/runs/{run_id}/rehydrate")
    except Exception:
        return False
    if data.get("status") == "not_found" or not data.get("tables"):
        return False
    st.session_state.run_id = run_id
    st.session_state.tables = data.get("tables", [])
    st.session_state.estate_summary = data.get("estate_summary", "")
    st.session_state.opportunities = data.get("opportunities", [])
    return True


def _go_back_to(step: int):
    """
    Reset state to return the user to a previous step.
    step 1 = Discover (clear visual state, keep run_id for checkpoint reuse)
    step 2 = Data Estate (clear estate_confirmed + downstream)
    step 3 = Approve Opportunity (keep opportunities, clear downstream)
    step 4 = Confirm Dry Run (keep dry_run_plan + brief, clear notebooks)
    """
    if step == 1:
        # Partial reset: keep run_id + URL so Discover can reuse the checkpoint.
        # _reset_all() is reserved for explicit "start over" (schema change / "New run").
        for k in ("tables", "estate_summary", "estate_confirmed", "opportunities",
                  "approved_opportunity", "dry_run_plan", "business_brief",
                  "risk_scorecard", "generated_notebooks", "exec_summary",
                  "bundle_written", "confirm_reset_pending", "pending_catalog", "pending_schema"):
            st.session_state[k] = _defaults[k]
        st.session_state.qa_history = {}
        # run_id and st.query_params intentionally kept for checkpoint reuse
    elif step == 2:
        st.session_state.estate_confirmed = False
        st.session_state.approved_opportunity = None
        st.session_state.dry_run_plan = None
        st.session_state.business_brief = ""
        st.session_state.risk_scorecard = None
        st.session_state.generated_notebooks = []
        st.session_state.exec_summary = ""
        st.session_state.bundle_written = False
    elif step == 3:
        st.session_state.approved_opportunity = None
        st.session_state.dry_run_plan = None
        st.session_state.business_brief = ""
        st.session_state.risk_scorecard = None
        st.session_state.generated_notebooks = []
        st.session_state.exec_summary = ""
        st.session_state.bundle_written = False
    elif step == 4:
        st.session_state.risk_scorecard = None
        st.session_state.generated_notebooks = []
        st.session_state.exec_summary = ""
        st.session_state.bundle_written = False


# ── Startup: restore run from URL query param (survives page refresh) ─────────
if not st.session_state.run_id and "run_id" in st.query_params:
    _try_rehydrate(st.query_params["run_id"])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Databricks ML Accelerator")
st.caption("Connect your Unity Catalog. Get production ML in days, not months.")

# ── Sidebar: workspace connection ─────────────────────────────────────────────
with st.sidebar:
    st.header("Workspace")

    # In Databricks Apps, identity comes from SSO — no manual credentials needed.
    # browse_* endpoints use the App's service principal (no token param needed).
    # OBO token is only forwarded as a header for run-level actions via _OBO_HEADERS.
    if st.session_state.obo_email:
        st.success(f"Signed in as\n**{st.session_state.obo_email}**")
        host = ""
        token = ""   # browse uses App identity; OBO token goes as header, not query param
    else:
        st.caption("Leave blank to use `~/.databrickscfg` DEFAULT profile.")
        host = st.text_input("Workspace URL", placeholder="https://your-workspace.cloud.databricks.com")
        token = st.text_input("Personal Access Token", type="password", placeholder="dapi...")

    st.divider()
    if st.button("🔗 Connect & Browse"):
        with st.spinner("Connecting..."):
            try:
                data = api_get("/browse/catalogs", params={"host": host, "token": token})
                st.session_state.catalogs = data.get("catalogs", [])
                st.session_state.schemas = []
                st.session_state.selected_catalog = ""
                st.session_state.selected_schema = ""
                st.success(f"Connected — {len(st.session_state.catalogs)} catalogs")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    if st.session_state.catalogs:
        new_cat = st.selectbox("Catalog", st.session_state.catalogs,
                               index=st.session_state.catalogs.index(st.session_state.selected_catalog)
                               if st.session_state.selected_catalog in st.session_state.catalogs else 0)

        # Schema change guard for catalog
        if new_cat != st.session_state.selected_catalog:
            if _has_active_run():
                st.session_state.confirm_reset_pending = True
                st.session_state.pending_catalog = new_cat
            else:
                st.session_state.selected_catalog = new_cat
                st.session_state.schemas = []
                try:
                    data = api_get("/browse/schemas", params={"catalog": new_cat, "host": host, "token": token})
                    st.session_state.schemas = data.get("schemas", [])
                except Exception as e:
                    st.error(f"Failed to load schemas: {e}")
                st.rerun()

        if st.session_state.schemas:
            new_sch = st.selectbox("Schema", st.session_state.schemas,
                                   index=st.session_state.schemas.index(st.session_state.selected_schema)
                                   if st.session_state.selected_schema in st.session_state.schemas else 0)

            # Schema change guard for schema
            if new_sch != st.session_state.selected_schema:
                if _has_active_run():
                    st.session_state.confirm_reset_pending = True
                    st.session_state.pending_schema = new_sch
                else:
                    st.session_state.selected_schema = new_sch
                    st.rerun()
    else:
        manual_cat = st.text_input("Catalog", placeholder="lakehouse_dev")
        manual_sch = st.text_input("Schema", placeholder="my_schema")
        if manual_cat:
            st.session_state.selected_catalog = manual_cat
        if manual_sch:
            st.session_state.selected_schema = manual_sch

    st.divider()
    if st.button("Health check"):
        try:
            st.json(api_get("/health"))
        except Exception as e:
            st.error(str(e))

# ── Schema change guard modal ─────────────────────────────────────────────────
if st.session_state.confirm_reset_pending:
    st.warning(
        "Changing catalog/schema will discard your current run progress. "
        "Are you sure you want to start over?"
    )
    col_yes, col_no = st.columns([1, 4])
    with col_yes:
        if st.button("Yes, start over", type="primary"):
            _reset_all()
            if st.session_state.pending_catalog:
                st.session_state.selected_catalog = st.session_state.pending_catalog
                st.session_state.schemas = []
                try:
                    data = api_get("/browse/schemas", params={
                        "catalog": st.session_state.pending_catalog,
                        "host": host, "token": token,
                    })
                    st.session_state.schemas = data.get("schemas", [])
                except Exception:
                    pass
            if st.session_state.pending_schema:
                st.session_state.selected_schema = st.session_state.pending_schema
            st.session_state.confirm_reset_pending = False
            st.session_state.pending_catalog = ""
            st.session_state.pending_schema = ""
            st.rerun()
    with col_no:
        if st.button("No, keep current run"):
            st.session_state.confirm_reset_pending = False
            st.session_state.pending_catalog = ""
            st.session_state.pending_schema = ""
            st.rerun()
    st.stop()

# ── Progress indicator ────────────────────────────────────────────────────────
catalog = st.session_state.selected_catalog
schema = st.session_state.selected_schema

phases = [
    "1 Discover",
    "2 Data Estate",
    "3 Approve Opportunity",
    "4 Confirm Dry Run",
    "5 Code Review",
    "6 Bundle Written",
]

current = 0
if st.session_state.tables:
    current = 1  # Data Estate view
if st.session_state.estate_confirmed:
    current = 2  # Approve Opportunity
if st.session_state.dry_run_plan:
    current = 3  # Confirm Dry Run
if st.session_state.generated_notebooks:
    current = 4  # Code Review
if st.session_state.bundle_written:
    current = 5  # Bundle Written

cols = st.columns(6)
for i, (col, phase) in enumerate(zip(cols, phases)):
    if i < current:
        col.success(f"✅ {phase}")
    elif i == current:
        col.info(f"▶ {phase}")
    else:
        col.write(f"⬜ {phase}")

st.divider()


# ── Step 1: Discover ──────────────────────────────────────────────────────────
if current == 0:
    st.header("Step 1 — Discover ML Opportunities")
    if catalog and schema:
        st.info(f"Target: `{catalog}.{schema}`")
    else:
        st.warning("Connect and select catalog + schema in the sidebar.")

    can_run = bool(catalog and schema)
    if st.button("🔍 Discover", type="primary", disabled=not can_run):
        # Fast path: if a prior checkpoint exists for this workspace, reuse it
        if st.session_state.run_id:
            with st.spinner("Restoring from previous discovery…"):
                if _restore_discovery_from_checkpoint(st.session_state.run_id):
                    st.rerun()
        # Slow path: full UC scan + LLM analysis
        with st.spinner("Analysing your data estate — this takes ~30–60 seconds…"):
            try:
                data = api_post("/runs", body={
                    "workspace": {"host": host, "token": token},
                    "catalog": catalog, "schema": schema,
                })
                if data.get("status") == "error":
                    field = data.get("field", "")
                    st.error(f"{'Catalog' if field=='catalog' else 'Schema' if field=='schema' else 'Error'}: {data.get('error')}")
                else:
                    st.session_state.run_id = data["run_id"]
                    st.session_state.tables = data.get("tables", [])
                    st.session_state.estate_summary = data.get("estate_summary", "")
                    st.session_state.opportunities = data.get("opportunities", [])
                    # Persist run_id in URL so page refresh can rehydrate from checkpoint
                    st.query_params["run_id"] = data["run_id"]
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Recent Runs (episodic memory) ─────────────────────────────────────────
    _STATUS_BADGE = {
        "awaiting_approval":            "🟡 Awaiting Approval",
        "awaiting_dry_run_confirmation": "🟡 Dry Run",
        "awaiting_code_review":         "🟡 Code Review",
        "completed":                    "🟢 Complete",
        "error":                        "🔴 Error",
        "running":                      "🔵 Running",
    }
    try:
        history_data = api_get("/runs/history")
        recent_runs = history_data.get("runs", [])
    except Exception:
        recent_runs = []

    if recent_runs:
        st.divider()
        st.subheader("Recent Runs")
        for run in recent_runs:
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])
                c1.caption(f"`{run['run_id'][:8]}…`")
                c2.caption(f"`{run['catalog']}.{run['schema']}`")
                c3.caption(run["use_case"] or "—")
                c4.caption(_STATUS_BADGE.get(run["status"], run["status"]))
                with c5:
                    if st.button("Resume", key=f"resume_{run['run_id']}"):
                        with st.spinner("Restoring run…"):
                            if _try_rehydrate(run["run_id"]):
                                st.query_params["run_id"] = run["run_id"]
                                st.rerun()
                            else:
                                st.error("Could not restore this run.")


# ── Step 2: Data Estate Overview ─────────────────────────────────────────────
elif current == 1:
    if st.button("← Back to Discover"):
        _go_back_to(1)
        st.rerun()

    tables = st.session_state.tables
    st.header("Step 2 — Data Estate Overview")
    st.caption(f"Run ID: `{st.session_state.run_id}`  |  `{catalog}.{schema}`")

    # Summary metrics
    n_managed = sum(1 for t in tables if t.get("table_type") == "MANAGED")
    n_external = sum(1 for t in tables if t.get("table_type") == "EXTERNAL")
    total_cols = sum(len(t.get("columns", [])) for t in tables)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tables Found", len(tables))
    m2.metric("Managed", n_managed)
    m3.metric("External", n_external)
    m4.metric("Total Columns", total_cols)

    st.divider()

    # Per-table cards
    for t in tables:
        with st.container(border=True):
            table_type = t.get("table_type", "")
            badge = "🟦" if table_type == "MANAGED" else "🟨"
            columns_list = t.get("columns", [])

            header_col, meta_col = st.columns([5, 1])
            with header_col:
                st.markdown(f"**{badge} `{t['name']}`**  &nbsp; `{table_type}`")
                if t.get("comment"):
                    st.caption(t["comment"])
            with meta_col:
                if t.get("row_count"):
                    st.caption(f"~{t['row_count']:,} rows")
                st.caption(f"{len(columns_list)} columns")

            if columns_list:
                with st.expander(f"Columns ({len(columns_list)})", expanded=False):
                    cols_per_row = 4
                    for i in range(0, len(columns_list), cols_per_row):
                        row_cols = st.columns(cols_per_row)
                        for j, col_w in enumerate(row_cols):
                            if i + j < len(columns_list):
                                c = columns_list[i + j]
                                nullable_flag = "" if c.get("nullable", True) else " ⚠️"
                                col_w.markdown(
                                    f"`{c['name']}`{nullable_flag}  \n*{c['type_text']}*"
                                )

    # AI estate analysis (appended to estate_summary after ANALYSIS: marker)
    summary = st.session_state.estate_summary
    if summary and "ANALYSIS:" in summary:
        analysis = summary.split("ANALYSIS:", 1)[1].strip()
        with st.expander("🤖 AI Analysis of your data estate", expanded=False):
            st.markdown(analysis)

    st.divider()
    if st.button("→ See ML Recommendations", type="primary"):
        st.session_state.estate_confirmed = True
        st.rerun()


# ── Step 3: Approve Opportunity ───────────────────────────────────────────────
elif current == 2:
    # Back button
    if st.button("← Back to Data Estate"):
        _go_back_to(2)
        st.rerun()

    st.header("Step 3 — Approve an ML Opportunity")
    st.caption(f"Run ID: `{st.session_state.run_id}`")
    st.write("Select one opportunity. The agent will explain exactly what will happen before generating any code.")

    for opp in st.session_state.opportunities:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.subheader(f"#{opp['rank']} — {opp['use_case']}")
                st.write(f"**Business value:** {opp['business_value']}")

                if opp.get("financial_impact"):
                    st.success(f"💰 **Financial impact:** {opp['financial_impact']}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Type", opp["ml_type"])
                m2.metric("Complexity", opp["complexity"])
                m3.metric("Est. AUC", opp["estimated_auc_range"])
                m4.metric("Confidence", opp.get("confidence", "medium").title())

                st.write(f"**Target:** `{opp['target_table']}` → `{opp['target_column']}`")
                with st.expander("Rationale"):
                    st.write(opp["rationale"])
            with c2:
                st.write("")
                st.write("")
                if st.button(f"✅ Approve #{opp['rank']}", key=f"app_{opp['rank']}", type="primary"):
                    with st.spinner("Generating dry run plan and business brief…"):
                        try:
                            data = api_post(
                                f"/runs/{st.session_state.run_id}/approve",
                                body={"selected_rank": opp["rank"]},
                                timeout=120,
                            )
                            if data.get("status") == "error":
                                st.error(data.get("error"))
                            else:
                                st.session_state.approved_opportunity = data.get("approved_opportunity")
                                st.session_state.dry_run_plan = data.get("dry_run_plan")
                                st.session_state.business_brief = data.get("exec_summary", "")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")


    st.divider()
    _render_qa_panel(
        "approve_opportunity",
        "e.g. Why was churn prediction ranked above demand forecasting?",
    )


# ── Step 4: Confirm Dry Run + CTO Business Brief ──────────────────────────────
elif current == 3:
    # Back button
    if st.button("← Back to Opportunities"):
        _go_back_to(3)
        st.rerun()

    opp = st.session_state.approved_opportunity or {}
    plan = st.session_state.dry_run_plan or {}

    st.header(f"Step 4 — Review Plan: {opp.get('use_case', '')}")
    st.caption(f"Run ID: `{st.session_state.run_id}`")
    st.info("Review exactly what will happen before code generation starts. No code has run yet.")

    # Two-tab layout: Business Brief for CTO | Technical Plan for engineers
    tab_brief, tab_technical = st.tabs(["📊 Business Brief (CTO view)", "🔧 Technical Plan (Engineer view)"])

    with tab_brief:
        brief = st.session_state.business_brief
        if brief:
            st.markdown(brief)
        else:
            # Render inline if brief not in session (fallback)
            st.subheader(opp.get("use_case", "ML Opportunity"))
            st.write(opp.get("business_value", ""))
            if opp.get("financial_impact"):
                st.success(f"💰 **Financial impact:** {opp['financial_impact']}")
            st.write(f"**Target:** `{opp.get('target_table', '')}` → predict `{opp.get('target_column', '')}`")
            st.write(f"**Estimated cost:** {plan.get('estimated_dbu_cost', 'N/A')}  |  "
                     f"**Estimated time:** {plan.get('estimated_run_time', 'N/A')}")

    with tab_technical:
        # Plain-English summary
        if plan.get("plain_english_summary"):
            with st.container(border=True):
                st.subheader("📋 What will happen")
                st.write(plan["plain_english_summary"])

        # Target column detail
        if plan.get("target_column_detail"):
            st.write(f"**Predicting:** {plan['target_column_detail']}")

        # Cost + time metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Estimated Cost", plan.get("estimated_dbu_cost", "—"))
        col2.metric("Estimated Time", plan.get("estimated_run_time", "—"))
        col3.metric("Est. Feature Rows", plan.get("estimated_row_count", "—"))
        col4.metric("Notebooks", "3")

        # Join keys
        if plan.get("join_keys"):
            st.write(f"**Join keys:** {', '.join(f'`{k}`' for k in plan['join_keys'])}")

        # Feature columns
        if plan.get("feature_columns"):
            with st.expander(f"🧮 Expected Feature Columns ({len(plan['feature_columns'])})", expanded=False):
                cols_per_row = 3
                feature_cols = plan["feature_columns"]
                for i in range(0, len(feature_cols), cols_per_row):
                    row = st.columns(cols_per_row)
                    for j, col_widget in enumerate(row):
                        if i + j < len(feature_cols):
                            col_widget.code(feature_cols[i + j], language=None)

        col_reads, col_writes = st.columns(2)
        with col_reads:
            with st.expander(f"📖 Tables to READ ({len(plan.get('tables_to_read', []))})", expanded=True):
                for t in plan.get("tables_to_read", []):
                    st.code(t, language=None)
        with col_writes:
            with st.expander(f"✍️ Tables to WRITE ({len(plan.get('tables_to_write', []))})", expanded=True):
                for t in plan.get("tables_to_write", []):
                    st.code(t, language=None)

        if plan.get("grant_statements"):
            with st.expander(f"🔐 Access Grants ({len(plan['grant_statements'])})"):
                for g in plan["grant_statements"]:
                    st.code(g, language="sql")

    _render_qa_panel(
        "dry_run",
        "e.g. Why are you joining 3 tables? What's the risk if the feature pipeline fails?",
    )

    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("✅ Confirm & Generate Code", type="primary"):
            with st.spinner("Generating 3 Databricks notebooks — ~2–3 minutes…"):
                try:
                    data = api_post(
                        f"/runs/{st.session_state.run_id}/confirm-dry-run",
                        timeout=600,
                    )
                    if data.get("status") == "error":
                        st.error(data.get("error"))
                    else:
                        st.session_state.risk_scorecard = data.get("risk_scorecard")
                        st.session_state.generated_notebooks = data.get("notebooks", [])
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    with col2:
        st.caption("3 notebooks (feature engineering, training, inference) + 3 job YAMLs will be generated.")


# ── Step 5: Code Review + Risk Scorecard ──────────────────────────────────────
elif current == 4:
    # Back button
    if st.button("← Back to Dry Run Plan"):
        _go_back_to(4)
        st.rerun()

    opp = st.session_state.approved_opportunity or {}
    scorecard = st.session_state.risk_scorecard or {}

    st.header(f"Step 5 — Review Generated Code: {opp.get('use_case', '')}")
    st.caption(f"Run ID: `{st.session_state.run_id}`")

    # Risk Scorecard banner
    overall = scorecard.get("overall", "")
    if overall == "ready":
        st.success(f"✅ **Deployment Readiness:** {scorecard.get('summary', 'All checks passed.')}")
    elif overall == "review_needed":
        st.warning(f"⚠️ **Deployment Readiness:** {scorecard.get('summary', 'Warnings to review.')}")
    elif overall == "blocked":
        st.error(f"❌ **Deployment Readiness:** {scorecard.get('summary', 'Critical issues found.')}")

    # Risk Scorecard table
    if scorecard.get("items"):
        with st.expander("🔍 Risk Scorecard — expand to see all checks", expanded=(overall != "ready")):
            status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
            header_cols = st.columns([3, 1, 6])
            header_cols[0].markdown("**Check**")
            header_cols[1].markdown("**Status**")
            header_cols[2].markdown("**Detail**")
            st.divider()
            for item in scorecard.get("items", []):
                row = st.columns([3, 1, 6])
                row[0].write(item["check"])
                row[1].write(status_icon.get(item["status"], item["status"]))
                row[2].write(item["detail"])

    st.divider()

    # Generated notebooks
    st.subheader("Generated Notebooks")
    for nb in st.session_state.generated_notebooks:
        with st.expander(f"📄 {nb['filename']}  —  `{nb['filepath']}`", expanded=False):
            st.code(nb["content"], language="python")

    _render_qa_panel(
        "code_review",
        "e.g. Why was XGBoost chosen? What does the GRANT statement give access to?",
    )

    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        disabled = overall == "blocked"
        if st.button("✅ Approve & Write Bundle", type="primary", disabled=disabled):
            with st.spinner("Writing notebooks and job YAMLs to bundles/…"):
                try:
                    data = api_post(
                        f"/runs/{st.session_state.run_id}/approve-code",
                        body={"action": "approve"},
                    )
                    if data.get("status") == "completed":
                        st.session_state.bundle_written = True
                        st.session_state.exec_summary = data.get("exec_summary", "")
                        st.rerun()
                    else:
                        st.error(data.get("error", "Bundle write failed"))
                except Exception as e:
                    st.error(f"Error: {e}")
        if disabled:
            st.caption("⛔ Fix critical issues before approving.")
    with col2:
        st.caption("Files will be written to `bundles/src/{use_case}/` and `bundles/resources/jobs/`")


# ── Step 6: Done ──────────────────────────────────────────────────────────────
elif current == 5:
    opp = st.session_state.approved_opportunity or {}
    slug = opp.get("use_case", "").lower().replace(" ", "_")

    st.header("✅ Bundle Ready")
    st.success(f"**{opp.get('use_case', 'ML Use Case')}** — notebooks and jobs written to `bundles/`")

    tab_summary, tab_files, tab_deploy = st.tabs(["📊 Executive Summary", "📁 Generated Files", "🚀 Deploy"])

    with tab_summary:
        exec_summary = st.session_state.exec_summary
        if exec_summary:
            st.markdown(exec_summary)
        else:
            st.info("Executive summary is in `bundles/SUMMARY.md`")

    with tab_files:
        files = [
            f"bundles/src/{slug}/01_feature_engineering.py",
            f"bundles/src/{slug}/02_training.py",
            f"bundles/src/{slug}/03_batch_inference.py",
            f"bundles/resources/jobs/{slug}_feature_pipeline_job.yml",
            f"bundles/resources/jobs/{slug}_training_job.yml",
            f"bundles/resources/jobs/{slug}_batch_inference_job.yml",
            "bundles/SUMMARY.md",
        ]
        for f in files:
            st.code(f, language=None)

    with tab_deploy:
        st.code(f"""# Install Databricks CLI if needed
pip install databricks-cli

# Deploy to dev workspace
databricks bundle deploy --target dev

# Run the feature pipeline first
databricks bundle run {slug}_feature_pipeline_job --target dev

# Then run training
databricks bundle run {slug}_training_job --target dev

# Then batch inference
databricks bundle run {slug}_batch_inference_job --target dev

# When ready for production
databricks bundle deploy --target prod""", language="bash")

    st.divider()
    _render_qa_panel(
        "done",
        "e.g. What should I monitor in the first 30 days after deployment?",
    )

    st.divider()
    if st.button("🔄 Start a new run"):
        _reset_all()
        st.rerun()
