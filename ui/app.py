"""
Streamlit UI for the Databricks ML Accelerator — Phase 1 + Step A improvements.

Features:
- Workspace URL + PAT configurable per session (different teams, different workspaces)
- Catalog and schema selected from live dropdowns (browse mode, no typos)
- UC validation errors surfaced clearly before running the agent
- Approve one opportunity to proceed to Phase 2 code generation
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Databricks ML Accelerator",
    page_icon="⚡",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("run_id", None),
    ("opportunities", []),
    ("approved", None),
    ("catalogs", []),
    ("schemas", []),
    ("selected_catalog", ""),
    ("selected_schema", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def api_get(path: str, params: dict = None) -> dict:
    r = httpx.get(f"{API_BASE}{path}", params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


def api_post(path: str, body: dict = None) -> dict:
    r = httpx.post(f"{API_BASE}{path}", json=body or {}, timeout=180)
    r.raise_for_status()
    return r.json()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Databricks ML Accelerator")
st.caption("Connect your Unity Catalog. Get production ML in days, not months.")

# ── Sidebar: workspace connection ─────────────────────────────────────────────
with st.sidebar:
    st.header("Workspace Connection")
    st.caption("Leave blank to use your local `~/.databrickscfg` profile.")

    host = st.text_input(
        "Workspace URL",
        placeholder="https://your-workspace.cloud.databricks.com",
        help="Leave blank to use the DEFAULT profile from ~/.databrickscfg",
    )
    token = st.text_input(
        "Personal Access Token",
        type="password",
        placeholder="dapi...",
        help="Leave blank to use the DEFAULT profile from ~/.databrickscfg",
    )

    st.divider()

    # Browse catalogs button
    if st.button("🔗 Connect & Browse Catalogs"):
        with st.spinner("Connecting..."):
            try:
                data = api_get("/browse/catalogs", params={"host": host, "token": token})
                st.session_state.catalogs = data.get("catalogs", [])
                st.session_state.schemas = []
                st.session_state.selected_catalog = ""
                st.session_state.selected_schema = ""
                st.success(f"Connected — {len(st.session_state.catalogs)} catalogs found")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Catalog dropdown (populated after connect)
    if st.session_state.catalogs:
        selected_catalog = st.selectbox(
            "Catalog",
            options=st.session_state.catalogs,
            index=st.session_state.catalogs.index(st.session_state.selected_catalog)
            if st.session_state.selected_catalog in st.session_state.catalogs else 0,
        )
        if selected_catalog != st.session_state.selected_catalog:
            st.session_state.selected_catalog = selected_catalog
            st.session_state.schemas = []
            st.session_state.selected_schema = ""
            # Auto-load schemas when catalog changes
            try:
                data = api_get("/browse/schemas", params={
                    "catalog": selected_catalog, "host": host, "token": token
                })
                st.session_state.schemas = data.get("schemas", [])
            except Exception as e:
                st.error(f"Failed to load schemas: {e}")
            st.rerun()

        # Schema dropdown
        if st.session_state.schemas:
            selected_schema = st.selectbox(
                "Schema",
                options=st.session_state.schemas,
                index=st.session_state.schemas.index(st.session_state.selected_schema)
                if st.session_state.selected_schema in st.session_state.schemas else 0,
            )
            st.session_state.selected_schema = selected_schema
    else:
        # Manual fallback (first-time, before connecting)
        st.caption("Or type manually (click Connect first for dropdowns):")
        manual_catalog = st.text_input("Catalog", placeholder="lakehouse_dev")
        manual_schema = st.text_input("Schema", placeholder="my_schema")
        if manual_catalog:
            st.session_state.selected_catalog = manual_catalog
        if manual_schema:
            st.session_state.selected_schema = manual_schema

    st.divider()
    st.caption(f"API: {API_BASE}")
    if st.button("Health check"):
        try:
            st.json(api_get("/health"))
        except Exception as e:
            st.error(str(e))


# ── Main: Step 1 — Discover ───────────────────────────────────────────────────
st.header("Step 1 — Discover ML Opportunities")

catalog = st.session_state.selected_catalog
schema = st.session_state.selected_schema

if catalog and schema:
    st.info(f"Target: `{catalog}.{schema}`")
else:
    st.warning("Connect and select a catalog + schema in the sidebar first.")

can_run = bool(catalog and schema) and st.session_state.run_id is None

if st.button("🔍 Discover ML Opportunities", type="primary", disabled=not can_run):
    with st.spinner(f"Connecting to `{catalog}.{schema}` and analyzing your data estate…"):
        try:
            data = api_post("/runs", body={
                "workspace": {"host": host, "token": token},
                "catalog": catalog,
                "schema": schema,
            })

            if data.get("status") == "error":
                field = data.get("field", "")
                msg = data.get("error", "Unknown error")
                if field == "catalog":
                    st.error(f"Catalog error: {msg}")
                elif field == "schema":
                    st.error(f"Schema error: {msg}")
                else:
                    st.error(f"Error: {msg}")
            else:
                st.session_state.run_id = data["run_id"]
                st.session_state.opportunities = data.get("opportunities", [])
                st.success(f"Discovery complete — Run ID: `{data['run_id']}`")
                st.rerun()

        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")


# ── Main: Step 2 — Review + Approve ──────────────────────────────────────────
if st.session_state.opportunities and not st.session_state.approved:
    st.header("Step 2 — Review ML Opportunities")
    st.write("The agent identified these ML opportunities in your data estate. Approve one to proceed.")

    for opp in st.session_state.opportunities:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.subheader(f"#{opp['rank']} — {opp['use_case']}")
                st.write(f"**Business value:** {opp['business_value']}")

                meta_col1, meta_col2, meta_col3 = st.columns(3)
                meta_col1.metric("ML Type", opp["ml_type"])
                meta_col2.metric("Complexity", opp["complexity"])
                meta_col3.metric("Est. AUC", opp["estimated_auc_range"])

                st.write(f"**Target:** `{opp['target_table']}` → predict `{opp['target_column']}`")
                if opp.get("feature_tables"):
                    st.write(f"**Feature tables:** {', '.join(f'`{t}`' for t in opp['feature_tables'])}")

                with st.expander("Rationale"):
                    st.write(opp["rationale"])

            with col2:
                st.write("")
                st.write("")
                st.write("")
                if st.button(f"✅ Approve #{opp['rank']}", key=f"approve_{opp['rank']}", type="primary"):
                    with st.spinner("Saving approval..."):
                        try:
                            data = api_post(
                                f"/runs/{st.session_state.run_id}/approve",
                                body={"selected_rank": opp["rank"]},
                            )
                            st.session_state.approved = data.get("approved_opportunity")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Approval failed: {e}")


# ── Main: Step 3 — Approved / Next Steps ─────────────────────────────────────
if st.session_state.approved:
    st.header("Step 3 — Approved ✅")
    opp = st.session_state.approved
    st.success(f"**{opp['use_case']}** approved and queued for code generation.")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("ML Type", opp["ml_type"])
        c2.metric("Complexity", opp["complexity"])
        c3.metric("Est. AUC", opp["estimated_auc_range"])
        st.write(f"**Target:** `{opp['target_table']}` → `{opp['target_column']}`")
        st.write(f"**Business value:** {opp['business_value']}")

    st.info("**Phase 2 coming next:** The agent will generate the feature pipeline, "
            "training job (MLflow tracked), and batch inference job as a Databricks Asset Bundle.")

    with st.expander("Full opportunity details (JSON)"):
        st.json(opp)

    if st.button("🔄 Start new run"):
        for key in ["run_id", "opportunities", "approved"]:
            st.session_state[key] = None if key != "opportunities" else []
        st.rerun()
