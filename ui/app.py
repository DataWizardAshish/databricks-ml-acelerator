"""
Streamlit UI for the ML Accelerator — Phase 1.
Talks to the FastAPI backend via HTTP.
"""

import time
import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Databricks ML Accelerator",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Databricks ML Accelerator")
st.caption("Connect your Unity Catalog. Get ML opportunities in minutes.")

# ── Sidebar: config ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    catalog = st.text_input("UC Catalog", value="lakehouse_dev")
    schema = st.text_input("UC Schema", value="ds_brand_funnel_ai_lhdev")
    st.divider()
    st.caption("API: " + API_BASE)
    if st.button("Check API health"):
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"API unreachable: {e}")

# ── Session state ─────────────────────────────────────────────────────────────
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "opportunities" not in st.session_state:
    st.session_state.opportunities = []
if "approved" not in st.session_state:
    st.session_state.approved = None

# ── Step 1: Start discovery ───────────────────────────────────────────────────
st.header("Step 1 — Discover Catalog")

if st.button("🔍 Discover ML Opportunities", type="primary", disabled=st.session_state.run_id is not None):
    with st.spinner("Connecting to Unity Catalog and analyzing your data estate..."):
        try:
            r = httpx.post(
                f"{API_BASE}/runs",
                json={"catalog": catalog, "schema": schema},
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()

            if data.get("status") == "error":
                st.error(f"Discovery failed: {data.get('error')}")
            else:
                st.session_state.run_id = data["run_id"]
                st.session_state.opportunities = data.get("opportunities", [])
                st.success(f"Discovery complete! Run ID: `{data['run_id']}`")
                st.rerun()
        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Step 2: Show opportunities and approve ────────────────────────────────────
if st.session_state.opportunities:
    st.header("Step 2 — Review ML Opportunities")
    st.write("The agent identified these ML opportunities in your data estate:")

    selected_rank = None
    for opp in st.session_state.opportunities:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"#{opp['rank']} — {opp['use_case']}")
                st.write(f"**Business value:** {opp['business_value']}")
                st.write(f"**ML type:** `{opp['ml_type']}`  |  **Complexity:** `{opp['complexity']}`  |  **Estimated AUC:** `{opp['estimated_auc_range']}`")
                st.write(f"**Target:** `{opp['target_table']}` → `{opp['target_column']}`")
                with st.expander("Rationale"):
                    st.write(opp["rationale"])
            with col2:
                st.write("")
                st.write("")
                if st.button(f"✅ Approve #{opp['rank']}", key=f"approve_{opp['rank']}"):
                    selected_rank = opp["rank"]

    if selected_rank is not None and st.session_state.approved is None:
        with st.spinner("Approving opportunity and saving decision..."):
            try:
                r = httpx.post(
                    f"{API_BASE}/runs/{st.session_state.run_id}/approve",
                    json={"selected_rank": selected_rank},
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                st.session_state.approved = data.get("approved_opportunity")
                st.rerun()
            except Exception as e:
                st.error(f"Approval failed: {e}")

# ── Step 3: Approved ──────────────────────────────────────────────────────────
if st.session_state.approved:
    st.header("Step 3 — Approved")
    opp = st.session_state.approved
    st.success(f"✅ Approved: **{opp['use_case']}**")
    st.info("Phase 2 coming next: the agent will generate feature pipeline, training job, and inference code.")

    with st.expander("Approved opportunity details"):
        st.json(opp)

    if st.button("🔄 Start new run"):
        st.session_state.run_id = None
        st.session_state.opportunities = []
        st.session_state.approved = None
        st.rerun()
