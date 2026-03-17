"""
Streamlit UI — Databricks ML Accelerator (Phase 1 + Phase 2).

Flow:
  Connect workspace → Browse catalog/schema → Discover → Approve opportunity
  → Code generation runs → Review notebooks → Approve → Bundle written to disk
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Databricks ML Accelerator", page_icon="⚡", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "run_id": None,
    "opportunities": [],
    "approved_opportunity": None,
    "generated_notebooks": [],
    "bundle_written": False,
    "catalogs": [],
    "schemas": [],
    "selected_catalog": "",
    "selected_schema": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def api_get(path, params=None):
    r = httpx.get(f"{API_BASE}{path}", params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


def api_post(path, body=None, timeout=300):
    r = httpx.post(f"{API_BASE}{path}", json=body or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Databricks ML Accelerator")
st.caption("Connect your Unity Catalog. Get production ML in days, not months.")

# ── Sidebar: workspace connection ─────────────────────────────────────────────
with st.sidebar:
    st.header("Workspace")
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
        cat = st.selectbox("Catalog", st.session_state.catalogs)
        if cat != st.session_state.selected_catalog:
            st.session_state.selected_catalog = cat
            st.session_state.schemas = []
            try:
                data = api_get("/browse/schemas", params={"catalog": cat, "host": host, "token": token})
                st.session_state.schemas = data.get("schemas", [])
            except Exception as e:
                st.error(f"Failed to load schemas: {e}")
            st.rerun()
        if st.session_state.schemas:
            sch = st.selectbox("Schema", st.session_state.schemas)
            st.session_state.selected_schema = sch
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


# ── Progress indicator ────────────────────────────────────────────────────────
catalog = st.session_state.selected_catalog
schema = st.session_state.selected_schema

phases = ["1 Discover", "2 Approve Opportunity", "3 Code Review", "4 Bundle Written"]
current = 0
if st.session_state.opportunities:
    current = 1
if st.session_state.approved_opportunity and not st.session_state.generated_notebooks:
    current = 2
if st.session_state.generated_notebooks:
    current = 2
if st.session_state.bundle_written:
    current = 3

cols = st.columns(4)
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
                    st.session_state.opportunities = data.get("opportunities", [])
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


# ── Step 2: Approve Opportunity ───────────────────────────────────────────────
elif current == 1:
    st.header("Step 2 — Approve an ML Opportunity")
    st.caption(f"Run ID: `{st.session_state.run_id}`")
    st.write("Select one opportunity to proceed with code generation.")

    for opp in st.session_state.opportunities:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.subheader(f"#{opp['rank']} — {opp['use_case']}")
                st.write(f"**Value:** {opp['business_value']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Type", opp["ml_type"])
                m2.metric("Complexity", opp["complexity"])
                m3.metric("Est. AUC", opp["estimated_auc_range"])
                st.write(f"**Target:** `{opp['target_table']}` → `{opp['target_column']}`")
                with st.expander("Rationale"):
                    st.write(opp["rationale"])
            with c2:
                st.write("")
                st.write("")
                if st.button(f"✅ Approve #{opp['rank']}", key=f"app_{opp['rank']}", type="primary"):
                    with st.spinner("Approved! Running feature planning + code generation — ~2–3 minutes…"):
                        try:
                            data = api_post(
                                f"/runs/{st.session_state.run_id}/approve",
                                body={"selected_rank": opp["rank"]},
                                timeout=600,
                            )
                            if data.get("status") == "error":
                                st.error(data.get("error"))
                            else:
                                st.session_state.approved_opportunity = data.get("approved_opportunity")
                                st.session_state.generated_notebooks = data.get("notebooks", [])
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")


# ── Step 3: Code Review ───────────────────────────────────────────────────────
elif current == 2 and not st.session_state.bundle_written:
    opp = st.session_state.approved_opportunity or {}
    st.header(f"Step 3 — Review Generated Code: {opp.get('use_case', '')}")
    st.caption(f"Run ID: `{st.session_state.run_id}`")
    st.info("Review the generated notebooks. Click **Approve & Write Bundle** to save to `bundles/`.")

    for nb in st.session_state.generated_notebooks:
        with st.expander(f"📄 {nb['filename']}  —  `{nb['filepath']}`", expanded=False):
            st.code(nb["content"], language="python")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("✅ Approve & Write Bundle", type="primary"):
            with st.spinner("Writing notebooks and job YAMLs to bundles/…"):
                try:
                    data = api_post(
                        f"/runs/{st.session_state.run_id}/approve-code",
                        body={"action": "approve"},
                    )
                    if data.get("status") == "completed":
                        st.session_state.bundle_written = True
                        st.rerun()
                    else:
                        st.error(data.get("error", "Bundle write failed"))
                except Exception as e:
                    st.error(f"Error: {e}")
    with col2:
        st.caption("Files will be written to `bundles/src/{use_case}/` and `bundles/resources/jobs/`")


# ── Step 4: Done ──────────────────────────────────────────────────────────────
elif current == 3:
    opp = st.session_state.approved_opportunity or {}
    st.header("✅ Bundle Ready")
    st.success(f"**{opp.get('use_case', 'ML Use Case')}** — notebooks and jobs written to `bundles/`")

    with st.container(border=True):
        st.subheader("Generated files")
        slug = opp.get("use_case", "").lower().replace(" ", "_")
        files = [
            f"bundles/src/{slug}/01_feature_engineering.py",
            f"bundles/src/{slug}/02_training.py",
            f"bundles/src/{slug}/03_batch_inference.py",
            f"bundles/resources/jobs/{slug}_feature_pipeline_job.yml",
            f"bundles/resources/jobs/{slug}_training_job.yml",
            f"bundles/resources/jobs/{slug}_batch_inference_job.yml",
        ]
        for f in files:
            st.code(f, language=None)

    with st.container(border=True):
        st.subheader("Next steps")
        st.code("""# Install Databricks CLI if needed
pip install databricks-cli

# Deploy to dev workspace
databricks bundle deploy --target dev

# Run the feature pipeline first
databricks bundle run {slug}_feature_pipeline_job --target dev

# Then run training
databricks bundle run {slug}_training_job --target dev

# Then batch inference
databricks bundle run {slug}_batch_inference_job --target dev""".format(slug=slug), language="bash")

    if st.button("🔄 Start a new run"):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()
