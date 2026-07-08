"""
Run: streamlit run dashboard/Home.py
"""

import sys, os
_dashboard_dir = os.path.dirname(os.path.abspath(__file__))
_project_root  = os.path.dirname(_dashboard_dir)
for _p in [_dashboard_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from api_client import health_check, model_health, get_inventory

st.set_page_config(
    page_title="Pharmacy Inventory",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
    .risk-high   { color: #C62828; font-weight: 700; }
    .risk-medium { color: #EF6C00; font-weight: 700; }
    .risk-low    { color: #2E7D32; font-weight: 700; }
    .api-online  { color: #2E7D32; font-weight: 600; }
    .api-offline { color: #C62828; font-weight: 600; }
    div[data-testid="stSidebarContent"] { background-color: #F5F7FA; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Rebex")
    st.caption("RL-powered medicine stockout prevention")
    st.divider()

    role = st.radio("Signed in as:", ["Pharmacy Technician", "Pharmacy Manager"])
    st.session_state["role"] = role
    st.divider()

    # API status badge
    api_status = health_check()
    ml_status  = model_health()
    if api_status.get("status") == "ok":
        st.markdown("<span class='api-online'>● API online</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='api-offline'>● API offline</span>", unsafe_allow_html=True)
        st.caption("Start with: `uvicorn api.main:app --port 8000`")

    if ml_status.get("dqn_model") == "loaded":
        st.markdown("<span class='api-online'>● DQN model loaded</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='api-offline'>● DQN model unavailable</span>",
                    unsafe_allow_html=True)

    st.divider()
    st.caption("District Hospital Pharmacy")
    st.caption("Eritrea — simulated dataset")
    st.caption("Literature-calibrated: Halibet 2018 [ERI],\nAsmara 2019 [ERI]")

# ── Main content ───────────────────────────────────────────────────────────
st.title("Pharmacy Inventory Dashboard")
st.caption(
    "Reinforcement-learning decision-support for essential medicine "
    "replenishment in Eritrean district hospital pharmacies."
)

# Live KPI metrics from API
inventory = get_inventory()
if inventory:
    n_meds     = len(inventory)
    n_high     = sum(1 for i in inventory if i["stockout_risk"] == "high")
    n_medium   = sum(1 for i in inventory if i["stockout_risk"] == "medium")
    n_low      = sum(1 for i in inventory if i["stockout_risk"] == "low")
else:
    n_meds = n_high = n_medium = n_low = 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Medicines tracked",         n_meds)
c2.metric("High stockout risk",     n_high,   help="< 7 days of cover")
c3.metric("Medium stockout risk",   n_medium, help="7–14 days of cover")
c4.metric("Low stockout risk",      n_low,    help="> 14 days of cover")

st.divider()

# Navigation guide
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
Pharmacy Technician workflow
1. Go to **Stock Count** in the sidebar
2. Select a medicine and enter today's count
3. Click **Get Recommendation** — the DQN agent responds instantly
4. Send the recommendation to your manager for approval
""")
with col_b:
    st.markdown("""
Pharmacy Manager
1. Go to **Recommendations** in the sidebar
2. Review pending orders with risk level and days of cover
3. Adjust quantity if needed, then Approve or Reject
4. View performance trends in **Reports**
""")

st.info(
    "All data shown is simulated using a literature-calibrated synthetic dataset. "
    "No real patient or hospital-identifying information is used. "
    "Recommendations are decision-support only — all orders require manager approval."
)