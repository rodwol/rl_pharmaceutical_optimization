import streamlit as st
from inventory_state import init_session_inventory, load_model

st.set_page_config(
    page_title="Pharmacy Inventory",
    page_icon="\U0001F48A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom styling: calm clinical palette, clear hierarchy ──
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .risk-high { color: #C62828; font-weight: 600; }
    .risk-medium { color: #EF6C00; font-weight: 600; }
    .risk-low { color: #2E7D32; font-weight: 600; }
    .stApp { background-color: #FAFBFC; }
</style>
""", unsafe_allow_html=True)

init_session_inventory()
model = load_model()

with st.sidebar:
    st.markdown("## \U0001F48A")
    st.caption("RL-powered medicine stockout prevention")
    st.divider()

    role = st.radio(
        "I am signed in as:",
        ["Pharmacy Technician", "Pharmacy Manager"],
        index=0,
    )
    st.divider()

    if model is None:
        st.warning("No trained model found at `models/dqn_agent.zip`. "
                   "Recommendations will show a fallback message until a model is trained.")
    else:
        st.success("DQN model loaded")

    st.caption("District Hospital Pharmacy — Eritrea (simulated)")
    st.caption("Data: literature-calibrated synthetic dataset. See README for sources.")

st.session_state["role"] = role

st.title("Pharmacy Inventory Dashboard")
st.caption(
    "A reinforcement-learning decision-support tool for essential medicine "
    "replenishment in district hospital pharmacies."
)

st.markdown("""
Use the navigation in the left sidebar (**Pages**) to move between:

- **Stock Count** — record today's stock and get an order recommendation
- **Recommendations** — review and approve pending orders (Manager)
- **Reports** — historical performance and DQN vs. EOQ comparison

This is a preliminary MVP for the Initial Software Demo. All data shown is
simulated; no real patient or hospital-identifying information is used.
""")

col1, col2, col3 = st.columns(3)
inv = st.session_state.inventory
n_meds = len(inv)
n_low_stock = sum(1 for v in inv.values() if v["stock_on_hand"] / max(v["base_daily_demand"], 1e-6) < 7)
n_pending = len(st.session_state.pending_recommendations)

col1.metric("Medicines tracked", n_meds)
col2.metric("At high stockout risk", n_low_stock, help="Less than 7 days of cover remaining")
col3.metric("Pending recommendations", n_pending)

st.info("Go to **Stock Count** in the sidebar to begin recording today's inventory.")