"""
inventory overview (live from API), DQN vs EOQ comparison,
and dataset validation summary.
"""

import sys, os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import client as api

st.set_page_config(page_title="Reports | Rebex", layout="wide")
st.title("Inventory Performance Reports")

tab1, tab2, tab3 = st.tabs([
    "Inventory Overview",
    "DQN vs. EOQ Baseline",
    "Dataset Validation",
])

# ── Tab 1: Live inventory from API ────────────────────────────────────────
with tab1:
    inventory = api.get_inventory()

    if not inventory:
        st.error("Cannot reach the API. Inventory data unavailable.")
    else:
        df = pd.DataFrame([{
            "Medicine":       i["name"],
            "Category":       i["category"],
            "Stock on hand":  i["stock_on_hand"],
            "Days of cover":  i["days_of_cover_pipeline"],
            "Pending qty":    i["pending_order_qty"],
            "Risk":           i["stockout_risk"].capitalize(),
        } for i in inventory])

        # KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("Total medicines",   len(df))
        k2.metric("High risk",
                   df[df["Risk"]=="High"].shape[0],
                   delta=None, help="< 7 days of cover")
        k3.metric("Avg days of cover",
                   f"{df['Days of cover'].mean():.1f}d")

        col_chart, col_pie = st.columns([3, 2])
        with col_chart:
            fig = px.bar(
                df.sort_values("Days of cover"),
                x="Medicine", y="Days of cover",
                color="Risk",
                color_discrete_map={"High":"#C62828","Medium":"#EF6C00","Low":"#2E7D32"},
                title="Days of cover by medicine",
                height=420,
            )
            fig.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig, use_container_width=True)

        with col_pie:
            risk_counts = df["Risk"].value_counts()
            fig2 = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={"High":"#C62828","Medium":"#EF6C00","Low":"#2E7D32"},
                title="Stockout risk distribution",
                height=420,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Full inventory table")
        st.dataframe(df, use_container_width=True, hide_index=True)

# ── Tab 2: DQN vs EOQ comparison ─────────────────────────────────────────
with tab2:
    st.subheader("DQN Agent vs. EOQ Baseline — Simulated Annual Performance")
    st.caption(
        "Evaluated over 10 held-out simulated years per medicine across "
        "all 15 ENLM medicines using the enriched environment "
        "(seasonal demand, fitted HMM, supplier stockouts, holding costs)."
    )

    # Pre-computed results from the most recent train.py run
    # Update these values after each training run
    results = {
        "DQN": {
            "mean_stockout_days": 29.5,
            "mean_service_level": 91.9,
            "mean_reward":        -3123.1,
        },
        "EOQ": {
            "mean_stockout_days": 267.0,
            "mean_service_level": 26.8,
            "mean_reward":        -2719.5,
        },
    }

    r1, r2, r3 = st.columns(3)
    r1.metric("Service level — DQN",
               f"{results['DQN']['mean_service_level']:.1f}%",
               delta=f"+{results['DQN']['mean_service_level']-results['EOQ']['mean_service_level']:.1f}pp vs EOQ")
    r2.metric("Stockout days/yr — DQN",
               f"{results['DQN']['mean_stockout_days']:.1f}",
               delta=f"{results['DQN']['mean_stockout_days']-results['EOQ']['mean_stockout_days']:.1f} vs EOQ",
               delta_color="inverse")
    r3.metric("Mean reward — DQN",
               f"{results['DQN']['mean_reward']:,.0f}",
               delta=f"{results['DQN']['mean_reward']-results['EOQ']['mean_reward']:,.0f} vs EOQ",
               help="Lower raw reward for DQN reflects proactive ordering costs. "
                    "Service level is the primary clinical metric.")

    fig = go.Figure()
    for agent, color in [("DQN","#4C72B0"), ("EOQ","#C44E52")]:
        fig.add_trace(go.Bar(
            name=agent,
            x=["Stockout days/yr", "Service level (%)"],
            y=[results[agent]["mean_stockout_days"],
               results[agent]["mean_service_level"]],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group", height=420,
        title="DQN vs. EOQ — Stockout days and Service level",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Interpreting the reward gap:** The DQN agent's raw reward is lower than "
        "EOQ's because it orders proactively (incurring holding and ordering costs) "
        "to maintain high availability. EOQ's better raw reward comes at the cost of "
        "267 stockout days per year — a clinically unacceptable outcome. "
        "Service level (91.9% vs 26.8%) is the primary performance metric for this "
        "problem domain."
    )

    with st.expander("About the evaluation methodology"):
        st.markdown("""
**Environment:** `PharmacyInventoryEnv` with the following features:
- Seasonal demand matching East Africa disease patterns (malaria peaks Apr/May, Nov)
- Day-of-week demand variation (weekends ~40% of weekday demand)
- HMM regime inference from observed demand history (stable/surge/disruption)
- Fitted HMM transition matrix from the validated synthetic dataset
- Supplier stockout behaviour: partial/delayed deliveries during disruption
- Fixed ordering cost K = 2.0 per order
- Daily holding cost h = 0.005 per unit
- Randomised initial inventory (3–30 days of cover)

**Baseline:** Economic Order Quantity (Q,R) policy with EOQ formula using the same
holding and ordering costs, reorder point at `mean_lead_time × base_demand + 7d safety stock`.

**Evaluation:** 10 held-out episodes per medicine × 15 medicines × 365 days.
""")

# ── Tab 3: Dataset Validation ─────────────────────────────────────────────
with tab3:
    st.subheader("Synthetic Dataset Validation Against Literature")
    st.caption(
        "The synthetic dataset was validated against Eritrean and regional "
        "Sub-Saharan African literature before being used to pre-train the HMM "
        "and train the DQN agent. Every parameter is tagged [ERI], [PROXY], or [ASSUM]."
    )

    validation_data = [
        ("Antibiotic point-in-time stockout rate",
         "15.1%", "12.5%", "Halibet 2018 [ERI]", "✓ PASS"),
        ("6-month stockout incidence",
         "76.7%", "60.0%", "Motta 2022 [PROXY, Ethiopia]", "~ BORDERLINE"),
        ("Mean stockout episode duration",
         "26.6 days", "38.8d (range 10–157d)", "Motta 2022 [PROXY, Ethiopia]", "~ BORDERLINE"),
        ("Procurement cycle length",
         "74.6 days (100% in 60–90d range)", "60–90 days", "Gubre 2025 [PROXY, Ethiopia]", "✓ PASS"),
        ("Lead time by regime (disruption > stable)",
         "Disruption 65.7d vs stable 16.6d", "Disruption longest", "Documented [ASSUM]", "✓ PASS"),
        ("Seasonal antimalarial peaks",
         "Peaks months 4, 11", "Apr/May and Nov", "East Africa malaria season", "✓ PASS"),
        ("Day-of-week demand pattern",
         "Weekend 25.5 vs weekday 60.9 units/day", "Weekends lower", "Clinic attendance pattern", "✓ PASS"),
        ("Antibiotic share of total demand",
         "30.9%", "53%", "Asmara 2019 [ERI]", "✗ FAIL (disclosed limitation)"),
    ]

    df_val = pd.DataFrame(validation_data, columns=[
        "Check", "Simulated", "Target", "Source", "Status"
    ])

    def colour_status(val):
        if "PASS" in val:   return "color: #2E7D32; font-weight: bold"
        if "FAIL" in val:   return "color: #C62828; font-weight: bold"
        if "BORDER" in val: return "color: #EF6C00; font-weight: bold"
        return ""

    st.dataframe(
        df_val.style.applymap(colour_status, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )

    st.warning(
        "This is disclosed in the methodology "
        "section and does not invalidate the RL training results."
    )