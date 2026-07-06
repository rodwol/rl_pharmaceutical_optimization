"""
Record today's stock count → DQN recommendation → send to manager.
Calls the FastAPI backend for both stock updates and recommendations.
"""

import sys, os, datetime
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import client as api

st.set_page_config(page_title="Stock Count | Rebex", layout="wide")

st.title("Daily Stock Count")
st.caption(f"Recording for {datetime.date.today().strftime('%A, %B %d, %Y')}")

# ── Load medicine list from API ────────────────────────────────────────────
medicines = api.get_medicines()
if not medicines:
    st.error("Cannot reach the API. Make sure `uvicorn api.main:app --port 8000` is running.")
    st.stop()

med_map = {m["medication_id"]: m for m in medicines}
med_options = {mid: f"{m['name']} ({m['category']})" for mid, m in med_map.items()}

selected_id = st.selectbox(
    "Select medicine",
    options=list(med_options.keys()),
    format_func=lambda mid: med_options[mid],
)
med = med_map[selected_id]

# ── Load current inventory state from API ─────────────────────────────────
inv_item = api.get_inventory_item(selected_id)
if not inv_item:
    st.error(f"Could not load inventory for {selected_id}.")
    st.stop()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Current inventory record")

    # Status badge
    risk = inv_item["stockout_risk"]
    risk_color = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(risk, "⚪")
    st.markdown(f"{risk_color} **Stockout risk: {risk.upper()}**")

    mc1, mc2 = st.columns(2)
    mc1.metric("Stock on hand",      f"{inv_item['stock_on_hand']:,.0f} {med['unit']}s")
    mc2.metric("Days of cover",      f"{inv_item['days_of_cover_pipeline']:.1f} days",
               help="Including pending orders in transit")

    mc3, mc4 = st.columns(2)
    mc3.metric("Pending order qty",  f"{inv_item['pending_order_qty']:,.0f} {med['unit']}s")
    mc4.metric("Days since order",   f"{inv_item['days_since_last_order']} days")

    st.write(f"**Average daily demand:** {inv_item['base_daily_demand']:.0f} {med['unit']}s/day")
    st.write(f"**7-day demand history:** {[int(x) for x in inv_item['demand_history_7d']]}")
    st.write(f"**Days to nearest expiry:** {inv_item['days_to_expiry']} days")

    st.divider()
    new_count = st.number_input(
        f"Enter today's physical count ({med['unit']}s)",
        min_value=0,
        value=int(inv_item["stock_on_hand"]),
        step=1,
        help="Count the actual units on the shelf right now.",
    )

    if st.button("📥 Submit count & get recommendation", type="primary",
                  use_container_width=True):
        with st.spinner("Calling DQN agent..."):
            # 1. Update stock count in API
            api.update_stock_count(selected_id, float(new_count))

            # 2. Get DQN recommendation
            rec = api.get_recommendation(
                medication_id=selected_id,
                medication_name=med["name"],
                stock_on_hand=float(new_count),
                base_daily_demand=inv_item["base_daily_demand"],
                days_since_last_order=inv_item["days_since_last_order"],
                pending_order_qty=inv_item["pending_order_qty"],
                demand_history_7d=inv_item["demand_history_7d"],
                days_to_expiry=inv_item["days_to_expiry"],
            )
        if rec:
            st.session_state["last_rec"] = {"med_id": selected_id, **rec}
            st.rerun()
        else:
            st.error("API returned no recommendation. Check that the DQN model is loaded.")

with col_right:
    st.subheader("DQN Recommendation")

    last_rec = st.session_state.get("last_rec")
    if last_rec and last_rec.get("med_id") == selected_id:
        rec = last_rec

        # Risk and urgency
        urgency_color = {"routine": "🟢", "low": "🟡",
                          "moderate": "🟠", "high": "🔴"}.get(rec.get("urgency"), "⚪")
        st.markdown(f"{urgency_color} **{rec['action_label']}**")
        st.markdown(f"Urgency: **{rec.get('urgency', '—').upper()}**")

        rc1, rc2 = st.columns(2)
        rc1.metric("Days of cover (now)",     f"{rec['days_of_cover_current']:.1f}d")
        rc2.metric("Days of cover (pipeline)",f"{rec['days_of_cover_with_pipeline']:.1f}d")

        if rec["recommended_order_qty"] > 0:
            st.metric("Recommended order quantity",
                       f"{rec['recommended_order_qty']:,.0f} {med['unit']}s")

        # HMM regime belief
        belief = rec.get("regime_belief", [0.8, 0.15, 0.05])
        with st.expander("HMM regime belief (from demand history)"):
            st.caption(
                "The Hidden Markov Model infers the current demand/supply regime "
                "from observed demand history. This belief is part of the DQN state."
            )
            b1, b2, b3 = st.columns(3)
            b1.metric("P(Stable)",      f"{belief[0]*100:.1f}%")
            b2.metric("P(Surge)",       f"{belief[1]*100:.1f}%")
            b3.metric("P(Disruption)",  f"{belief[2]*100:.1f}%")

        # DQN state vector (for transparency/auditability)
        with st.expander("DQN observation vector (8 features)"):
            obs = rec.get("observation_vector", [])
            labels = ["Stock (norm)", "Days since order (norm)", "Pending qty (norm)",
                      "7d demand signal", "P(Stable)", "P(Surge)", "P(Disruption)",
                      "Days to expiry (norm)"]
            for label, val in zip(labels, obs):
                st.write(f"`{label}`: {val:.4f}")

        st.divider()
        if rec["recommended_order_qty"] > 0:
            adj_qty = st.number_input(
                "Adjust quantity before sending (optional)",
                min_value=0,
                value=int(rec["recommended_order_qty"]),
                step=10,
            )
            if st.button(" Send to Pharmacy Manager for approval",
                          use_container_width=True):
                result = api.submit_order(
                    medication_id=selected_id,
                    medication_name=med["name"],
                    order_qty=float(adj_qty),
                    dqn_action=rec["action"],
                    stockout_risk=rec["stockout_risk"],
                    days_of_cover=rec["days_of_cover_current"],
                    submitted_by="tech_demo",
                    notes=f"DQN action {rec['action']}: {rec['action_label']}",
                )
                if result:
                    st.success(
                        f"Order #{result['order_id']} submitted for "
                        f"{rec['recommended_order_qty']:,.0f} {med['unit']}s "
                        f"of {med['name']}. Awaiting manager approval."
                    )
                else:
                    st.error("Failed to submit order. Check API connection.")
        else:
            st.info("No order needed — stock levels are adequate.")
    else:
        st.info("Submit a stock count on the left to see the DQN recommendation here.")

st.divider()
st.caption(
    "Recommendations are generated by a DQN reinforcement-learning agent trained on a "
    "This is a decision-support tool — all orders require Pharmacy Manager approval. "
)