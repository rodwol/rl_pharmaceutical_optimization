"""
Review, adjust, and approve/reject pending DQN-generated orders.
Reads from and writes to the FastAPI orders endpoint.
"""

import sys, os, datetime
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import client as api

st.set_page_config(page_title="Recommendations | Rebex", layout="wide")

st.title("Pending Recommendations")
st.caption("Review and approve replenishment recommendations submitted by pharmacy technicians.")

role = st.session_state.get("role", "Pharmacy Technician")
if role != "Pharmacy Manager":
    st.warning(
        "This page is for the **Pharmacy Manager** role. "
        "Switch roles in the sidebar to approve orders."
    )

# ── Pending orders from API ────────────────────────────────────────────────
pending = api.get_orders(status="pending")

if not pending:
    st.info(
        "No pending recommendations. "
        "Recommendations submitted from the **Stock Count** page will appear here."
    )
else:
    st.markdown(f"**{len(pending)} order(s) awaiting approval**")
    st.divider()

    for rec in pending:
        risk_icon = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(
            rec.get("stockout_risk", "low"), "⚪")

        with st.container(border=True):
            h1, h2, h3 = st.columns([4, 2, 2])
            with h1:
                st.markdown(f"**{rec['medication_name']}**  "
                             f"(`{rec['medication_id']}`)")
                st.caption(f"Submitted by **{rec['submitted_by']}** "
                            f"on {rec['submitted_date']}")
                if rec.get("notes"):
                    st.caption(f"Note: {rec['notes']}")
            with h2:
                st.markdown(f"{risk_icon} Stockout risk: **{rec.get('stockout_risk','—').upper()}**")
                st.write(f"Days of cover: **{rec.get('days_of_cover', '—')}d**")
            with h3:
                dqn_labels = {0:"No order", 1:"Small", 2:"Medium", 3:"Large"}
                st.write(f"DQN action: **{dqn_labels.get(rec.get('dqn_action',0), '—')}**")
                st.write(f"Recommended qty: **{rec['order_qty']:,.0f} units**")

            st.divider()
            ac1, ac2, ac3 = st.columns([3, 1, 1])
            with ac1:
                adj_qty = st.number_input(
                    "Final order quantity (adjust if needed)",
                    min_value=0,
                    value=int(rec["order_qty"]),
                    step=10,
                    key=f"qty_{rec['order_id']}",
                )
            with ac2:
                if st.button(" Approve", key=f"approve_{rec['order_id']}",
                              type="primary", use_container_width=True,
                              disabled=(role != "Pharmacy Manager")):
                    result = api.approve_order(
                        order_id=rec["order_id"],
                        approved_by="manager_demo",
                        adjusted_qty=float(adj_qty) if adj_qty != rec["order_qty"] else None,
                    )
                    if result:
                        st.success(f"Order #{rec['order_id']} approved.")
                        st.rerun()
                    else:
                        st.error("Approval failed. Check API connection.")
            with ac3:
                if st.button(" Reject", key=f"reject_{rec['order_id']}",
                              use_container_width=True,
                              disabled=(role != "Pharmacy Manager")):
                    result = api.reject_order(rec["order_id"], approved_by="manager_demo")
                    if result:
                        st.info(f"Order #{rec['order_id']} rejected.")
                        st.rerun()
                    else:
                        st.error("Rejection failed. Check API connection.")

# ── Order history ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Order history")

tab_approved, tab_rejected, tab_all = st.tabs(["Approved", "Rejected", "All orders"])

with tab_approved:
    approved = api.get_orders(status="approved")
    if approved:
        st.dataframe(pd.DataFrame(approved), use_container_width=True, hide_index=True)
    else:
        st.caption("No approved orders yet.")

with tab_rejected:
    rejected = api.get_orders(status="rejected")
    if rejected:
        st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)
    else:
        st.caption("No rejected orders yet.")

with tab_all:
    all_orders = api.get_orders()
    if all_orders:
        st.dataframe(pd.DataFrame(all_orders), use_container_width=True, hide_index=True)
    else:
        st.caption("No orders yet.")