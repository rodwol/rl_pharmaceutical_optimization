import os
import requests
import streamlit as st
from typing import Optional

API_BASE = os.getenv("Rebex_API_URL", "http://localhost:8000")
TIMEOUT  = 5   # seconds


def _get(path: str, params: dict = None) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        st.session_state["api_status"] = "online"
        return r.json()
    except Exception as e:
        st.session_state["api_status"] = "offline"
        return None


def _post(path: str, body: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=TIMEOUT)
        r.raise_for_status()
        st.session_state["api_status"] = "online"
        return r.json()
    except Exception as e:
        st.session_state["api_status"] = "offline"
        return None


def _patch(path: str, body: dict = None) -> Optional[dict]:
    try:
        r = requests.patch(f"{API_BASE}{path}", json=body or {}, timeout=TIMEOUT)
        r.raise_for_status()
        st.session_state["api_status"] = "online"
        return r.json()
    except Exception as e:
        st.session_state["api_status"] = "offline"
        return None


# Public API 

def health_check() -> dict:
    result = _get("/health")
    return result or {"status": "offline"}


def model_health() -> dict:
    result = _get("/api/recommend/health")
    return result or {"dqn_model": "unavailable", "hmm_inferrer": "unavailable", "overall": "offline"}


def get_medicines() -> list:
    result = _get("/api/medicines")
    return result or []


def get_inventory() -> list:
    result = _get("/api/inventory")
    return result or []


def get_inventory_item(medication_id: str) -> Optional[dict]:
    return _get(f"/api/inventory/{medication_id}")


def update_stock_count(medication_id: str, stock_on_hand: float,
                        submitted_by: str = "tech_demo") -> Optional[dict]:
    return _patch(
        f"/api/inventory/{medication_id}",
        {"stock_on_hand": stock_on_hand, "submitted_by": submitted_by},
    )


def get_recommendation(medication_id: str, medication_name: str,
                         stock_on_hand: float, base_daily_demand: float,
                         days_since_last_order: int, pending_order_qty: float,
                         demand_history_7d: list, days_to_expiry: int) -> Optional[dict]:
    return _post("/api/recommend", {
        "medication_id":        medication_id,
        "medication_name":      medication_name,
        "stock_on_hand":        stock_on_hand,
        "base_daily_demand":    base_daily_demand,
        "days_since_last_order": days_since_last_order,
        "pending_order_qty":    pending_order_qty,
        "demand_history_7d":    demand_history_7d,
        "days_to_expiry":       days_to_expiry,
    })


def submit_order(medication_id: str, medication_name: str, order_qty: float,
                  dqn_action: int, stockout_risk: str, days_of_cover: float,
                  submitted_by: str = "tech_demo",
                  notes: str = None) -> Optional[dict]:
    return _post("/api/orders", {
        "medication_id":   medication_id,
        "medication_name": medication_name,
        "order_qty":       order_qty,
        "dqn_action":      dqn_action,
        "stockout_risk":   stockout_risk,
        "days_of_cover":   days_of_cover,
        "submitted_by":    submitted_by,
        "notes":           notes,
    })


def get_orders(status: str = None) -> list:
    params = {"status": status} if status else None
    result = _get("/api/orders", params=params)
    return result or []


def approve_order(order_id: int, approved_by: str = "manager_demo",
                   adjusted_qty: float = None) -> Optional[dict]:
    body = {"approved_by": approved_by}
    if adjusted_qty is not None:
        body["adjusted_qty"] = adjusted_qty
    return _patch(f"/api/orders/{order_id}/approve", body)


def reject_order(order_id: int, approved_by: str = "manager_demo") -> Optional[dict]:
    return _patch(f"/api/orders/{order_id}/reject", {"approved_by": approved_by})

