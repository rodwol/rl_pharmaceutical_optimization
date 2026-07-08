import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stable_baselines3 import DQN
from rl.environment import PharmacyInventoryEnv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dqn_agent.zip")
MEDS_PATH = os.path.join(os.path.dirname(__file__), "..", "medications.csv")

ACTION_LABELS = {
    0: "No order needed",
    1: "Order SMALL batch (~14 days supply)",
    2: "Order MEDIUM batch (~30 days supply)",
    3: "Order LARGE batch (~60 days supply)",
}

ACTION_RISK_COLOR = {0: "#2E7D32", 1: "#F9A825", 2: "#EF6C00", 3: "#C62828"}


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return DQN.load(MODEL_PATH)
    return None


@st.cache_data
def load_medications():
    if os.path.exists(MEDS_PATH):
        return pd.read_csv(MEDS_PATH)
    # Fallback minimal list if medications.csv isn't found in this environment
    return pd.DataFrame([
        {"medication_id": "M001", "name": "Artemether-Lumefantrine 20/120mg", "category": "Antimalarial", "unit": "tablet"},
        {"medication_id": "M003", "name": "Amoxicillin 500mg", "category": "Antibiotic", "unit": "capsule"},
        {"medication_id": "M011", "name": "Paracetamol 500mg", "category": "Analgesic", "unit": "tablet"},
    ])


def init_session_inventory():
    """Initializes a simple in-memory inventory state for the demo session.
    Note: per artifact storage constraints, this dashboard runs as a Python
    (not browser-artifact) Streamlit app, so normal Python session_state is
    appropriate here -- this is not subject to the no-localStorage rule that
    applies to in-browser HTML/React artifacts."""
    if "inventory" not in st.session_state:
        meds = load_medications()
        rng = np.random.default_rng(7)
        st.session_state.inventory = {
            row["medication_id"]: {
                "name": row["name"],
                "category": row["category"],
                "unit": row.get("unit", "unit"),
                "stock_on_hand": int(rng.uniform(200, 900)),
                "base_daily_demand": float(rng.uniform(20, 90)),
                "days_since_last_order": int(rng.integers(0, 20)),
                "pending_qty": 0,
                "demand_history_7d": [float(rng.uniform(20, 90)) for _ in range(7)],
                "regime_belief": [0.8, 0.15, 0.05],
                "days_to_expiry": int(rng.integers(60, 700)),
            }
            for _, row in meds.iterrows()
        }
    if "pending_recommendations" not in st.session_state:
        st.session_state.pending_recommendations = []
    if "order_history" not in st.session_state:
        st.session_state.order_history = []


def build_observation(item: dict) -> np.ndarray:
    """Builds the 8-dim observation vector matching environment.py's
    _get_obs(), from a dashboard inventory item dict."""
    base_demand = max(item["base_daily_demand"], 1e-6)
    norm_stock = item["stock_on_hand"] / (base_demand * 60)
    norm_days_since_order = item["days_since_last_order"] / 60.0
    norm_pending = item["pending_qty"] / (base_demand * 60)
    norm_demand_signal = np.mean(item["demand_history_7d"][-7:]) / base_demand
    belief = item["regime_belief"]
    norm_expiry = item["days_to_expiry"] / 365.0

    obs = np.array([
        norm_stock, norm_days_since_order, norm_pending, norm_demand_signal,
        belief[0], belief[1], belief[2], norm_expiry,
    ], dtype=np.float32)
    return np.clip(obs, 0.0, 10.0)


def get_recommendation(item: dict, model) -> dict:
    """Runs the DQN model on an inventory item's current state and returns
    a structured recommendation."""
    if model is None:
        return {"action": 0, "label": "Model not available (demo fallback: no order)",
                "order_qty": 0, "risk": "unknown"}

    obs = build_observation(item)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    base_demand = item["base_daily_demand"]
    qty_map = {0: 0, 1: base_demand * 7, 2: base_demand * 21, 3: base_demand * 45}
    order_qty = round(qty_map[action])

    days_of_cover = item["stock_on_hand"] / max(base_demand, 1e-6)
    if days_of_cover < 7:
        risk = "high"
    elif days_of_cover < 14:
        risk = "medium"
    else:
        risk = "low"

    return {
        "action": action,
        "label": ACTION_LABELS[action],
        "order_qty": order_qty,
        "risk": risk,
        "days_of_cover": round(days_of_cover, 1),
    }