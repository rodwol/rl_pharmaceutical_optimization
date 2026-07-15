"""
Accepts current inventory state for a single medicine, builds the 8-dim
observation vector matching environment.py, runs the frozen DQN model,
and returns the recommended order action with full transparency.
"""

import os
import sys
import numpy as np
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from api.models.database import get_db, RecommendationLog

router = APIRouter()

# ── Module-level state, populated once by load_models() at app startup ────
_state = {"model": None, "inferrer": None, "load_error": None}


def load_models():
    """
    Called once from api/main.py's @app.on_event("startup") handler.
    Loads the DQN model and HMM inferrer into module-level state so every
    request handler can access them instantly without any import or I/O cost.
    """
    try:
        from stable_baselines3 import DQN
        candidates = [
            os.path.join(project_root, "rl", "models", "dqn_agent.zip"),
            os.path.join(project_root, "models", "dqn_agent.zip"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(
                "Model not found in rl/models/ or models/. Run rl/train.py first."
            )
        _state["model"] = DQN.load(path)
    except Exception as e:
        _state["load_error"] = f"DQN model unavailable: {e}"

    try:
        sys.path.insert(0, os.path.join(project_root, "rl"))
        from hmm_demand import RegimeBeliefInferrer
        hmm_candidates = [
            os.path.join(project_root, "rl", "models", "pretrained_hmm.pkl"),
            os.path.join(project_root, "models", "pretrained_hmm.pkl"),
        ]
        hmm_path = next((p for p in hmm_candidates if os.path.exists(p)), hmm_candidates[0])
        _state["inferrer"] = RegimeBeliefInferrer(load_path=hmm_path)
    except Exception:
        _state["inferrer"] = None   # falls back to prior [0.8, 0.15, 0.05]


def _get_model():
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=_state["load_error"] or "Model not loaded yet.",
        )
    return _state["model"]


def _get_inferrer():
    return _state["inferrer"]


# Constants

ACTION_LABELS = {
    0: "No order needed",
    1: "Order SMALL batch (~7 days supply)",
    2: "Order MEDIUM batch (~21 days supply)",
    3: "Order LARGE batch (~45 days supply)",
}

ACTION_URGENCY = {
    0: "routine",
    1: "low",
    2: "moderate",
    3: "high",
}


# Pydantic schemas 
class InventoryStateRequest(BaseModel):
    medication_id: str = Field(..., example="M003",
        description="ENLM medicine ID (M001–M015)")
    medication_name: str = Field(..., example="Amoxicillin 500mg")
    stock_on_hand: float = Field(..., ge=0, example=420.0,
        description="Current units physically in stock")
    base_daily_demand: float = Field(..., gt=0, example=83.0,
        description="Average daily consumption (stable regime, weekday)")
    days_since_last_order: int = Field(..., ge=0, example=8,
        description="Days since the last order was placed")
    pending_order_qty: float = Field(0.0, ge=0, example=0.0,
        description="Units already on order (in transit)")
    demand_history_7d: List[float] = Field(
        ..., min_length=7, max_length=7,
        example=[78, 85, 72, 91, 80, 45, 22],
        description="Daily demand for the past 7 days (most recent last)")
    days_to_expiry: int = Field(365, ge=0, example=280,
        description="Days until the nearest batch expires")

    model_config = {"json_schema_extra": {"examples": [{
        "medication_id": "M003",
        "medication_name": "Amoxicillin 500mg",
        "stock_on_hand": 420.0,
        "base_daily_demand": 83.0,
        "days_since_last_order": 8,
        "pending_order_qty": 0.0,
        "demand_history_7d": [78, 85, 72, 91, 80, 45, 22],
        "days_to_expiry": 280,
    }]}}


class RecommendationResponse(BaseModel):
    medication_id: str
    medication_name: str
    action: int = Field(...,
        description="DQN action: 0=no order, 1=small, 2=medium, 3=large")
    action_label: str
    urgency: str = Field(...,
        description="routine / low / moderate / high")
    recommended_order_qty: float = Field(...,
        description="Recommended units to order")
    days_of_cover_current: float = Field(...,
        description="Days of cover from stock on hand only")
    days_of_cover_with_pipeline: float = Field(...,
        description="Days of cover including pending orders")
    stockout_risk: str = Field(...,
        description="high (<7d cover) / medium (<14d) / low (>=14d)")
    regime_belief: List[float] = Field(...,
        description="[P(stable), P(surge), P(disruption)] from HMM inferrer")
    observation_vector: List[float] = Field(...,
        description="Full 8-dim state vector fed to DQN (for auditability)")
    disclaimer: str = Field(
        default="This recommendation requires Pharmacy Manager approval "
                "before an order is submitted. RxGuard is a decision-support "
                "tool, not an automated ordering system.",
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_observation(req: InventoryStateRequest,
                        belief: np.ndarray) -> np.ndarray:
    base = max(req.base_daily_demand, 1e-6)
    norm_stock         = req.stock_on_hand / (base * 60)
    norm_days_order    = req.days_since_last_order / 60.0
    norm_pending       = req.pending_order_qty / (base * 60)
    norm_demand_signal = np.mean(req.demand_history_7d) / base
    norm_expiry        = req.days_to_expiry / 365.0

    obs = np.array([
        norm_stock, norm_days_order, norm_pending, norm_demand_signal,
        belief[0], belief[1], belief[2], norm_expiry,
    ], dtype=np.float32)
    return np.clip(obs, 0.0, 10.0)


def _stockout_risk(days_cover: float) -> str:
    if days_cover < 14:   return "high"
    if days_cover < 30:  return "medium"
    return "low"


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get DQN replenishment recommendation",
    description=(
        "Accepts the current inventory state for a single medicine and returns "
        "recommendation.\n\n"
        "The recommendation is generated by a Deep Q-Network agent trained on a "
        "literature-calibrated synthetic dataset of Eritrean district hospital "
        "pharmacy demand patterns. "
        "**All recommendations require Pharmacy Manager approval before "
        "submission to the district medical store.**"
    ),
)
def recommend(req: InventoryStateRequest, db: Session = Depends(get_db)):
    model    = _get_model()
    inferrer = _get_inferrer()

    # Get HMM belief (falls back to prior if inferrer unavailable)
    if inferrer is not None:
        belief = inferrer.belief(req.demand_history_7d, t=req.days_since_last_order)
    else:
        belief = np.array([0.80, 0.15, 0.05], dtype=np.float32)

    obs    = _build_observation(req, belief)
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # Order quantity — matches environment.py action map
    base      = req.base_daily_demand
    qty_map   = {0: 0.0, 1: base * 7, 2: base * 21, 3: base * 45}
    order_qty = round(qty_map[action])

    days_cover          = req.stock_on_hand / max(base, 1e-6)
    days_cover_pipeline = (req.stock_on_hand + req.pending_order_qty) / max(base, 1e-6)

    # Log the recommendation for audit/analysis (production DB persistence)
    try:
        db.add(RecommendationLog(
            medication_id=req.medication_id,
            action=action,
            recommended_qty=order_qty,
            stockout_risk=_stockout_risk(days_cover_pipeline),
            days_of_cover=round(days_cover, 1),
            regime_belief_stable=float(belief[0]),
            regime_belief_surge=float(belief[1]),
            regime_belief_disruption=float(belief[2]),
            created_at=datetime.utcnow(),
        ))
        db.commit()
    except Exception:
        db.rollback()   # logging failure should never break the recommendation

    return RecommendationResponse(
        medication_id=req.medication_id,
        medication_name=req.medication_name,
        action=action,
        action_label=ACTION_LABELS[action],
        urgency=ACTION_URGENCY[action],
        recommended_order_qty=order_qty,
        days_of_cover_current=round(days_cover, 1),
        days_of_cover_with_pipeline=round(days_cover_pipeline, 1),
        stockout_risk=_stockout_risk(days_cover_pipeline),
        regime_belief=belief.tolist(),
        observation_vector=obs.tolist(),
    )


@router.get(
    "/recommend/health",
    summary="Check model and HMM availability",
    tags=["Health"],
)
def recommend_health():
    status = {}
    try:
        _get_model()
        status["dqn_model"] = "loaded"
    except Exception as e:
        status["dqn_model"] = f"unavailable: {e}"

    inferrer = _get_inferrer()
    status["hmm_inferrer"] = "loaded" if inferrer is not None else "unavailable (using prior)"
    status["overall"] = "ok" if status["dqn_model"] == "loaded" else "degraded"
    return status