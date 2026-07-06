"""
Inventory snapshot endpoints — used by the Streamlit dashboard to
populate the Reports page without re-running the DQN model.

GET /api/inventory          current stock snapshot (all medicines)
GET /api/inventory/{id}     single medicine state
GET /api/medicines          static medicine catalogue (from ENLM 2015)
"""

import os
import sys
import numpy as np
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

router = APIRouter()

# ── Medicine catalogue (from medications.csv / ENLM 2015) ─────────────────
MEDICINES = [
    {"medication_id": "M001", "name": "Artemether-Lumefantrine 20/120mg", "category": "Antimalarial",  "unit": "tablet",  "unit_cost_usd": 0.12, "shelf_life_days": 730},
    {"medication_id": "M002", "name": "Quinine Sulfate 300mg",            "category": "Antimalarial",  "unit": "tablet",  "unit_cost_usd": 0.08, "shelf_life_days": 730},
    {"medication_id": "M003", "name": "Amoxicillin 500mg",                "category": "Antibiotic",    "unit": "capsule", "unit_cost_usd": 0.04, "shelf_life_days": 730},
    {"medication_id": "M004", "name": "Ceftriaxone 1g Injection",         "category": "Antibiotic",    "unit": "vial",    "unit_cost_usd": 0.90, "shelf_life_days": 730},
    {"medication_id": "M005", "name": "Metronidazole 400mg",              "category": "Antibiotic",    "unit": "tablet",  "unit_cost_usd": 0.03, "shelf_life_days": 730},
    {"medication_id": "M006", "name": "Ciprofloxacin 500mg",              "category": "Antibiotic",    "unit": "tablet",  "unit_cost_usd": 0.07, "shelf_life_days": 730},
    {"medication_id": "M007", "name": "Co-trimoxazole 480mg",             "category": "Antibiotic",    "unit": "tablet",  "unit_cost_usd": 0.02, "shelf_life_days": 730},
    {"medication_id": "M008", "name": "Oxytocin 10IU Injection",          "category": "MCH",           "unit": "ampoule", "unit_cost_usd": 0.45, "shelf_life_days": 365},
    {"medication_id": "M009", "name": "ORS Sachet",                       "category": "MCH",           "unit": "sachet",  "unit_cost_usd": 0.05, "shelf_life_days": 1460},
    {"medication_id": "M010", "name": "Ferrous Sulfate + Folic Acid",     "category": "MCH",           "unit": "tablet",  "unit_cost_usd": 0.02, "shelf_life_days": 730},
    {"medication_id": "M011", "name": "Paracetamol 500mg",                "category": "Analgesic",     "unit": "tablet",  "unit_cost_usd": 0.01, "shelf_life_days": 730},
    {"medication_id": "M012", "name": "Ibuprofen 400mg",                  "category": "Analgesic",     "unit": "tablet",  "unit_cost_usd": 0.02, "shelf_life_days": 730},
    {"medication_id": "M013", "name": "Metformin 500mg",                  "category": "Diabetes",      "unit": "tablet",  "unit_cost_usd": 0.03, "shelf_life_days": 730},
    {"medication_id": "M014", "name": "Amlodipine 5mg",                   "category": "Cardiovascular","unit": "tablet",  "unit_cost_usd": 0.04, "shelf_life_days": 730},
    {"medication_id": "M015", "name": "RDT Malaria Test Kit",             "category": "Diagnostics",   "unit": "kit",     "unit_cost_usd": 0.55, "shelf_life_days": 730},
]
MEDICINE_MAP = {m["medication_id"]: m for m in MEDICINES}

# Base daily demands from environment.py (stable regime, weekday avg)
BASE_DEMANDS = {
    "M001": 67.0,  "M002": 14.7,  "M003": 83.2,
    "M004": 18.1,  "M005": 41.8,  "M006": 33.6,
    "M007": 58.5,  "M008": 10.5,  "M009": 41.7,
    "M010": 62.5,  "M011": 111.1, "M012": 61.3,
    "M013": 41.0,  "M014": 35.9,  "M015": 84.0,
}

# Simulated current inventory (demo values — replace with DB query in production)
_rng = np.random.default_rng(99)
_INVENTORY = {
    mid: {
        "stock_on_hand": float(int(_rng.uniform(100, 2500))),
        "pending_order_qty": float(int(_rng.choice([0, 0, 0, int(_rng.uniform(200, 800))]))),
        "days_since_last_order": int(_rng.integers(0, 25)),
        "days_to_expiry": int(_rng.integers(60, 700)),
        "demand_history_7d": [
            float(int(_rng.poisson(BASE_DEMANDS[mid]))) for _ in range(7)
        ],
    }
    for mid in BASE_DEMANDS
}


# ── Schemas ───────────────────────────────────────────────────────────────

class MedicineInfo(BaseModel):
    medication_id: str
    name: str
    category: str
    unit: str
    unit_cost_usd: float
    shelf_life_days: int


class InventoryItem(BaseModel):
    medication_id: str
    name: str
    category: str
    unit: str
    base_daily_demand: float
    stock_on_hand: float
    pending_order_qty: float
    days_since_last_order: int
    days_to_expiry: int
    demand_history_7d: List[float]
    days_of_cover: float        = Field(..., description="Stock / base demand")
    days_of_cover_pipeline: float = Field(..., description="(Stock + pending) / base demand")
    stockout_risk: str          = Field(..., description="high / medium / low")


def _build_item(mid: str) -> InventoryItem:
    med  = MEDICINE_MAP[mid]
    inv  = _INVENTORY[mid]
    base = BASE_DEMANDS[mid]
    doc  = inv["stock_on_hand"] / max(base, 1e-6)
    docp = (inv["stock_on_hand"] + inv["pending_order_qty"]) / max(base, 1e-6)
    risk = "high" if docp < 7 else ("medium" if docp < 14 else "low")
    return InventoryItem(
        medication_id=mid,
        name=med["name"],
        category=med["category"],
        unit=med["unit"],
        base_daily_demand=base,
        stock_on_hand=inv["stock_on_hand"],
        pending_order_qty=inv["pending_order_qty"],
        days_since_last_order=inv["days_since_last_order"],
        days_to_expiry=inv["days_to_expiry"],
        demand_history_7d=inv["demand_history_7d"],
        days_of_cover=round(doc, 1),
        days_of_cover_pipeline=round(docp, 1),
        stockout_risk=risk,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get(
    "/medicines",
    response_model=List[MedicineInfo],
    summary="List all tracked medicines (ENLM 2015 subset)",
)
def list_medicines():
    return [MedicineInfo(**m) for m in MEDICINES]


@router.get(
    "/inventory",
    response_model=List[InventoryItem],
    summary="Current inventory snapshot (all 15 medicines)",
    description=(
        "Returns the current stock level, days of cover, pending orders, "
        "and stockout risk for all tracked medicines. "
        "In this MVP, values are simulated; in production these would be "
        "pulled from the PostgreSQL database populated by pharmacy technician "
        "stock count submissions."
    ),
)
def get_inventory():
    return [_build_item(mid) for mid in BASE_DEMANDS]


@router.get(
    "/inventory/{medication_id}",
    response_model=InventoryItem,
    summary="Get inventory state for a single medicine",
)
def get_inventory_item(medication_id: str):
    if medication_id not in _INVENTORY:
        raise HTTPException(
            status_code=404,
            detail=f"Medicine '{medication_id}' not found. "
                   f"Valid IDs: {list(BASE_DEMANDS.keys())}"
        )
    return _build_item(medication_id)


@router.patch(
    "/inventory/{medication_id}",
    response_model=InventoryItem,
    summary="Update stock count (submitted by pharmacy technician)",
    description="Records a physical stock count and updates the in-memory state.",
)
def update_stock_count(medication_id: str, stock_on_hand: float, submitted_by: str = "tech_demo"):
    if medication_id not in _INVENTORY:
        raise HTTPException(status_code=404,
                            detail=f"Medicine '{medication_id}' not found.")
    if stock_on_hand < 0:
        raise HTTPException(status_code=400,
                            detail="stock_on_hand cannot be negative.")
    _INVENTORY[medication_id]["stock_on_hand"] = stock_on_hand
    return _build_item(medication_id)
