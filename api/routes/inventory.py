"""
Inventory endpoints

GET   /api/medicines            static medicine catalogue (from DB)
GET   /api/inventory             current inventory snapshot (all medicines)
GET   /api/inventory/{id}        single medicine state
PATCH /api/inventory/{id}        submit stock count (updates DB + logs audit trail)
"""

import json
from datetime import date, datetime
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.models.database import get_db, Medicine, InventoryState, StockCount

router = APIRouter()


# Schemas

class MedicineInfo(BaseModel):
    medication_id: str
    name: str
    category: str
    unit: str
    unit_cost_usd: float
    shelf_life_days: int

    model_config = {"from_attributes": True}


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
    days_of_cover: float
    days_of_cover_pipeline: float
    stockout_risk: str


class StockCountRequest(BaseModel):
    stock_on_hand: float = Field(..., ge=0)
    submitted_by: str    = Field("tech_demo")


def _build_item(med: Medicine, inv: InventoryState) -> InventoryItem:
    base = max(med.base_daily_demand, 1e-6)
    doc  = inv.stock_on_hand / base
    docp = (inv.stock_on_hand + inv.pending_order_qty) / base
    risk = "high" if docp < 7 else ("medium" if docp < 14 else "low")
    history = json.loads(inv.demand_history_7d_json)

    return InventoryItem(
        medication_id=med.medication_id,
        name=med.name,
        category=med.category,
        unit=med.unit,
        base_daily_demand=med.base_daily_demand,
        stock_on_hand=inv.stock_on_hand,
        pending_order_qty=inv.pending_order_qty,
        days_since_last_order=inv.days_since_last_order,
        days_to_expiry=inv.days_to_expiry,
        demand_history_7d=history,
        days_of_cover=round(doc, 1),
        days_of_cover_pipeline=round(docp, 1),
        stockout_risk=risk,
    )


# Endpoints

@router.get("/medicines", response_model=List[MedicineInfo],
            summary="List all tracked medicines (ENLM 2015 subset)")
def list_medicines(db: Session = Depends(get_db)):
    return db.query(Medicine).order_by(Medicine.medication_id).all()


@router.get("/inventory", response_model=List[InventoryItem],
            summary="Current inventory snapshot (all 15 medicines)")
def get_inventory(db: Session = Depends(get_db)):
    meds = {m.medication_id: m for m in db.query(Medicine).all()}
    states = db.query(InventoryState).all()
    return [_build_item(meds[s.medication_id], s) for s in states
            if s.medication_id in meds]


@router.get("/inventory/{medication_id}", response_model=InventoryItem,
            summary="Get inventory state for a single medicine")
def get_inventory_item(medication_id: str, db: Session = Depends(get_db)):
    med = db.query(Medicine).filter(Medicine.medication_id == medication_id).first()
    inv = db.query(InventoryState).filter(InventoryState.medication_id == medication_id).first()
    if med is None or inv is None:
        raise HTTPException(status_code=404,
                             detail=f"Medicine '{medication_id}' not found.")
    return _build_item(med, inv)


@router.patch("/inventory/{medication_id}", response_model=InventoryItem,
              summary="Submit a stock count (pharmacy technician)",
              description=(
                  "Updates the live inventory state AND writes a permanent "
                  "audit record to the stock_counts table."
              ))
def update_stock_count(medication_id: str, body: StockCountRequest,
                        db: Session = Depends(get_db)):
    med = db.query(Medicine).filter(Medicine.medication_id == medication_id).first()
    inv = db.query(InventoryState).filter(InventoryState.medication_id == medication_id).first()
    if med is None or inv is None:
        raise HTTPException(status_code=404,
                             detail=f"Medicine '{medication_id}' not found.")

    # Update live state
    inv.stock_on_hand = body.stock_on_hand
    inv.last_updated  = datetime.utcnow()

    # Append to audit log
    db.add(StockCount(
        medication_id=medication_id,
        count_date=date.today(),
        stock_on_hand=body.stock_on_hand,
        submitted_by=body.submitted_by,
    ))
    db.commit()
    db.refresh(inv)
    return _build_item(med, inv)


@router.get("/inventory/{medication_id}/history",
            summary="Stock count audit history for a medicine")
def get_stock_history(medication_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(StockCount)
        .filter(StockCount.medication_id == medication_id)
        .order_by(StockCount.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": r.id,
            "count_date": r.count_date,
            "stock_on_hand": r.stock_on_hand,
            "submitted_by": r.submitted_by,
        }
        for r in records
    ]