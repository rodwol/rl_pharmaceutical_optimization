"""
POST   /api/orders              submit order for manager approval
GET    /api/orders              list orders (filterable by status)
GET    /api/orders/{id}         get single order
PATCH  /api/orders/{id}/approve approve a pending order
PATCH  /api/orders/{id}/reject  reject a pending order
DELETE /api/orders/{id}         delete an order (admin only)

MVP: orders stored in-memory. Replace _store with SQLAlchemy
session (see api/models/database.py) for production deployment.
"""

from datetime import date
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

# ── In-memory store (MVP) ─────────────────────────────────────────────────
_store: List[dict] = []
_next_id: int = 1


# ── Schemas ───────────────────────────────────────────────────────────────

class OrderSubmitRequest(BaseModel):
    medication_id: str      = Field(..., example="M003")
    medication_name: str    = Field(..., example="Amoxicillin 500mg")
    order_qty: float        = Field(..., gt=0, example=1162.0,
                                    description="Units to order")
    dqn_action: int         = Field(..., ge=0, le=3,
                                    description="DQN action that generated this recommendation")
    stockout_risk: str      = Field(..., example="high",
                                    description="high / medium / low")
    days_of_cover: float    = Field(..., example=5.1,
                                    description="Days of cover at time of recommendation")
    submitted_by: str       = Field(..., example="tech_demo")
    notes: Optional[str]    = Field(None, example="DQN recommended large batch; stock critically low")

    model_config = {"json_schema_extra": {"examples": [{
        "medication_id": "M003",
        "medication_name": "Amoxicillin 500mg",
        "order_qty": 1162.0,
        "dqn_action": 3,
        "stockout_risk": "high",
        "days_of_cover": 5.1,
        "submitted_by": "tech_demo",
        "notes": "DQN recommended large batch; stock critically low",
    }]}}


class OrderResponse(BaseModel):
    order_id: int
    medication_id: str
    medication_name: str
    order_qty: float
    dqn_action: int
    stockout_risk: str
    days_of_cover: float
    submitted_by: str
    status: str
    submitted_date: str
    approved_by: Optional[str]  = None
    approved_date: Optional[str] = None
    notes: Optional[str]        = None


class ApproveRequest(BaseModel):
    approved_by: str         = Field("manager_demo", example="manager_demo")
    adjusted_qty: Optional[float] = Field(None, example=1200.0,
        description="Override order quantity if manager adjusts it; "
                    "omit to keep DQN-recommended quantity")


# ── Helpers ───────────────────────────────────────────────────────────────

def _find(order_id: int) -> dict:
    for o in _store:
        if o["order_id"] == order_id:
            return o
    raise HTTPException(status_code=404, detail=f"Order {order_id} not found")


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post(
    "/orders",
    response_model=OrderResponse,
    status_code=201,
    summary="Submit an order for Pharmacy Manager approval",
)
def submit_order(req: OrderSubmitRequest):
    global _next_id
    order = {
        "order_id":      _next_id,
        "medication_id": req.medication_id,
        "medication_name": req.medication_name,
        "order_qty":     req.order_qty,
        "dqn_action":    req.dqn_action,
        "stockout_risk": req.stockout_risk,
        "days_of_cover": req.days_of_cover,
        "submitted_by":  req.submitted_by,
        "status":        "pending",
        "submitted_date": str(date.today()),
        "approved_by":   None,
        "approved_date": None,
        "notes":         req.notes,
    }
    _store.append(order)
    _next_id += 1
    return OrderResponse(**order)


@router.get(
    "/orders",
    response_model=List[OrderResponse],
    summary="List all orders",
    description="Returns all orders. Filter by status using ?status=pending|approved|rejected",
)
def list_orders(
    status: Optional[str] = Query(None, example="pending",
                                   description="Filter: pending / approved / rejected"),
):
    results = _store if status is None else [o for o in _store if o["status"] == status]
    return [OrderResponse(**o) for o in results]


@router.get(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Get a single order by ID",
)
def get_order(order_id: int):
    return OrderResponse(**_find(order_id))


@router.patch(
    "/orders/{order_id}/approve",
    response_model=OrderResponse,
    summary="Approve a pending order (Pharmacy Manager)",
    description=(
        "Approves the order and optionally adjusts the quantity. "
        "Only pending orders can be approved."
    ),
)
def approve_order(order_id: int, body: ApproveRequest):
    order = _find(order_id)
    if order["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Order {order_id} is already {order['status']} — "
                   f"only pending orders can be approved."
        )
    order["status"]        = "approved"
    order["approved_by"]   = body.approved_by
    order["approved_date"] = str(date.today())
    if body.adjusted_qty is not None:
        order["order_qty"] = body.adjusted_qty
        if order["notes"]:
            order["notes"] += f" | Manager adjusted qty to {body.adjusted_qty}"
        else:
            order["notes"] = f"Manager adjusted qty to {body.adjusted_qty}"
    return OrderResponse(**order)


@router.patch(
    "/orders/{order_id}/reject",
    response_model=OrderResponse,
    summary="Reject a pending order (Pharmacy Manager)",
)
def reject_order(order_id: int, body: ApproveRequest):
    order = _find(order_id)
    if order["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Order {order_id} is already {order['status']}."
        )
    order["status"]        = "rejected"
    order["approved_by"]   = body.approved_by
    order["approved_date"] = str(date.today())
    return OrderResponse(**order)


@router.delete(
    "/orders/{order_id}",
    summary="Delete an order record",
    status_code=204,
)
def delete_order(order_id: int):
    global _store
    order = _find(order_id)
    _store = [o for o in _store if o["order_id"] != order_id]
    return None
