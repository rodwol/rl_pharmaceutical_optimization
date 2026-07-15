"""
Order management endpoints.

POST   /api/orders              submit order for manager approval
GET    /api/orders              list orders (filterable by status)
GET    /api/orders/{id}         get single order
PATCH  /api/orders/{id}/approve approve a pending order
PATCH  /api/orders/{id}/reject  reject a pending order
DELETE /api/orders/{id}         delete an order record
"""

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.models.database import get_db, Order

router = APIRouter()


# Schemas

class OrderSubmitRequest(BaseModel):
    medication_id: str    = Field(..., examples=["M003"])
    medication_name: str  = Field(..., examples=["Amoxicillin 500mg"])
    order_qty: float       = Field(..., gt=0, examples=[1162.0])
    dqn_action: int         = Field(..., ge=0, le=3)
    stockout_risk: str      = Field(..., examples=["high"])
    days_of_cover: float     = Field(..., examples=[5.1])
    submitted_by: str        = Field(..., examples=["tech_demo"])
    notes: Optional[str]      = Field(None, examples=["DQN recommended large batch"])


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
    submitted_date: date
    approved_by: Optional[str]  = None
    approved_date: Optional[date] = None
    notes: Optional[str]        = None

    model_config = {"from_attributes": True}   # allows .from_orm(sqlalchemy_obj)


class ApproveRequest(BaseModel):
    approved_by: str              = Field("manager_demo")
    adjusted_qty: Optional[float]  = Field(None, examples=[1200.0])


# Helpers

def _find(order_id: int, db: Session) -> Order:
    order = db.query(Order).filter(Order.order_id == order_id).first()
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


# Endpoints

@router.post("/orders", response_model=OrderResponse, status_code=201,
             summary="Submit an order for Pharmacy Manager approval")
def submit_order(req: OrderSubmitRequest, db: Session = Depends(get_db)):
    order = Order(
        medication_id=req.medication_id,
        medication_name=req.medication_name,
        order_qty=req.order_qty,
        dqn_action=req.dqn_action,
        stockout_risk=req.stockout_risk,
        days_of_cover=req.days_of_cover,
        submitted_by=req.submitted_by,
        status="pending",
        submitted_date=date.today(),
        notes=req.notes,
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


@router.get("/orders", response_model=List[OrderResponse],
            summary="List all orders",
            description="Filter by status using ?status=pending|approved|rejected")
def list_orders(
    status: Optional[str] = Query(None, examples=["pending"]),
    db: Session = Depends(get_db),
):
    q = db.query(Order)
    if status:
        q = q.filter(Order.status == status)
    return q.order_by(Order.order_id.desc()).all()


@router.get("/orders/{order_id}", response_model=OrderResponse,
            summary="Get a single order by ID")
def get_order(order_id: int, db: Session = Depends(get_db)):
    return _find(order_id, db)


@router.patch("/orders/{order_id}/approve", response_model=OrderResponse,
              summary="Approve a pending order (Pharmacy Manager)")
def approve_order(order_id: int, body: ApproveRequest, db: Session = Depends(get_db)):
    order = _find(order_id, db)
    if order.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Order {order_id} is already {order.status} — "
                   f"only pending orders can be approved.",
        )
    order.status = "approved"
    order.approved_by = body.approved_by
    order.approved_date = date.today()
    if body.adjusted_qty is not None:
        note = f"Manager adjusted qty to {body.adjusted_qty}"
        order.notes = f"{order.notes} | {note}" if order.notes else note
        order.order_qty = body.adjusted_qty
    db.commit()
    db.refresh(order)
    return order


@router.patch("/orders/{order_id}/reject", response_model=OrderResponse,
              summary="Reject a pending order (Pharmacy Manager)")
def reject_order(order_id: int, body: ApproveRequest, db: Session = Depends(get_db)):
    order = _find(order_id, db)
    if order.status != "pending":
        raise HTTPException(status_code=400,
                             detail=f"Order {order_id} is already {order.status}.")
    order.status = "rejected"
    order.approved_by = body.approved_by
    order.approved_date = date.today()
    db.commit()
    db.refresh(order)
    return order


@router.delete("/orders/{order_id}", status_code=204,
                summary="Delete an order record")
def delete_order(order_id: int, db: Session = Depends(get_db)):
    order = _find(order_id, db)
    db.delete(order)
    db.commit()
    return None