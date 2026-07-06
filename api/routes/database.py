"""
SQLAlchemy ORM models + PostgreSQL session setup for production deployment.

In the MVP, routes use in-memory storage (see routes/orders.py).
Swap the in-memory _store for DB session calls when deploying with Docker
Compose (docker-compose.yml defines the PostgreSQL service).

Usage:
    from api.models.database import get_db, Order, StockCount
    
    @router.post("/orders")
    def submit(req, db: Session = Depends(get_db)):
        order = Order(**req.dict())
        db.add(order); db.commit(); db.refresh(order)
        return order
"""

import os
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    Column, Integer, String, Float, Date, Boolean, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
    raise RuntimeError("DATABASE_URL not found.")


engine       = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class Order(Base):
    __tablename__ = "orders"
    id             = Column(Integer, primary_key=True, index=True)
    medication_id  = Column(String, nullable=False, index=True)
    medication_name= Column(String, nullable=False)
    order_qty      = Column(Float, nullable=False)
    dqn_action     = Column(Integer, nullable=False)
    stockout_risk  = Column(String, nullable=False)
    days_of_cover  = Column(Float, nullable=False)
    submitted_by   = Column(String, nullable=False)
    status         = Column(String, default="pending")
    submitted_date = Column(Date)
    approved_by    = Column(String, nullable=True)
    approved_date  = Column(Date, nullable=True)
    notes          = Column(String, nullable=True)


class StockCount(Base):
    __tablename__ = "stock_counts"
    id             = Column(Integer, primary_key=True, index=True)
    medication_id  = Column(String, nullable=False, index=True)
    count_date     = Column(Date, nullable=False)
    stock_on_hand  = Column(Float, nullable=False)
    submitted_by   = Column(String, nullable=False)


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"
    id              = Column(Integer, primary_key=True, index=True)
    medication_id   = Column(String, nullable=False, index=True)
    action          = Column(Integer, nullable=False)
    recommended_qty = Column(Float, nullable=False)
    stockout_risk   = Column(String, nullable=False)
    days_of_cover   = Column(Float, nullable=False)
    regime_belief_stable     = Column(Float)
    regime_belief_surge      = Column(Float)
    regime_belief_disruption = Column(Float)
    requested_by    = Column(String)
    created_date    = Column(Date)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call on application startup in production."""
    Base.metadata.create_all(bind=engine)