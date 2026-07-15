"""
SQLAlchemy ORM models and PostgreSQL session management.

All routes now use these models via get_db() dependency injection —
data persists across API restarts and is queryable directly with SQL.

Environment variable DATABASE_URL controls the connection string.
Falls back to a sensible local default if not set.
"""

import os
from datetime import date, datetime
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, create_engine, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# pool_pre_ping avoids stale-connection errors after DB restarts/idling
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# MODELS

class Medicine(Base):
    """Static catalogue seeded once from the ENLM 2015 medicine list."""
    __tablename__ = "medicines"

    medication_id    = Column(String, primary_key=True)
    name              = Column(String, nullable=False)
    category          = Column(String, nullable=False)
    unit              = Column(String, nullable=False)
    unit_cost_usd     = Column(Float, nullable=False)
    shelf_life_days   = Column(Integer, nullable=False)
    base_daily_demand = Column(Float, nullable=False)


class InventoryState(Base):
    """Current live inventory state per medicine — one row per medicine,
    updated in place as stock counts and orders are processed."""
    __tablename__ = "inventory_state"

    medication_id          = Column(String, primary_key=True)
    stock_on_hand           = Column(Float, nullable=False, default=0.0)
    pending_order_qty        = Column(Float, nullable=False, default=0.0)
    days_since_last_order    = Column(Integer, nullable=False, default=0)
    days_to_expiry           = Column(Integer, nullable=False, default=365)
    demand_history_7d_json   = Column(String, nullable=False, default="[]")
    last_updated             = Column(DateTime, default=datetime.utcnow,
                                       onupdate=datetime.utcnow)


class StockCount(Base):
    """Audit log of every stock count submission by a technician."""
    __tablename__ = "stock_counts"

    id             = Column(Integer, primary_key=True, index=True, autoincrement=True)
    medication_id  = Column(String, nullable=False, index=True)
    count_date     = Column(Date, nullable=False, default=date.today)
    stock_on_hand  = Column(Float, nullable=False)
    submitted_by   = Column(String, nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)


class Order(Base):
    """Order lifecycle: pending -> approved/rejected."""
    __tablename__ = "orders"

    order_id         = Column(Integer, primary_key=True, index=True, autoincrement=True)
    medication_id    = Column(String, nullable=False, index=True)
    medication_name  = Column(String, nullable=False)
    order_qty        = Column(Float, nullable=False)
    dqn_action       = Column(Integer, nullable=False)
    stockout_risk    = Column(String, nullable=False)
    days_of_cover    = Column(Float, nullable=False)
    submitted_by     = Column(String, nullable=False)
    status           = Column(String, nullable=False, default="pending", index=True)
    submitted_date   = Column(Date, default=date.today)
    approved_by      = Column(String, nullable=True)
    approved_date    = Column(Date, nullable=True)
    notes            = Column(String, nullable=True)


class RecommendationLog(Base):
    """Audit log of every DQN recommendation served, for later analysis
    of model behaviour in production (drift detection, usage patterns)."""
    __tablename__ = "recommendation_logs"

    id                        = Column(Integer, primary_key=True, index=True, autoincrement=True)
    medication_id             = Column(String, nullable=False, index=True)
    action                    = Column(Integer, nullable=False)
    recommended_qty           = Column(Float, nullable=False)
    stockout_risk             = Column(String, nullable=False)
    days_of_cover             = Column(Float, nullable=False)
    regime_belief_stable      = Column(Float)
    regime_belief_surge       = Column(Float)
    regime_belief_disruption  = Column(Float)
    created_at                = Column(DateTime, default=datetime.utcnow)

# SESSION MANAGEMENT

def get_db():
    """FastAPI dependency — yields a session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist. Called on app startup."""
    Base.metadata.create_all(bind=engine)


def seed_medicines(db: Session):
    """Seed the medicines and inventory_state tables on first run.
    Safe to call repeatedly — skips medicines that already exist."""
    import json

    MEDICINES = [
        {"medication_id": "M001", "name": "Artemether-Lumefantrine 20/120mg", "category": "Antimalarial",  "unit": "tablet",  "unit_cost_usd": 0.12, "shelf_life_days": 730, "base_daily_demand": 67.0},
        {"medication_id": "M002", "name": "Quinine Sulfate 300mg",            "category": "Antimalarial",  "unit": "tablet",  "unit_cost_usd": 0.08, "shelf_life_days": 730, "base_daily_demand": 14.7},
        {"medication_id": "M003", "name": "Amoxicillin 500mg",                "category": "Antibiotic",    "unit": "capsule", "unit_cost_usd": 0.04, "shelf_life_days": 730, "base_daily_demand": 83.2},
        {"medication_id": "M004", "name": "Ceftriaxone 1g Injection",         "category": "Antibiotic",    "unit": "vial",    "unit_cost_usd": 0.90, "shelf_life_days": 730, "base_daily_demand": 18.1},
        {"medication_id": "M005", "name": "Metronidazole 400mg",              "category": "Antibiotic",    "unit": "tablet",  "unit_cost_usd": 0.03, "shelf_life_days": 730, "base_daily_demand": 41.8},
        {"medication_id": "M006", "name": "Ciprofloxacin 500mg",              "category": "Antibiotic",    "unit": "tablet",  "unit_cost_usd": 0.07, "shelf_life_days": 730, "base_daily_demand": 33.6},
        {"medication_id": "M007", "name": "Co-trimoxazole 480mg",             "category": "Antibiotic",    "unit": "tablet",  "unit_cost_usd": 0.02, "shelf_life_days": 730, "base_daily_demand": 58.5},
        {"medication_id": "M008", "name": "Oxytocin 10IU Injection",          "category": "MCH",           "unit": "ampoule", "unit_cost_usd": 0.45, "shelf_life_days": 365, "base_daily_demand": 10.5},
        {"medication_id": "M009", "name": "ORS Sachet",                       "category": "MCH",           "unit": "sachet",  "unit_cost_usd": 0.05, "shelf_life_days": 1460,"base_daily_demand": 41.7},
        {"medication_id": "M010", "name": "Ferrous Sulfate + Folic Acid",     "category": "MCH",           "unit": "tablet",  "unit_cost_usd": 0.02, "shelf_life_days": 730, "base_daily_demand": 62.5},
        {"medication_id": "M011", "name": "Paracetamol 500mg",                "category": "Analgesic",     "unit": "tablet",  "unit_cost_usd": 0.01, "shelf_life_days": 730, "base_daily_demand": 111.1},
        {"medication_id": "M012", "name": "Ibuprofen 400mg",                  "category": "Analgesic",     "unit": "tablet",  "unit_cost_usd": 0.02, "shelf_life_days": 730, "base_daily_demand": 61.3},
        {"medication_id": "M013", "name": "Metformin 500mg",                  "category": "Diabetes",      "unit": "tablet",  "unit_cost_usd": 0.03, "shelf_life_days": 730, "base_daily_demand": 41.0},
        {"medication_id": "M014", "name": "Amlodipine 5mg",                   "category": "Cardiovascular","unit": "tablet",  "unit_cost_usd": 0.04, "shelf_life_days": 730, "base_daily_demand": 35.9},
        {"medication_id": "M015", "name": "RDT Malaria Test Kit",             "category": "Diagnostics",   "unit": "kit",     "unit_cost_usd": 0.55, "shelf_life_days": 730, "base_daily_demand": 84.0},
    ]

    import numpy as np
    rng = np.random.default_rng(99)

    existing_ids = {m.medication_id for m in db.query(Medicine).all()}

    for med in MEDICINES:
        if med["medication_id"] not in existing_ids:
            db.add(Medicine(**med))

            stock = float(int(rng.uniform(100, 2500)))
            pending = float(int(rng.choice([0, 0, 0, int(rng.uniform(200, 800))])))
            history = [float(int(rng.poisson(med["base_daily_demand"]))) for _ in range(7)]

            db.add(InventoryState(
                medication_id=med["medication_id"],
                stock_on_hand=stock,
                pending_order_qty=pending,
                days_since_last_order=int(rng.integers(0, 25)),
                days_to_expiry=int(rng.integers(60, 700)),
                demand_history_7d_json=json.dumps(history),
            ))
    db.commit()