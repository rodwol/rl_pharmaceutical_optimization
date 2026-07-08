"""
Endpoints:
  GET  /                      health check
  GET  /health                health check (explicit)
  POST /api/recommend         DQN replenishment recommendation
  GET  /api/recommend/health  model availability check
  POST /api/orders            submit an order for manager approval
  GET  /api/orders            list all orders (filter by status)
  GET  /api/orders/{id}       get single order
  PATCH /api/orders/{id}/approve  approve a pending order
  PATCH /api/orders/{id}/reject   reject a pending order
  GET  /api/inventory         current inventory snapshot (for dashboard)
  GET  /api/medicines         list of all tracked medicines
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import recommend, orders, inventory

app = FastAPI(
    title="Rebex",
    description=(
        "**Reinforcement-learning powered replenishment recommendation API** "
        "for essential medicine stockout prevention in Eritrean district hospital "
        "pharmacies.\n\n"
        "The DQN agent was trained on a literature-calibrated synthetic dataset "
        "using a Hidden Markov Model for demand regime inference "
        "(stable / surge / disruption). "
        "All orders require Pharmacy Manager approval before submission — "
        "this is a decision-support tool, not an automated ordering system.\n\n"
        "**Literature sources:** Halibet 2018 [ERI], Asmara 2019 [ERI], "
        "Motta 2022 [PROXY], Gubre 2025 [PROXY]."
    ),
    version="1.0.0",
    contact={
        "name": "Project R",
    },
    license_info={"name": "MIT"},
)

# ── CORS — allows the Streamlit dashboard to call the API ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers 
app.include_router(recommend.router,  prefix="/api", tags=["Recommendations"])
app.include_router(orders.router,     prefix="/api", tags=["Orders"])
app.include_router(inventory.router,  prefix="/api", tags=["Inventory"])


@app.get("/", tags=["Health"], summary="Root health check")
def root():
    return {
        "status": "ok",
        "service": "R API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", tags=["Health"], summary="Health check")
def health():
    return {"status": "ok"}

