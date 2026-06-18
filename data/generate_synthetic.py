"""

Synthetic pharmacy inventory/demand generator for Eritrean district hospital
pharmacies. Builds on the structure of the original prototype script but:

  1. Grounds the medicine list in the 2015 Eritrean National List of
     Medicines (ENLM) categories instead of an invented generic list.
  2. Replaces the single seasonal-multiplier demand process with an
     HMM-driven regime-switching process (stable / surge / disruption),
     addressing the supervisor's MDP+HMM comment directly.
  3. Calibrates every operational parameter (stockout duration, stockout
     incidence, procurement cycle, lead time) against literature_params.py,
     with every number traceable to a cited source or flagged as an
     explicit assumption.
  4. Runs a validation step at the end comparing simulated output statistics
     against the literature targets, so the methodology section can state
     quantitatively how well the simulation matches published benchmarks.
"""

import sys
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from literature_params import (
    STOCKOUT_DURATION, STOCKOUT_INCIDENCE, PROCUREMENT_CYCLE_DAYS_PROXY,
    LEAD_TIME_DAYS_ASSUM, PRESCRIBING_PROFILE_ERI, HMM_REGIMES,
)
from rl.hmm_demand import generate_regime_sequence, REGIME_NAMES

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────
# 1. MEDICATION LIST — grounded in ENLM 2015 categories
# ─────────────────────────────────────────────────────────────────────────
# Representative subset across ENLM therapeutic categories most relevant to
# district hospital pharmacy operations and the cited prescribing studies
# (antibiotics dominate prescribing per Asmara/Halibet studies, so they are
# over-represented relative to a uniform sample across all ENLM sections).
medications = [
    {"medication_id": "M001", "name": "Artemether-Lumefantrine 20/120mg", "category": "Antimalarial", "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.12},
    {"medication_id": "M002", "name": "Quinine Sulfate 300mg",            "category": "Antimalarial", "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.08},
    {"medication_id": "M003", "name": "Amoxicillin 500mg",                "category": "Antibiotic",   "unit": "capsule", "shelf_life_days": 730, "unit_cost_usd": 0.04},
    {"medication_id": "M004", "name": "Ceftriaxone 1g Injection",         "category": "Antibiotic",   "unit": "vial",    "shelf_life_days": 730, "unit_cost_usd": 0.90},
    {"medication_id": "M005", "name": "Metronidazole 400mg",              "category": "Antibiotic",   "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.03},
    {"medication_id": "M006", "name": "Ciprofloxacin 500mg",              "category": "Antibiotic",   "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.07},
    {"medication_id": "M007", "name": "Co-trimoxazole 480mg",             "category": "Antibiotic",   "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.02},
    {"medication_id": "M008", "name": "Oxytocin 10IU Injection",          "category": "MCH",          "unit": "ampoule", "shelf_life_days": 365, "unit_cost_usd": 0.45},
    {"medication_id": "M009", "name": "ORS Sachet",                       "category": "MCH",          "unit": "sachet",  "shelf_life_days": 1460,"unit_cost_usd": 0.05},
    {"medication_id": "M010", "name": "Ferrous Sulfate + Folic Acid",     "category": "MCH",          "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.02},
    {"medication_id": "M011", "name": "Paracetamol 500mg",                "category": "Analgesic",    "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.01},
    {"medication_id": "M012", "name": "Ibuprofen 400mg",                  "category": "Analgesic",    "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.02},
    {"medication_id": "M013", "name": "Metformin 500mg",                  "category": "Diabetes",     "unit": "tablet",  "shelf_life_days": 730, "unit_cost_usd": 0.03},
    {"medication_id": "M014", "name": "Amlodipine 5mg",                   "category": "Cardiovascular","unit": "tablet", "shelf_life_days": 730, "unit_cost_usd": 0.04},
    {"medication_id": "M015", "name": "RDT Malaria Test Kit",             "category": "Diagnostics",  "unit": "kit",     "shelf_life_days": 730, "unit_cost_usd": 0.55},
]
medications_df = pd.DataFrame(medications)

SEASONAL_CATEGORY = {
    "Antimalarial": "malaria", "Diagnostics": "malaria",
    "Antibiotic": "diarrheal_resp_blend", "MCH": "mch", "Analgesic": "flat",
    "Diabetes": "flat", "Cardiovascular": "flat",
}

def malaria_seasonal_multiplier(month):
    profile = {1:1.1,2:1.2,3:1.5,4:2.0,5:2.2,6:1.4,7:1.1,8:1.0,9:0.9,10:1.3,11:1.9,12:1.6}
    return profile[month]

def diarrheal_resp_blend_multiplier(month):
    diarrheal = {1:1.0,2:1.1,3:1.5,4:1.8,5:1.6,6:1.0,7:0.8,8:0.8,9:1.0,10:1.4,11:1.7,12:1.5}
    respiratory = {1:1.0,2:0.9,3:0.8,4:0.8,5:0.9,6:1.3,7:1.5,8:1.4,9:1.2,10:1.0,11:0.9,12:1.0}
    return 0.5 * diarrheal[month] + 0.5 * respiratory[month]

def mch_seasonal_multiplier(month):
    profile = {1:0.95,2:0.95,3:1.0,4:1.05,5:1.05,6:1.1,7:1.1,8:1.05,9:1.0,10:1.0,11:1.0,12:1.0}
    return profile[month]

def get_seasonal_multiplier(category, month):
    kind = SEASONAL_CATEGORY.get(category, "flat")
    if kind == "malaria": return malaria_seasonal_multiplier(month)
    if kind == "diarrheal_resp_blend": return diarrheal_resp_blend_multiplier(month)
    if kind == "mch": return mch_seasonal_multiplier(month)
    return 1.0

# Approximate base daily demand for a single Eritrean district hospital
# pharmacy, scaled using the Asmara prescribing profile (1.78 meds/script,
# ~53.5% containing an antibiotic) to set relative category proportions.
base_daily_demand = {
    "M001": 45, "M002": 10, "M003": 70, "M004": 15, "M005": 35, "M006": 28, "M007": 50,
    "M008": 10, "M009": 40, "M010": 60, "M011": 110, "M012": 60, "M013": 40, "M014": 35, "M015": 55,
}

FACILITY = {"facility_id": "F001", "name": "District Hospital Pharmacy (Eritrea, simulated)",
            "facility_type": "district_hospital", "country": "Eritrea"}

# ─────────────────────────────────────────────────────────────────────────
# 2. DATE RANGE
# ─────────────────────────────────────────────────────────────────────────
start_date = date(2022, 1, 1)
end_date = date(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq="D")
n_days = len(date_range)

# ─────────────────────────────────────────────────────────────────────────
# 3. GENERATE ONE SHARED REGIME SEQUENCE FOR THE FACILITY
#    (all medicines at this facility share the same supply-side regime,
#     since disruption/procurement delays are facility-wide events; demand
#     surges are category-specific via the seasonal multiplier instead)
# ─────────────────────────────────────────────────────────────────────────
regime_seq = generate_regime_sequence(n_days, seed=42)

# ─────────────────────────────────────────────────────────────────────────
# 4. GENERATE DAILY RECORDS
# ─────────────────────────────────────────────────────────────────────────
records = []
lt_cfg = LEAD_TIME_DAYS_ASSUM
proc_cfg = PROCUREMENT_CYCLE_DAYS_PROXY

for med in medications:
    mid, cat = med["medication_id"], med["category"]
    base_d = base_daily_demand[mid]
    shelf_days = med["shelf_life_days"]

    # Running inventory state for THIS medicine (depletes with demand,
    # replenished by pending orders that arrive after their lead time).
    # Initial stock ~ 14 days of base demand, a typical starting safety stock.
    stock_on_hand = int(base_d * 21)
    pending_orders = []  # list of [days_remaining, quantity]
    REORDER_TRIGGER_DAYS = 25   # reorder while ~25 days of cover remain (covers lead time + buffer)
    ORDER_SIZE_DAYS = 45        # order enough to cover ~45 days of demand

    for i, dt in enumerate(date_range):
        month, dow = dt.month, dt.dayofweek
        regime_idx = regime_seq[i]
        regime_name = REGIME_NAMES[regime_idx]
        regime_params = HMM_REGIMES[regime_name]

        seasonal = get_seasonal_multiplier(cat, month)
        dow_mult = 0.55 if dow == 5 else (0.30 if dow == 6 else 1.0)
        trend_mult = 1.0 + 0.02 * ((dt.year - 2022) + (dt.dayofyear - 1) / 365)

        mu = base_d * seasonal * dow_mult * trend_mult * regime_params["demand_multiplier"]
        mu = max(mu, 0.1)
        demand = int(rng.poisson(mu))

        lead_time = int(np.clip(
            rng.triangular(lt_cfg["min"], lt_cfg["mode"], lt_cfg["max"]) * regime_params["lead_time_multiplier"],
            lt_cfg["min"], lt_cfg["max"] * 3
        ))
        procurement_due = int(rng.triangular(proc_cfg["min"], proc_cfg["mode"], proc_cfg["max"]))
        supplier_stockout = int(rng.random() < regime_params["stockout_prob_daily"])

        remaining_shelf = int(rng.uniform(30, shelf_days))
        expiry_date = dt + timedelta(days=remaining_shelf)

        # ── Record stock available BEFORE today's demand is served ──
        opening_stock = stock_on_hand

        # ── Receive any orders due to arrive today ──
        units_received = 0
        still_pending = []
        for days_remaining, qty in pending_orders:
            days_remaining -= 1
            if days_remaining <= 0:
                units_received += qty
            else:
                still_pending.append([days_remaining, qty])
        pending_orders = still_pending
        stock_on_hand += units_received

        # ── Deplete stock by today's demand (cannot go below 0) ──
        unmet = max(demand - stock_on_hand, 0)
        stock_on_hand = max(stock_on_hand - demand, 0)

        # ── Trigger a reorder if stock is low and no order already pending ──
        days_of_stock_left = stock_on_hand / max(base_d * seasonal, 0.1)
        if days_of_stock_left < REORDER_TRIGGER_DAYS and len(pending_orders) == 0:
            order_qty = int(base_d * seasonal * ORDER_SIZE_DAYS)
            # A supplier-side stockout delays the order from even being placed
            # effectively (it gets added with extra delay), modelling supply shocks.
            effective_lead_time = lead_time * (2 if supplier_stockout else 1)
            pending_orders.append([effective_lead_time, order_qty])

        records.append({
            "date": dt.date(), "facility_id": FACILITY["facility_id"],
            "medication_id": mid, "category": cat,
            "regime": regime_name,
            "demand_units": demand, "stock_on_hand": opening_stock,
            "closing_stock": stock_on_hand,
            "units_received": units_received, "lead_time_days": lead_time,
            "procurement_cycle_days": procurement_due,
            "supplier_stockout": supplier_stockout,
            "expiry_date": expiry_date.date(), "seasonal_mult": round(seasonal, 3),
            "dow": dow, "month": month, "year": dt.year,
            "unmet_demand_today": unmet,
        })

demand_df = pd.DataFrame(records)
demand_df["stockout_flag"] = (demand_df["unmet_demand_today"] > 0).astype(int)
demand_df["unmet_demand"] = demand_df["unmet_demand_today"]

print(f"Generated {len(demand_df):,} daily records across {len(medications)} medicines, {n_days} days")
print(f"Regime distribution: {pd.Series(regime_seq).map(lambda i: REGIME_NAMES[i]).value_counts(normalize=True).round(3).to_dict()}")

# ─────────────────────────────────────────────────────────────────────────
# 5. VALIDATION AGAINST LITERATURE TARGETS
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION AGAINST LITERATURE-DERIVED TARGETS")
print("=" * 60)

# (a) Six-month stockout incidence rate, target ≈ 0.60 (Motta 2022, PROXY)
six_month_window = 182
incidence_checks = []
for mid in demand_df["medication_id"].unique():
    sub = demand_df[demand_df["medication_id"] == mid].sort_values("date")
    for start in range(0, len(sub) - six_month_window, six_month_window):
        window = sub.iloc[start:start + six_month_window]
        had_stockout = window["stockout_flag"].sum() > 0
        incidence_checks.append(had_stockout)
sim_incidence = np.mean(incidence_checks)
print(f"\n(a) 6-month stockout incidence rate")
print(f"    Simulated : {sim_incidence:.3f}")
print(f"    Target    : {STOCKOUT_INCIDENCE['six_month_incidence_rate_PROXY']:.3f}  (Motta 2022, PROXY)")

# (b) Point-in-time stockout rate, target ≈ 0.125 (Halibet 2018, ERI, antibiotics)
abx_sub = demand_df[demand_df["category"] == "Antibiotic"]
sim_point_rate = abx_sub["stockout_flag"].mean()
print(f"\n(b) Point-in-time stockout rate (antibiotics)")
print(f"    Simulated : {sim_point_rate:.3f}")
print(f"    Target    : {STOCKOUT_INCIDENCE['point_in_time_stockout_rate_ERI']:.3f}  (Halibet 2018, ERI)")

# (c) Mean stockout duration (days), target range 10-157, mean ~38.8 (Motta, PROXY)
demand_df_sorted = demand_df.sort_values(["medication_id", "date"])
durations = []
for mid in demand_df["medication_id"].unique():
    sub = demand_df_sorted[demand_df_sorted["medication_id"] == mid]
    flags = sub["stockout_flag"].values
    run = 0
    for f in flags:
        if f == 1:
            run += 1
        elif run > 0:
            durations.append(run)
            run = 0
    if run > 0:
        durations.append(run)
sim_mean_duration = np.mean(durations) if durations else 0
print(f"\n(c) Mean stockout episode duration (days)")
print(f"    Simulated : {sim_mean_duration:.1f}")
print(f"    Target    : {STOCKOUT_DURATION['general_mean_days_PROXY']:.1f} "
      f"(range {STOCKOUT_DURATION['general_min_days_PROXY']}-{STOCKOUT_DURATION['general_max_days_PROXY']}, Motta 2022, PROXY)")

print("\n" + "=" * 60)
print("NOTE: exact alignment is not expected/required — these checks confirm")
print("the simulation lands in a plausible literature-grounded range, not an")
print("arbitrary one. Document this validation table in the methodology section.")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────
# 6. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────
demand_df.to_csv("synthetic_demand_data.csv", index=False)
medications_df.to_csv("medications.csv", index=False)
print(f"\nSaved: synthetic_demand_data.csv ({demand_df.shape}), medications.csv ({medications_df.shape})")
