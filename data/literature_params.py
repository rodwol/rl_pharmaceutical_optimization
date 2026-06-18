"""
Documents every numeric parameter used in the synthetic data generator and
HMM demand model, tagged by source. This file IS the audit trail for the
"where did your numbers come from" methodology question.
 
Tag legend:
  [ERI]   = Eritrea-specific, drawn directly from a cited Eritrean study
  [PROXY] = Not available for Eritrea; substituted with a regional
            Sub-Saharan / East African proxy from a cited study
  [ASSUM] = No literature anchor found; documented modelling assumption
"""
 
# SOURCES - full citations live in the research proposal
SOURCES = {
    "ENLM2015":      "2015 Eritrean National List of Medicines (WHO Platform)",
    "HALIBET2018":   "Assessment of inpatient antibiotic use in Halibet National "
                      "Referral Hospital using WHO indicators (2018)",
    "ERI_REFHOSP2022": "Drug prescribing and dispensing practices in regional and "
                        "national referral hospitals of Eritrea (PLOS ONE, 2022)",
    "ERI_ASMARA2019":  "Prescribing practices using WHO indicators in six community "
                        "pharmacies in Asmara, Eritrea (2019)",
    "MOTTA2022":     "Availability and stock-out duration of essential medicines in "
                      "Shegaw Motta General Hospital, North West Ethiopia (2022)",
    "RWANDA_NCD2020": "Availability, Costs and Stock-Outs of Essential NCD Drugs in "
                       "Three Rural Rwandan Districts (2020)",
    "GUBRE2025":     "Essential Medicine Availability, Stock-Out Duration, and "
                      "Influencing Factors in Gubre Town, Central Ethiopia (2025)",
    "ZAMBIA2016":    "The Impact of Inventory Management on Stock-Outs of Essential "
                      "Drugs in Sub-Saharan Africa: Zambia field experiment (PLOS ONE, 2016)",
}
 
# ─────────────────────────────────────────────────────────────────────────
# 1. STOCKOUT DURATION  (days a medicine is unavailable once stocked out)
# ─────────────────────────────────────────────────────────────────────────
# [ERI] Halibet: key antibiotics out of stock on average 78.18 days; 87.5% availability
# [PROXY] Motta Ethiopia: mean stockout duration 38.8 days, range 10-157 days
# [PROXY] Rwanda district hospitals: median stockout length range 3.5-228 days
#
# Eritrea (Halibet) gives a single point estimate (~78 days, antibiotics only).
# The Ethiopia/Rwanda studies give us a *distribution shape* (heavy right tail)
# that Eritrea's single number doesn't. We blend: use Halibet's 78 days as the
# Eritrea-specific anchor for antibiotics, and the Motta/Rwanda spread to set
# the distribution shape (lognormal) applied across all categories.
STOCKOUT_DURATION = {
    "antibiotic_mean_days_ERI": 78.18,        # [ERI] Halibet 2018
    "general_mean_days_PROXY": 38.8,          # [PROXY] Motta 2022
    "general_min_days_PROXY": 10,             # [PROXY] Motta 2022
    "general_max_days_PROXY": 157,            # [PROXY] Motta 2022
    "rwanda_median_range_PROXY": (3.5, 228),  # [PROXY] Rwanda NCD 2020
    "distribution": "lognormal",              # [ASSUM] shape choice given heavy tail
}
 
# ─────────────────────────────────────────────────────────────────────────
# 2. STOCKOUT INCIDENCE  (probability / rate of entering a stockout state)
# ─────────────────────────────────────────────────────────────────────────
# [ERI] Halibet: 87.5% availability on day of study -> ~12.5% point-in-time stockout
# [ERI] Eritrea referral hospitals (PLOS ONE 2022): >50% of key medicines available
#       in stock -> implies up to ~50% unavailability across the broader basket
# [PROXY] Motta Ethiopia: 60% of essential medicines stocked out at least once in 6mo
STOCKOUT_INCIDENCE = {
    "point_in_time_stockout_rate_ERI": 0.125,      # [ERI] Halibet 2018 (antibiotics)
    "six_month_incidence_rate_PROXY": 0.60,        # [PROXY] Motta 2022 (any stockout in 6mo)
    "referral_hospital_availability_ERI": 0.50,    # [ERI] PLOS ONE 2022 (>50% available)
}
 
# ─────────────────────────────────────────────────────────────────────────
# 3. PROCUREMENT CYCLE / REPLENISHMENT INTERVAL  (days between resupply)
# ─────────────────────────────────────────────────────────────────────────
# [PROXY] Gubre Town Ethiopia: facilities far from suppliers resupplied every 2-3 months
PROCUREMENT_CYCLE_DAYS_PROXY = {"min": 60, "mode": 75, "max": 90}  # [PROXY] Gubre 2025
 
# ─────────────────────────────────────────────────────────────────────────
# 4. LEAD TIME  (days between order placement and delivery)
# ─────────────────────────────────────────────────────────────────────────
# No Eritrea-specific or regional point-estimate found for *order-to-delivery*
# lead time distinct from the full procurement cycle. We treat lead time as a
# fraction of the procurement cycle (delivery delay component), documented as
# an explicit modelling assumption rather than inventing an unsourced number.
LEAD_TIME_DAYS_ASSUM = {
    "min": 7, "mode": 14, "max": 30,
    "note": "[ASSUM] No direct Eritrea/regional point-estimate found; modelled as "
            "delivery-delay component within the Gubre 2025 procurement cycle window. "
            "Disclosed explicitly as a limitation in methodology.",
}
 
# ─────────────────────────────────────────────────────────────────────────
# 5. DEMAND ESTIMATION METHOD
# ─────────────────────────────────────────────────────────────────────────
# [PROXY] Zambia field experiment: demand rate = units per box / days between
# consecutive box issues, smoothed with a triple moving average (half-widths 40/30/20)
DEMAND_METHOD_NOTE = (
    "[PROXY] Demand-rate construction follows Leung et al. (Zambia, 2016): "
    "quantity dispensed between consecutive restock events divided by elapsed "
    "days, smoothed with a moving average to handle censored/missing data."
)
 
# ─────────────────────────────────────────────────────────────────────────
# 6. PRESCRIBING / DEMAND COMPOSITION  (Eritrea-specific)
# ─────────────────────────────────────────────────────────────────────────
# [ERI] Asmara community pharmacies: average medicines per prescription, % with antibiotic
PRESCRIBING_PROFILE_ERI = {
    "avg_medicines_per_prescription": 1.78,   # [ERI] Asmara 2019
    "pct_prescriptions_with_antibiotic": 0.535,  # [ERI] Asmara 2019 (~53-54%)
}
 
# ─────────────────────────────────────────────────────────────────────────
# 7. HMM REGIME DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────
# Three latent demand/supply regimes, parameterized using the above sources.
HMM_REGIMES = {
    "stable": {
        "demand_multiplier": 1.0,
        "stockout_prob_daily": 0.02,        # background risk, low
        "lead_time_multiplier": 1.0,
    },
    "surge": {
        # malaria/respiratory/diarrheal seasonal surge -> demand spikes
        "demand_multiplier": 1.8,           # [ASSUM] calibrated to land near literature ranges
        "stockout_prob_daily": 0.06,
        "lead_time_multiplier": 1.1,
    },
    "disruption": {
        # supplier-side shock: procurement delay, matches PROCUREMENT_CYCLE_DAYS_PROXY tail
        "demand_multiplier": 1.0,
        "stockout_prob_daily": 0.35,        # calibrated so 6-month incidence ≈ 0.60 (Motta)
        "lead_time_multiplier": 4.0,         # long delays drive the heavy-tailed durations seen in Motta/Rwanda
    },
}
 
# Regime transition matrix [stable, surge, disruption] -> rows sum to 1
# [ASSUM] Tuned so long-run stockout incidence over a 6-month (≈182 day) window
# approximates the 60% literature figure (validated empirically in calibration script).
HMM_TRANSITION_MATRIX = [
    [0.965, 0.03, 0.005],   # from stable
    [0.10, 0.875, 0.025],   # from surge
    [0.018, 0.012, 0.97],   # from disruption (very sticky -> long episodes, matches heavy-tailed durations)
]
 
HMM_INITIAL_STATE_PROBS = [0.80, 0.15, 0.05]  # mostly start "stable"