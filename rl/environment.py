"""
Enriched Gymnasium environment matching the validated synthetic dataset.

Key improvements over the initial version:
  1. Demand uses the SAME seasonal + weekday + regime logic as
     generate_synthetic.py — not just regime multipliers alone.
  2. The fitted HMM (pretrained_hmm.pkl) supplies both the regime
     transition dynamics AND the emission parameters — no hand-tuned
     transition matrix.
  3. Supplier stockout behaviour: orders can be delayed or partially
     fulfilled, matching the supplier_stockout column in the dataset.
  4. A small fixed ordering cost discourages unnecessary orders.
  5. A daily holding cost provides a smooth economic signal against
     over-stocking (matching the H term in the EOQ baseline formula).
  6. Initial inventory is randomised across episodes to force
     generalisation (domain randomisation).

State vector (8 features — unchanged from proposal):
  [0] normalised stock on hand
  [1] normalised days since last order
  [2] normalised pipeline (pending order quantity)
  [3] normalised 7-day trailing demand signal
  [4] P(stable   | demand history)  ← HMM belief
  [5] P(surge    | demand history)  ← HMM belief
  [6] P(disruption | demand history) ← HMM belief
  [7] normalised days to nearest-batch expiry

Action space (discrete, 4):
  0 → no order
  1 → Q_min  (~7  days of base demand)
  2 → Q_mid  (~21 days of base demand)
  3 → Q_max  (~45 days of base demand)

Reward per day:
  +1.0   full-availability day (no stockout)
  -10.0  stockout day (any unmet demand)
  -h     holding cost per unit on hand (h = 0.005 per unit/day)
  -K     fixed ordering cost per order placed (K = 2.0)
  -5.0   per expired unit
"""


import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import date, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
module_dir = os.path.dirname(os.path.abspath(__file__))
for path in (project_root, module_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from .hmm_demand import RegimeBeliefInferrer, HMM_CACHE_PATH
except ImportError:  # pragma: no cover - fallback for direct script execution
    from hmm_demand import RegimeBeliefInferrer, HMM_CACHE_PATH

try:
    from data.literature_params import (
        LEAD_TIME_DAYS_ASSUM,
        ORDER_QUANTITY_DAYS_ASSUM,
        HMM_REGIMES,
        PROCUREMENT_CYCLE_DAYS_PROXY,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from literature_params import (
        LEAD_TIME_DAYS_ASSUM,
        ORDER_QUANTITY_DAYS_ASSUM,
        HMM_REGIMES,
        PROCUREMENT_CYCLE_DAYS_PROXY,
    )

# ─────────────────────────────────────────────────────────────────────────
# SEASONAL MULTIPLIERS — extracted directly from the validated dataset
# (generate_synthetic.py, confirmed in validation Section 5)
# ─────────────────────────────────────────────────────────────────────────
SEASONAL_PROFILES = {
    "Antimalarial":  [1.1, 1.2, 1.5, 2.0, 2.2, 1.4, 1.1, 1.0, 0.9, 1.3, 1.9, 1.6],
    "Antibiotic":    [1.0, 1.0, 1.15, 1.3, 1.25, 1.15, 1.15, 1.1, 1.1, 1.2, 1.3, 1.25],
    "MCH":           [0.95, 0.95, 1.0, 1.05, 1.05, 1.1, 1.1, 1.05, 1.0, 1.0, 1.0, 1.0],
    "Analgesic":     [1.0] * 12,
    "Diabetes":      [1.0] * 12,
    "Cardiovascular":[1.0] * 12,
    "Diagnostics":   [1.1, 1.2, 1.5, 2.0, 2.2, 1.4, 1.1, 1.0, 0.9, 1.3, 1.9, 1.6],
}

# Day-of-week multipliers (0=Mon … 6=Sun), extracted from dataset
DOW_FACTORS = [0.9957, 1.0097, 0.9921, 1.0052, 0.9974, 0.5398, 0.2977]

# Regime parameters are kept in sync with the literature-parameter source of truth.
REGIME_DEMAND_MULT = {
    regime: params["demand_multiplier"] for regime, params in HMM_REGIMES.items()
}

# Lead-time multipliers by regime (from literature_params.py [ASSUM])
REGIME_LEAD_MULT = {
    regime: params["lead_time_multiplier"] for regime, params in HMM_REGIMES.items()
}

# Supplier stockout probability by regime (from literature_params.py)
REGIME_SUPPLIER_SO = {
    regime: params["stockout_prob_daily"] for regime, params in HMM_REGIMES.items()
}

# Base daily demand per medicine (stable regime, weekday average)
# Extracted directly from the validated synthetic dataset
BASE_DEMAND = {
    "M001": 67.0,  # Artemether-Lumefantrine   (Antimalarial)
    "M002": 14.7,  # Quinine Sulfate            (Antimalarial)
    "M003": 83.2,  # Amoxicillin 500mg          (Antibiotic)
    "M004": 18.1,  # Ceftriaxone 1g             (Antibiotic)
    "M005": 41.8,  # Metronidazole 400mg        (Antibiotic)
    "M006": 33.6,  # Ciprofloxacin 500mg        (Antibiotic)
    "M007": 58.5,  # Co-trimoxazole 480mg       (Antibiotic)
    "M008": 10.5,  # Oxytocin 10IU              (MCH)
    "M009": 41.7,  # ORS Sachet                 (MCH)
    "M010": 62.5,  # Ferrous Sulfate+Folic Acid (MCH)
    "M011": 111.1, # Paracetamol 500mg          (Analgesic)
    "M012": 61.3,  # Ibuprofen 400mg            (Analgesic)
    "M013": 41.0,  # Metformin 500mg            (Diabetes)
    "M014": 35.9,  # Amlodipine 5mg             (Cardiovascular)
    "M015": 84.0,  # RDT Malaria Test Kit       (Diagnostics)
}

MED_CATEGORY = {
    "M001": "Antimalarial", "M002": "Antimalarial",
    "M003": "Antibiotic",   "M004": "Antibiotic",
    "M005": "Antibiotic",   "M006": "Antibiotic",   "M007": "Antibiotic",
    "M008": "MCH",          "M009": "MCH",           "M010": "MCH",
    "M011": "Analgesic",    "M012": "Analgesic",
    "M013": "Diabetes",     "M014": "Cardiovascular","M015": "Diagnostics",
}

REGIME_NAMES = ["stable", "surge", "disruption"]

# Lead time distribution parameters [ASSUM] — triangular
LT_MIN, LT_MODE, LT_MAX = (
    LEAD_TIME_DAYS_ASSUM["min"],
    LEAD_TIME_DAYS_ASSUM["mode"],
    LEAD_TIME_DAYS_ASSUM["max"],
)

# Discrete action quantities (days of base demand covered by each bucket)
Q_DAYS = {
    "min": ORDER_QUANTITY_DAYS_ASSUM["min"],
    "mid": ORDER_QUANTITY_DAYS_ASSUM["mid"],
    "max": ORDER_QUANTITY_DAYS_ASSUM["max"],
}

# Economic parameters
HOLDING_COST_PER_UNIT_PER_DAY = 0.005   # h
ORDERING_COST_PER_ORDER       = 2.0     # K
STOCKOUT_PENALTY              = 10.0    # per stockout day
EXPIRY_PENALTY                = 5.0     # per expired unit
AVAILABILITY_REWARD           = 1.0     # per full day


class PharmacyInventoryEnv(gym.Env):
    """
    medication_id : str
        ENLM medicine ID (M001-M015). Determines base demand,
        category, and seasonal profile.
    episode_length_days : int
        Simulated days per episode (default 365 = 1 year).
    seed : int | None
        RNG seed for reproducibility.
    shelf_life_days : int
        Default batch shelf life in days.
    hmm_inferrer : RegimeBeliefInferrer | None
        Pre-trained frozen HMM — injected from train.py so it is
        loaded once and shared across all environment instances.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        medication_id: str = "M003",
        episode_length_days: int = 365,
        seed: int = None,
        shelf_life_days: int = 365,
        hmm_inferrer: RegimeBeliefInferrer = None,
    ):
        super().__init__()
        self.medication_id = medication_id
        self.category = MED_CATEGORY.get(medication_id, "Analgesic")
        self.base_daily_demand = BASE_DEMAND.get(medication_id, 50.0)
        self.seasonal_profile = SEASONAL_PROFILES.get(self.category, [1.0] * 12)
        self.episode_length_days = episode_length_days
        self.shelf_life_days = shelf_life_days
        self._seed = seed
        self._hmm_inferrer = hmm_inferrer
        self.procurement_cycle_days_proxy = PROCUREMENT_CYCLE_DAYS_PROXY["mode"]

        # Action space: 4 discrete order sizes
        self.q_min = self.base_daily_demand * Q_DAYS["min"]
        self.q_mid = self.base_daily_demand * Q_DAYS["mid"]
        self.q_max = self.base_daily_demand * Q_DAYS["max"]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(8,), dtype=np.float32
        )
        self.rng = np.random.default_rng(seed)
        self._reset_state()

    # ── internal helpers ──────────────────────────────────────────────────

    def _sim_date(self) -> tuple:
        """Returns (month 1-12, day-of-week 0-6) for current timestep."""
        base = date(2022, 1, 1) + timedelta(days=self.t)
        return base.month, base.weekday()

    def _demand_mu(self, regime: str) -> float:
        """
        Compute the Poisson rate for today's demand using the same
        three-factor model as generate_synthetic.py:
          μ = base_demand × seasonal_mult × dow_mult × regime_mult
        """
        month, dow = self._sim_date()
        seasonal = self.seasonal_profile[month - 1]
        dow_mult = DOW_FACTORS[dow]
        regime_mult = REGIME_DEMAND_MULT.get(regime, 1.0)
        return max(self.base_daily_demand * seasonal * dow_mult * regime_mult, 0.1)

    def _sample_regime(self, prev_regime_idx: int) -> int:
        """
        Sample next regime using the FITTED HMM transition matrix
        (from pretrained_hmm.pkl), not the hand-tuned one.
        Falls back to literature_params if inferrer not available.
        """
        if self._hmm_inferrer is not None and hasattr(self._hmm_inferrer, '_model'):
            T = self._hmm_inferrer._model.transmat_
        else:
            from data.literature_params import HMM_TRANSITION_MATRIX
            T = np.array(HMM_TRANSITION_MATRIX)
        return int(self.rng.choice(3, p=T[prev_regime_idx]))

    def _get_belief(self) -> np.ndarray:
        if self._hmm_inferrer is not None:
            return self._hmm_inferrer.belief(self.demand_history, self.t)
        return np.array([0.80, 0.15, 0.05], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        norm_stock = self.stock_on_hand / (self.base_daily_demand * 60)
        norm_days_order = self.days_since_last_order / 60.0
        pending_qty = sum(q for _, q, _ in self.pending_orders)
        norm_pending = pending_qty / (self.base_daily_demand * 60)
        norm_demand = (np.mean(self.demand_history[-7:])
                       / max(self.base_daily_demand, 1e-6))
        belief = self._get_belief()
        nearest_expiry = min(
            (d for d, q in self.batches if q > 0),
            default=self.shelf_life_days
        )
        norm_expiry = nearest_expiry / self.shelf_life_days
        obs = np.array([
            norm_stock, norm_days_order, norm_pending, norm_demand,
            belief[0], belief[1], belief[2], norm_expiry,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 10.0)

    def _reset_state(self):
        self.t = 0
        self.days_since_last_order = 0

        # ── Randomised initial inventory (domain randomisation) ──
        # Drawn uniformly between 3 and 30 days of base demand so the
        # agent must generalise across low, normal, and high starting stock.
        init_cover_days = self.rng.uniform(3, 30)
        self.stock_on_hand = float(self.base_daily_demand * init_cover_days)
        self.batches = [[self.shelf_life_days, self.stock_on_hand]]

        # pending_orders: list of [days_remaining, qty, partial_fill_pct]
        self.pending_orders = []

        # Warm-start demand history with base demand
        self.demand_history = [self.base_daily_demand] * 7

        # Initial regime — sample from fitted HMM start probs if available
        if self._hmm_inferrer is not None and hasattr(self._hmm_inferrer, '_model'):
            start = self._hmm_inferrer._model.startprob_
            # startprob_ can be [0,0,1] (degenerate) if HMM converged
            # oddly — fall back to sensible prior in that case
            if np.any(start > 0.5):
                self.regime_idx = int(np.argmax(start))
            else:
                self.regime_idx = int(self.rng.choice(3, p=start))
        else:
            self.regime_idx = 0  # start stable

        self.cumulative_stockout_days = 0
        self.cumulative_reward = 0.0

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        # ── Regime transition (using fitted HMM) ──
        self.regime_idx = self._sample_regime(self.regime_idx)
        regime = REGIME_NAMES[self.regime_idx]

        # ── Demand realisation: seasonal × weekday × regime ──
        mu = self._demand_mu(regime)
        demand = int(self.rng.poisson(mu))
        self.demand_history.append(demand)

        # ── Receive pending orders (with supplier stockout behaviour) ──
        units_received = 0.0
        still_pending = []
        for days_rem, qty, fill_pct in self.pending_orders:
            days_rem -= 1
            if days_rem <= 0:
                # Supplier stockout: partial fulfilment (50–90% of order)
                # or further delay (add extra lead time)
                supplier_so = self.rng.random() < REGIME_SUPPLIER_SO[regime]
                if supplier_so:
                    # 50% chance: partial delivery (60-90% of qty)
                    # 50% chance: delayed (add 7-21 days)
                    if self.rng.random() < 0.5:
                        actual_fill = self.rng.uniform(0.6, 0.9)
                        delivered = qty * actual_fill
                        remainder = qty * (1 - actual_fill)
                        units_received += delivered
                        if remainder > 1:
                            # Back-order the remainder
                            extra_delay = int(self.rng.integers(7, 22))
                            still_pending.append([extra_delay, remainder, 1.0])
                    else:
                        # Delay the whole order
                        extra_delay = int(self.rng.integers(7, 22))
                        still_pending.append([extra_delay, qty, fill_pct])
                else:
                    units_received += qty
            else:
                still_pending.append([days_rem, qty, fill_pct])

        self.pending_orders = still_pending
        if units_received > 0:
            self.batches.append([self.shelf_life_days, units_received])
        self.stock_on_hand += units_received

        # ── Serve demand (FIFO batches) ──
        remaining = demand
        for batch in self.batches:
            if remaining <= 0:
                break
            take = min(batch[1], remaining)
            batch[1] -= take
            remaining -= take
        unmet = max(remaining, 0)
        self.stock_on_hand = max(self.stock_on_hand - demand, 0)

        # ── Age batches; collect expiries ──
        expired_units = 0
        for batch in self.batches:
            batch[0] -= 1
            if batch[0] <= 0 and batch[1] > 0:
                expired_units += batch[1]
                self.stock_on_hand = max(self.stock_on_hand - batch[1], 0)
                batch[1] = 0
        self.batches = [b for b in self.batches if b[1] > 0]

        # ── Place order (action) ──
        order_map = {0: 0.0, 1: self.q_min, 2: self.q_mid, 3: self.q_max}
        order_qty = order_map[int(action)]
        ordering_cost = 0.0
        pending_qty = sum(q for _, q, _ in self.pending_orders)
        total_covered = self.stock_on_hand + pending_qty
        can_order = (
            order_qty > 0
            and total_covered < self.base_daily_demand * 75
            and len(self.pending_orders) < 2
        )
        if can_order:
            lt_base = float(np.clip(
                self.rng.triangular(LT_MIN, LT_MODE, LT_MAX)
                * REGIME_LEAD_MULT[regime],
                LT_MIN, LT_MAX * 4,
            ))
            self.pending_orders.append([int(lt_base), order_qty, 1.0])
            self.days_since_last_order = 0
            ordering_cost = ORDERING_COST_PER_ORDER     # fixed cost K
        else:
            self.days_since_last_order += 1

        # ── Reward ──────────────────────────────────────────────────────
        stockout_today = unmet > 0
        reward = AVAILABILITY_REWARD if not stockout_today else -STOCKOUT_PENALTY
        reward -= HOLDING_COST_PER_UNIT_PER_DAY * self.stock_on_hand   # holding
        reward -= ordering_cost                                          # ordering
        reward -= EXPIRY_PENALTY * expired_units                        # expiry

        self.cumulative_stockout_days += int(stockout_today)
        self.cumulative_reward += reward
        self.t += 1

        info = {
            "demand": demand,
            "unmet_demand": unmet,
            "stockout": stockout_today,
            "expired_units": expired_units,
            "regime": regime,
            "stock_on_hand": self.stock_on_hand,
            "units_received": units_received,
            "order_qty": order_qty,
            "ordering_cost": ordering_cost,
            "holding_cost": HOLDING_COST_PER_UNIT_PER_DAY * self.stock_on_hand,
            "regime_belief": self._get_belief().tolist(),
            "procurement_cycle_days_proxy": self.procurement_cycle_days_proxy,
        }
        return (
            self._get_obs(), reward, False,
            self.t >= self.episode_length_days, info,
        )

    def render(self):
        belief = self._get_belief()
        month, dow = self._sim_date()
        seasonal = self.seasonal_profile[month - 1]
        print(
            f"Day {self.t:3d} | {REGIME_NAMES[self.regime_idx]:12s} | "
            f"stock={self.stock_on_hand:7.0f} | "
            f"belief=[{belief[0]:.2f},{belief[1]:.2f},{belief[2]:.2f}] | "
            f"seasonal={seasonal:.2f} | "
            f"cum_reward={self.cumulative_reward:8.1f}"
        )