"""
Custom Gymnasium environment for the essential-medicine replenishment MDP.

State (per proposal):
    [0] normalized stock level
    [1] normalized days since last order
    [2] normalized pending order quantity
    [3] normalized 7-day trailing demand signal
    [4:7] HMM regime belief (posterior over stable/surge/disruption)
    [7] normalized days until nearest-batch expiry

Action (discrete, per proposal):
    0 -> order 0
    1 -> order Q_min   (≈ 7 days of base demand)
    2 -> order Q_mid   (≈ 21 days of base demand)
    3 -> order Q_max   (≈ 45 days of base demand)

Reward (per proposal):
    -10  per stockout day (any unmet demand that day)
    -0.05 per unit overstocked (stock above a defined ceiling)
    -5   per expired unit (batch reaching its expiry with stock remaining)
    +1   per day of full demand availability (no stockout)

The environment re-uses the literature-calibrated regime model from
hmm_demand.py so the demand/supply dynamics the agent trains against are
the same ones validated in generate_synthetic.py.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from literature_params import HMM_REGIMES, LEAD_TIME_DAYS_ASSUM
from hmm_demand import generate_regime_sequence, REGIME_NAMES, fit_hmm_to_demand, regime_belief_features


class PharmacyInventoryEnv(gym.Env):
    """
    Single-medicine inventory replenishment environment.

    One step = one day. One episode = `episode_length_days` days
    (default 365, i.e. one simulated year) of a single medicine's
    demand/supply process, regime-switching per the HMM.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, base_daily_demand: float = 50.0, episode_length_days: int = 365,
                 seed: int = None, overstock_ceiling_days: float = 60.0,
                 shelf_life_days: int = 365):
        super().__init__()
        self.base_daily_demand = base_daily_demand
        self.episode_length_days = episode_length_days
        self.overstock_ceiling = base_daily_demand * overstock_ceiling_days
        self.shelf_life_days = shelf_life_days
        self._seed = seed

        # Action space: 4 discrete order sizes
        self.q_min = base_daily_demand * 7
        self.q_mid = base_daily_demand * 21
        self.q_max = base_daily_demand * 45
        self.action_space = spaces.Discrete(4)

        # Observation space: 8 continuous features, normalized to roughly [0,1]+
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(8,), dtype=np.float32)

        self.lt_cfg = LEAD_TIME_DAYS_ASSUM
        self.rng = np.random.default_rng(seed)

        self._reset_state()

    def _reset_state(self):
        self.t = 0
        self.stock_on_hand = self.base_daily_demand * 21  # start with ~3 weeks cover
        self.days_since_last_order = 0
        self.pending_orders = []          # list of [days_remaining, qty]
        self.batches = [[self.shelf_life_days, self.stock_on_hand]]  # [days_to_expiry, qty]
        self.demand_history = [self.base_daily_demand] * 7  # rolling 7-day window
        self.regime_seq = generate_regime_sequence(self.episode_length_days + 1, seed=self._seed)
        self.regime_belief_model = None
        self._observed_demand_for_hmm = []
        self._cached_belief = None
        self.cumulative_stockout_days = 0
        self.cumulative_reward = 0.0

    def _current_regime_belief(self) -> np.ndarray:
        """
        Returns a belief vector over [stable, surge, disruption].
        Early in the episode (insufficient history to fit an HMM), falls
        back to a uniform/neutral prior. Once enough demand history is
        observed, fits a fresh HMM periodically (every 7 days, not every
        single step -- refitting per-step is computationally wasteful and
        the regime belief does not need single-day resolution to be useful
        to the agent) and caches the posterior for the current timestep.
        """
        if len(self._observed_demand_for_hmm) < 30:
            return np.array([0.8, 0.15, 0.05], dtype=np.float32)  # prior, matches HMM_INITIAL_STATE_PROBS

        refit_due = (self.t % 7 == 0) or (self._cached_belief is None)
        if refit_due:
            try:
                demand_arr = np.array(self._observed_demand_for_hmm[-180:])  # cap history window for speed
                model, _ = fit_hmm_to_demand(demand_arr, n_states=3)
                posteriors = regime_belief_features(model, demand_arr)
                self._cached_belief = posteriors[-1].astype(np.float32)
            except Exception:
                if self._cached_belief is None:
                    self._cached_belief = np.array([0.8, 0.15, 0.05], dtype=np.float32)
        return self._cached_belief

    def _get_obs(self) -> np.ndarray:
        norm_stock = self.stock_on_hand / (self.base_daily_demand * 60)
        norm_days_since_order = self.days_since_last_order / 60.0
        pending_qty = sum(q for _, q in self.pending_orders)
        norm_pending = pending_qty / (self.base_daily_demand * 60)
        norm_demand_signal = np.mean(self.demand_history[-7:]) / max(self.base_daily_demand, 1e-6)
        belief = self._current_regime_belief()
        nearest_expiry = min((d for d, q in self.batches if q > 0), default=self.shelf_life_days)
        norm_expiry = nearest_expiry / self.shelf_life_days

        obs = np.array([
            norm_stock, norm_days_since_order, norm_pending, norm_demand_signal,
            belief[0], belief[1], belief[2], norm_expiry,
        ], dtype=np.float32)
        obs = np.clip(obs, 0.0, 10.0)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        regime_idx = self.regime_seq[self.t]
        regime_name = REGIME_NAMES[regime_idx]
        regime_params = HMM_REGIMES[regime_name]

        # ── Demand realization for today ──
        mu = max(self.base_daily_demand * regime_params["demand_multiplier"], 0.1)
        demand = int(self.rng.poisson(mu))
        self.demand_history.append(demand)
        self._observed_demand_for_hmm.append(demand)

        # ── Receive any orders due today ──
        units_received = 0
        still_pending = []
        for days_remaining, qty in self.pending_orders:
            days_remaining -= 1
            if days_remaining <= 0:
                units_received += qty
            else:
                still_pending.append([days_remaining, qty])
        self.pending_orders = still_pending
        if units_received > 0:
            self.batches.append([self.shelf_life_days, units_received])
        self.stock_on_hand += units_received

        # ── Deplete stock by demand (FIFO across batches) ──
        remaining_demand = demand
        for batch in self.batches:
            if remaining_demand <= 0:
                break
            take = min(batch[1], remaining_demand)
            batch[1] -= take
            remaining_demand -= take
        unmet = max(remaining_demand, 0)
        self.stock_on_hand = max(self.stock_on_hand - demand, 0)

        # ── Age batches by one day; collect expired units ──
        expired_units = 0
        for batch in self.batches:
            batch[0] -= 1
            if batch[0] <= 0 and batch[1] > 0:
                expired_units += batch[1]
                self.stock_on_hand = max(self.stock_on_hand - batch[1], 0)
                batch[1] = 0
        self.batches = [b for b in self.batches if b[1] > 0]

        # ── Apply the agent's action: place a new order ──
        order_map = {0: 0.0, 1: self.q_min, 2: self.q_mid, 3: self.q_max}
        order_qty = order_map[int(action)]
        pending_qty = sum(q for _, q in self.pending_orders)
        total_inbound_and_onhand = self.stock_on_hand + pending_qty
        # A real pharmacy will not place a new order if stock + what's
        # already in the pipeline already covers a generous horizon
        # (here, ~75 days of demand) -- this caps unbounded stacking
        # regardless of how many orders are formally "pending".
        max_reasonable_stock = self.base_daily_demand * 75
        can_order = (order_qty > 0) and (total_inbound_and_onhand < max_reasonable_stock) and (len(self.pending_orders) < 2)
        if can_order:
            lead_time = int(np.clip(
                self.rng.triangular(self.lt_cfg["min"], self.lt_cfg["mode"], self.lt_cfg["max"])
                * regime_params["lead_time_multiplier"],
                self.lt_cfg["min"], self.lt_cfg["max"] * 4
            ))
            self.pending_orders.append([lead_time, order_qty])
            self.days_since_last_order = 0
        else:
            self.days_since_last_order += 1

        # ── Reward ──
        reward = 0.0
        stockout_today = unmet > 0
        if stockout_today:
            reward -= 10.0
            self.cumulative_stockout_days += 1
        else:
            reward += 1.0
        overstock_units = max(self.stock_on_hand - self.overstock_ceiling, 0)
        reward -= 0.01 * overstock_units
        reward -= 5.0 * expired_units

        self.cumulative_reward += reward
        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_length_days

        obs = self._get_obs()
        info = {
            "demand": demand, "unmet_demand": unmet, "stockout": stockout_today,
            "expired_units": expired_units, "regime": regime_name,
            "stock_on_hand": self.stock_on_hand, "order_qty": order_qty,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Day {self.t:3d} | stock={self.stock_on_hand:7.1f} | "
              f"regime={REGIME_NAMES[self.regime_seq[min(self.t, len(self.regime_seq)-1)]]:10s} | "
              f"cum_reward={self.cumulative_reward:8.1f} | stockout_days={self.cumulative_stockout_days}")


if __name__ == "__main__":
    # Smoke test: random policy for one episode
    env = PharmacyInventoryEnv(base_daily_demand=50.0, episode_length_days=60, seed=42)
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(60):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"\nRandom-policy smoke test over 60 days: total_reward={total_reward:.1f}, "
          f"stockout_days={env.cumulative_stockout_days}")
    print(f"Final observation: {obs}")