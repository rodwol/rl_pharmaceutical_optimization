"""
Economic Order Quantity (EOQ) baseline policy, used as the comparison
benchmark against the DQN agent per the proposal's evaluation design.

Classic EOQ formula:
    Q* = sqrt( (2 * D * S) / H )
where:
    D = annual demand
    S = fixed ordering cost per order
    H = annual holding cost per unit

A reorder point R = average lead-time demand + safety stock determines
WHEN to order; EOQ determines HOW MUCH to order each time.
"""

import numpy as np
from environment import PharmacyInventoryEnv


def compute_eoq(annual_demand: float, ordering_cost: float = 15.0,
                 holding_cost_per_unit_per_year: float = 0.5) -> float:
    """Classic EOQ formula. Costs are illustrative placeholders -- document
    as [ASSUM] in methodology since no Eritrea-specific ordering/holding
    cost figures were found in the literature reviewed."""
    return np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit_per_year)


def eoq_action_from_qty(order_qty: float, env: PharmacyInventoryEnv) -> int:
    """Maps a continuous EOQ quantity onto the env's nearest discrete action
    bucket, so the baseline operates under the same action space as the DQN
    agent for a fair comparison."""
    options = {0: 0.0, 1: env.q_min, 2: env.q_mid, 3: env.q_max}
    closest = min(options.items(), key=lambda kv: abs(kv[1] - order_qty))
    return closest[0]


def run_eoq_policy(env: PharmacyInventoryEnv, eoq_qty: float, reorder_point: float,
                    n_steps: int = 365):
    """
    Runs a (Q, R) policy: order `eoq_qty` whenever stock falls to or below
    `reorder_point` and no order is currently pending.
    """
    obs, _ = env.reset()
    total_reward, stockout_days, expired_total = 0.0, 0, 0
    action_for_order = eoq_action_from_qty(eoq_qty, env)

    for _ in range(n_steps):
        stock = obs[0] * (env.base_daily_demand * 60)
        pending = len(env.pending_orders)
        action = action_for_order if (stock <= reorder_point and pending == 0) else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        expired_total += info["expired_units"]
        if info["stockout"]:
            stockout_days += 1
        if terminated or truncated:
            break

    service_level = 1 - stockout_days / n_steps
    return {
        "total_reward": total_reward,
        "stockout_days": stockout_days,
        "service_level": service_level,
        "expired_units": expired_total,
    }


if __name__ == "__main__":
    base_demand = 50.0
    annual_demand = base_demand * 365

    eoq_qty = compute_eoq(annual_demand)
    # Reorder point: ~ (mean lead time * base demand) + safety stock buffer
    mean_lead_time_days = 18  # midpoint of LEAD_TIME_DAYS_ASSUM (7-30, mode 14)
    safety_stock_days = 7
    reorder_point = base_demand * (mean_lead_time_days + safety_stock_days)

    print(f"EOQ quantity (continuous):     {eoq_qty:.1f} units")
    print(f"Reorder point:                  {reorder_point:.1f} units")

    env = PharmacyInventoryEnv(base_daily_demand=base_demand, episode_length_days=365, seed=42)
    results = run_eoq_policy(env, eoq_qty, reorder_point)

    print("\nEOQ Baseline — 1 year simulation")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k:15s}: {v}")