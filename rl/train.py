"""
Trains the DQN agent on the pharmacy replenishment environment and
evaluates it against the EOQ baseline, reporting the metrics required by
the ML Track ("Initial Performance metrics").

Usage:
    python train.py --episodes 200 --save-path models/dqn_agent.zip
"""

import argparse
import os
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback

from agent import build_env, build_agent
from eoq_baseline import compute_eoq, run_eoq_policy
from environment import PharmacyInventoryEnv


def evaluate_policy(model, env_factory, n_episodes: int = 10, episode_length: int = 365):
    """Runs the trained DQN policy (deterministic) over several episodes
    and reports averaged metrics."""
    rewards, stockout_rates, expired_totals = [], [], []
    for ep in range(n_episodes):
        env = env_factory(seed=1000 + ep)
        obs, _ = env.reset()
        total_reward, stockout_days, expired_total = 0.0, 0, 0
        for _ in range(episode_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            expired_total += info["expired_units"]
            if info["stockout"]:
                stockout_days += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)
        stockout_rates.append(stockout_days / episode_length)
        expired_totals.append(expired_total)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_stockout_days": float(np.mean(stockout_rates) * episode_length),
        "mean_service_level": float(1 - np.mean(stockout_rates)),
        "mean_expired_units": float(np.mean(expired_totals)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200,
                         help="Number of training episodes (each 365 steps)")
    parser.add_argument("--save-path", type=str, default="models/dqn_agent.zip")
    parser.add_argument("--base-demand", type=float, default=50.0)
    parser.add_argument("--resume-from", type=str, default=None,
                         help="Path to an existing model .zip to continue training from")
    args = parser.parse_args()

    episode_length = 365
    total_timesteps = args.episodes * episode_length

    print(f"Training DQN for {args.episodes} episodes ({total_timesteps:,} timesteps)...")
    train_env = build_env(base_daily_demand=args.base_demand, episode_length_days=episode_length, seed=42)

    if args.resume_from and os.path.exists(args.resume_from):
        from stable_baselines3 import DQN as DQNClass
        print(f"Resuming from {args.resume_from}")
        model = DQNClass.load(args.resume_from, env=train_env)
    else:
        model = build_agent(train_env)

    checkpoint_dir = os.path.dirname(args.save_path) or "."
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(episode_length * 25, 1000),  # checkpoint roughly every 25 episodes
        save_path=checkpoint_dir,
        name_prefix="dqn_checkpoint",
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=checkpoint_callback)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    # ── Evaluate DQN ──
    print("\nEvaluating trained DQN agent over 10 held-out episodes...")
    dqn_metrics = evaluate_policy(
        model,
        lambda seed: PharmacyInventoryEnv(base_daily_demand=args.base_demand,
                                           episode_length_days=episode_length, seed=seed),
        n_episodes=10, episode_length=episode_length,
    )

    # ── Evaluate EOQ baseline ──
    print("Evaluating EOQ baseline over 10 held-out episodes...")
    annual_demand = args.base_demand * 365
    eoq_qty = compute_eoq(annual_demand)
    reorder_point = args.base_demand * (18 + 7)  # mean lead time + safety stock
    eoq_results = []
    for seed in range(1000, 1010):
        eoq_env = PharmacyInventoryEnv(base_daily_demand=args.base_demand,
                                        episode_length_days=episode_length, seed=seed)
        eoq_results.append(run_eoq_policy(eoq_env, eoq_qty, reorder_point, n_steps=episode_length))
    eoq_metrics = {
        "mean_reward": float(np.mean([r["total_reward"] for r in eoq_results])),
        "mean_stockout_days": float(np.mean([r["stockout_days"] for r in eoq_results])),
        "mean_service_level": float(np.mean([r["service_level"] for r in eoq_results])),
        "mean_expired_units": float(np.mean([r["expired_units"] for r in eoq_results])),
    }

    # ── Report ──
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: DQN vs. EOQ Baseline")
    print("=" * 60)
    print(f"{'Metric':<25}{'DQN Agent':>15}{'EOQ Baseline':>18}")
    print("-" * 60)
    print(f"{'Mean reward/year':<25}{dqn_metrics['mean_reward']:>15.1f}{eoq_metrics['mean_reward']:>18.1f}")
    print(f"{'Mean stockout days/yr':<25}{dqn_metrics['mean_stockout_days']:>15.1f}{eoq_metrics['mean_stockout_days']:>18.1f}")
    print(f"{'Mean service level':<25}{dqn_metrics['mean_service_level']*100:>14.1f}%{eoq_metrics['mean_service_level']*100:>17.1f}%")
    print(f"{'Mean expired units/yr':<25}{dqn_metrics['mean_expired_units']:>15.1f}{eoq_metrics['mean_expired_units']:>18.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()