"""
  - Load & validate synthetic_demand_data.csv
  - pre-train HMM once on the dataset (or load cached)
  - load the frozen HMM inferrer into environment
  - train DQN across multiple medicines
  - evaluate DQN vs. EOQ baseline
"""

import argparse, os, numpy as np, pandas as pd, sys
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import Env

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from hmm_demand import pretrain_hmm, RegimeBeliefInferrer, HMM_CACHE_PATH
from environment import PharmacyInventoryEnv, BASE_DEMAND, HOLDING_COST_PER_UNIT_PER_DAY


# EOQ Baseline 

def compute_eoq(annual_demand, ordering_cost=2.0,
                holding_cost_per_unit_per_year=None):
    if holding_cost_per_unit_per_year is None:
        holding_cost_per_unit_per_year = HOLDING_COST_PER_UNIT_PER_DAY * 365
    return np.sqrt((2 * annual_demand * ordering_cost)
                   / holding_cost_per_unit_per_year)


def run_eoq_episode(env: PharmacyInventoryEnv) -> dict:
    """
    (Q, R) policy:  order EOQ units whenever stock falls below
    the reorder point (mean_lead_time × base_demand + safety_stock).
    Operates under the same environment dynamics as the DQN agent.
    """
    base = env.base_daily_demand
    annual_demand = base * 365
    eoq_qty = compute_eoq(annual_demand)
    mean_lead_time = 18   # days (mode of triangular distribution)
    safety_stock = base * 7
    reorder_point = base * mean_lead_time + safety_stock

    # Snap EOQ to nearest discrete action
    q_options = {0: 0.0, 1: env.q_min, 2: env.q_mid, 3: env.q_max}
    action = min(q_options, key=lambda a: abs(q_options[a] - eoq_qty))

    obs, _ = env.reset()
    total_reward, stockout_days, expired = 0.0, 0, 0
    for _ in range(env.episode_length_days):
        stock = obs[0] * (base * 60)
        pending_orders = len(env.pending_orders)
        chosen = action if (stock <= reorder_point and pending_orders == 0) else 0
        obs, reward, terminated, truncated, info = env.step(chosen)
        total_reward += reward
        expired += info["expired_units"]
        if info["stockout"]:
            stockout_days += 1
        if terminated or truncated:
            break
    return {
        "total_reward": total_reward,
        "stockout_days": stockout_days,
        "service_level": 1 - stockout_days / env.episode_length_days,
        "expired_units": expired,
    }


# ── Dataset Validation ────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> bool:
    from data.literature_params import STOCKOUT_INCIDENCE, STOCKOUT_DURATION
    print("\n── Dataset Validation ──────────────────────────────────────")

    abx_pt = df[df["category"] == "Antibiotic"]["stockout_flag"].mean()
    target_pt = STOCKOUT_INCIDENCE["point_in_time_stockout_rate_ERI"]
    pt_ok = abs(abx_pt - target_pt) < 0.10
    print(f"  Antibiotic point-in-time stockout: "
          f"{abx_pt:.3f}  (target ≈ {target_pt:.3f})  {'✓' if pt_ok else '!'}")

    six_month = 182
    checks = []
    for mid in df["medication_id"].unique():
        sub = df[df["medication_id"] == mid].sort_values("date")
        for start in range(0, len(sub) - six_month, six_month):
            checks.append(sub.iloc[start:start + six_month]["stockout_flag"].sum() > 0)
    incidence = np.mean(checks)
    inc_ok = abs(incidence - 0.60) < 0.25
    print(f"  6-month stockout incidence:        "
          f"{incidence:.3f}  (target ≈ 0.600)  {'✓' if inc_ok else '!'}")

    durations = []
    for mid in df["medication_id"].unique():
        flags = df[df["medication_id"] == mid].sort_values("date")["stockout_flag"].values
        run = 0
        for f in flags:
            if f == 1:
                run += 1
            elif run > 0:
                durations.append(run); run = 0
        if run > 0:
            durations.append(run)
    mean_dur = np.mean(durations) if durations else 0
    dur_ok = 5 < mean_dur < 100
    print(f"  Mean stockout duration:            "
          f"{mean_dur:.1f}d  (target ≈ 38.8d, range 10-157d)  "
          f"{'✓' if dur_ok else '!'}")
    print(f"  Records: {len(df):,}  | Medicines: {df['medication_id'].nunique()}")
    print("────────────────────────────────────────────────────────────\n")
    return pt_ok and inc_ok and dur_ok


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(model, inferrer, meds, n_per_med=3, ep_len=365):
    """Evaluate DQN and EOQ across all medicines, n_per_med seeds each."""
    dqn_r, dqn_so, dqn_sl = [], [], []
    eoq_r, eoq_so, eoq_sl = [], [], []

    for med_id in meds:
        for seed in range(2000, 2000 + n_per_med):
            # DQN
            env = PharmacyInventoryEnv(medication_id=med_id, episode_length_days=ep_len,
                                        seed=seed, hmm_inferrer=inferrer)
            obs, _ = env.reset()
            total_r, so_days = 0.0, 0
            for _ in range(ep_len):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(action))
                total_r += reward
                if info["stockout"]: so_days += 1
                if term or trunc: break
            dqn_r.append(total_r); dqn_so.append(so_days)
            dqn_sl.append(1 - so_days / ep_len)

            # EOQ
            eoq_env = PharmacyInventoryEnv(medication_id=med_id,
                                             episode_length_days=ep_len, seed=seed)
            eoq_res = run_eoq_episode(eoq_env)
            eoq_r.append(eoq_res["total_reward"])
            eoq_so.append(eoq_res["stockout_days"])
            eoq_sl.append(eoq_res["service_level"])

    return (
        {"mean_reward": np.mean(dqn_r), "std_reward": np.std(dqn_r),
         "mean_stockout_days": np.mean(dqn_so), "mean_service_level": np.mean(dqn_sl)},
        {"mean_reward": np.mean(eoq_r), "std_reward": np.std(eoq_r),
         "mean_stockout_days": np.mean(eoq_so), "mean_service_level": np.mean(eoq_sl)},
    )


# ── Multi-medicine training wrapper ──────────────────────────────────────

class MultiMedEnv(Env):
    """
    Wraps PharmacyInventoryEnv to randomly select a medicine at the
    start of each episode. This forces the DQN agent to learn a
    generalised ordering policy rather than one specific to a single
    medicine's demand level.
    """
    def __init__(self, meds, ep_len, inferrer, base_seed=42):
        super().__init__()
        self.meds = meds
        self.ep_len = ep_len
        self.inferrer = inferrer
        self.rng = np.random.default_rng(base_seed)
        self._make_env(meds[0], base_seed)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _make_env(self, med_id, seed):
        self._env = PharmacyInventoryEnv(
            medication_id=med_id, episode_length_days=self.ep_len,
            seed=seed, hmm_inferrer=self.inferrer,
        )

    def reset(self, seed=None, options=None):
        med_id = self.meds[int(self.rng.integers(len(self.meds)))]
        ep_seed = int(self.rng.integers(10000))
        self._make_env(med_id, ep_seed)
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def render(self): self._env.render()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500,
                         help="Total training episodes")
    parser.add_argument("--save-path", type=str, default="models/dqn_agent.zip")
    parser.add_argument("--data-path", type=str,
                         default="synthetic_demand_data.csv")
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()

    ep_len = 365
    total_timesteps = args.episodes * ep_len

    # Step 1: Load & validate dataset
    print(f"Loading {args.data_path}...")
    df = pd.read_csv(args.data_path)
    validate_dataset(df)

    # Step 2: Pre-train HMM
    if args.skip_pretrain and os.path.exists(HMM_CACHE_PATH):
        print(f"Loading cached HMM from {HMM_CACHE_PATH}")
    else:
        print("Pre-training HMM on demand dataset...")
        pretrain_hmm(df, save_path=HMM_CACHE_PATH)

    # Step 3: Load frozen inferrer
    inferrer = RegimeBeliefInferrer(load_path=HMM_CACHE_PATH)
    print("HMM inferrer ready (frozen)\n")

    # Step 4: Train DQN across all medicines
    meds = list(BASE_DEMAND.keys())
    train_env = Monitor(MultiMedEnv(meds, ep_len, inferrer, base_seed=42))
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.95,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.08,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=ep_len * 50,
        save_path=os.path.dirname(args.save_path),
        name_prefix="dqn_ckpt",
    )
    print(f"Training DQN: {args.episodes} episodes "
          f"({total_timesteps:,} timesteps, multi-medicine)...")
    model.learn(total_timesteps=total_timesteps,
                progress_bar=False, callback=checkpoint_cb)
    model.save(args.save_path)
    print(f"\nModel saved → {args.save_path}")

    # Step 5: Evaluate
    print("\nEvaluating across all 15 medicines (3 seeds each)...")
    dqn, eoq = evaluate(model, inferrer, meds, n_per_med=3, ep_len=ep_len)

    print("\n" + "=" * 64)
    print("RESULTS: DQN (enriched env) vs. EOQ Baseline")
    print("=" * 64)
    rows = [
        ("Mean reward/year",       "mean_reward",        False),
        ("Mean stockout days/yr",  "mean_stockout_days", True),
        ("Mean service level",     "mean_service_level", False),
    ]
    print(f"  {'Metric':<26}{'DQN':>12}{'EOQ':>12}{'Δ (DQN better)':>16}")
    print("  " + "-" * 66)
    for label, key, lower_is_better in rows:
        d, e = dqn[key], eoq[key]
        if key == "mean_service_level":
            ds, es = f"{d*100:.1f}%", f"{e*100:.1f}%"
            delta = f"{(d-e)*100:+.1f}pp"
        else:
            ds, es = f"{d:.1f}", f"{e:.1f}"
            pct = (e-d)/abs(e)*100 if lower_is_better else (d-e)/abs(e)*100
            delta = f"{pct:+.1f}%"
        print(f"  {label:<26}{ds:>12}{es:>12}{delta:>16}")
    print("=" * 64)
    print(f"\n  DQN std reward: ±{dqn['std_reward']:.1f}   "
          f"EOQ std reward: ±{eoq['std_reward']:.1f}")


if __name__ == "__main__":
    main()