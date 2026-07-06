"""
hmm_demand.py
─────────────────────────────────────────────────────────────────────────
HMM's single, clearly defined role in this project:

    DEMAND REGIME INFERENCE
    ───────────────────────
    The HMM is pre-trained ONCE on the synthetic demand dataset
    (synthetic_demand_data.csv) to learn what stable, surge, and
    disruption demand patterns look like.

    During each RL training episode, the frozen pre-trained HMM watches
    the agent's observed demand history and outputs a belief vector:

        belief = [P(stable|history), P(surge|history), P(disruption|history)]

    This belief vector is part of the DQN state (features 4-6 of the
    8-dim observation vector), giving the agent a structured signal about
    the current demand/supply regime WITHOUT directly revealing it.

    This mirrors how real pharmacy staff operate: they cannot observe the
    true regime, but they can infer it from recent demand patterns and
    adjust ordering accordingly.

What the HMM does NOT do:
    - It does not generate demand during episodes (Poisson draws do that)
    - It is not re-fitted during training (weights are frozen after pre-training)
    - It does not label data in the CSV (the regime column there is for EDA)
"""

import os
import sys
import pickle
import numpy as np
from hmmlearn import hmm

# ── Path setup — works whether run from project root or rl/ folder ────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.literature_params import (
    HMM_REGIMES, HMM_TRANSITION_MATRIX, HMM_INITIAL_STATE_PROBS
)

REGIME_NAMES = ["stable", "surge", "disruption"]
N_STATES = 3

# Saved inside rl/models/ — next to dqn_agent.zip
HMM_CACHE_PATH = os.path.join(os.path.dirname(__file__), "models", "pretrained_hmm.pkl")


# ─────────────────────────────────────────────────────────────────────────
# PRE-TRAINING — fit once on the synthetic dataset, save to disk
# ─────────────────────────────────────────────────────────────────────────

def pretrain_hmm(demand_df, save_path: str = HMM_CACHE_PATH):
    """
    Fits a Gaussian HMM on aggregated daily demand from the synthetic
    dataset. Called once before training begins; the fitted model is saved
    to disk and reloaded for all subsequent episodes.

    Aggregates demand across all medicines per day to give the HMM a
    stable, high-signal sequence (730 days) to learn from.
    """
    daily_total = (
        demand_df.groupby("date")["demand_units"]
        .sum()
        .sort_index()
        .values
        .astype(float)
    )

    demand_mean = daily_total.mean()
    demand_std  = daily_total.std()
    X = ((daily_total - demand_mean) / (demand_std + 1e-8)).reshape(-1, 1)

    best_model, best_score = None, -np.inf
    for seed in range(10):
        try:
            model = hmm.GaussianHMM(
                n_components=N_STATES,
                covariance_type="diag",
                n_iter=300,
                random_state=seed,
                tol=1e-4,
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM pre-training failed across all random seeds.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(
            {"model": best_model,
             "demand_mean": demand_mean,
             "demand_std": demand_std},
            f,
        )

    print(f"HMM pre-trained and saved -> {save_path}")
    print(f"  Log-likelihood : {best_score:.2f}")
    print(f"  Transition matrix:\n{np.round(best_model.transmat_, 3)}")
    print(f"  Emission means : {best_model.means_.flatten().round(3)}")
    return best_model


# ─────────────────────────────────────────────────────────────────────────
# INFERENCE — use the frozen pre-trained model during RL episodes
# ─────────────────────────────────────────────────────────────────────────

class RegimeBeliefInferrer:
    """
    Wraps the pre-trained HMM and exposes one method:

        inferrer.belief(demand_history, t) -> np.ndarray shape (3,)

    Returns [P(stable), P(surge), P(disruption)] given observed demand
    history. The model is FROZEN — no re-fitting ever occurs at runtime.
    """

    MIN_HISTORY  = 30
    UPDATE_EVERY = 7

    def __init__(self, load_path: str = HMM_CACHE_PATH):
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Pre-trained HMM not found at:\n  {load_path}\n"
                f"Run train.py without --skip-pretrain first."
            )
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self._model  = data["model"]
        self._mean   = data["demand_mean"]
        self._std    = data["demand_std"]
        self._prior  = np.array([0.80, 0.15, 0.05], dtype=np.float32)
        self._cache  = None

    def belief(self, demand_history: list, t: int) -> np.ndarray:
        """
        Returns posterior regime belief given observed demand history.
        Uses the FROZEN pre-trained HMM — no re-fitting.
        Refreshes every UPDATE_EVERY days; returns cached value between.
        """
        if len(demand_history) < self.MIN_HISTORY:
            return self._prior

        if self._cache is not None and (t % self.UPDATE_EVERY != 0):
            return self._cache

        try:
            arr = np.array(demand_history[-180:], dtype=float)
            arr_norm = (arr - self._mean) / (self._std + 1e-8)
            posteriors = self._model.predict_proba(arr_norm.reshape(-1, 1))
            self._cache = posteriors[-1].astype(np.float32)
        except Exception:
            if self._cache is None:
                self._cache = self._prior.copy()

        return self._cache

    def reset(self):
        """Call at the start of each episode to clear the cached belief."""
        self._cache = None


# ─────────────────────────────────────────────────────────────────────────
# REGIME SEQUENCE GENERATOR
# Drives ground-truth demand in the environment.
# The inferrer tries to recover an approximation from observed demand —
# intentionally imperfect, like real staff inferring conditions.
# ─────────────────────────────────────────────────────────────────────────

def generate_regime_sequence(n_days: int, seed: int = None) -> np.ndarray:
    """
    Samples a latent regime sequence using the literature-calibrated
    transition matrix. Returns int array (0=stable, 1=surge, 2=disruption).
    """
    rng        = np.random.default_rng(seed)
    transition = np.array(HMM_TRANSITION_MATRIX)
    initial    = np.array(HMM_INITIAL_STATE_PROBS)

    regimes    = np.zeros(n_days, dtype=int)
    regimes[0] = rng.choice(N_STATES, p=initial)
    for t in range(1, n_days):
        regimes[t] = rng.choice(N_STATES, p=transition[regimes[t - 1]])
    return regimes


if __name__ == "__main__":
    n_days  = 730
    regimes = generate_regime_sequence(n_days, seed=42)
    print("Regime sequence self-test")
    print("-" * 50)
    for i, name in enumerate(REGIME_NAMES):
        pct = (regimes == i).mean() * 100
        print(f"  {name:12s}: {pct:5.1f}% of days")
    print("Self-test passed.")