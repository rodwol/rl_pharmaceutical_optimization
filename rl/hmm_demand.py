"""
Hidden Markov Model layer for latent demand regime detection and
generation. This addresses the supervisor's comment directly:
 
    "Is there a way you can make it a Markov Decision Process and use HMMs?"
 
Two uses of the HMM here:
  1. GENERATIVE: sample a regime-switching sequence (stable/surge/disruption)
     to drive the synthetic data generator, replacing the purely seasonal-
     multiplier approach used in the previous script.
  2. INFERENTIAL: fit an HMM on observed demand sequences to recover regime
     *beliefs* (posterior probabilities), which become part of the MDP state
     fed to the DQN agent later.
 
Regime parameters are imported from literature_params.py so every number
traces back to a cited source.
"""
 
import os
import sys
import numpy as np
from hmmlearn import hmm
from data.literature_params import HMM_REGIMES, HMM_TRANSITION_MATRIX, HMM_INITIAL_STATE_PROBS

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
 
REGIME_NAMES = ["stable", "surge", "disruption"]
rng = np.random.default_rng(7)
 
 
def generate_regime_sequence(n_days: int, seed: int = None) -> np.ndarray:
    """
    Sample a sequence of latent regimes over n_days using the literature-
    calibrated transition matrix. Returns an array of ints (0=stable,
    1=surge, 2=disruption).
    """
    local_rng = np.random.default_rng(seed) if seed is not None else rng
    transition = np.array(HMM_TRANSITION_MATRIX)
    initial = np.array(HMM_INITIAL_STATE_PROBS)
 
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = local_rng.choice(3, p=initial)
    for t in range(1, n_days):
        regimes[t] = local_rng.choice(3, p=transition[regimes[t - 1]])
    return regimes
 
 
def regime_to_params(regime_idx: int) -> dict:
    """Map a regime index to its literature-calibrated parameter dict."""
    return HMM_REGIMES[REGIME_NAMES[regime_idx]]
 
 
def fit_hmm_to_demand(demand_sequence: np.ndarray, n_states: int = 3):
    """
    Fit a Gaussian HMM to an observed (or synthetic) 1-D demand sequence to
    recover latent regime structure. Used both to (a) sanity-check that our
    generated sequences are recoverable, and (b) later, to infer regime
    beliefs from real pharmacy demand logs if/when they become available.
 
    Returns the fitted model and the most-likely state sequence (Viterbi).
    """
    X = demand_sequence.reshape(-1, 1).astype(float)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                             n_iter=200, random_state=7)
    model.fit(X)
    hidden_states = model.predict(X)
    return model, hidden_states
 
 
def regime_belief_features(model, demand_sequence: np.ndarray) -> np.ndarray:
    """
    Returns the posterior probability (belief) over hidden states for each
    timestep — this is what gets appended to the MDP state vector for the
    DQN agent, per the proposal's state design:
    'Stock level, days since last order, pending order qty, 7-day demand
    signal, HMM regime belief'.
    """
    X = demand_sequence.reshape(-1, 1).astype(float)
    posteriors = model.predict_proba(X)  # shape (n_days, n_states)
    return posteriors
 
 
if __name__ == "__main__":
    # Quick self-test: generate a regime sequence and report its statistics
    n_days = 730  # 2 years, matches the synthetic generator's date range
    regimes = generate_regime_sequence(n_days, seed=42)
 
    print("Regime sequence self-test")
    print("─" * 50)
    for i, name in enumerate(REGIME_NAMES):
        pct = (regimes == i).mean() * 100
        print(f"  {name:12s}: {pct:5.1f}% of days")
 
    # Fit an HMM back onto a noisy demand series driven by these regimes to
    # confirm the regimes are recoverable from observed demand alone
    demand = np.array([
        rng.poisson(50 * HMM_REGIMES[REGIME_NAMES[r]]["demand_multiplier"])
        for r in regimes
    ])
    model, hidden = fit_hmm_to_demand(demand)
    print(f"\nFitted HMM recovered {len(set(hidden))} distinct states from demand alone")
    print(f"Learned transition matrix:\n{np.round(model.transmat_, 3)}")
