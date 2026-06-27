"""
DQN agent definition for the pharmacy replenishment MDP, using
Stable-Baselines3's DQN implementation.

Architecture (per proposal's "Model Architecture" requirement):
    Input layer  : 8 features (state vector from environment.py)
    Hidden layer1: 64 units, ReLU
    Hidden layer2: 64 units, ReLU
    Output layer : 4 units (Q-values for each discrete action)
    Optimizer    : Adam
    Loss         : Smooth L1 (Huber) loss on TD error (SB3 default for DQN)
    Exploration  : epsilon-greedy, linearly annealed
"""

import sys
import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from environment import PharmacyInventoryEnv

# Network architecture: two hidden layers of 64 units each (ReLU by default in SB3 MLP policy)
POLICY_KWARGS = dict(net_arch=[64, 64])

DQN_HYPERPARAMS = dict(
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.95,                 # discount factor, per proposal's MDP table
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=POLICY_KWARGS,
    verbose=1,
)


def build_env(base_daily_demand: float = 50.0, episode_length_days: int = 365, seed: int = 42):
    env = PharmacyInventoryEnv(base_daily_demand=base_daily_demand,
                                episode_length_days=episode_length_days, seed=seed)
    return Monitor(env)


def build_agent(env, **override_kwargs):
    params = {**DQN_HYPERPARAMS, **override_kwargs}
    model = DQN("MlpPolicy", env, **params)
    return model


if __name__ == "__main__":
    env = build_env()
    model = build_agent(env)
    print("DQN agent built successfully.")
    print(f"Policy network: {model.policy}")
