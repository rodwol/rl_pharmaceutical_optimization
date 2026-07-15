import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "training_rewards.csv")
df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Reward"])

plt.xlabel("Training Episode")
plt.ylabel("Episode Reward")
plt.title("DQN Training Reward")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_reward.png"), dpi=300)

plt.show()