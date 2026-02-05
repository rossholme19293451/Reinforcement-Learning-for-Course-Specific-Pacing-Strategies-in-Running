import matplotlib.pyplot as plt
import pandas as pd

from env.hybrid_keller_env import *
from agents.PPO_agent import *

profile = np.loadtxt("../data/elevation_profiles/Brading_10k.csv", delimiter=",", skiprows=1)

r = 0.892  # s
Fmax = 12.2  # m/s^2
sigma = 41.54  # j/(kg*s)
E0 = 2405.8  # j/kg
tau = 337  # s

sRw_list = [0.7]
tRw_list = [30, 32 ,35, 37, 40]

results = []

for sRw in sRw_list:
    for tRw in tRw_list:
        env = hybrid_keller_env(profile, r, Fmax, sigma, E0, tau, sRw, tRw)
        agent = PPO_Agent(env, device="cpu")
        agent.train()

        episodes_data = agent.run(10)

        rewards = []
        times = []
        mean_forces = []
        mean_velocities = []
        mean_powers = []

        for ep in episodes_data:
            velocities = np.array(ep["velocity"]) * env.v_max
            actions = np.array(ep["action"])[:, 0]
            forces = (actions + 1.0) / 2 * env.Fmax

            rewards.append(np.sum(ep["reward"]))
            times.append(ep["time"][-1])
            mean_forces.append(forces.mean())
            mean_velocities.append(velocities.mean())
            mean_powers.append((forces * velocities).mean())

        results.append({
            "sRw": sRw,
            "tRw": tRw,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "mean_force": np.mean(mean_forces),
            "mean_velocities": np.mean(mean_velocities),
            "mean_powers": np.mean(mean_powers),
        })


df = pd.DataFrame(results)
print(df.to_string())


