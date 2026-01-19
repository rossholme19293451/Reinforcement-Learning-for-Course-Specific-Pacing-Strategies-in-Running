import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from env.hybrid_keller_env import *
from agents.PPO_agent import *

profile = np.loadtxt("../data/elevation_profiles/Ryde_10.csv", delimiter=",", skiprows=1)

r = 0.892  # s
Fmax = 12.2  # m/s^2
sigma = 41.54  # j/(kg*s)
E0 = 2405.8  # j/kg
tau = 337  # s
k = 15

env = hybrid_keller_env(profile, r, Fmax, sigma, E0, tau, k)

agent = PPO_Agent(env, device="cpu")
agent.train()
episodes_data = agent.run()

for ep_data in episodes_data:
    rewards = ep_data["reward"]
    plt.plot(rewards, label="Episode reward")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Reward vs Timestep")
plt.legend()
plt.show()

ep_data = episodes_data[0]

distances = np.array(ep_data["distance"]) * env.total_distance

velocities = np.array(ep_data["velocity"]) * env.v_max
velocities = savgol_filter(velocities, window_length=500, polyorder=3)

energies = np.array(ep_data["energy"]) * env.E0

actions = []
for action in np.array(ep_data["action"]):
    actions.append((action[0] + 1.0) / 2 * env.Fmax)

actions = np.array(actions)
actions = savgol_filter(actions, window_length=500, polyorder=3)
print(actions)

print(f"Distance reached: {distances[-1]}, Timestep: {len(distances)/(1/env.dt)}, Time: {len(distances)/60/(1/env.dt)}")

elevations = np.interp(distances, profile[:, 0], profile[:, 1])

# plot elevation and velocity
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(distances, elevations, color='red', label="Elevation (m)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)", color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(distances, velocities, color='blue', label="Velocity (m/s)")
ax2.set_ylabel("Velocity (m/s)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title("Velocity mapped onto Elevation Profile")
fig.tight_layout()
plt.show()

# plot elevation and energy
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(distances, elevations, color='red', label="Elevation (m)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)", color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(distances, energies, color='green', label="Energy (J/kg)")
ax2.set_ylabel("Energy (J/kg)", color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title("Energy mapped onto Elevation Profile")
fig.tight_layout()
plt.show()

# plot energy and velocity
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(distances, velocities, color='blue', label="Velocity (m/s)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Velocity (m/s)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(distances, energies, color='green', label="Energy (J/kg)")
ax2.set_ylabel("Energy (J/kg)", color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title("Energy mapped onto Velocity Profile")
fig.tight_layout()
plt.show()

# plot force (actions) and elevation
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(distances, elevations, color='red', label="Elevation (m)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)", color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax2 = ax1.twinx()

ax2.plot(distances, actions, color='blue', label="Force (m/s)")
ax2.set_ylabel("Force (m/s)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
plt.title("Force mapped onto Elevation Profile")
fig.tight_layout()
plt.show()
