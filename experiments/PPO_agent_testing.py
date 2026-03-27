from scipy.signal import savgol_filter
from env.hybrid_keller_env import *
from agents.PPO_agent import *

#load elevation profile
profile = np.loadtxt("../data/elevation_profiles/Brading_10k.csv", delimiter=",", skiprows=1)

#physiological parameters
r = 0.892  # s
Fmax = 12.2  # m/s^2
sigma = 41.54  # j/(kg*s)
E0 = 2405.8  # j/kg
tau = 337  # s
sRw = 0.40
tRw = 40

#initalise environment
env = hybrid_keller_env(profile, r, Fmax, sigma, E0, tau, sRw, tRw)

#initalise and train PPO agent
agent = PPO_Agent(env, device="cpu")
agent.train()   #train for specified number of frames

#run trained policy and collect trajectory data
episodes_data = agent.run()

#plot reward progression during episode
for ep_data in episodes_data:
    rewards = ep_data["reward"]
    plt.plot(rewards, label="Episode reward")

plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Reward vs Timestep")
plt.legend()
plt.show()

#extract and denormalise trajectory data
ep_data = episodes_data[0]

distances = np.array(ep_data["distance"]) * env.total_distance

velocities = np.array(ep_data["velocity"]) * env.v_max

#smooth velocity for visualisation
velocities = savgol_filter(velocities, window_length=500, polyorder=3)

energies = np.array(ep_data["energy"]) * env.E0

#convert actions back to force values
actions = []
for action in np.array(ep_data["action"]):
    actions.append(float(action) * env.Fmax)

actions = np.array(actions)

#smooth forces (actions) for visualisation
actions = savgol_filter(actions, window_length=500, polyorder=3)
print(actions)

#performance summary
print(f"Distance reached: {distances[-1]}, Timestep: {ep_data['time'][-1]}, Time: {ep_data['time'][-1]/60}")

#performance metrics
print("Mean force: ", actions.mean())
print("Std Force: ", actions.std())
print("Mean velocity: ", velocities.mean())
print("Std velocity: ", velocities.std())
print("Mean power: ", (actions * velocities).mean())
print("Final Energy: ", energies[-1])
print("Sigma: ", sigma)

#interpolate elevation at trajectory points
elevations = np.interp(distances, profile[:, 0], profile[:, 1])

# plot elevation and velocity
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(distances, elevations, color="red", label="Elevation (m)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.grid(True)

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
ax1.grid(True)

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
ax1.grid(True)

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
ax1.grid(True)

ax2.plot(distances, actions, color='purple', label="Force (m/s)")
ax2.set_ylabel("Force (m/s)", color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title("Force mapped onto Elevation Profile")
fig.tight_layout()
plt.show()

#compute average energy use per segment
energy_usage_windows = {}

window_length = len(energies) // 10

for i in range(10):
    window_start = i * window_length
    window_end = window_start + window_length
    window_midpoint = (window_start + window_end) // 2
    #energy used = energy at start - energy at end
    energy_usage_windows[window_midpoint] = energies[window_start] - energies[window_end]

#extract segment midpoints
mid_distances = np.array([distances[idx] for idx in energy_usage_windows.keys()])
mid_energies = np.array(list(energy_usage_windows.values()))

#segmental energy use and elevation
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(distances, elevations, color='red', alpha=0.4, label="Elevation (m)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)", color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(mid_distances, mid_energies, color='green', marker='o', linewidth=2)
ax2.set_ylabel("Avg Energy Usage (J/kg)", color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title("Energy Usage per Segment mapped onto Elevation Profile")
fig.tight_layout()
plt.show()