import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from env.hybrid_keller_env import *

profile = np.loadtxt("../data/elevation_profiles/Ryde_10.csv", delimiter=",", skiprows=1)

r = 0.892 #s
Fmax = 12.2  #m/s^2
sigma = 41.54  #j/(kg*s)
E0 = 2405.8 #j/kg
tau = 337 #s
sRw = 0.4
tRw = 40

env = hybrid_keller_env(profile, r, Fmax, sigma, E0, tau, sRw, tRw)

obs, _ = env.reset()
done = False
reward = 0

actions, distances, velocities, energies, elevations= [], [], [], [], []

while not done:
    f = Fmax * 0.5614
    action = [(f / Fmax * 2) - 1.0]
    obs, temp_reward, terminated, truncated, info = env.step(action)
    print(info)
    env.render()

    actions.append(f)
    distance = obs[0] * env.total_distance
    velocity = obs[1] * env.v_max
    energy = obs[2] * env.E0

    distances.append(distance)
    velocities.append(velocity)
    energies.append(energy)

    elevations.append(np.interp(distance, profile[:,0], profile[:,1]))

    reward += temp_reward
    print(f"Cumulative Reward: {reward}")

    done = terminated or truncated

print(f"Finished in {info['time']}s / {info['time']/60} mins")

velocities = savgol_filter(velocities, window_length=500, polyorder=3)

#plot elevation and velocity
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(distances, elevations, color='red', label="Elevation (m)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)", color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(distances, velocities, color='blue', label="Velocity (m/s)")
ax2.set_ylabel("Velocity (m/s)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title("Velocity mapped onto Elevation Profile")
fig.tight_layout()
plt.show()

#plot elevation and energy
fig, ax1 = plt.subplots(figsize=(12,6))
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

#plot energy and velocity
fig, ax1 = plt.subplots(figsize=(12,6))
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
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(distances, actions, color='purple', label="Force (m/s)")
ax2.set_ylabel("Force (m/s)", color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title("Force mapped onto Elevation Profile")
fig.tight_layout()
plt.show()

energy_usage_windows = {}

window_length = len(distances) // 10

for i in range(10):
    window_start = i * window_length
    window_end = window_start + window_length
    window_midpoint = (window_start + window_end) // 2
    energy_usage_windows[window_midpoint] = energies[window_start] - energies[window_end]

mid_distances = np.array([distances[idx] for idx in energy_usage_windows.keys()])
mid_energies = np.array(list(energy_usage_windows.values()))

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