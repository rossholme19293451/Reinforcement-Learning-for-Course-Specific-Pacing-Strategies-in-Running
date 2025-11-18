import matplotlib.pyplot as plt
from env.hybrid_keller_env import *

profile = np.loadtxt("../data/elevation_profiles/Ryde_10.csv", delimiter=",", skiprows=1)

r = 0.892 #s
Fmax = 12.2  #m/s^2
sigma = 41.54  #j/(kg*s)
E0 = 2405.8 #j/kg
tau = 337 #s

env = hybrid_keller_env(profile, r, Fmax, sigma, E0, tau)

obs, _ = env.reset()
done = False
reward = 0

distances, velocities, energies, elevations= [], [], [], []

while not done:
    action = [Fmax * 0.561]
    obs, temp_reward, terminated, truncated, info = env.step(action)
    print(info)
    env.render()

    distances.append(obs[0])
    velocities.append(obs[1])
    energies.append(obs[2])

    elevations.append(np.interp(obs[0], profile[:,0], profile[:,1]))

    reward += temp_reward
    print(f"Cumulative Reward: {reward}")

    done = terminated or truncated

print(f"Finished in {info['time']}s / {info['time']/60} mins")

#plot elevation and velocity
fig, ax1 = plt.subplots(figsize=(12,6))
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

#plot elevation and energy
fig, ax1 = plt.subplots(figsize=(12,6))
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

#plot energy and velocity
fig, ax1 = plt.subplots(figsize=(12,6))
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
