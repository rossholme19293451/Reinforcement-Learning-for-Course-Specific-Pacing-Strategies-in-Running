import matplotlib.pyplot as plt
from env.hybrid_keller_env import *

profile = np.loadtxt("../data/elevation_profiles/Ryde_10.csv", delimiter=",", skiprows=1)

r = 0.892 #s
Fmax = 12.2  #m/s^2
sigma = 41.54  #j/(kg*s)
E0 = 2405.8 #j/kg
tau = 337 #s
k = 10

env = hybrid_keller_env(profile, r, Fmax, sigma, E0, tau, k)

obs, _ = env.reset()
done = False

distances, velocities, energies, elevations= [], [], [], []
reward = 0

v_target = 6.1027

while not done:
    #get current grade
    current_grade = np.interp(obs[0], profile[:,0], env.grades)

    #compute force for constant velocity
    f = v_target / env.r + env.g * current_grade
    f = np.clip(f, 0.0, env.Fmax)

    action = f
    obs, temp_reward, terminated, truncated, info = env.step(action)
    print(info)
    env.render()

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




