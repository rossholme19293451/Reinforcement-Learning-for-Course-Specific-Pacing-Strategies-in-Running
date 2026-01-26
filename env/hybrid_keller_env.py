import numpy as np
import gymnasium as gym
from gymnasium import spaces

g = 9.81 #gravity

class hybrid_keller_env(gym.Env):
    def __init__(
        self,
        elevation_profile,
        r, #s
        Fmax, #m/s^2
        sigma, #j/(kg*s)
        E0, #j/kg
        tau, #s
        k, #reward weight
        dt = 0.2, #s
        max_time = 1*3600,
        v_max = 13.0,
        grade_max = 0.35,
    ):

        super().__init__()
        self.elevation_profile = np.array(elevation_profile)
        self.distances = self.elevation_profile[:, 0]
        self.elevations = self.elevation_profile[:, 1]

        #compute grade at each distance index
        if len(self.elevations) > 1:
            self.grades = np.gradient(self.elevations, self.distances)
        else:
            self.grades = np.zeros_like(self.elevations)

        self.total_distance = float(self.distances[-1])
        self.max_time = max_time

        self.r = float(r)
        self.Fmax = float(Fmax)
        self.sigma = float(sigma)
        self.E0 = float(E0)
        self.dt = float(dt)
        self.g = float(g)
        self.recovery_rate = 1 - np.exp(-dt/tau)
        self.k = float(k)
        self.v_max = float(v_max)
        self.grade_max = np.max(np.abs(self.grades))


        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (1,),
            dtype=np.float32)

        #observation space normalised
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.distance = 0.0
        self.velocity = 0.0
        self.energy = self.E0
        self.time = 0.0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_grade(self, distance):
        return float(np.interp(distance, self.distances, self.grades))

    def _get_obs(self):
        distance_scaled = self.distance / self.total_distance
        velocity_scaled = self.velocity / self.v_max
        energy_scaled = self.energy / self.E0
        grade_scaled = self._get_grade(self.distance) / self.grade_max
        return np.clip(
            np.array(
                [distance_scaled, velocity_scaled, energy_scaled, grade_scaled], dtype=np.float32
            ),
            [0.0, 0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0, 1.0]
        )

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        f = (a + 1.0) / 2 * self.Fmax

        grade = self._get_grade(self.distance)
        grade_effect = self.g * grade

        #keller dynamics to calculate velocity
        dv = (f - self.velocity/self.r - grade_effect) * self.dt
        self.velocity = max(0.0, self.velocity + dv)

        #distance update
        dx = self.velocity * self.dt
        self.distance = min(self.distance + dx, self.total_distance)

        #energy update
        dE = (self.sigma - (f * self.velocity)) * self.dt
        if dE > 0:
            dE *= self.recovery_rate
        self.energy = min(self.energy + dE, self.E0)

        #time update
        self.time += self.dt

        #termination
        terminated = self.distance >= self.total_distance
        truncated = self.energy <= 0.0 or self.time > self.max_time

        #reward
        energy_used = min(0.0, dE) #energy used is >= 0.0
        reward = 0.01 * (dx + energy_used)

        #success
        if terminated:
            reward += 50
            reward += -self.k * (self.energy / self.E0) ** 2

        #failure
        if truncated and not terminated:
            reward = -100.0


        obs = self._get_obs()
        info = {"time": self.time, "grade": grade}
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"t = {self.time:.1f}s "
              f"| x = {self.distance:.1f}m "
              f"| v = {self.velocity:.1f}m/s "
              f"| E = {self.energy:.1f} J/kg")

