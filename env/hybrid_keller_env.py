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
        sigma, #cal/(kg*s)
        E0, #cal/kg
        dt = 1.0,
        max_time = 3*3600
    ):

        super().__init__()
        self.elevation_profile = np.array(elevation_profile)

        self.distances = self.profile[:, 0]
        self.elevations = self.profile[:, 1]

        #compute grade at each distance index
        if len(self.elevations) > 1:
            self.grades = np.gradient(self.elevations, self.distances)
        else:
            self.grades = np.zeros_like(self.elevations)

        self.total_distance = float(self.distances[-1])
        self.max_time = max_time

        self.r = r
        self.Fmax = Fmax
        self.sigma = sigma
        self.E0 = E0
        self.dt = dt
        self.g = g

        # 0 <= f(t) <= Fmax
        self.action_space = spaces.Box(low = np.array([0.0]), high = np.array([self.Fmax]), dtype=np.float32)

        #0 <= distance <= total_distance, 0 <= velocity, 0 <= energy <= E0
        high = np.array([self.total_distance, np.finfo(np.float32).max, self.E0], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.distance = 0.0
        self.velocity = 0.0
        self.energy = self.E0
        self.time = 0.0
        obs = np.array([self.distance, self.velocity, self.energy], dtype=np.float32)
        return obs

    def _get_grade(self, distance):
        return np.interp(distance, self.distances, self.grades)

    def step(self, action):
        f = np.clip(action, 0.0, self.Fmax)

        grade = self._get_grade(self.distance)
        grade_effect = self.g * grade

        #keller dynamics to calculate velocity
        dv = (f - self.velocity/self.r - grade_effect ) * self.dt
        self.velocity = max(0.0, self.velocity + dv)

        #distance update
        dx = self.velocity * self.dt
        self.distance = min(self.distance + dx, self.total_distance)

        #energy update
        dE = (-self.sigma * self.velocity) * self.dt
        self.energy = max(0.0, self.energy + dE)

        #time update
        self.time += self.dt

        #termination
        terminated = self.energy <= 0.0 or self.distance >= self.total_distance
        truncated = self.time > self.max_time

        #reward
        reward = -0.1

        obs = np.array([self.distance, self.velocity, self.energy], dtype=np.float32)
        info = {"time": self.time, "grade": grade}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"t = {self.time:.1f}s "
              f"| x = {self.distance:.1f}m "
              f"| v = {self.velocity:.1f}m/s "
              f"| E = {self.energy:.1f} cal/kg")

