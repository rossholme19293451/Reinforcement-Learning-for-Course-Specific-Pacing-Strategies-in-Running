import numpy as np
import gymnasium as gym
from gymnasium import spaces

g = 9.81 #gravity acceleration

class hybrid_keller_env(gym.Env):
    """
    Gymnasium environment for running pacing optimisation.
    Extends Keller's 1973 model with elevation dependent forces and
    realistic energy reconstitution based on Skiba et al. 2012.
    """
    def __init__(
        self,
        elevation_profile,
        r, #resistance coefficient, s
        Fmax, #max force per unit mass, m/s^2
        sigma, #aerobic energy supply rate, j/(kg*s)
        E0, #inital anaerobic energy store, j/kg
        tau, #energy recovery time const, s
        sRw, #step reward weight
        tRw, # terminal reward weight
        dt = 0.2, #environment time step, s
    ):

        super().__init__()
        #load and process elevation profile
        self.elevation_profile = np.array(elevation_profile)
        self.distances = self.elevation_profile[:, 0]
        self.elevations = self.elevation_profile[:, 1]

        #compute grade at each distance index
        if len(self.elevations) > 1:
            self.grades = np.gradient(self.elevations, self.distances)
        else:
            self.grades = np.zeros_like(self.elevations)

        self.total_distance = float(self.distances[-1])

        #physiological and environment parameters
        self.r = float(r)
        self.Fmax = float(Fmax)
        self.sigma = float(sigma)
        self.E0 = float(E0)
        self.dt = float(dt)
        self.g = float(g)

        #exponential energy recovery rate
        self.recovery_rate = 1 - np.exp(-dt/tau)

        #sacle reward weights by course distance
        self.sRw = sRw * (self.total_distance / 10000)
        self.tRw = float(tRw)

        #max velocity calculation
        self.v_max = self.Fmax * r

        #normalisation const for grade obs
        if np.max(np.abs(self.elevations)) > 0:
            self.grade_max = np.max(np.abs(self.grades))
        else:
            self.grade_max = 0.001 #prevent division by 0

        #max time limit, 4x theoretical min time
        self.max_time = 4 * self.total_distance / self.v_max

        #action: force per unit mass, bounded to [-1, 1] then scaled to [0, Fmax]
        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (1,),
            dtype=np.float32)

        #observation: [distance_progress, velocity, energy, gradient, previous_force]
        #observation space normalised to [0, 1] except grade which is [-1, 1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        """
        super().reset(seed=seed, options=options)
        self.distance = 0.0
        self.velocity = 0.0
        self.energy = self.E0
        self.time = 0.0
        self.prev_f = 0.0 #previous force for action smoothing
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_grade(self, distance):
        """
        Get grade at current distance using linear interpolation.
        """
        return float(np.interp(distance, self.distances, self.grades))

    def _get_obs(self):
        """
        Return normalised obs vector.
        """
        distance_scaled = self.distance / self.total_distance
        velocity_scaled = self.velocity / self.v_max
        energy_scaled = self.energy / self.E0
        grade_scaled = self._get_grade(self.distance) / self.grade_max
        prev_f_scaled = self.prev_f / self.Fmax

        #clip to ensure obs are within bounds
        return np.clip(
            np.array(
                [distance_scaled, velocity_scaled, energy_scaled, grade_scaled, prev_f_scaled], dtype=np.float32
            ),
            [0.0, 0.0, 0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def step(self, action):
        """
        Execute one time step of the gym environment.
        Applies Keller like dynamics with elevation effects and energy reconstitution.
        """
        #scale action from [-1, 1] to [0, Fmax]
        a = float(np.clip(action[0], -1.0, 1.0))
        f_raw = (a + 1.0) / 2 * self.Fmax

        #exponetial smoothing to prevent abrupt force changes
        alpha = 0.2
        f = self.prev_f + alpha * (f_raw - self.prev_f)

        self.prev_f = f

        grade = self._get_grade(self.distance)
        grade_effect = self.g * grade   #gravitaional resistance

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

        #termination conditions
        terminated = self.distance >= self.total_distance   #course completed
        truncated = self.energy <= 0.0 or self.time > self.max_time  #failure

        #reward, encourages distance progress and efficient energy use
        energy_used = min(0.0, dE) #energy used is <= 0.0
        reward = 0.015 * (dx + (self.sRw * energy_used))

        #success
        if terminated:
            reward += 50    #large bonus for completion
            reward += -self.tRw * (self.energy / self.E0) ** 2  #penalty for leftover energy

        #failure
        if truncated and not terminated:
            reward = -100.0


        obs = self._get_obs()
        info = {"time": self.time, "grade": grade}
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Print current state for debugging.
        """
        print(f"t = {self.time:.1f}s "
              f"| x = {self.distance:.1f}m "
              f"| v = {self.velocity:.1f}m/s "
              f"| E = {self.energy:.1f} J/kg")