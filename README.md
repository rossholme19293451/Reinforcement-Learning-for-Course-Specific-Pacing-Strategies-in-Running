# Reinforcement Learning for Course-Specific Pacing Strategies in Running

Undergraduate dissertation project applying Proximal Policy Optimization (PPO) to generate optimal pacing strategies for distance running that account for elevation changes.

## Overview

This project extends Keller's 1973 mathematical model of competitive running to incorporate realistic elevation profiles and asymmetric energy recovery dynamics (Skiba et al. 2012). A PPO agent (Schulman et al. 2017) using GAE (Schulman et al. 2018) to compute advantages, learns to optimize force application throughout a race course to minimize total time while respecting physiological constraints.

## Key Components

* **Environment** (`env/hybrid\_keller\_env.py`): Gymnasium environment implementing extended Keller dynamics with gradient-dependent forces
* **Agent** (`agents/PPO\_agent.py`): PPO implementation with Actor-Critic architecture and GAE
* **Data Processing** (`scripts/gpx\_to\_csv.py`): Converts GPX files to uniform 1m elevation profiles
* **Baselines** (`scripts/constant\_force.py`, `scripts/constant\_velocity.py`): Comparison strategies

## Requirements

* Python 3.8+
* gymnasium
* torch
* numpy
* scipy
* matplotlib
* pandas
* gpxpy

## Usage

Train a PPO agent on a course:

```python
from env.hybrid\_keller\_env import hybrid\_keller\_env
from agents.PPO\_agent import PPO\_Agent
import numpy as np

profile = np.loadtxt("data/elevation\_profiles/course.csv", delimiter=",", skiprows=1)
env = hybrid\_keller\_env(profile, r=0.892, Fmax=12.2, sigma=41.54, E0=2405.8, tau=337, sRw=0.4, tRw=40)

agent = PPO\_Agent(env, device="cpu")
agent.train()
episodes\_data = agent.run()
```

## References

* Keller, J.B. (1973). A Theory of Competitive Running. *Physics Today*
* Skiba et al. (2012). Modeling the Expenditure and Reconstitution of Work Capacity above Critical Power
* Schulman et al. (2017). Proximal Policy Optimization Algorithms
* Schulman et al. (2018). High-dimensional continuous control using generalized advantage estimation

## Author

Ross Holme <br>Oxford Brookes University <br>Undergraduate Dissertation, 2025

