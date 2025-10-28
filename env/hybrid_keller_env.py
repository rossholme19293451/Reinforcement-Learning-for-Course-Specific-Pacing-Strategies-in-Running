import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym



#create the environment
env = gym.make('HybridKeller-v0')
env.reset()

# play 10 games
for i in range(10):
    # initialize the variables
    done = False
    game_rew = 0
    while not done:
        # choose a random action
        action = env.action_space.sample()
        # take a step in the environment
        new_obs, rew, done, truncated, info = env.step(action)
        game_rew += rew
        # when is done, print the cumulative reward of the game and reset the environment
        if done or truncated:
            print('Episode %d finished, reward:%d' % (i, game_rew))
            env.reset()
