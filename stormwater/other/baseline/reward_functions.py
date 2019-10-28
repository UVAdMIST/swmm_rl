"""
Reward functions for DQN RL

Written by Ben Bowes, May 6, 2019
"""

import numpy as np


def reward_function1(depth, flood):
    # depth reward
    depth = [-1 if i < 1.0 else 1 for i in depth]
    depth = [-3 if i > 4.0 else i for i in depth]
    weights = [1, 1]
    depth_reward = np.dot(depth, np.transpose(weights))
    # flooding reward
    flood = [-10 if i > 0.0 else 1 for i in flood]
    weights = [1, 1, 1]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = depth_reward + flood_reward
    return total_reward


def reward_function2(depth, flood):
    # # depth reward
    # depth = [-1 if i < 1.0 else 1 for i in depth]
    # depth = [-3 if i > 4.0 else i for i in depth]
    # weights = [1, 1]
    # depth_reward = np.dot(depth, np.transpose(weights))
    # flooding reward
    flood = [-(2**(i)) if i > 0.0 else 0 for i in flood]
    weights = [2, 2, 2, 2, 2, 2, 1, 1, 1]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = flood_reward
    return total_reward

def reward_function3(depth, flood):
    return -sum(flood)