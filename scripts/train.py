import os
import numpy as np
import envs
import gymnasium as gym

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "scenes"))
env = gym.make('PickAndPlace-v0', xml_path=os.path.join(BASE_DIR, "scene.xml"))

observation, info = env.reset()

episode_over = False
total_reward = 0.0

while not episode_over:
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += float(reward)
    
    episode_over = terminated or truncated

env.close()