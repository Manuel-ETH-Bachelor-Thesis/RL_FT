import os
import numpy as np
import torch
import random
from torch.distributions import Normal
import envs
import gymnasium as gym
from models.architecture.Policy_Network import REINFORCE
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env = gym.make('Move-v0', xml_path=os.path.join(BASE_DIR, "resources", "robots", "unitree_g1", "scene.xml"))


total_episodes = int(5e3)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

plt.ion()
fig, ax = plt.subplots()

for seed in [42]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    agent = REINFORCE(obs_dim, action_dim, lr=3e-4)

    rewards_history = []

    for episode in range(total_episodes):
        obs, _ = env.reset(seed=seed + episode)

        max_ep_steps = 1000
        episode_reward = 0.0
        for step in range(max_ep_steps):
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            agent.rewards.append(reward)
            episode_reward += reward
            if terminated or truncated:
                break

        agent.update()

        if episode % 10 == 0:
            rewards_history.append(episode_reward)
            print(f"Episode {episode} - Total reward: {episode_reward}")
        if episode % 500 == 0:
            ax.clear()
            ax.plot(range(0, episode + 1, 10), rewards_history, label=f"Seed {seed} Training", color='blue')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("REINFORCE Learning Progress")
            ax.legend()
            plt.pause(0.0001)