import os
# os.environ["MUJOCO_GL"] = "osmesa" # "egl" on euler cluster
import numpy as np
import envs
import hydra
import gymnasium as gym
from omegaconf import DictConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources", "custom"))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    env = gym.make('PickAndPlace-v0', xml_path=os.path.join(BASE_DIR, "custom_scene.xml"), cfg=cfg)
    for i in range(10):
        episode(env)

def episode(env):
    observation, info = env.reset()

    episode_over = False
    total_reward = 0.0

    while not episode_over:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        
        episode_over = terminated or truncated
        print(observation)

train()
