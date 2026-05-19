"""
Script to start fine-tuning of pre-trained diffusion model using PPO
"""

import os
import sys
import pretty_errors

import math
import hydra
from omegaconf import OmegaConf

from envs.pick_and_place import PickAndPlace
from models.utils.gym_utils import make_async
from models.trainers.train_ppo_diffusion_agent import TrainPPODiffusionAgent

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
        raise ValueError(f"Base policy checkpoint not found at {cfg.base_policy_path}.")

    def make_env():
        return PickAndPlace(cfg, render_mode=None)

    env = make_async(
        env_fn=make_env, 
        num_envs=cfg.env.n_envs,
        asynchronous=True
    )

    agent = TrainPPODiffusionAgent(cfg)
    
    agent.venv = env
    
    agent.run()

if __name__ == "__main__":
    main()