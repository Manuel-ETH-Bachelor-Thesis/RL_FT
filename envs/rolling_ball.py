import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from mujoco import viewer
from gymnasium.spaces import Box

# Dependent on upstream IL setup
class PickAndPlace(MujocoEnv):
    def __init__(self, xml_path: str ):
        self.frame_skip = 1
        
        observation_space = Box(
            low=0.0,
            high=0.0,
            shape=(0,),
            dtype=np.float32,
        )

        super().__init__(
            xml_path,
            frame_skip=self.frame_skip,
            observation_space=observation_space,
            render_mode="human",
        )

        self.action_space = Box(
            low=0.0,
            high=0.0,
            shape=(0,),
            dtype=np.float32,
        )

    def step(self, action):
        self.do_simulation(np.zeros(0), self.frame_skip)
        self.render()
        return np.zeros(0, dtype=np.float64), np.float64(0.0), False, False, {"a": np.float64(0.0)} 
    
    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return np.zeros(0, dtype=np.float64)