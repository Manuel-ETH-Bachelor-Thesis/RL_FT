import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

class Move(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 250,
    }
    def __init__(self, xml_path: str, render_mode=None, camera_name="side_tracker"):
        self.frame_skip = 1
        
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(71,),
            dtype=np.float64,
        )

        super().__init__(
            xml_path,
            frame_skip=self.frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name=camera_name
        )

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(31,),
            dtype=np.float64,
        )
        self.prev_action = None
        
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        if self.render_mode is not None:
            self.render()

        if self.prev_action is None:
            self.prev_action = action
        
        target_height = 0.75
        current_height = self.state_vector()[2]
        posture_reward = np.exp(-10.0 * (current_height - target_height)**2)


        reward = self.data.qvel[0] - 0.005 * np.sum(np.square(self.prev_action - action)) + posture_reward
        self.prev_action = action

        return self.state_vector(), float(reward), False, False, {"a": np.float64(0.0)}

    def reset_model(self):
        self.prev_action = None
        self.set_state(self.init_qpos, self.init_qvel)
        return self.state_vector()
