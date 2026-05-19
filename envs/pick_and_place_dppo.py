import numpy as np
import mujoco
from gymnasium.spaces import Box, Dict
from omegaconf import DictConfig
from envs.base_env import BaseRandomizedMujocoEnv

# Environment for pick-and-place dppo fine tuning, using sparse reward

# Dependent on upstream IL setup
class PickAndPlace(BaseRandomizedMujocoEnv):
    def __init__(self, xml_path: str, cfg: DictConfig, randomize: bool = True, **kwargs):
        self.frame_skip = 10
        self.img_size = 128 # Camera
        self.step_count = 0
        self.max_episode_steps = 2000
        self.success = False

        observation_space = Dict({
            # 62: 31 upper qpos  + 31 upper qvel
            "proprio": Box(low=-1e10, high=1e10, shape=(62,), dtype=np.float32), 
            "rgb": Box(low=0, high=255, shape=(3, self.img_size, self.img_size), dtype=np.uint8),
            "depth": Box(low=0, high=255, shape=(1, self.img_size, self.img_size), dtype=np.uint8)
        })

        super().__init__(
            xml_path,
            frame_skip=self.frame_skip,
            observation_space=observation_space,
            cfg=cfg,
            randomize=randomize,
            render_mode="rgb_array",
            **kwargs
        )

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds[:, 0], bounds[:, 1]
        
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        self.obs_renderer = mujoco.Renderer(self.model, height=self.img_size, width=self.img_size)

    def _get_obs(self):
        # --- Joints ---
        upper_qpos = self.data.qpos[19:50].astype(np.float32) # Exclude leg joint positions
        upper_qvel = self.data.qvel[18:49].astype(np.float32) # Exclude leg joint velocities
        proprio = np.concatenate([upper_qpos, upper_qvel]).astype(np.float32)

        # --- Vision ---
        self.obs_renderer.update_scene(self.data, camera="head_camera")
        
        rgb_raw = self.obs_renderer.render()
        rgb = np.transpose(rgb_raw, (2, 0, 1)).astype(np.uint8)

        self.obs_renderer.enable_depth_rendering()
        depth_raw = self.obs_renderer.render()
        self.obs_renderer.disable_depth_rendering()
        
        max_depth = 2.0 
        depth_norm = np.clip(depth_raw / max_depth, 0.0, 1.0)
        depth_scaled = (depth_norm * 255).astype(np.uint8)
        depth = np.expand_dims(depth_scaled, axis=0)

        return {
            "proprio": proprio,
            "rgb": rgb,
            "depth": depth
        }

    def _get_reward(self, done):
        object_pos = self.data.geom("object_geom").xpos
        target_pos = self.data.site("target_site").xpos

        dist = np.linalg.norm(object_pos - target_pos)

        reward = 0.0 

        if dist < 0.05 and not self.success:
            reward = 1.0
            self.success = True

        return reward, {
            "dist_obj_target": float(dist)
        }

    def step(self, action):
        if action.ndim == 1:
            actions = np.expand_dims(action, axis=0)
        else:
            actions = action

        total_reward = 0.0
        
        for act in actions:
            self.step_count += 1
            self.do_simulation(act, self.frame_skip)
            
            terminated = self.success
            truncated = self.step_count >= self.max_episode_steps
            done = terminated or truncated

            step_reward, reward_info = self._get_reward(done)
            total_reward += step_reward

            if done:
                break
        
        obs = self._get_obs()

        info = {
            "reward_info": reward_info
        }

        return obs, total_reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        qpos = self.randomize_env(qpos)
        self.step_count = 0
        self.success = False

        self.set_state(qpos, qvel)
        return self._get_obs()