import numpy as np
import mujoco
from gymnasium.spaces import Box, Dict
from omegaconf import DictConfig
from envs.base_env import BaseRandomizedMujocoEnv

# Dependent on upstream IL setup
class PickAndPlace(BaseRandomizedMujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"], 
        "render_fps": 25
    }
    
    def __init__(self, xml_path: str, cfg: DictConfig, randomize: bool = True, **kwargs):
        self.frame_skip = 10
        self.img_size = 128 # Camera


        observation_space = Dict({
            # 62: 31 upper qpos  + 31 upper qvel
            "proprio": Box(low=-1e10, high=1e10, shape=(62,), dtype=np.float32), # 31 upper qpos + 31 upper qvel
            "rgb": Box(low=0, high=255, shape=(3, self.img_size, self.img_size), dtype=np.uint8),
            "depth": Box(low=0, high=255, shape=(1, self.img_size, self.img_size), dtype=np.uint8)
        })

        super().__init__(
            xml_path,
            frame_skip=self.frame_skip,
            observation_space=observation_space,
            cfg=cfg,
            randomize=randomize,
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

    def _get_reward(self, obs, action):
        w_reach = 1.0
        r_reach = self._reward_reach(obs, action)
        w_grasp = 2.0
        r_grasp = self._reward_grasp(obs, action)
        w_place = 3.0
        r_place = self._reward_place(obs, action, r_grasp)
        w_penalty = 0.5
        penalty_action = -np.linalg.norm(action)

        total_reward = w_reach * r_reach + w_grasp * r_grasp + w_place * r_place + w_penalty * penalty_action
        return total_reward


    def _reward_reach(self, obs, action):
        left_grasp_pos = self.data.site("left_grasp_center").xpos
        right_grasp_pos = self.data.site("right_grasp_center").xpos
        object_pos = self.data.geom("object_geom").xpos

        dist_left = np.linalg.norm(left_grasp_pos - object_pos)
        dist_right = np.linalg.norm(right_grasp_pos - object_pos)

        closest_dist = min(dist_left, dist_right)
        reward_reach = -closest_dist

        furthest_dist = max(dist_left, dist_right)
        exclusion_radius = 0.15 #15cm
        
        penalty_two_hands = 0.0
        if furthest_dist < exclusion_radius:
            penalty_two_hands = -5.0* (exclusion_radius - furthest_dist)

        return reward_reach + penalty_two_hands

    def _reward_grasp(self, obs, action):
        grasp_threshold = 0.005 #0.5cm

        obj_pos = self.data.geom("object_geom").xpos
        obj_radius = self.model.geom("object_geom").size[0]

        def is_grasping(site_name):
            site_pos = self.data.site(site_name).xpos
            dist_to_surface = np.linalg.norm(site_pos - obj_pos) - obj_radius
            return dist_to_surface < grasp_threshold

        left_num_grasp = sum([is_grasping("left_thumb_tip"), is_grasping("left_index_tip"), is_grasping("left_middle_tip")])
        right_num_grasp = sum([is_grasping("right_thumb_tip"), is_grasping("right_index_tip"), is_grasping("right_middle_tip")])

        best_hand_grasp = max(left_num_grasp, right_num_grasp)

        if best_hand_grasp == 3:
            return 1.0
        elif best_hand_grasp == 2:
            return 0.5
        return 0.0

    def _reward_place(self, obs, action, r_grasp):
        object_pos = self.data.geom("object_geom").xpos
        target_pos = self.data.site("target_site").xpos

        dist_to_target = np.linalg.norm(object_pos - target_pos)

        reward_place = 0.0
        if r_grasp > 0.0:
            reward_place = -dist_to_target

        success_bonus = 0.0
        if dist_to_target < 0.05: #5cm
            success_bonus = 10.0

        return reward_place + success_bonus

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        qpos = self.randomize_env(qpos)

        self.set_state(qpos, qvel)
        return self._get_obs()