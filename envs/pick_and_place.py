import numpy as np
import mujoco
from gymnasium.spaces import Box, Dict
from omegaconf import DictConfig
from envs.base_env import BaseRandomizedMujocoEnv

# Dependent on upstream IL setup
class PickAndPlace(BaseRandomizedMujocoEnv):
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

    def _get_reward(self, obs, action):
        w_reach = 1.0
        r_reach = self._reward_reach(obs, action)
        
        w_posture = 0.2
        r_posture = self._reward_posture(obs, action)
        
        w_grasp = 5.0
        r_grasp = self._reward_grasp(obs, action)
        
        w_place = 3.0
        r_place = self._reward_place(obs, action, r_grasp)
        
        w_penalty = 0.1
        penalty_action = -np.linalg.norm(action)

        w_vel = 0.005
        r_vel = self._reward_velocity()

        w_lift = 2.0
        r_lift = self._reward_lift(obs, action, r_grasp)

        total_reward = (w_reach * r_reach) + (w_posture * r_posture) + (w_grasp * r_grasp) + (w_place * r_place) + (w_penalty * penalty_action) + (w_vel * r_vel) + (w_lift * r_lift)

        reward_info = {
            "r_reach": r_reach,
            "w_r_reach": w_reach * r_reach,
            "r_posture": r_posture,
            "w_r_posture": w_posture * r_posture,
            "r_grasp": r_grasp,
            "w_r_grasp": w_grasp * r_grasp,
            "r_place": r_place,
            "w_r_place": w_place * r_place,
            "r_action_penalty": penalty_action,
            "w_r_action_penalty": w_penalty * penalty_action,
            "r_velocity": r_vel,
            "w_r_velocity": w_vel * r_vel,
            "r_lift": r_lift,
            "w_r_lift": w_lift * r_lift
        }

        return total_reward, reward_info


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

    def _reward_posture(self, obs, action):
        waist_yaw = self.data.joint("waist_yaw_joint").qpos[0]
        waist_roll = self.data.joint("waist_roll_joint").qpos[0]
        waist_pitch = self.data.joint("waist_pitch_joint").qpos[0]
        
        posture_penalty = -(waist_yaw**2 + waist_roll**2 + waist_pitch**2)
        return posture_penalty

    def _reward_velocity(self):
        upper_qvel = self.data.qvel[18:49]    
        return -np.sum(np.square(upper_qvel))

    def _reward_grasp(self, obs, action):
        obj_pos = self.data.geom("object_geom").xpos

        def get_dist(site_name):
            return np.linalg.norm(self.data.site(site_name).xpos - obj_pos)

        l_thumb = get_dist("left_thumb_tip")
        l_index = get_dist("left_index_tip")
        l_middle = get_dist("left_middle_tip")
        l_mean_dist = (l_thumb + l_index + l_middle) / 3.0

        r_thumb = get_dist("right_thumb_tip")
        r_index = get_dist("right_index_tip")
        r_middle = get_dist("right_middle_tip")
        r_mean_dist = (r_thumb + r_index + r_middle) / 3.0

        best_hand_dist = min(l_mean_dist, r_mean_dist)

        dense_grasp = np.exp(-4.0 * best_hand_dist)

        sparse_bonus = 0.0
        if best_hand_dist < 0.045: # 4.5cm
            sparse_bonus = 1.0

        return dense_grasp + sparse_bonus

    def _reward_lift(self, obs, action, r_grasp):
        object_pos = self.data.geom("object_geom").xpos
        table_center_z = self.data.geom("table_surface").xpos[2] 
        table_half_height = self.model.geom("table_surface").size[2]
        table_top_z = table_center_z + table_half_height
        
        height_above_surface = object_pos[2] - table_top_z
        
        reward_lift = 0.0
        
        if r_grasp > 1.0:
            target_lift = 0.05 
            
            lifted_amount = np.clip(height_above_surface, 0.0, target_lift)
            reward_lift = lifted_amount / target_lift
            
            if lifted_amount >= target_lift:
                reward_lift += 2.0 

        return reward_lift

    def _reward_place(self, obs, action, r_grasp):
        object_pos = self.data.geom("object_geom").xpos
        target_pos = self.data.site("target_site").xpos

        dist_to_target = np.linalg.norm(object_pos - target_pos)

        reward_place = 0.0
        success_bonus = 0.0
        if r_grasp > 1.0:
            reward_place = -dist_to_target
            if dist_to_target < 0.05: #5cm
                success_bonus = 10.0

        return reward_place + success_bonus

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs()

        reward, reward_info = self._get_reward(obs, action)
        terminated = False
        truncated = False
        info = {}
        info["reward_info"] = reward_info

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        qpos = self.randomize_env(qpos)

        self.set_state(qpos, qvel)
        return self._get_obs()