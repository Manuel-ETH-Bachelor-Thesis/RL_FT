import numpy as np
import mujoco
from mujoco import mj_name2id as mujoco_mj_name2id
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from omegaconf import DictConfig, OmegaConf

class BaseRandomizedMujocoEnv(MujocoEnv):
    def __init__(self, xml_path: str, frame_skip: int, observation_space, cfg: DictConfig, randomize: bool = True, **kwargs):
        self.cfg = cfg
        self.randomize = randomize
        super().__init__(
            xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            **kwargs
        )

    def _sample_noise(self, noise_cfg):
        if noise_cfg is None:
            return 0.0
        dist_type = noise_cfg['type'] 
        dist_func = getattr(np.random, dist_type)
        kwargs = noise_cfg['kwargs']
        return dist_func(**kwargs)

    def randomize_env(self, qpos):
        if not self.randomize:
            return qpos

        rand_cfg = self.cfg.get("scene", None)
        if not rand_cfg:
            return qpos        
        
        obj_type_map = {
            "light": mujoco.mjtObj.mjOBJ_LIGHT,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "cam": mujoco.mjtObj.mjOBJ_CAMERA,
        }
        
        for obj_type_str, objects in rand_cfg.items():
            if obj_type_str not in obj_type_map:
                continue

            mj_obj_type = obj_type_map[obj_type_str]
            for obj_name, props in objects.items():
                obj_id = mujoco_mj_name2id(self.model, mj_obj_type, obj_name)

                if obj_id == -1:
                    print(f"Warning: {obj_type_str} '{obj_name}' specified for randomization not found in the scene.")
                    continue
                
                for prop_name, prop_cfg in props.items():
                    default_val = np.array(prop_cfg['default'])

                    noise = prop_cfg.get("noise", None)
                    noise_val = self._sample_noise(noise) if noise else 0.0
                    new_val = default_val + noise_val
                    
                    if "clip" in prop_cfg:
                        clip_cfg = prop_cfg['clip']
                        new_val = np.clip(new_val, clip_cfg[0], clip_cfg[1])
                        
                    if obj_type_str == "joint" and prop_name == "qpos":
                        qpos_adr = self.model.jnt_qposadr[obj_id]
                        val_len = len(default_val)
                        qpos[qpos_adr : qpos_adr + val_len] = new_val
                    else:
                        mj_attr_name = f"{obj_type_str}_{prop_name}"
                        if hasattr(self.model, mj_attr_name):
                            attr_array = getattr(self.model, mj_attr_name)
                            attr_array[obj_id] = new_val
                        else:
                            print(f"Warning: {mj_attr_name} specified for randomization is not a valid MuJoCo attribute for {obj_type_str}.")

                            
        return qpos