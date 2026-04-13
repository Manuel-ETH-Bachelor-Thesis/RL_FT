import os
import gymnasium as gym
import envs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gymnasium.wrappers import RescaleAction
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

class MyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                r = info["episode"]["r"]
                self.episode += 1
                
                if self.episode % 10 == 0:
                    print(f"Episode {self.episode} | Last Reward: {r:.2f}")
        return True

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_env(cfg: DictConfig):
    def _init():
        env = gym.make('PickAndPlace-v0', xml_path=os.path.join(BASE_DIR, "resources", "custom", "custom_scene.xml"), cfg=cfg)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        return Monitor(env)
    return _init


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    num_envs = 6
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    env = DummyVecEnv([make_env(cfg_dict) for _ in range(num_envs)])

    model = SAC(
        "MultiInputPolicy",
        env, 
        verbose=0, 
        seed=42, 
        device="auto",
        tensorboard_log=os.path.join(BASE_DIR, "logs", "sac_tensorboard")
    )
    
    total_timesteps = 5_000_000
    print(f"Starting training on {num_envs} CPU cores with GPU Neural Network updates...")
    
    model.learn(total_timesteps=total_timesteps, callback=MyCallback())
    model.save(os.path.join(BASE_DIR, "models", "sac_pick_and_place"))

    env.close()

if __name__ == "__main__":
    main()