import os
import sys
import signal
import gymnasium as gym
import envs
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import RescaleAction
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_env(cfg: DictConfig):
    def _init():
        env = gym.make('PickAndPlace-v0', xml_path=os.path.join(BASE_DIR, "resources", "custom", "custom_scene.xml"), cfg=cfg)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        return Monitor(env)
    return _init

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    num_envs = 32
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    env = SubprocVecEnv([make_env(cfg_dict) for _ in range(num_envs)])

    model = SAC(
        "MultiInputPolicy",
        env, 
        verbose=0,
        buffer_size=100_000, 
        seed=42, 
        device="auto",
        tensorboard_log=os.path.join(BASE_DIR, "logs", "sac_tensorboard")
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // num_envs, 1), 
        save_path=os.path.join(BASE_DIR, "models", "checkpoints"),
        name_prefix="sac_model_checkpoint"
    )

    def handle_termination(sig, frame):
        print(f"\n[!] Job termination signal ({sig}) caught! Saving emergency backup...")
        model.save(os.path.join(BASE_DIR, "models", "sac_pick_and_place_interrupted"))
        env.close()
        print("Backup saved successfully. Shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)
    
    total_timesteps = 5_000_000
    print(f"Starting training on {num_envs} CPU cores with GPU Neural Network updates...")
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    except Exception as e:
        print(f"Training crashed with exception: {e}")
    finally:
        model.save(os.path.join(BASE_DIR, "models", "sac_pick_and_place_final"))
        env.close()
        print("Training complete. Final model saved.")

if __name__ == "__main__":
    main()