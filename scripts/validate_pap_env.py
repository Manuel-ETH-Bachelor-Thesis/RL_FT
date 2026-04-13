import os
import time
import gymnasium as gym
import envs
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import hydra
from omegaconf import DictConfig, OmegaConf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_env(cfg: DictConfig):
    def _init():
        env = gym.make('PickAndPlace-v0', xml_path=os.path.join(BASE_DIR, "resources", "custom", "custom_scene.xml"), cfg=cfg)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        return Monitor(env)
    return _init

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    print("Initializing a single environment...")
    env = make_env(cfg_dict)() 

    print("\n--- Running SB3 Environment Checker ---")
    check_env(env.unwrapped) 
    print("✅ Environment architecture is SB3-compliant!")

    print("\n--- Starting Random Action Stress Test ---")
    obs, info = env.reset()
    
    total_steps = 1000
    start_time = time.time()
    
    for step in range(1, total_steps + 1):
        random_action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        if step % 100 == 0:
            print(f"Step {step}/{total_steps} | Simulated Time: {env.unwrapped.data.time:.2f}s | Reward: {reward:.3f}")
            
        if terminated or truncated:
            print(f"♻️ Episode finished! Resetting environment...")
            obs, info = env.reset()
            
    duration = time.time() - start_time
    fps = total_steps / duration
    
    print(f"\n🚀 Validation Complete!")
    print(f"Processed {total_steps} steps in {duration:.2f} seconds.")
    print(f"M2 Performance: {fps:.1f} Environment Steps Per Second (SPS)")
    
    env.close()

if __name__ == "__main__":
    main()