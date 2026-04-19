import os
import gymnasium as gym
import envs
from stable_baselines3 import SAC
from gymnasium.wrappers import RescaleAction, RecordVideo
import hydra
from omegaconf import DictConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def record(cfg: DictConfig):
    xml_path = os.path.join(BASE_DIR, "resources", "custom", "custom_scene.xml")
    
    env = gym.make(
        'PickAndPlace-v0', 
        xml_path=xml_path, 
        cfg=cfg, 
        render_mode="rgb_array",
        camera_name="top_down_camera" 
    )
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    video_dir = os.path.join(BASE_DIR, "videos")
    env = RecordVideo(
        env, 
        video_folder=video_dir, 
        name_prefix="g1_topdown_4M_steps",
        episode_trigger=lambda x: True
    )

    model_path = os.path.join(BASE_DIR, "models", "checkpoints", "sac_model_checkpoint_4000000_steps")
    print(f"Loading brain from: {model_path}.zip...")
    
    model = SAC.load(model_path, env=env)

    print("Recording 3 demonstration episodes...")
    episodes = 3
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {ep + 1} finished | Total Reward: {total_reward:.2f}")

    env.close()
    print(f"\nSuccess! Videos saved to: {video_dir}")

if __name__ == "__main__":
    record()