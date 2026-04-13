import os
import gymnasium as gym
import envs  # Registers your custom Move-v0
from stable_baselines3 import SAC

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
xml_path = os.path.join(BASE_DIR, "resources", "scenes", "scene_with_cam.xml")
model_path = os.path.join(BASE_DIR, "models", "pre_trained", "sac_move1")

env = gym.make('Move-v0', xml_path=xml_path, render_mode="human", camera_name="side_tracker")

model = SAC.load(model_path)

num_test_episodes = 5

for ep in range(num_test_episodes):
    obs, _ = env.reset()
    episode_reward = 0.0
    done = False
    
   
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        
        done = terminated or truncated
        
    print(f"Episode {ep + 1} Complete! Total Reward: {episode_reward:.2f}")

env.close()
