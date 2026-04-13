import os
import numpy as np
import envs
import gymnasium as gym


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env = gym.make('Move-v0', xml_path=os.path.join(BASE_DIR, "resources", "custom", "custom_scene.xml"), render_mode="human")

obs, _ = env.reset()
for i in range(1000):
    action = np.zeros(env.action_space.shape[0])
    # wave = np.sin(i/10)
    # action[15] = wave
    # action[22] = -wave
    #action[16:18] = wave
    #action[23:25] = -wave
    obs, _, _, _, _ = env.step(action)

env.close()