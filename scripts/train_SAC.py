import os
import gymnasium as gym
import envs
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

class MyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode = 0
        self.rewards = []
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            r = self.locals["infos"][0]["episode"]["r"]
            self.rewards.append(r)
            print(f"Episode {self.episode} - Reward: {r}")
            if self.episode % 50 == 0:
                self.ax.clear()
                self.ax.plot(self.rewards, label="SAC Training", color='blue')
                self.ax.set_xlabel("Episode")
                self.ax.set_ylabel("Total Reward")
                self.ax.set_title("Stable Baselines3 SAC Progress")
                self.ax.legend()
                plt.savefig("progress_plot.png")
                plt.pause(0.001)
            self.episode += 1
        return True


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_env():
    def _init():
        env = gym.make('Move-v0', xml_path=os.path.join(BASE_DIR, "resources", "robots", "unitree_g1", "scene.xml"))
        return Monitor(env)
    return _init

if __name__ == "__main__":
    num_envs = 6
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = SAC("MlpPolicy", env, verbose=0, seed=42, device="cpu")
    total_episodes = 5_000_000
    print(f"Starting training on {num_envs} cores")
    model.learn(total_timesteps=total_episodes, callback=MyCallback())
    model.save(os.path.join(BASE_DIR, "models", "sac_move1"))

    env.close()