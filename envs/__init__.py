from gymnasium import register

register(
    id="PickAndPlace-v0",
    entry_point="envs.pick_and_place:PickAndPlace",
    max_episode_steps=2000,
)

register(
    id="RollingBall-v0",
    entry_point="envs.rolling_ball:RollingBall",
    max_episode_steps=2000,
)