from gymnasium import register

register(
    id="PickAndPlace-v0",
    entry_point="envs.pick_and_place:PickAndPlace"
)