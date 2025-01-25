from gymnasium.envs.registration import register

register(
    id="ur5_env/GridWorld-v0",
    entry_point="ur5_env.envs:GridWorldEnv",
)
