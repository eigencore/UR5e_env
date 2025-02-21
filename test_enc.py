from ur5_env.envs import UR5eEnv            


if __name__ == "__main__":
    env = UR5eEnv(render_mode="human", model_path="model/scene.xml")
    env.reset()
    print("observation_space:", env.action_space.sample())