import numpy as np
from ur5_env.envs import UR5eEnv   

def main():
    env = UR5eEnv(render_mode="human", model_path="model/scene.xml")
    
    b = [-np.pi/9, np.pi/7, -np.pi/7, 0.001, np.pi/8, -np.pi/6]
    c = [-np.pi/8, -np.pi/6, -np.pi/8, 0.001, np.pi/10, -np.pi/5]
    w = [np.pi/4, np.pi/10, np.pi/7, 0.001, np.pi/6, np.pi/8]
    steps = 10000
    duration = 20  # Duración de la simulación en segundos

    # trajectory = env.generate_trajectory(b, c, w, steps, duration)
    trajectory = env.generate_trajectory(steps, duration)
    env.reset()
    env.execute_trajectory(trajectory)
    env.plot_logs()
    env.close()

if __name__ == "__main__":
    main()
