import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import ur5_env  # Importa tu paquete para registrar el entorno
import time

# Registra el entorno personalizado
gym.register(
    id="UR5eEnv-v0",
    entry_point="ur5_env.envs:UR5eEnv",
)

# Directorio para guardar logs y modelos
log_dir = "ur5e_ppo_logs"
os.makedirs(log_dir, exist_ok=True)

# Semillas para entrenamiento
seeds = [0]

# Bucle sobre las semillas
for seed in seeds:
    def make_env():
        # Crea el entorno personalizado
        env = gym.make("UR5eEnv-v0", render_mode="human", model_path="model/scene.xml")
        env.reset(seed=seed)  # Fija la semilla para reproducibilidad
        return env

    # Crea un entorno vectorizado (DummyVecEnv para un solo entorno)
    vec_env = DummyVecEnv([make_env])

    model = PPO.load("ur5e_ppo_logs/ppo_ur5e_seed_0.zip", device="cpu")
    
    # Evalúa el modelo
    obs = vec_env.reset()
    time.sleep(1)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        
    # Cierra el entorno
    vec_env.close()
    
print("Evaluation complete.")