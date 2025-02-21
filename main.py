import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import ur5_env  # Importa tu paquete para registrar el entorno

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
        env = gym.make("UR5eEnv-v0", render_mode=None, model_path="model/scene.xml")
        env.reset(seed=seed)  # Fija la semilla para reproducibilidad
        # Configura el monitor para guardar logs
        log_file = os.path.join(log_dir, f"sb3_ur5e_seed_{seed}.csv")
        env = Monitor(env, log_file)
        return env

    # Crea un entorno vectorizado (DummyVecEnv para un solo entorno)
    vec_env = DummyVecEnv([make_env])

    # Entrena el modelo PPO
    model = PPO(
        "MlpPolicy",  # Usa "MlpPolicy" para observaciones vectorizadas
        vec_env,
        seed=seed,  # Fija la semilla para reproducibilidad
        verbose=0,  # Muestra logs durante el entrenamiento
        tensorboard_log=log_dir,  # Guarda logs de TensorBoard
        device="cpu",  # Usa "cuda" para GPU
    )

    print(f"Training with seed {seed}...")
    model.learn(total_timesteps=int(200_000), progress_bar=True)  # Entrena por 100,000 pasos

    # Guarda el modelo entrenado
    model_path = os.path.join(log_dir, f"ppo_ur5e_seed_{seed}.zip")
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Cierra el entorno
    vec_env.close()

print("Training complete for all seeds.")