from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer
import numpy as np
import mujoco


class URe5Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, model_path=None):
        self.window_size = 512  # The size of the PyGame window
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # We have 7 actions: 6 for the ctrl of the robot and 1 for the gripper 
        self.action_space = spaces.Box(
            low=np.array([-2*np.float32(np.pi), -2*np.float32(np.pi), -np.float32(np.pi), -2*np.float32(np.pi), -2*np.float32(np.pi), -2*np.float32(np.pi), 0], dtype=np.float32),
            high=np.array([2*np.float32(np.pi), 2*np.float32(np.pi), np.float32(np.pi), 2*np.float32(np.pi), 2*np.float32(np.pi), 2*np.float32(np.pi), 255], dtype=np.float32),
            dtype=np.float32
        )
        
        # The observation space is the qpos and qvel of the robot 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.concatenate([self.data.qpos[:-4], self.data.qvel[:12]])
        
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._cube_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        keyframe_name = "home"
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
                
        if keyframe_id != -1:
            self.data.qpos[:] = self.model.key_qpos[keyframe_id]
            self.data.qvel[:] = self.model.key_qvel[keyframe_id]
            self.data.ctrl[:] = self.model.key_ctrl[keyframe_id]
        else:
            raise ValueError(f"Keyframe '{keyframe_name}' not found")
        
        mujoco.mj_forward(self.model, self.data)

        
        # We will sample the target's location randomly until it doesn't
        # coincide with the agent's location
        self._cube_location = self.data.body("green_cube").xpos.copy()
        self._target_location = self.data.body("target_zone").xpos.copy()


        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        terminated = np.array_equal(self._cube_location, self._target_location)
        observation = self._get_obs()
        info = self._get_info()
        reward = -info["distance"]
        done = terminated or info["distance"] < 0.001
        return observation, reward, done, terminated, info
    
    
    def _render_frame(self):
        # Crear un contexto de renderizado si no existe
        if not hasattr(self, "ctx"):
            self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        
        # Configurar la cámara
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        
        # Configurar la vista
        viewport = mujoco.MjrRect(0, 0, self.window_size, self.window_size)
        
        # Crear una escena
        scn = mujoco.MjvScene(self.model, maxgeom=10000)
        opt = mujoco.MjvOption()
        
        # Renderizar la escena
        mujoco.mjv_updateScene(
            self.model, self.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn
        )
        
        # Crear un array para almacenar la imagen renderizada
        rgb_array = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        
        # Renderizar la escena en el array
        mujoco.mjr_render(viewport, scn, self.ctx)
        mujoco.mjr_readPixels(rgb_array, None, viewport, self.ctx)
        
        # Devolver el array de píxeles (en formato RGB)
        return np.transpose(rgb_array, axes=(1, 0, 2))  # Transponer para que coincida con el formato de PyGame

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            if self.viewer is None:
                # Abrir el visor nativo de MuJoCo
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                # Sincronizar el visor con los datos actuales
                self.viewer.sync()

    def close(self):
        # Cerrar el visor nativo de MuJoCo si está abierto
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None  # Asegurarse de que el visor se marque como cerrado

        # Liberar el contexto de renderizado si existe
        if hasattr(self, "ctx") and self.ctx is not None:
            self.ctx.free()  # Liberar el contexto de renderizado
            self.ctx = None  # Asegurarse de que el contexto se marque como liberado

    
if __name__ == "__main__":
    # Crear el entorno
    env = URe5Env(render_mode="human", model_path="model/scene.xml")
    
    # Reiniciar el entorno
    observation, info = env.reset()
    print("Observación inicial:", observation)
    print("Información inicial:", info)
    
    # Ejecutar algunos pasos de la simulación
    while True:
        # Tomar una acción aleatoria
        action = env.action_space.sample()
        
        # Avanzar un paso en la simulación
        observation, reward, done, terminated, info = env.step(action)
        
        # Mostrar los resultados
        print("Observación:", observation)
        print("Recompensa:", reward)
        print("¿Terminado?", done)
        print("Información:", info)
        
        # Renderizar el entorno (si está en modo "human")
        env.render()
        
        # Salir si el episodio ha terminado
        if done:
            print("¡Episodio terminado!")
            break
    
    # Cerrar el entorno
    env.close()