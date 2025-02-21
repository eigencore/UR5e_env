from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import mujoco.viewer
import numpy as np
import mujoco
from control import PIDController  # Asegúrate de tener esta librería instalada

class UR5eEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, model_path=None):
        super().__init__()
        self.window_size = 512 
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.max_steps = 1e4
        self.step_count = 0
        
        keyframe_name = "home"
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        
        if keyframe_id != -1:
            self.qpos_home = self.model.key_qpos[keyframe_id]
            self.qvel_home = self.model.key_qvel[keyframe_id]
            self.ctrl_home = self.model.key_ctrl[keyframe_id]
        else:
            raise ValueError(f"Keyframe '{keyframe_name}' not found")

        # Acciones: 6 posiciones articulares deseadas (q_desired) + 1 para el gripper
        # Reducir el rango de acciones a ±π/4 alrededor de home
        self.action_space = spaces.Box(
            low=np.array([-np.pi/4]*6 + [0], dtype=np.float32),
            high=np.array([np.pi/4]*6 + [255], dtype=np.float32),
            dtype=np.float32
        )
        
        # Espacio de observación: qpos, qvel, posición del cubo, posición del target, etc.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Inicializar controladores PID para cada articulación
        self.pids = [PIDController(Kp=50, Ki=5, Kd=30, dt=0.0002) for _ in range(6)]

        self.window = None
        self.clock = None

    def _get_obs(self):
        # Original robot state
        robot_qpos = self.data.qpos[:12]  # Last 14 qpos are for the cube and target 
        robot_qvel = self.data.qvel[:12]  # First 12 DOFs for UR5
        
        # Cube information
        cube_pos = self.data.body("green_cube").xpos.copy()
        cube_quat = self.data.body("green_cube").xquat.copy()  # Quaternion [w,x,y,z]
        
        # Target information
        target_pos = self.data.body("target_zone").xpos.copy()
        
        # Relative calculations
        cube_to_target = cube_pos - target_pos
        cube_target_dist = np.linalg.norm(cube_to_target)
        

        # Combine all features into single array
        obs = np.concatenate([
            robot_qpos,
            robot_qvel,
            cube_pos,
            cube_quat,
            target_pos,
            cube_to_target,
            [cube_target_dist]
        ]).astype(np.float32)
        
        assert obs.shape == (38,), f"Observation shape mismatch: {obs.shape}"
        assert obs.dtype == np.float32, f"Observation dtype mismatch: {obs.dtype}"
        
        return obs
        
    def _get_info(self):
        cube_pos = self.data.body("green_cube").xpos.copy()
        target_pos = self.data.body("target_zone").xpos.copy()
        cube_to_target = cube_pos - target_pos
        cube_target_dist = np.linalg.norm(cube_to_target)
        
        return {
            "cube_pos": cube_pos,
            "target_pos": target_pos,
            "distance": cube_target_dist
        }
            
    def is_gripper_touching_cube(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Obtener los nombres de los cuerpos en contacto
            body1 = self.model.geom(contact.geom1).bodyid
            body2 = self.model.geom(contact.geom2).bodyid
            
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)
            
            # Verificar si el contacto es entre el gripper y el cubo
            if ("left_pad" in body1_name or "right_pad" in body1_name) and "green_cube" in body2_name:
                return True
            if ("left_pad" in body2_name or "right_pad" in body2_name) and "green_cube" in body1_name:
                return True
        
        return False
    
    # def reward(self):
    #     # Get info
    #     cube_pos = self.data.body("green_cube").xpos.copy()
    #     target_pos = self.data.body("target_zone").xpos.copy()
    #     left_pad_pos = self.data.body("left_pad").xpos
    #     right_pad_pos = self.data.body("right_pad").xpos
        
    #     shoulder_lift_home = self.qpos_home[:12]
    #     shoulder_lift_current = self.data.qpos[:12]

    #     shoulder_lift_dist = np.linalg.norm(shoulder_lift_current - shoulder_lift_home)
        

    #     gripper_pos = (left_pad_pos + right_pad_pos) / 2
        
    #     # Calculate distances
    #     cube_target_dist = np.linalg.norm(cube_pos - target_pos)
    #     gripper_cube_dist = np.linalg.norm(gripper_pos - cube_pos)
        
    #     # Reward components
    #     dist_reward = 1 / (1 + cube_target_dist)  # Recompensa por acercarse al target
    #     gripper_reward = 1 / (1 + gripper_cube_dist)  # Recompensa por acercarse al cubo
        
    #     # Penalización por inactividad (velocidad cercana a cero)
    #     velocity_penalty = -0.01 if np.linalg.norm(self.data.qvel[:6]) < 0.01 else 0.0
        
    #     # Success condition
    #     terminated = cube_target_dist < 0.05
    #     success_bonus = 10.0 if terminated else 0.0
        
    #     # Gripper touching cube
    #     gripper_touching_cube = 1.0 if self.is_gripper_touching_cube() else -1.0
        
    #     # Penalización por caída del cubo
    #     cube_fell = self.is_cube_fallen()
    #     cube_fall_penalty = -10.0 if cube_fell else 0.0
        
    #     # Penalización por caída del robot
    #     # robot_fell = self.is_robot_fallen()
    #     # robot_fall_penalty = -10.0 if robot_fell else 0.0
        
    #     # Penalización por colisión con la mesa
    #     collision_with_table = self.is_robot_colliding_with_table()
    #     collision_penalty = -10.0 if collision_with_table else 0.0
        
    #     # Combine components
    #     reward = (
    #         #0.1 * dist_reward +
    #         #1.0 * gripper_reward +
    #         #0.2 * gripper_touching_cube -
    #         -1.0 * shoulder_lift_dist #+
    #         #velocity_penalty +
    #         #success_bonus +
    #         #cube_fall_penalty +
    #         #robot_fall_penalty +
    #         #collision_penalty
    #     )
        
    #     # Termination conditions
    #     truncated = self.step_count >= self.max_steps
    #     terminated = terminated or cube_fell  or collision_with_table # or robot_fell
        
    #     # Info for debugging
    #     info = {
    #         "distance": cube_target_dist,
    #         "gripper_distance": gripper_cube_dist,
    #         "is_success": terminated,
    #         "cube_fell": cube_fell,
    #         #"robot_fell": robot_fell,
    #         "collision_with_table": collision_with_table,
    #         "reward_components": {
    #             "distance": dist_reward,
    #             "gripper": gripper_reward,
    #             "success_bonus": success_bonus,
    #             "velocity_penalty": velocity_penalty,
    #             "cube_fall_penalty": cube_fall_penalty,
    #          #   "robot_fall_penalty": robot_fall_penalty,
    #             "collision_penalty": collision_penalty
    #         }
    #     }
        
    #     return reward, info, terminated, truncated

    def reward(self):
        # Calcular error para todas las articulaciones
        q_error = np.linalg.norm(self.data.qpos[1] - self.qpos_home[6])
        
        q_error_penalty = -1.0 * q_error
        
        # Componentes de recompensa
        position_reward = 1 / (1 + q_error)  # Recompensa por estar cerca de home
        velocity_penalty = 0.01 * np.linalg.norm(self.data.qvel[:6])  # Penalizar movimientos rápidos
        
        reward = (
            5.0 * position_reward +
            q_error_penalty 
            - 1.0 * velocity_penalty
            - 100000.0 * self.is_cube_fallen()
            - 100000.0 * self.is_robot_colliding_with_table()
        )
        
        # Condiciones de terminación
        terminated = self.is_cube_fallen() or self.is_robot_colliding_with_table()
        truncated = self.step_count >= self.max_steps
        
        return reward, {}, terminated, truncated

    def is_robot_colliding_with_table(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            body1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom1])
            body2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[contact.geom2])
            
            # print(f"Contacto entre: {body1} y {body2}")  # Depuración
            
            if (body1 is not None and body2 is not None):
                if ("upper_arm_link" in body1.lower() and "table" in body2.lower()) or ("upper_arm_link" in body2.lower() and "table" in body1.lower()):
                    return True
        
        return False
    
    def is_cube_fallen(self):
        cube_pos = self.data.body("green_cube").xpos.copy()
        return cube_pos[2] < 0.1  # Umbral para considerar que el cubo cayó al suel
    
    def is_robot_fallen(self):
        base_pos = self.data.body("base").xpos.copy()
        return base_pos[2] < 0.1  # Umbral para considerar que el robot cayó al suelo
    

    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        for pid in self.pids:
            pid.reset()
        
        self.data.qpos[:] = self.qpos_home
        self.data.qvel[:] = self.qvel_home
        self.data.ctrl[:] = self.ctrl_home
   
        mujoco.mj_forward(self.model, self.data)
        
        self._cube_location = self.data.body("green_cube").xpos.copy()
        self._target_location = self.data.body("target_zone").xpos.copy()

        observation = self._get_obs()
        info = self._get_info()
        self.step_count = 0

        return observation, info

    def step(self, action):
        self.step_count += 1
        
        # 1. RL genera las posiciones articulares deseadas y la acción del gripper
        q_desired = action[:6]  # Primeros 6 valores: posiciones articulares
        gripper_action = action[6]  # Último valor: acción del gripper (0-255)
        # 2. PID sigue la trayectoria
        current_q = self.data.qpos[:6]
        error = q_desired - current_q
        tau = np.zeros(6)
        for i in range(6):
            tau[i] = np.clip(self.pids[i].update(error[i]), -20, 20) 

        # 3. Aplicar torque al robot
        self.data.ctrl[:6] = tau
        
        # 4. Aplicar acción del gripper
        self.data.ctrl[6] = gripper_action  # Controlar el actuador del gripper
        
        # 5. Avanzar simulación
        mujoco.mj_step(self.model, self.data)
        
        # 6. Calcular recompensa y condiciones de terminación
        reward, info, terminated, truncated = self.reward()
        
        return self._get_obs(), reward, terminated, truncated, info
    
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