import numpy as np
import mujoco
import mujoco.viewer
from control import PIDController  
import time


def inverse_kinematics(target_pos, current_q):
    return np.array([-3.9,-3.0,1.95,-2.2,-1.76,0.189])


class UR5eEnv:
    def __init__(self, model_path):
        # Cargar el modelo y los datos de MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Configurar el controlador PID para cada articulación
        self.Kp = 800.0  # Ganancia proporcional
        self.Ki = 5.0    # Ganancia integral
        self.Kd = 50.0   # Ganancia derivativa
        self.dt = 0.002  # Paso de tiempo de MuJoCo (por defecto)

        # Inicializar un PID para cada articulación (6 en total)
        self.pids = [PIDController(self.Kp, self.Ki, self.Kd, self.dt) for _ in range(6)]

        # Posición objetivo del efector final (ejemplo)
        self.target_pos = np.array([0.5, 0.2, 0.3])  # x, y, z en metros

    def inverse_kinematics(self,target_pos, current_q):
        return np.array([-3.14, -1.56, 1.58, -1.57, 0, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003])

    def run(self):
        # Inicializar el visualizador
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Loop de control
        i = 0
        while self.viewer.is_running():
             
            # 1. Obtener q_current (posiciones actuales de las articulaciones)
            current_q = self.data.qpos[:12]

            # 2. Calcular q_desired usando cinemática inversa temporal
            q_desired = self.inverse_kinematics(self.target_pos, current_q)

            # 3. Calcular error para cada articulación
            error = q_desired - current_q

            # 4. Calcular torque con PID
            tau = np.zeros(6)
            for i in range(6):
                tau[i] = self.pids[i].update(error[i])

            # 5. Aplicar torque al robot (asignar a data.ctrl)
            self.data.ctrl[:6] = tau
            

            # 6. Avanzar simulación (actualiza data.qpos y data.qvel)
            mujoco.mj_step(self.model, self.data)

            # 7. Sincronizar la visualización
            self.viewer.sync()


        # Cerrar el visualizador al salir del loop
        self.viewer.close()

# Punto de entrada del programa
if __name__ == "__main__":
    # Ruta al archivo XML del modelo
    model_path = "model/scene.xml"  # Reemplaza con la ruta correcta

    # Crear y ejecutar el entorno
    env = UR5eEnv(model_path)
    env.run()