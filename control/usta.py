import numpy as np

class uSTAController:
    def __init__(self, k1, k2, k3, k4, dt):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.dt = dt
        self.z = np.zeros(6)  # Inicializar z como un vector de 6 elementos
        self.z_dot = np.zeros(6)  # Inicializar z_dot como un vector de 6 elementos

    def calculate_z_dot(self, e):
        norm_e = np.linalg.norm(e)
        if norm_e == 0:
            norm_e = 1e-6  # Evitar división por cero
        z_dot = -self.k3 * (e / norm_e) - self.k4 * e
        return z_dot

    def calculate_ust(self, e):
        norm_e = np.linalg.norm(e)
        if norm_e == 0:
            norm_e = 1e-9  # Evitar división por cero
        ust = -self.k1 * (e / (norm_e ** 0.5)) - self.k2 * e + self.z
        return ust

    def update(self, e):
        self.z_dot = self.calculate_z_dot(e)
        self.z += self.z_dot * self.dt
        ust = -1*self.calculate_ust(e)
        return ust

    def reset(self):
        self.z = np.zeros(6)  # Reiniciar z como un vector de 6 elementos
        self.z_dot = np.zeros(6)  # Reiniciar z_dot como un vector de 6 elementos
