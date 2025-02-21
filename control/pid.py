import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, integral_limit=100.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.previous_error = 0.0
        self.integral_limit = integral_limit  # Anti-windup

    def update(self, error):
        # Término proporcional
        P = self.Kp * error
        
        # Término integral (con límite anti-windup)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        I = self.Ki * self.integral
        
        # Término derivativo
        derivative = (error - self.previous_error) / self.dt
        D = self.Kd * derivative
        
        # Actualizar error anterior
        self.previous_error = error
        
        # Salida total
        output = P + I + D
        return output
    
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        