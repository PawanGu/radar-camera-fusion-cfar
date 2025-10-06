import numpy as np

class KalmanFilterCV:
    def __init__(self, dt=0.1, process_var=1e-3, meas_var=0.1):
        self.F = np.block([[np.eye(2), dt*np.eye(2)],
                           [np.zeros((2,2)), np.eye(2)]])
        self.H = np.block([[np.eye(2), np.zeros((2,2))]])
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var
        self.P = np.eye(4)
        self.x = np.zeros((4,1))

    def init_state(self, pos):
        self.x[:2] = np.array(pos).reshape(2,1)
        self.x[2:] = 0.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].ravel()

    def update(self, z):
        z = np.array(z).reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].ravel()
