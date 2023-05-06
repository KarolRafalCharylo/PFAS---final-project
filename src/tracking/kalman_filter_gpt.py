import numpy as np

class KalmanFilter3D:
    def __init__(self, x0, P0=None, dt=1.0, acc_std=1.0):
        """
        Initialize Kalman filter.

        :param x0: Initial state vector.
        :param P0: Initial covariance matrix (default is identity matrix).
        :param dt: Time step (default is 1.0).
        :param acc_std: Standard deviation of acceleration noise (default is 1.0).
        """
        self.x = x0
        self.P = np.eye(9) if P0 is None else P0
        self.A = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0, 0],
                           [0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt, 0],
                           [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt*dt],
                           [0, 0, 0, 1, 0, 0, dt, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.B = np.array([[0.5*dt*dt, 0, 0],
                           [0, 0.5*dt*dt, 0],
                           [0, 0, 0.5*dt*dt],
                           [dt, 0, 0],
                           [0, dt, 0],
                           [0, 0, dt],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        self.Q = np.zeros((9, 9))
        self.Q[:6, :6] = np.eye(6) * acc_std**2
        self.R = np.eye(3)

    def predict(self, u=None):
        """
        Predict next state of the system.

        :param u: Control input vector.
        :return: Predicted state vector.
        """
        if u is not None:
            self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.A, self.x)

        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # Compute acceleration vector
        ax, ay, az = self.x[6:9]

        # Update state transition matrix with acceleration
        self.A[0, 3] = self.A[1, 4] = self.A[2, 5] = self.A[3, 6] = self.A[4, 7] = self.A[5, 8] = ax * self.dt
        self.A[0, 6] = self.A[1, 7] = self.A[2, 8] = 0.5 * ax * self.dt ** 2
        self.A[3, 6] = self.A[4, 7] = self.A[5, 8] = 0.5 * ax * self.dt ** 2

        # Update process noise covariance matrix with acceleration
        self.Q[:3, :3] = np.eye(3) * (0.5 * ax * self.dt) ** 2
        self.Q[3:6, 3:6] = np.eye(3) * (ax * self.dt)
        self.Q[:3, 3:6] = self.Q[3:6, :3] = np.eye(3) * (0.5 * ax * self.dt ** 2)

        return self.x

    def update(self, z):
        """
        Update the state estimate with a new measurement.

        :param z: Measurement vector.
        :return: Updated state vector.
        """
        # Compute innovation
        y = z - np.dot(self.H, self.x.T)

        # Compute innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Compute Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state estimate
        self.x = self.x + np.dot(K, y)

        # Update state covariance
        self.P = np.dot(np.eye(self.n) - np.dot(K, self.H), self.P)

        # Compute acceleration vector
        ax, ay, az = self.x[6:9]

        # Update measurement matrix with acceleration
        self.H[0, 6] = self.H[3, 7] = self.H[6, 8] = ax

        return self.x