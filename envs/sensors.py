"""IMU noise model and gyro-integrating attitude estimator: what the real drone measures."""
import numpy as np

from configs import Config

SMALL_ANGLE = 1e-12  # rad; a rotation below this is the identity


class ImuModel:
    """Gyro and accelerometer with a per-episode bias and white noise."""

    def __init__(self, cfg: Config):
        """cfg: the `sensors` section of configs/env.yaml."""
        self.cfg = cfg
        self.gyro_bias: np.ndarray = np.zeros(3)
        self.accel_bias: np.ndarray = np.zeros(3)

    def reset(self, rng: np.random.Generator):
        """Draw this episode's biases (constant over an episode: a few seconds)."""
        self.rng = rng
        self.gyro_bias = rng.uniform(-self.cfg.gyro_bias, self.cfg.gyro_bias, 3)
        self.accel_bias = rng.uniform(-self.cfg.accel_bias, self.cfg.accel_bias, 3)

    def read(self, gyro: np.ndarray, accel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ideal readings -> measured readings (same units: rad/s, m/s^2)."""
        gyro = gyro + self.gyro_bias + self.rng.normal(0, self.cfg.gyro_noise, 3)
        accel = accel + self.accel_bias + self.rng.normal(0, self.cfg.accel_noise, 3)
        return gyro, accel


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def axis_angle_quat(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector [rad] -> quaternion [w, x, y, z]."""
    angle = np.linalg.norm(rotvec)
    if angle < SMALL_ANGLE:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * rotvec / angle])


class AttitudeEstimator:
    """Attitude from integrating the gyro alone (no accelerometer correction)."""

    def reset(self, q: np.ndarray, roll_progress: float = 0.0):
        """Start from attitude q (body -> world, [w, x, y, z])."""
        self.q = np.array(q, float)
        self.roll_progress = roll_progress

    def update(self, gyro: np.ndarray, dt: float):
        """Integrate one gyro reading [rad/s, body frame] over dt [s] (exact for a constant rate over dt)."""
        self.q = quat_multiply(self.q, axis_angle_quat(gyro * dt))
        self.q /= np.linalg.norm(self.q)
        self.roll_progress += gyro[0] * dt

    @property
    def gravity_body(self) -> np.ndarray:
        """World up in body frame (last row of R): (0, 0, 1) when level."""
        w, x, y, z = self.q
        return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
