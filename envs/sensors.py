"""What the real drone measures, and what it can compute from it.

The policy runs on an ESP32 next to a Betaflight flight controller, which sends it the IMU readings (MSP_RAW_IMU)
and its attitude estimate (MSP_ATTITUDE). So the policy gets no position and no velocity: only gyro, accelerometer
and an attitude that comes from integrating the gyro.

ImuModel:           ideal MuJoCo readings + a bias drawn per episode + white noise (configs/env.yaml, `sensors`).
AttitudeEstimator:  integrates the noisy gyro, like Betaflight does during a flip (it ignores the accelerometer when
                    |acc| is far from 1 g, which is the whole flip). Also integrates the roll rate, which counts the
                    flip's progress. Plain numpy, so it ports directly to the ESP32.
"""
import numpy as np


class ImuModel:
    def __init__(self, cfg):
        """cfg: the `sensors` section of configs/env.yaml."""
        self.cfg = cfg
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def reset(self, rng):
        """Draw this episode's biases (constant over an episode: a few seconds)."""
        self.rng = rng
        self.gyro_bias = rng.uniform(-self.cfg.gyro_bias, self.cfg.gyro_bias, 3)
        self.accel_bias = rng.uniform(-self.cfg.accel_bias, self.cfg.accel_bias, 3)

    def read(self, gyro, accel):
        """Ideal readings -> measured readings (same units: rad/s, m/s^2)."""
        gyro = gyro + self.gyro_bias + self.rng.normal(0, self.cfg.gyro_noise, 3)
        accel = accel + self.accel_bias + self.rng.normal(0, self.cfg.accel_noise, 3)
        return gyro, accel


def quat_multiply(a, b):
    """Hamilton product of quaternions [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def axis_angle_quat(rotvec):
    """Rotation vector [rad] -> quaternion [w, x, y, z]."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * rotvec / angle])


class AttitudeEstimator:
    def reset(self, q, roll_progress=0.0):
        """Start from attitude q (body -> world, [w, x, y, z]); on the real drone, Betaflight's estimate when the
        flip is triggered."""
        self.q = np.array(q, float)
        self.roll_progress = roll_progress

    def update(self, gyro, dt):
        """Integrate one gyro reading [rad/s, body frame] over dt [s] (exact for a constant rate over dt)."""
        self.q = quat_multiply(self.q, axis_angle_quat(gyro * dt))
        self.q /= np.linalg.norm(self.q)
        self.roll_progress += gyro[0] * dt

    @property
    def gravity_body(self):
        """World up (z) in body frame, R^T e_z = the last row of R: (0, 0, 1) when level, (0, 0, -1) upside down.
        Yaw-free. On the real drone, computed from Betaflight's roll and pitch with sines and cosines, so it has no
        jump at roll = ±180 deg (Betaflight's axis and sign conventions differ from ours: check them on the bench)."""
        w, x, y, z = self.q
        return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
