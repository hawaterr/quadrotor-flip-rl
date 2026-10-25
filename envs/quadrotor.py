"""Minimal quadrotor rigid-body simulator with a CTBR (collective thrust + body rates) interface.

State:
    p   position, world frame [m]            (z up)
    v   velocity, world frame [m/s]
    q   orientation quaternion [w, x, y, z]   (body -> world)
    w   angular velocity, body frame [rad/s]

Simplifications (fine for a first version, improve later):
    - The rate controller outputs torques directly (no motor mixing, no motor saturation).
    - First-order lag on collective thrust only.
    - No drag, no ground contact.
"""
import numpy as np

G = 9.81


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Quadrotor:
    def __init__(self, cfg):
        """cfg: the `dynamics` section of configs/env.yaml."""
        self.m = cfg.mass
        self.I = np.diag(cfg.inertia)
        self.I_inv = np.linalg.inv(self.I)
        self.max_thrust_acc = cfg.thrust_to_weight * G  # max collective thrust / mass
        self.max_rate = cfg.max_rate
        self.kp = np.array(cfg.rate_kp)
        self.thrust_tau = cfg.thrust_tau
        self.dt = cfg.sim_dt
        self.reset()

    def reset(self, p=None, v=None, q=None, w=None):
        self.p = np.zeros(3) if p is None else np.array(p, float)
        self.v = np.zeros(3) if v is None else np.array(v, float)
        self.q = np.array([1.0, 0, 0, 0]) if q is None else np.array(q, float)
        self.w = np.zeros(3) if w is None else np.array(w, float)
        self.thrust_acc = G  # start at hover thrust

    @property
    def R(self):
        return quat_to_rot(self.q)

    def step(self, thrust_acc_cmd, rate_cmd):
        """One physics step. thrust_acc_cmd: collective thrust / mass [m/s^2]. rate_cmd: body rates [rad/s]."""
        dt = self.dt

        # Clip high values
        thrust_acc_cmd = np.clip(thrust_acc_cmd, 0.0, self.max_thrust_acc)
        rate_cmd = np.clip(rate_cmd, -self.max_rate, self.max_rate)

        # Thrust lag (motors can't change instantly): first order low pass filter
        self.thrust_acc += (thrust_acc_cmd - self.thrust_acc) * dt / self.thrust_tau

        # Rate controller: P on rate error + gyroscopic compensation
        Iw = self.I @ self.w
        torque = self.I @ (self.kp * (rate_cmd - self.w)) + np.cross(self.w, Iw)
        # beta flight has similar low level controller: apply torque proportional to error in angular velocity
        # then do compensation for gyroscopic effects
        # NOTE: rate is using low level controller but thrust low pass filter, a bit odd

        # Translational dynamics (world frame)
        acc = self.R @ np.array([0.0, 0.0, self.thrust_acc]) - np.array([0.0, 0.0, G])

        # Rotational dynamics (body frame): Euler's equation
        w_dot = self.I_inv @ (torque - np.cross(self.w, Iw))

        # Integrate (semi-implicit Euler)
        self.v += acc * dt
        self.p += self.v * dt
        self.w += w_dot * dt
        self.q += 0.5 * quat_mul(self.q, np.array([0.0, *self.w])) * dt
        self.q /= np.linalg.norm(self.q)
        return acc


if __name__ == "__main__":
    # Sanity checks
    from configs import load_env
    quad = Quadrotor(load_env("hover").dynamics)

    quad.reset(p=[0, 0, 10])
    quad.thrust_acc = 0.0
    for _ in range(500):  # 1 s free fall
        quad.step(0.0, np.zeros(3))
    print(f"free fall 1s: z={quad.p[2]:.2f} (expect ~{10 - 0.5 * G:.2f} minus a bit of thrust lag)")

    quad.reset(p=[0, 0, 2])
    for _ in range(1000):  # 2 s hover
        quad.step(G, np.zeros(3))
    print(f"hover 2s: z={quad.p[2]:.3f} (expect 2.000)")

    quad.reset(p=[0, 0, 2])
    for _ in range(int(2 * np.pi / 10.0 / quad.dt)):  # roll at 10 rad/s for one revolution
        quad.step(G, np.array([10.0, 0, 0]))
    print(f"one roll rev: body z-axis world z = {quad.R[2, 2]:.2f} (expect ~1, slightly less from controller lag)")
