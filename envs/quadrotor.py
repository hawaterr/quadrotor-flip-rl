"""Quadrotor on MuJoCo (model: envs/quadrotor.xml), with a CTBR (collective thrust + body rates) interface.

State, stored in MuJoCo's free joint:
    p   position, world frame [m]            (z up)          qpos[:3]
    v   velocity, world frame [m/s]                          qvel[:3]
    q   orientation quaternion [w, x, y, z]   (body -> world) qpos[3:7]
    w   angular velocity, body frame [rad/s]                 qvel[3:6]

Simplifications:
    - The rate controller outputs torques directly (no motor mixing, no motor saturation).
    - First-order lag on collective thrust only.
    - No drag, no ground contact.
"""
import os

import mujoco
import numpy as np

G = 9.81
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quadrotor.xml")


class Quadrotor:
    def __init__(self, cfg):
        """cfg: the `dynamics` section of configs/env.yaml."""
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quad")
        self.model.body_mass[self.body] = cfg.mass
        self.model.body_inertia[self.body] = cfg.inertia
        self.model.opt.timestep = cfg.sim_dt
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self.data = mujoco.MjData(self.model)

        self.m = cfg.mass
        self.I = np.array(cfg.inertia, float)
        self.max_thrust_acc = cfg.thrust_to_weight * G  # max collective thrust / mass
        self.max_rate = cfg.max_rate
        self.kp = np.array(cfg.rate_kp)
        self.thrust_tau = cfg.thrust_tau
        self.dt = cfg.sim_dt
        self.reset()

    def reset(self, p=None, v=None, q=None, w=None):
        mujoco.mj_resetData(self.model, self.data)
        self.p = np.zeros(3) if p is None else p
        self.v = np.zeros(3) if v is None else v
        self.q = np.array([1.0, 0, 0, 0]) if q is None else q
        self.w = np.zeros(3) if w is None else w
        self.thrust_acc = G  # start at hover thrust

    @property
    def p(self):
        return self.data.qpos[:3]

    @p.setter
    def p(self, value):
        self.data.qpos[:3] = value

    @property
    def v(self):
        return self.data.qvel[:3]

    @v.setter
    def v(self, value):
        self.data.qvel[:3] = value

    @property
    def q(self):
        return self.data.qpos[3:7]

    @q.setter
    def q(self, value):
        self.data.qpos[3:7] = value

    @property
    def w(self):
        return self.data.qvel[3:6]

    @w.setter
    def w(self, value):
        self.data.qvel[3:6] = value

    @property
    def R(self):
        """Rotation matrix (body -> world), from qpos: after mj_step, data.xmat still holds the pose from before
        the integration."""
        out = np.empty(9)
        mujoco.mju_quat2Mat(out, self.data.qpos[3:7])
        return out.reshape(3, 3)

    def step(self, thrust_acc_cmd, rate_cmd):
        """One physics step. thrust_acc_cmd: collective thrust / mass [m/s^2]. rate_cmd: body rates [rad/s]."""
        dt = self.dt

        thrust_acc_cmd = np.clip(thrust_acc_cmd, 0.0, self.max_thrust_acc)
        rate_cmd = np.clip(rate_cmd, -self.max_rate, self.max_rate)

        # Thrust lag (motors can't change instantly): first order low pass filter
        self.thrust_acc += (thrust_acc_cmd - self.thrust_acc) * dt / self.thrust_tau

        # Rate controller: P on rate error + gyroscopic compensation (MuJoCo adds the gyroscopic term itself,
        # so the two cancel)
        w = self.w
        torque = self.I * (self.kp * (rate_cmd - w)) + np.cross(w, self.I * w)

        # xfrc_applied is in the world frame
        R = self.R
        self.data.xfrc_applied[self.body, :3] = self.m * self.thrust_acc * R[:, 2]
        self.data.xfrc_applied[self.body, 3:] = R @ torque
        mujoco.mj_step(self.model, self.data)


if __name__ == "__main__":
    # Sanity checks: python -m envs.quadrotor
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
