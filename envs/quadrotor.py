"""Quadrotor on MuJoCo (model: envs/quadrotor.xml), with a CTBR (collective thrust + body rates) interface.

State, stored in MuJoCo's free joint:
    p   position, world frame [m]            (z up)          qpos[:3]
    v   velocity, world frame [m/s]                          qvel[:3]
    q   orientation quaternion [w, x, y, z]   (body -> world) qpos[3:7]
    w   angular velocity, body frame [rad/s]                 qvel[3:6]

Control: collective thrust and body rates -> rate controller -> mixer -> four motors, each with a first-order lag and
a thrust limit. When the motors can't deliver everything, the priority is roll and pitch, then collective, then yaw.

Simplifications:
    - Motor thrust is commanded directly: no battery, propeller or motor speed model, no rotor inertia.
    - No drag, no ground contact (episodes end above the floor).
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
        # override xml with config values: (since each model saves the config, and we can override in command line, and for future domain randomisation)
        self.model.body_mass[self.body] = cfg.mass
        self.model.body_inertia[self.body] = cfg.inertia
        self.model.opt.timestep = cfg.sim_dt

        # X layout, in the order of quadrotor.xml. Diagonal motors spin the same way, so yaw torque adds no roll or pitch
        offset = cfg.arm_length / np.sqrt(2)
        x, y = offset * np.array([1.0, 1, -1, -1]), offset * np.array([1.0, -1, -1, 1])
        yaw = cfg.yaw_torque_per_thrust * np.array([1.0, -1, 1, -1])
        self.model.site_pos[:, 0], self.model.site_pos[:, 1] = x, y
        self.model.actuator_gear[:, 5] = yaw
        self.model.actuator_dynprm[:, 0] = cfg.thrust_tau
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self.data = mujoco.MjData(self.model)

        # (collective, roll, pitch, yaw torque) = allocation @ motor thrusts. Only the torque columns of the inverse
        # are kept: _mix() handles the collective itself
        allocation = np.array([np.ones(4), y, -x, yaw])
        self.torque_to_motors = np.linalg.inv(allocation)[:, 1:]

        self.m = cfg.mass
        self.I = np.array(cfg.inertia, float)
        self.max_thrust_acc = cfg.thrust_to_weight * G  # max collective thrust / mass
        self.max_motor_thrust = self.m * self.max_thrust_acc / 4  # N
        self.max_rate = cfg.max_rate
        self.kp = np.array(cfg.rate_kp)
        self.dt = cfg.sim_dt
        self.reset()

    def reset(self, p=None, v=None, q=None, w=None):
        mujoco.mj_resetData(self.model, self.data)
        self.p = np.zeros(3) if p is None else p
        self.v = np.zeros(3) if v is None else v
        self.q = np.array([1.0, 0, 0, 0]) if q is None else q
        self.w = np.zeros(3) if w is None else w
        self.thrust_acc = G  # start at hover thrust
        self.saturation = 0

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

    @property
    def thrust_acc(self):
        """Actual collective thrust / mass, after the motor lag [m/s^2]."""
        return self.data.act.sum() / self.m

    @thrust_acc.setter
    def thrust_acc(self, value):
        self.data.act[:] = self.m * value / 4

    def _mix(self, thrust, torque):
        """Motor thrust commands [N] for a collective thrust [N] and a body torque [N m], each motor between 0 and
        max_motor_thrust. When they don't all fit, the priority is roll and pitch, then collective, then yaw, similar
        to PX4's roll/pitch airmode. Yaw comes last because it is weak: a small yaw error asks for large thrust
        differences, and would otherwise take the range the other axes need.

        self.saturation: the highest priority that wasn't delivered as asked. 0 = none, 1 = yaw, 2 = collective,
        3 = roll and pitch."""
        limit = self.max_motor_thrust
        self.saturation = 0

        # Roll and pitch: scaled down if too large, which keeps the axis of rotation
        roll_pitch = self.torque_to_motors[:, :2] @ torque[:2]
        spread = roll_pitch.max() - roll_pitch.min()
        if spread > limit:
            roll_pitch = roll_pitch * limit / spread
            self.saturation = 3

        # Collective: moved as little as needed to fit roll and pitch
        per_motor = np.clip(thrust / 4, -roll_pitch.min(), limit - roll_pitch.max())
        if per_motor != thrust / 4 and self.saturation == 0:
            self.saturation = 2
        motors = per_motor + roll_pitch

        # Yaw: scaled down to the range that is left on the motors it pushes up or down
        yaw = self.torque_to_motors[:, 2] * torque[2]
        if torque[2] != 0:
            room = np.where(yaw > 0, limit - motors, motors)
            needed = np.abs(yaw)
            if np.any(needed > room):
                yaw = yaw * max(0.0, np.min(room / needed))
                self.saturation = self.saturation or 1
        return np.clip(motors + yaw, 0.0, limit)  # the clip only removes rounding errors

    def step(self, thrust_acc_cmd, rate_cmd):
        """One physics step. thrust_acc_cmd: collective thrust / mass [m/s^2]. rate_cmd: body rates [rad/s]."""
        thrust_acc_cmd = np.clip(thrust_acc_cmd, 0.0, self.max_thrust_acc)
        rate_cmd = np.clip(rate_cmd, -self.max_rate, self.max_rate)

        # Rate controller: P on rate error + gyroscopic compensation
        w = self.w
        torque = self.I * (self.kp * (rate_cmd - w)) + np.cross(w, self.I * w)

        self.data.ctrl[:] = self._mix(self.m * thrust_acc_cmd, torque)
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
