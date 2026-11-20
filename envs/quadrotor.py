"""Quadrotor on MuJoCo with a collective thrust + body rates (CTBR) interface."""
import os

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from configs import Config

G = 9.81
N_MOTORS = 4
SAT_NONE, SAT_YAW, SAT_COLLECTIVE, SAT_ROLL_PITCH = 0, 1, 2, 3  # highest priority not delivered
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quadrotor.xml")


class Quadrotor:
    """MuJoCo quadrotor flown through a rate controller: collective thrust and body rate commands."""

    def __init__(self, cfg: Config):
        """cfg: the `dynamics` section of configs/env.yaml."""
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quad")
        # config values override the xml
        self.model.body_mass[self.body] = cfg.mass
        self.model.body_inertia[self.body] = cfg.inertia
        self.model.opt.timestep = cfg.sim_dt

        # X layout, in quadrotor.xml order
        offset = cfg.arm_length / np.sqrt(2)
        x, y = offset * np.array([1.0, 1, -1, -1]), offset * np.array([1.0, -1, -1, 1])
        yaw = cfg.yaw_torque_per_thrust * np.array([1.0, -1, 1, -1])
        motors = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"motor{i}") for i in range(N_MOTORS)]
        self.model.site_pos[motors, 0], self.model.site_pos[motors, 1] = x, y
        self.model.actuator_gear[:, 5] = yaw
        self.model.actuator_dynprm[:, 0] = cfg.thrust_tau
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self.data = mujoco.MjData(self.model)

        # (collective, roll, pitch, yaw torque) = allocation @ motor thrusts; _mix() handles the collective
        allocation = np.array([np.ones(N_MOTORS), y, -x, yaw])
        self.torque_to_motors = np.linalg.inv(allocation)[:, 1:]

        self.m = cfg.mass
        self.I = np.array(cfg.inertia, float)  # assumed by the rate controller
        self.max_thrust_acc = cfg.thrust_to_weight * G  # max collective thrust / mass
        self.max_motor_thrust = self.m * self.max_thrust_acc / N_MOTORS  # N
        self.hover_cmd = 1.0 / cfg.thrust_to_weight  # motor command at hover
        self.max_rate = cfg.max_rate
        self.drag_nominal = np.array(cfg.drag, float)
        self.thrust_tau = cfg.thrust_tau
        self.kp = np.array(cfg.rate_kp)
        self.dt = cfg.sim_dt
        self.set_params()
        self.reset()

    def set_params(self, hover_thrust: float = 1.0, thrust_curve: float = 0.0, inertia: ArrayLike = (1.0, 1.0, 1.0),
                   motor_tau: float = 1.0, drag: ArrayLike = (1.0, 1.0, 1.0)):
        """Set the plant's true dynamics, as multiples of nominal; the controller keeps assuming nominal."""
        self.hover_thrust, self.thrust_curve = hover_thrust, thrust_curve
        inertia = self.I * np.asarray(inertia)
        if not np.array_equal(inertia, self.model.body_inertia[self.body]):
            self.model.body_inertia[self.body] = inertia
            # Recompute MuJoCo's precomputed mass matrix, on scratch data (mj_setConst overwrites its state)
            mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self.model.actuator_dynprm[:, 0] = self.thrust_tau * motor_tau
        self.drag = self.drag_nominal * np.asarray(drag)

    def _motor_thrust(self, motors: np.ndarray) -> np.ndarray:
        """Nominal motor thrust commands [N] -> true thrust [N]."""
        c = motors / self.max_motor_thrust
        return motors * self.hover_thrust * (1.0 + self.thrust_curve * (c - self.hover_cmd))

    def reset(self, p: ArrayLike | None = None, v: ArrayLike | None = None, q: ArrayLike | None = None,
              w: ArrayLike | None = None):
        mujoco.mj_resetData(self.model, self.data)
        self.p = np.zeros(3) if p is None else p
        self.v = np.zeros(3) if v is None else v
        self.q = np.array([1.0, 0, 0, 0]) if q is None else q
        self.w = np.zeros(3) if w is None else w
        self.thrust_acc = G  # start at hover thrust
        self.saturation = SAT_NONE

    @property
    def p(self) -> np.ndarray:
        return self.data.qpos[:3]

    @p.setter
    def p(self, value: ArrayLike):
        self.data.qpos[:3] = value

    @property
    def v(self) -> np.ndarray:
        return self.data.qvel[:3]

    @v.setter
    def v(self, value: ArrayLike):
        self.data.qvel[:3] = value

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[3:7]

    @q.setter
    def q(self, value: ArrayLike):
        self.data.qpos[3:7] = value

    @property
    def w(self) -> np.ndarray:
        return self.data.qvel[3:6]

    @w.setter
    def w(self, value: ArrayLike):
        self.data.qvel[3:6] = value

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix (body -> world), from qpos (data.xmat lags one step)."""
        out = np.empty(9)
        mujoco.mju_quat2Mat(out, self.data.qpos[3:7])
        return out.reshape(3, 3)

    @property
    def imu(self) -> tuple[np.ndarray, np.ndarray]:
        """Ideal IMU readings in body frame: gyro [rad/s], specific force [m/s^2]; they lag one physics step."""
        return self.data.sensordata[:3].copy(), self.data.sensordata[3:6].copy()

    def forward(self):
        """Recompute the derived quantities (sensor readings) for the current state, without stepping."""
        mujoco.mj_forward(self.model, self.data)

    @property
    def thrust_acc(self) -> float:
        """Actual collective thrust / mass, after the motor lag [m/s^2]."""
        return self.data.act.sum() / self.m

    @thrust_acc.setter
    def thrust_acc(self, value: float):
        self.data.act[:] = self.m * value / N_MOTORS

    def _mix(self, thrust: float, torque: np.ndarray) -> np.ndarray:
        """Motor thrusts [N] for a collective thrust [N] and body torque [N m]; priority roll/pitch, collective, yaw."""
        limit = self.max_motor_thrust
        self.saturation = SAT_NONE

        # Roll and pitch: scaled down to fit, keeping the rotation axis
        roll_pitch = self.torque_to_motors[:, :2] @ torque[:2]
        spread = roll_pitch.max() - roll_pitch.min()
        if spread > limit:
            roll_pitch = roll_pitch * limit / spread
            self.saturation = SAT_ROLL_PITCH

        # Collective: moved as little as needed
        per_motor = np.clip(thrust / N_MOTORS, -roll_pitch.min(), limit - roll_pitch.max())
        if per_motor != thrust / N_MOTORS and self.saturation == SAT_NONE:
            self.saturation = SAT_COLLECTIVE
        motors = per_motor + roll_pitch

        # Yaw: scaled down to the range left
        yaw = self.torque_to_motors[:, 2] * torque[2]
        if torque[2] != 0:
            room = np.where(yaw > 0, limit - motors, motors)
            needed = np.abs(yaw)
            if np.any(needed > room):
                yaw = yaw * max(0.0, np.min(room / needed))
                self.saturation = self.saturation or SAT_YAW
        return np.clip(motors + yaw, 0.0, limit)  # rounding errors only

    def step(self, thrust_acc_cmd: float, rate_cmd: np.ndarray):
        """One physics step. thrust_acc_cmd: collective thrust / mass [m/s^2]. rate_cmd: body rates [rad/s]."""
        thrust_acc_cmd = np.clip(thrust_acc_cmd, 0.0, self.max_thrust_acc)
        rate_cmd = np.clip(rate_cmd, -self.max_rate, self.max_rate)

        # Rate controller: P + gyroscopic compensation
        w = self.w
        torque = self.I * (self.kp * (rate_cmd - w)) + np.cross(w, self.I * w)

        self.data.ctrl[:] = self._motor_thrust(self._mix(self.m * thrust_acc_cmd, torque))
        R = self.R
        self.data.xfrc_applied[self.body, :3] = R @ (-self.m * self.drag * (R.T @ self.v))  # drag, world frame
        mujoco.mj_step(self.model, self.data)
