"""Quadrotor physics, motor mixer and attitude estimator."""
from collections.abc import Callable

import mujoco
import numpy as np

from configs import Config, load_env
from envs.quad_env import QuadEnv
from envs.quadrotor import N_MOTORS, SAT_NONE, SAT_ROLL_PITCH, SAT_YAW, G, Quadrotor


def test_physics_sanity():
    quad = Quadrotor(load_env("hover").dynamics)

    # Free fall from rest, thrust off: z(t) with linear vertical drag k
    quad.reset(p=[0, 0, 10])
    quad.thrust_acc = 0.0
    t, k = 1.0, quad.drag[2]
    for _ in range(round(t / quad.dt)):
        quad.step(0.0, np.zeros(3))
    expected = 10 - G / k * (t - (1 - np.exp(-k * t)) / k)
    assert abs(quad.p[2] - expected) < 0.05

    # Hover thrust holds altitude
    quad.reset(p=[0, 0, 2])
    for _ in range(round(2.0 / quad.dt)):
        quad.step(G, np.zeros(3))
    assert abs(quad.p[2] - 2.0) < 0.01

    # One commanded roll revolution ends close to level
    quad.reset(p=[0, 0, 2])
    rate = 10.0
    for _ in range(round(2 * np.pi / rate / quad.dt)):
        quad.step(G, np.array([rate, 0, 0]))
    assert quad.R[2, 2] > 0.9


def test_mixer():
    """Motor thrusts, mapped through the MuJoCo model's geometry, deliver the command; saturation priorities."""
    quad = Quadrotor(load_env("hover").dynamics)
    m = quad.model
    sites = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"motor{i}") for i in range(N_MOTORS)]
    x, y = m.site_pos[sites, 0], m.site_pos[sites, 1]
    allocation = np.array([np.ones(N_MOTORS), y, -x, m.actuator_gear[:, 5]])
    hover = quad.m * G

    torque = np.array([0.01, -0.02, 0.005])
    np.testing.assert_allclose(allocation @ quad._mix(hover, torque), [hover, *torque], atol=1e-9)
    assert quad.saturation == SAT_NONE

    # Too much roll/pitch: scaled down, keeping the rotation axis
    delivered = allocation @ quad._mix(hover, np.array([10.0, 5.0, 0.0]))
    assert quad.saturation == SAT_ROLL_PITCH
    assert np.isclose(delivered[1] / delivered[2], 2.0)

    # Too much yaw: yaw gives way, collective is kept
    delivered = allocation @ quad._mix(hover, np.array([0.0, 0.0, 10.0]))
    assert quad.saturation == SAT_YAW
    assert np.isclose(delivered[0], hover)
    assert 0 < delivered[3] < 10.0


def test_estimator_tracks_attitude(env_cfg: Callable[..., Config]):
    """With an ideal gyro, the integrated attitude follows the true one through a full roll (plus pitch and yaw)."""
    env = QuadEnv(cfg=env_cfg("flip", ideal=True))
    env.reset(seed=0)
    for _ in range(40):  # 0.8 s at full roll rate: over one revolution
        env.step(np.array([0.0, 1.0, 0.3, 0.2]))
        # The gyro lags one physics step, so the error stays below max_rate * dt
        np.testing.assert_allclose(env.estimator.gravity_body, env.quad.R[2, :], atol=0.05)
        assert abs(env.estimator.roll_progress - env.roll_progress) < 0.05
    assert env.roll_progress > 2 * np.pi
