"""Gymnasium environment: hover at a fixed target point, starting from a perturbed state.

Action (4, in [-1, 1]): collective thrust, roll rate, pitch rate, yaw rate  -> scaled to CTBR.
    Thrust is centred on hover: a[0] = 0 -> thrust = g, so a zero-mean initial policy roughly hovers.
Observation: ground-truth state for now. Later, swap in your IMU-based estimate.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quadrotor import G, Quadrotor


class HoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ctrl_dt=0.01, episode_time=5.0, target=(0.0, 0.0, 3.0),
                 min_height=0.5, max_dist=3.0, init_noise=1.0):
        self.quad = Quadrotor()
        self.substeps = int(round(ctrl_dt / self.quad.dt))
        self.ctrl_dt = ctrl_dt
        self.max_steps = int(episode_time / ctrl_dt)
        self.target = np.array(target, float)
        self.min_height = min_height
        self.max_dist = max_dist
        self.init_noise = init_noise  # scales the initial perturbation (0 = start exactly at hover)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # rot matrix (9) + body rates (3) + velocity (3) + position error (3) + previous action (4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32)

    def _scale_action(self, a):
        # a[0] = -1 -> 0 thrust, 0 -> g, 1 -> max thrust
        if a[0] >= 0:
            thrust = G + a[0] * (self.quad.max_thrust_acc - G)
        else:
            thrust = G * (1.0 + a[0])

        
        rates = a[1:] * self.quad.max_rate
        return thrust, rates

    def _obs(self):
        q = self.quad
        return np.concatenate([
            q.R.flatten(),
            q.w / 10.0,
            q.v / 5.0,
            (q.p - self.target) / 2.0,
            self.prev_action,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        s = self.init_noise
        # Random small tilt as an axis-angle -> quaternion
        axis_angle = rng.normal(0, 0.2 * s, 3)
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle if angle > 1e-8 else np.array([1.0, 0, 0])
        q0 = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
        self.quad.reset(
            p=self.target + rng.uniform(-0.5 * s, 0.5 * s, 3),
            v=rng.normal(0, 0.3 * s, 3),
            q=q0,
            w=rng.normal(0, 0.5 * s, 3),
        )
        self.prev_action = np.zeros(4)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1) 
        thrust, rates = self._scale_action(action)
        for _ in range(self.substeps):
            self.quad.step(thrust, rates)
        self.t += 1

        q = self.quad
        dist = np.linalg.norm(q.p - self.target)
        upright = q.R[2, 2]  # 1 = level, -1 = upside down

        # Positive "alive" reward that is largest at the target, so surviving is always worth it
        reward = 1.0 - 0.5 * dist
        reward += 0.2 * upright
        reward -= 0.05 * np.linalg.norm(q.w)
        reward -= 0.05 * np.linalg.norm(q.v)
        reward -= 0.01 * np.sum((action - self.prev_action) ** 2)
        self.prev_action = action

        terminated = bool(q.p[2] < self.min_height or dist > self.max_dist)
        if terminated:
            reward -= 10.0
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {"dist": dist}
