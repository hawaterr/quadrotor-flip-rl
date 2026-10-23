"""Gymnasium environment: do one roll flip, then recover to level and stop rotating.

Action (4, in [-1, 1]): collective thrust, roll rate, pitch rate, yaw rate  -> scaled to CTBR.
Observation: ground-truth state for now. Later, swap in your IMU-based estimate.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quadrotor import G, Quadrotor


class FlipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ctrl_dt=0.01, episode_time=2.0, start_height=3.0, min_height=0.5):
        self.quad = Quadrotor()
        self.substeps = int(round(ctrl_dt / self.quad.dt))
        self.ctrl_dt = ctrl_dt
        self.max_steps = int(episode_time / ctrl_dt)
        self.start_height = start_height
        self.min_height = min_height

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # rot matrix (9) + body rates (3) + velocity (3) + height error (1)
        # + flip progress (1) + flip done flag (1) + previous action (4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32)

    def _scale_action(self, a):
        thrust = (a[0] + 1.0) / 2.0 * self.quad.max_thrust_acc
        rates = a[1:] * self.quad.max_rate
        return thrust, rates

    def _obs(self):
        q = self.quad
        return np.concatenate([
            q.R.flatten(),
            q.w / 10.0,
            q.v / 5.0,
            [(q.p[2] - self.start_height) / 2.0],
            # Attitude alone can't tell "before flip" from "after flip" (both upright),
            # so the agent needs to know how far it has rotated.
            [self.roll_progress / (2 * np.pi)],
            [float(self.flip_done)],
            self.prev_action,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.quad.reset(
            p=[0, 0, self.start_height],
            v=rng.normal(0, 0.1, 3),
            w=rng.normal(0, 0.1, 3),
        )
        self.roll_progress = 0.0
        self.flip_done = False
        self.prev_action = np.zeros(4)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        thrust, rates = self._scale_action(action)
        for _ in range(self.substeps):
            self.quad.step(thrust, rates)
            self.roll_progress += self.quad.w[0] * self.quad.dt
        self.t += 1

        q = self.quad
        upright = q.R[2, 2]  # 1 = level, -1 = upside down
        reward = 0.0

        if not self.flip_done:
            # Reward rotating about body x in the positive direction
            reward += 1.0 * q.w[0] * self.ctrl_dt
            if self.roll_progress >= 2 * np.pi * 0.95:
                self.flip_done = True
                reward += 10.0  # completion bonus
        else:
            # Recovery: be level, stop rotating, stop falling
            reward += 0.5 * upright
            reward -= 0.05 * np.linalg.norm(q.w)
            reward -= 0.05 * abs(q.v[2])

        # Small penalties: don't lose too much altitude, keep actions smooth
        reward -= 0.1 * max(0.0, self.start_height - q.p[2])
        reward -= 0.01 * np.sum((action - self.prev_action) ** 2)
        self.prev_action = action

        terminated = bool(q.p[2] < self.min_height)
        if terminated:
            reward -= 10.0
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {
            "flip_done": self.flip_done, "roll_progress": self.roll_progress}
