"""Gymnasium quadrotor environment with selectable task.

    task="hover": hold a fixed target point, starting from a perturbed state.
    task="flip":  do one roll flip, then recover to level and stop rotating.

Action (4, in [-1, 1]): collective thrust, roll rate, pitch rate, yaw rate  -> scaled to CTBR.
    Thrust is centred on hover: a[0] = 0 -> thrust = g, so a zero-mean initial policy roughly hovers.
Observation: ground-truth state for now. Later, swap in your IMU-based estimate.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quadrotor import G, Quadrotor

TASKS = ("hover", "flip")
# Per-task defaults, used when the matching constructor argument is None
EPISODE_TIME = {"hover": 5.0, "flip": 2.0}
INIT_NOISE = {"hover": 1.0, "flip": 0.0}
MAX_DIST = {"hover": 3.0, "flip": 6.0}  # a flip needs room to climb first


class QuadEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task="hover", ctrl_dt=0.01, episode_time=None, target=(0.0, 0.0, 3.0),
                 min_height=0.5, max_dist=None, init_noise=None, mid_flip_prob=0.0):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        self.task = task
        self.quad = Quadrotor()
        self.substeps = int(round(ctrl_dt / self.quad.dt))
        self.ctrl_dt = ctrl_dt
        self.max_steps = int((episode_time or EPISODE_TIME[task]) / ctrl_dt)
        self.target = np.array(target, float)
        self.min_height = min_height
        self.max_dist = MAX_DIST[task] if max_dist is None else max_dist
        # scales the initial perturbation (0 = start exactly at hover, plus tiny v/w noise)
        self.init_noise = INIT_NOISE[task] if init_noise is None else init_noise
        # flip only: fraction of episodes that start part-way through a roll (training curriculum,
        # so the agent sees "finish the flip and recover" without first discovering a full roll)
        self.mid_flip_prob = mid_flip_prob

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # rot matrix (9) + body rates (3) + velocity (3) + position error (3) + previous action (4)
        # flip only: + flip progress (1) + flip done flag (1)
        n_obs = 22 + (2 if task == "flip" else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

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
        parts = [
            q.R.flatten(),
            q.w / 10.0,
            q.v / 5.0,
            (q.p - self.target) / 2.0,
            self.prev_action,
        ]
        if self.task == "flip":
            # Attitude alone can't tell "before flip" from "after flip" (both upright),
            # so the agent needs to know how far it has rotated.
            parts += [[self.roll_progress / (2 * np.pi)], [float(self.flip_done)]]
        return np.concatenate(parts).astype(np.float32)

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
            v=rng.normal(0, max(0.3 * s, 0.1), 3),
            q=q0,
            w=rng.normal(0, max(0.5 * s, 0.1), 3),
        )
        self.roll_progress = 0.0
        self.flip_done = False
        if self.task == "flip" and rng.random() < self.mid_flip_prob:
            # Already rolling at a random angle, with some climb speed to carry it through
            phi = rng.uniform(0, 0.9 * 2 * np.pi)
            self.quad.q = np.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])
            self.quad.w = np.array([10.0, 0, 0]) + rng.normal(0, 0.5, 3)
            self.quad.v = np.array([0, 0, 2.0]) + rng.normal(0, 0.3, 3)
            self.roll_progress = phi
        self.best_progress = np.clip(self.roll_progress, 0, 2 * np.pi)
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
        dist = np.linalg.norm(q.p - self.target)
        if self.task == "hover":
            reward, terminated = self._hover_reward(action, dist)
        else:
            reward, terminated = self._flip_reward(action, dist)
        self.prev_action = action

        if terminated:
            reward -= 10.0
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {
            "dist": dist, "flip_done": self.flip_done, "roll_progress": self.roll_progress}

    def _hover_reward(self, action, dist):
        q = self.quad
        upright = q.R[2, 2]  # 1 = level, -1 = upside down
        # Positive "alive" reward that is largest at the target, so surviving is always worth it
        reward = 1.0 - 0.5 * dist
        reward += 0.2 * upright
        reward -= 0.05 * np.linalg.norm(q.w)
        reward -= 0.05 * np.linalg.norm(q.v)
        reward -= 0.01 * np.sum((action - self.prev_action) ** 2)
        terminated = bool(q.p[2] < self.min_height or dist > self.max_dist)
        return reward, terminated

    def _flip_reward(self, action, dist):
        q = self.quad
        upright = q.R[2, 2]
        reward = 0.0
        if not self.flip_done:
            # Reward new roll progress only, so rolling back and forth can't be farmed
            progress = np.clip(self.roll_progress, 0, 2 * np.pi)
            reward += 3.0 * max(0.0, progress - self.best_progress)
            self.best_progress = max(self.best_progress, progress)
            if self.roll_progress >= 2 * np.pi * 0.95:
                self.flip_done = True
                reward += 10.0  # completion bonus
        else:
            # Recovery: be level, stop rotating, stop falling
            reward += 0.5 * upright
            reward -= 0.05 * np.linalg.norm(q.w)
            reward -= 0.05 * abs(q.v[2])

        # Stay near the start point (climbing away is not free), keep actions smooth.
        # The first 3 m are free: a flip needs to climb first and ends up above the start.
        reward -= 0.5 * max(0.0, dist - 3.0)
        reward -= 0.01 * np.sum((action - self.prev_action) ** 2)
        terminated = bool(q.p[2] < self.min_height or dist > self.max_dist)
        return reward, terminated
