"""Gymnasium quadrotor env: flip and hover tasks, with an asymmetric actor/critic observation."""
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs import Config, load_env
from envs.quadrotor import G, Quadrotor
from envs.sensors import AttitudeEstimator, ImuModel, axis_angle_quat, quat_multiply

ACT_DIM = 4                     # collective thrust, roll, pitch, yaw rate
ACTOR_BASE_DIM = 3 + 1          # gravity in body frame, elapsed share of the episode
FRAME_DIM = 3 + 3 + ACT_DIM     # gyro, accelerometer, action
STATE_DIM = 9 + 3 + 3 + 3       # rotation matrix, body rates, velocity, position error
FLIP_DIM = 1 + 1                # roll progress, flip done flag
N_PARAM_OBS = 1 + 1 + 3 + 1 + 1 + 1 + 1  # hover thrust, thrust curve, inertia, motor lag, drag xy, drag z, latency


class QuadEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task="flip", cfg: Config | None = None, training=False):
        """cfg: env config (default configs/env.yaml); training: enables mid-flip starts."""
        self.cfg = cfg if cfg is not None else load_env(task)
        self.task = self.cfg.task
        self.training = training
        self.quad = Quadrotor(self.cfg.dynamics)
        self.policy_dt = self.cfg.policy_dt
        self.substeps = int(round(self.policy_dt / self.quad.dt))
        self.max_steps = int(self.cfg.episode_time / self.policy_dt)
        self.target = np.array(self.cfg.target, float)
        self.imu = ImuModel(self.cfg.sensors)
        self.estimator = AttitudeEstimator()
        self.history_len = self.cfg.history
        scale = self.cfg.obs_scale
        self.w_scale, self.v_scale, self.pos_scale = scale.body_rate, scale.velocity, scale.position
        self.latency_range = [int(round(s / self.quad.dt)) for s in self.cfg.latency]
        self.randomisation = self.cfg.randomisation
        self.params = {}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        flip_dim = FLIP_DIM if self.task == "flip" else 0
        self.n_actor_obs = ACTOR_BASE_DIM + FRAME_DIM * self.history_len + flip_dim
        n_obs = self.n_actor_obs + STATE_DIM + flip_dim + (N_PARAM_OBS if self.randomisation else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

    def scale_action(self, a: np.ndarray) -> tuple[float, np.ndarray]:
        """Action in [-1, 1] -> (thrust / mass [m/s^2], body rates [rad/s]); a[0] = 0 is hover."""
        if a[0] >= 0:
            thrust = G + a[0] * (self.quad.max_thrust_acc - G)
        else:
            thrust = G * (1.0 + a[0])
        rates = a[1:] * self.quad.max_rate
        return thrust, rates

    def _obs(self) -> np.ndarray:
        q, est = self.quad, self.estimator
        actor = [est.gravity_body, [self.t / self.max_steps], *self.history]
        privileged = [q.R.flatten(), q.w / self.w_scale, q.v / self.v_scale,
                      (q.p - self.target) / self.pos_scale]
        if self.task == "flip":
            # Before and after the flip look the same, so the agent needs its roll progress
            actor += [[est.roll_progress / (2 * np.pi)], [float(self.est_flip_done)]]
            privileged += [[self.roll_progress / (2 * np.pi)], [float(self.flip_done)]]
        if self.randomisation:
            privileged.append(self.param_obs)
        return np.concatenate(actor + privileged).astype(np.float32)

    def _push_frame(self, gyro: np.ndarray, accel: np.ndarray, action: np.ndarray):
        """Prepend the newest IMU reading and action to the history."""
        self.history.insert(0, np.concatenate([gyro / self.w_scale, accel / G, action]))
        del self.history[self.history_len:]

    def _read_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """Noisy IMU reading, also fed to the attitude estimator."""
        gyro, accel = self.imu.read(*self.quad.imu)
        self.estimator.update(gyro, self.quad.dt)
        return gyro, accel

    def _update_est_flip_done(self):
        if self.task == "flip":
            self.est_flip_done |= self.estimator.roll_progress >= 2 * np.pi * self.cfg.reward.done_frac

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """options={"exact": True}: no minimum v/w spread (evaluation)."""
        super().reset(seed=seed)
        self._random_start(exact=(options or {}).get("exact", False))
        self.roll_progress = 0.0
        self.flip_done = False
        mid = self.cfg.mid_flip
        prob = mid.prob if (self.training and mid is not None) else 0.0
        if self.task == "flip" and self.np_random.random() < prob:
            self._mid_flip_start()
        self.best_progress = np.clip(self.roll_progress, 0, 2 * np.pi)
        self.prev_action = np.zeros(ACT_DIM)
        self.t = 0

        rng = self.np_random
        self.quad.forward()  # sensor readings for the start state
        self.imu.reset(rng)
        error = axis_angle_quat(rng.normal(0, self.cfg.sensors.att_init_std, 3))
        self.estimator.reset(quat_multiply(self.quad.q, error), self.roll_progress)
        self.est_flip_done = False
        self._update_est_flip_done()
        gyro, accel = self.imu.read(*self.quad.imu)
        self.history = []
        for _ in range(self.history_len):
            self._push_frame(gyro, accel, self.prev_action)

        # Until the first command arrives: hover thrust at the current body rates
        lo, hi = self.latency_range
        self.delay = lo if lo == hi else int(rng.integers(lo, hi + 1))
        held = (G, np.clip(self.quad.w, -self.quad.max_rate, self.quad.max_rate))
        self.commands = deque([held] * self.delay)
        if self.randomisation:  # drawn last, so starts don't depend on it
            self._randomise_dynamics()
        return self._obs(), {}

    def _randomise_dynamics(self):
        """Draw this episode's true dynamics; param_obs holds them scaled to [-1, 1] for the critic."""
        rng, dr = self.np_random, self.randomisation
        draw = lambda name, n=None: rng.uniform(*dr[name], n)
        drag_xy, drag_z = draw("drag_xy"), draw("drag_z")
        p = self.params = dict(hover_thrust=draw("hover_thrust"), thrust_curve=draw("thrust_curve"),
                               inertia=draw("inertia", 3), motor_tau=draw("motor_tau"),
                               drag=np.array([drag_xy, drag_xy, drag_z]))
        self.quad.set_params(**p)

        def scaled(value, lo, hi):
            return np.zeros_like(value) if hi == lo else (np.asarray(value) - (lo + hi) / 2) / ((hi - lo) / 2)
        self.param_obs = np.concatenate([
            [scaled(p["hover_thrust"], *dr.hover_thrust), scaled(p["thrust_curve"], *dr.thrust_curve)],
            scaled(p["inertia"], *dr.inertia),
            [scaled(p["motor_tau"], *dr.motor_tau), scaled(drag_xy, *dr.drag_xy), scaled(drag_z, *dr.drag_z),
             scaled(self.delay, *self.latency_range)]])

    def _random_start(self, exact: bool):
        """Random state around the target; exact drops the minimum v/w spread."""
        rng, start = self.np_random, self.cfg.start
        min_std = 0.0 if exact else start.min_std
        axis_angle = rng.normal(0, start.tilt_std, 3)
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle if angle > 1e-8 else np.array([1.0, 0, 0])
        q0 = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
        self.quad.reset(
            p=self.target + rng.uniform(-start.pos_range, start.pos_range, 3),
            v=rng.normal(0, max(start.v_std, min_std), 3),
            q=q0,
            w=rng.normal(0, max(start.w_std, min_std), 3),
        )

    def _mid_flip_start(self):
        """Curriculum: start part-way through a rolling, climbing flip."""
        rng, mid = self.np_random, self.cfg.mid_flip
        phi = rng.uniform(0, mid.max_angle_frac * 2 * np.pi)
        self.quad.q = np.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])
        self.quad.w = np.array([mid.roll_rate, 0, 0]) + rng.normal(0, mid.rate_std, 3)
        self.quad.v = np.array([0, 0, mid.climb_speed]) + rng.normal(0, mid.climb_std, 3)
        self.roll_progress = phi

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1, 1)
        thrust, rates = self.scale_action(action)
        for _ in range(self.substeps):
            self.commands.append((thrust, rates))
            self.quad.step(*self.commands.popleft())  # the command sent `delay` physics steps ago
            self.roll_progress += self.quad.w[0] * self.quad.dt
            gyro, accel = self._read_imu()
        self._update_est_flip_done()
        self._push_frame(gyro, accel, action)
        self.t += 1

        q = self.quad
        dist = np.linalg.norm(q.p - self.target)
        if self.task == "hover":
            reward, terminated = self._hover_reward(action, dist)
        else:
            reward, terminated = self._flip_reward(action, dist)
        self.prev_action = action

        crashed = terminated
        if crashed:
            reward -= self.cfg.reward.crash
        truncated = self.t >= self.max_steps and not crashed
        if truncated and self.task == "flip":
            # Handover to the pilot: a true end, with a bonus for being back at the start
            if self.flip_done:
                reward += self.cfg.reward.handover * np.exp(-dist / self.cfg.reward.handover_scale)
            terminated, truncated = True, False
        return self._obs(), float(reward), terminated, truncated, {
            "dist": dist, "flip_done": self.flip_done, "roll_progress": self.roll_progress, "crashed": crashed}

    def _terminated(self, dist: float) -> bool:
        return bool(self.quad.p[2] < self.cfg.min_height or dist > self.cfg.max_dist)

    def _hover_reward(self, action: np.ndarray, dist: float) -> tuple[float, bool]:
        q, r = self.quad, self.cfg.reward
        upright = q.R[2, 2]  # 1 = level, -1 = upside down
        reward = r.alive - r.dist * dist
        reward += r.upright * upright
        reward -= r.body_rate * np.linalg.norm(q.w)
        reward -= r.speed * np.linalg.norm(q.v)
        reward -= r.smooth * np.sum((action - self.prev_action) ** 2)
        return reward, self._terminated(dist)

    def _flip_reward(self, action: np.ndarray, dist: float) -> tuple[float, bool]:
        q, r = self.quad, self.cfg.reward
        upright = q.R[2, 2]
        reward = 0.0
        if not self.flip_done:
            # New progress only, so rolling back and forth can't be farmed
            progress = np.clip(self.roll_progress, 0, 2 * np.pi)
            reward += r.roll_progress * max(0.0, progress - self.best_progress)
            self.best_progress = max(self.best_progress, progress)
            if self.roll_progress >= 2 * np.pi * r.done_frac:
                self.flip_done = True
                reward += r.completion_bonus
        else:

            reward += r.upright * upright
            reward += r.target * np.exp(-dist / r.target_scale)
            reward += r.still * np.exp(-np.linalg.norm(q.v) / r.still_scale)
            reward -= r.body_rate * np.linalg.norm(q.w)
            reward -= r.speed * min(np.linalg.norm(q.v), r.speed_cap)
            reward -= r.thrust_effort * action[0] ** 2

        # The first free_dist metres are free: a flip needs to climb
        reward -= r.dist * max(0.0, dist - r.free_dist)
        reward -= r.smooth * np.sum((action - self.prev_action) ** 2)
        return reward, self._terminated(dist)
