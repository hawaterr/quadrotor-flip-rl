"""Gymnasium quadrotor environment with selectable task.

    task="flip":  do one roll flip, then recover to a still, level hover at the start height.
    task="hover": hold a fixed target point, starting from a perturbed state (a quick sanity check).

Action (4, in [-1, 1]): collective thrust, roll rate, pitch rate, yaw rate  -> scaled to CTBR.
    Thrust is centred on hover: a[0] = 0 -> thrust = g, so a zero-mean initial policy roughly hovers.

Observation = [actor part, privileged part] (asymmetric actor-critic, ppo/asymmetric.py):
    actor part (the first n_actor_obs entries): only what the real drone has (envs/sensors.py), so the policy can be
        deployed: estimated gravity direction in body frame (3), then the last `history` frames, newest first, of
        [noisy gyro (3), noisy accelerometer (3), action taken (4)]; flip only: + estimated roll progress (1) +
        estimated flip done flag (1). No position, no velocity: the drone can't measure them.
    privileged part: the ground-truth state, read only by the critic (training in sim): rotation matrix (9), body
        rates (3), velocity (3), position error (3); flip only: + true roll progress (1) + true flip done flag (1).
The reward is computed from the ground truth (training only).

All settings (dynamics, episode length, limits, start noise, curriculum, reward weights) come from
configs/env.yaml (configs.load_env).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs import load_env
from envs.quadrotor import G, Quadrotor
from envs.sensors import AttitudeEstimator, ImuModel, axis_angle_quat, quat_multiply

# Observation scaling, to keep inputs roughly in [-1, 1]. Fixed, not a config setting: trained models
# depend on it, so changing it would silently break them.
W_SCALE = 10.0    # rad/s
V_SCALE = 5.0     # m/s
POS_SCALE = 2.0   # m


class QuadEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task="flip", cfg=None, training=False):
        """cfg: the env config (cfg.env of a loaded config); by default configs/env.yaml for `task`.
        training=True turns on the flip's mid-roll starts (curriculum); evaluation always starts from rest."""
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

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        flip_extra = 2 if self.task == "flip" else 0  # roll progress + flip done flag
        self.n_actor_obs = 3 + 10 * self.history_len + flip_extra
        n_obs = self.n_actor_obs + 18 + flip_extra
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

    def scale_action(self, a):
        """Action in [-1, 1] -> (collective thrust / mass [m/s^2], body rates [rad/s]).
        a[0] = -1 -> 0 thrust, 0 -> g, 1 -> max thrust."""
        if a[0] >= 0:
            thrust = G + a[0] * (self.quad.max_thrust_acc - G)
        else:
            thrust = G * (1.0 + a[0])
        rates = a[1:] * self.quad.max_rate
        return thrust, rates

    def _obs(self):
        q, est = self.quad, self.estimator
        actor = [est.gravity_body, *self.history]
        privileged = [q.R.flatten(), q.w / W_SCALE, q.v / V_SCALE, (q.p - self.target) / POS_SCALE]
        if self.task == "flip":
            # Attitude alone can't tell "before flip" from "after flip" (both upright),
            # so the agent needs to know how far it has rotated.
            actor += [[est.roll_progress / (2 * np.pi)], [float(self.est_flip_done)]]
            privileged += [[self.roll_progress / (2 * np.pi)], [float(self.flip_done)]]
        return np.concatenate(actor + privileged).astype(np.float32)

    def _push_frame(self, gyro, accel, action):
        """Newest IMU reading and the action taken, at the front of the history."""
        self.history.insert(0, np.concatenate([gyro / W_SCALE, accel / G, action]))
        del self.history[self.history_len:]

    def _read_imu(self):
        """Noisy IMU readings, integrated by the attitude estimator."""
        gyro, accel = self.imu.read(*self.quad.imu)
        self.estimator.update(gyro, self.quad.dt)
        return gyro, accel

    def _update_est_flip_done(self):
        if self.task == "flip":
            self.est_flip_done |= self.estimator.roll_progress >= 2 * np.pi * self.cfg.reward.done_frac

    def reset(self, seed=None, options=None):
        """options={"exact": True} (used for evaluation): drop the minimum v/w spread, so a start with zero
        noise (the flip's) is exactly at rest."""
        super().reset(seed=seed)
        self._random_start(exact=(options or {}).get("exact", False))
        self.roll_progress = 0.0
        self.flip_done = False
        mid = self.cfg.mid_flip
        prob = mid.prob if (self.training and mid is not None) else 0.0
        if self.task == "flip" and self.np_random.random() < prob:
            self._mid_flip_start()
        self.best_progress = np.clip(self.roll_progress, 0, 2 * np.pi)
        self.prev_action = np.zeros(4)
        self.t = 0
        self.z0 = self.quad.p[2]

        # Sensors: the estimate starts from the true attitude with a small error (on the real drone: Betaflight's
        # estimate when the flip is triggered)
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
        return self._obs(), {}

    def _random_start(self, exact):
        """Reset the quadrotor to a random state around the target, with the spread in configs/env.yaml (`start`).
        During training, velocity and body rates get at least min_std, so even a zero-noise start differs a little
        each episode; `exact` removes that minimum. A spread of 0 means rng.normal returns exactly its mean, 0."""
        rng, start = self.np_random, self.cfg.start
        min_std = 0.0 if exact else start.min_std
        # Random small tilt as an axis-angle -> quaternion
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
        """Flip training curriculum: start part-way through a roll, so the agent practises "finish the flip and
        recover" before it can do a whole flip. Already rolling at a random angle, with some climb speed to carry
        it through."""
        rng, mid = self.np_random, self.cfg.mid_flip
        phi = rng.uniform(0, mid.max_angle_frac * 2 * np.pi)
        self.quad.q = np.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])
        self.quad.w = np.array([mid.roll_rate, 0, 0]) + rng.normal(0, mid.rate_std, 3)
        self.quad.v = np.array([0, 0, mid.climb_speed]) + rng.normal(0, mid.climb_std, 3)
        self.roll_progress = phi

    def step(self, action):
        action = np.clip(action, -1, 1)
        thrust, rates = self.scale_action(action)
        for _ in range(self.substeps):
            self.quad.step(thrust, rates)
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

        if terminated:
            reward -= self.cfg.reward.crash
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {
            "dist": dist, "flip_done": self.flip_done, "roll_progress": self.roll_progress}

    def _terminated(self, dist):
        return bool(self.quad.p[2] < self.cfg.min_height or dist > self.cfg.max_dist)

    def _hover_reward(self, action, dist):
        q, r = self.quad, self.cfg.reward
        upright = q.R[2, 2]  # 1 = level, -1 = upside down
        # Positive "alive" reward that is largest at the target, so surviving is always worth it
        reward = r.alive - r.dist * dist
        reward += r.upright * upright
        reward -= r.body_rate * np.linalg.norm(q.w)
        reward -= r.speed * np.linalg.norm(q.v)
        reward -= r.smooth * np.sum((action - self.prev_action) ** 2)
        return reward, self._terminated(dist)

    def _flip_reward(self, action, dist):
        q, r = self.quad, self.cfg.reward
        upright = q.R[2, 2]
        reward = 0.0
        if not self.flip_done:
            # Reward new roll progress only, so rolling back and forth can't be farmed
            progress = np.clip(self.roll_progress, 0, 2 * np.pi)
            reward += r.roll_progress * max(0.0, progress - self.best_progress)
            self.best_progress = max(self.best_progress, progress)
            if self.roll_progress >= 2 * np.pi * r.done_frac:
                self.flip_done = True
                reward += r.completion_bonus
        else:
            # Recovery: be level, stop rotating, stop moving and hold the start height. Not "fly back to the start
            # point": the drone can't measure its position. Both pulls are positive and largest at the goal, and
            # still matter far from it, so recovering always beats crashing early.
            reward += r.upright * upright
            reward += r.still * np.exp(-np.linalg.norm(q.v) / r.still_scale)
            reward += r.height * np.exp(-abs(q.p[2] - self.z0) / r.height_scale)
            reward -= r.body_rate * np.linalg.norm(q.w)
            # Capped, so being fast right after the flip can't outweigh the positive terms (crashing would
            # then pay)
            reward -= r.speed * min(np.linalg.norm(q.v), r.speed_cap)
            # Thrust away from hover (action 0): free when hovering steadily, costly when bobbing on full-on /
            # full-off thrust
            reward -= r.thrust_effort * action[0] ** 2

        # Stay near the start point (climbing away is not free), keep actions smooth.
        # The first free_dist metres are free: a flip needs to climb first and ends up above the start.
        reward -= r.dist * max(0.0, dist - r.free_dist)
        reward -= r.smooth * np.sum((action - self.prev_action) ** 2)
        return reward, self._terminated(dist)
