"""Gymnasium quadrotor environment with selectable task.

    task="flip":  do one roll flip, then recover to level, not rotating, slow, roughly back at the start point; then
                  the pilot takes over (on IMU alone the drone can't hold its position for long).
    task="hover": hold a fixed target point, starting from a perturbed state (a quick sanity check).

Action (4, in [-1, 1]): collective thrust, roll rate, pitch rate, yaw rate  -> scaled to CTBR.
    Thrust is centred on hover: a[0] = 0 -> thrust = g, so a zero-mean initial policy roughly hovers.
    Latency: each command reaches the rate controller a delay after the policy sends it, drawn once per episode
    from `latency` (the whole loop: reading the IMU, inference, sending the command, Betaflight). The observation
    records the action when it is sent, as the ESP32 would.

Observation = [actor part, privileged part] (asymmetric actor-critic, ppo/asymmetric.py):
    actor part (the first n_actor_obs entries): only what the real drone has (envs/sensors.py), so the policy can be
        deployed: estimated gravity direction in body frame (3), elapsed share of the episode (1; the ESP32 knows
        when it started the flip), then the last `history` frames, newest first, of [noisy gyro (3), noisy
        accelerometer (3), action taken (4)]; flip only: + estimated roll progress (1) + estimated flip done flag (1).
        No position, no velocity: the drone can't measure them.
    privileged part: the ground-truth state, read only by the critic (training in sim): rotation matrix (9), body
        rates (3), velocity (3), position error (3); flip only: + true roll progress (1) + true flip done flag (1);
        with domain randomisation and critic_params: + this episode's dynamics (PARAM_OBS).
    Body rates, velocity and position are divided by `obs_scale` (configs/env.yaml).
The reward is computed from the ground truth (training only).

Domain randomisation (`randomisation` in configs/env.yaml, null = off): each episode draws the true dynamics (thrust
at the hover command, thrust curve, inertia, motor lag, drag), while the rate controller, the mixer and the action
scaling keep assuming the nominal ones (envs/quadrotor.py, set_params). The actor has to infer them from its IMU and
action history. Applied in evaluation too (fixed seeds, so fixed draws).

Flip episodes end at the handover (episode_time): the pilot takes over, so it is a true end of the task (terminated,
not truncated: no value bootstrapped beyond it), with a bonus for being back at the start point. info["crashed"] tells
a crash from a handover.

All settings (dynamics, episode length, limits, start noise, curriculum, reward weights) come from
configs/env.yaml (configs.load_env).
"""
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from configs import load_env
from envs.quadrotor import G, Quadrotor
from envs.sensors import AttitudeEstimator, ImuModel, axis_angle_quat, quat_multiply

N_PARAM_OBS = 9


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
        scale = self.cfg.obs_scale
        self.w_scale, self.v_scale, self.pos_scale = scale.body_rate, scale.velocity, scale.position
        # Latency, in physics steps. Models saved before it existed have no `latency`: none
        latency = self.cfg.get("latency") or [0.0, 0.0]
        self.latency_range = [int(round(s / self.quad.dt)) for s in latency]
        # Domain randomisation: None (also for models saved before it existed) = nominal dynamics
        self.randomisation = self.cfg.get("randomisation")
        self.params = {}
        dr = self.randomisation
        self.critic_params = bool(dr and dr.critic_params)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        flip_extra = 2 if self.task == "flip" else 0  # roll progress + flip done flag
        self.n_actor_obs = 4 + 10 * self.history_len + flip_extra
        n_obs = self.n_actor_obs + 18 + flip_extra + (N_PARAM_OBS if self.critic_params else 0)
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
        actor = [est.gravity_body, [self.t / self.max_steps], *self.history]
        privileged = [q.R.flatten(), q.w / self.w_scale, q.v / self.v_scale,
                      (q.p - self.target) / self.pos_scale]
        if self.task == "flip":
            # Attitude alone can't tell "before flip" from "after flip" (both upright),
            # so the agent needs to know how far it has rotated.
            actor += [[est.roll_progress / (2 * np.pi)], [float(self.est_flip_done)]]
            privileged += [[self.roll_progress / (2 * np.pi)], [float(self.flip_done)]]
        if self.critic_params:
            privileged.append(self.param_obs)
        return np.concatenate(actor + privileged).astype(np.float32)

    def _push_frame(self, gyro, accel, action):
        """Newest IMU reading and the action taken, at the front of the history."""
        self.history.insert(0, np.concatenate([gyro / self.w_scale, accel / G, action]))
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

        # Latency: commands reach the motors `delay` physics steps after the policy sends them. Until then the
        # command from before the start is in effect: hover thrust at the current body rates (on the real drone:
        # the pilot's). Drawn last, so the other random draws don't depend on it; no draw for a fixed latency
        lo, hi = self.latency_range
        self.delay = lo if lo == hi else int(rng.integers(lo, hi + 1))
        held = (G, np.clip(self.quad.w, -self.quad.max_rate, self.quad.max_rate))
        self.commands = deque([held] * self.delay)
        if self.randomisation:  # drawn after everything else, so the starts are the same with and without it
            self._randomise_dynamics()
        return self._obs(), {}

    def _randomise_dynamics(self):
        """Draw this episode's true dynamics, uniformly in the ranges of `randomisation` (configs/env.yaml), and give
        them to the plant. The controller and the actor don't know them; the critic does (critic_params), scaled to
        [-1, 1] over their ranges."""
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
            # Handover to the pilot: the task ends here. Bonus for being back at the start point, only after a flip
            # (otherwise hovering in place would compete with flipping)
            if self.flip_done:
                reward += self.cfg.reward.handover * np.exp(-dist / self.cfg.reward.handover_scale)
            terminated, truncated = True, False
        return self._obs(), float(reward), terminated, truncated, {
            "dist": dist, "flip_done": self.flip_done, "roll_progress": self.roll_progress, "crashed": crashed}

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
            # Recovery: be level, stop rotating, stop moving, roughly back at the start point. The drone can't measure
            # its position, but over a short episode it can undo its own displacement from what it commanded and
            # felt (thrust history, drag). Both pulls are positive and largest at the goal, and still matter far
            # from it, so recovering always beats crashing early.
            reward += r.upright * upright
            reward += r.target * np.exp(-dist / r.target_scale)
            reward += r.still * np.exp(-np.linalg.norm(q.v) / r.still_scale)
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
