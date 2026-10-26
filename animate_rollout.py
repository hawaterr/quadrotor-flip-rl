"""Run one episode of any policy on QuadEnv and animate it: 3D view plus command and attitude plots.

The policy name also picks the task: <name>_flip runs the flip env, everything else the hover env.
Settings: configs/animate_rollout.yaml; override any of them as key=value.

    python animate_rollout.py                           # hand-coded hover controller (tests env + physics, no RL)
    python animate_rollout.py policy=scripted_flip      # hand-coded flip
    python animate_rollout.py policy=zero               # thrust off: should fall
    python animate_rollout.py policy=hover              # constant hover thrust, no feedback: drifts from the perturbed start
    python animate_rollout.py policy=random
    python animate_rollout.py policy=ppo_hover          # trained model from ppo/train_ppo.py (models/ppo_hover.zip)
    python animate_rollout.py policy=ppo_flip           # trained model from ppo/train_ppo.py task=flip (models/ppo_flip.zip)
    python animate_rollout.py policy=tdmpc2_flip        # trained agent from tdmpc2/train_tdmpc2.py task=flip, with MPPI
    python animate_rollout.py policy=tdmpc2_flip plan=false    # same agent, policy prior only (no MPPI)
    python animate_rollout.py policy=ppo_flip_v1        # any model file in models/ named ppo_* or tdmpc2_*
    python animate_rollout.py save=plots/hover.gif      # also save the animation
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from configs import env_from_checkpoint, load_config, load_env
from envs.evaluation import SETTLE_WINDOW
from envs.quad_env import TASKS, QuadEnv
from envs.quadrotor import G

ARM = 0.15  # visual arm length [m]
FRAME_DT = 0.02  # s per animation frame (50 fps); frames are picked so playback is about real time
BASELINES = ["zero", "hover", "random"] + [f"scripted_{task}" for task in TASKS]
MODEL_FILES = {"ppo_": ".zip", "tdmpc2_": ".pt"}  # trained models in models/: name prefix -> file extension


def trained_models():
    """Names of the trained models in models/, e.g. ppo_flip, ppo_flip_v1, tdmpc2_flip_prev."""
    if not os.path.isdir("models"):
        return []
    names = []
    for f in sorted(os.listdir("models")):
        stem, ext = os.path.splitext(f)
        if any(stem.startswith(p) and ext == e for p, e in MODEL_FILES.items()) and not stem.endswith(".tmp"):
            names.append(stem)
    return names


def task_of(policy):
    """The first task named in the policy (ppo_flip_v1 -> flip); open-loop baselines run in the hover env."""
    return next((part for part in policy.split("_") if part in TASKS), "hover")


def env_config_for(policy):
    """Trained models run in the env they were trained in (saved with the model); baselines use env.yaml."""
    task = task_of(policy)
    if policy.startswith("tdmpc2_"):
        import torch
        ckpt = torch.load(f"models/{policy}.pt", weights_only=False)
        return env_from_checkpoint(ckpt, ckpt.get("task", task))
    if policy.startswith("ppo_"):
        from ppo.train_ppo import saved_env_config
        return saved_env_config(task, policy)
    return load_env(task)


def make_policy(name, env, plan=True, scripted=None):
    """scripted: the `scripted` section of configs/animate_rollout.yaml (gains of the hand-written controllers)."""
    if name == "zero":
        return lambda obs, t: np.array([-1.0, 0, 0, 0])          # thrust off
    if name == "hover":
        return lambda obs, t: np.array([0.0, 0, 0, 0])           # thrust = g
    if name == "random":
        return lambda obs, t: env.action_space.sample()
    if name.startswith("ppo_"):
        from stable_baselines3 import PPO
        model = PPO.load(f"models/{name}")  # models/ppo_<task>.zip, as saved by ppo/train_ppo.py
        return lambda obs, t: model.predict(obs, deterministic=True)[0]
    if name.startswith("tdmpc2_"):
        from tdmpc2.agent import TDMPC2
        agent, _ = TDMPC2.load(f"models/{name}.pt")  # models/tdmpc2_<task>.pt, as saved by tdmpc2/train_tdmpc2.py
        return lambda obs, t: agent.act(obs, t0=t == 0, eval_mode=True, plan=plan)

    # scripted_hover: cascaded PD controller. Desired acceleration from position/velocity error sets the
    # desired thrust direction; body rates rotate body z towards it.
    def pd(obs, t):
        q = env.quad
        a_des = -scripted.pd_kp * (q.p - env.target) - scripted.pd_kd * q.v + np.array([0, 0, G])
        z_des = a_des / np.linalg.norm(a_des)
        rates = scripted.att_gain * np.cross([0, 0, 1], q.R.T @ z_des)
        thrust = a_des @ q.R[:, 2]
        a0 = (thrust - G) / (q.max_thrust_acc - G) if thrust >= G else thrust / G - 1.0
        return np.clip(np.array([a0, *(rates / q.max_rate)]), -1, 1)
    if name == "scripted_hover":
        return pd

    # scripted_flip: climb, roll until one revolution, then recover with the PD controller
    climb_steps = round(scripted.flip_climb_time / env.policy_dt)

    def flip(obs, t):
        if t < climb_steps:
            return np.array([1.0, 0, 0, 0])
        if not env.flip_done:
            return np.array([scripted.flip_thrust, 1.0, 0, 0])
        return pd(obs, t)
    return flip


def rollout(env, policy, seed):
    """Log the state at t = 0 (after reset) and after every step. Commands are logged per step and held over
    the step that follows, so index i of every series is time i * policy_dt."""
    obs, _ = env.reset(seed=seed, options={"exact": True})
    log = {"p": [], "R": [], "r": [], "dist": [], "revs": [], "done": [], "thrust": [],
           "thrust_cmd": [], "rate_cmd": [], "w": [], "speed": [], "mass": env.quad.m, "task": env.task}
    total = 0.0

    def record():
        q = env.quad
        log["p"].append(q.p.copy())
        log["R"].append(q.R.copy())
        log["r"].append(total)
        log["dist"].append(np.linalg.norm(q.p - env.target))
        log["revs"].append(env.roll_progress / (2 * np.pi))
        log["done"].append(env.flip_done)
        log["thrust"].append(q.m * q.thrust_acc)  # actual thrust force [N] (after motor lag)
        log["w"].append(q.w.copy())
        log["speed"].append(np.linalg.norm(q.v))

    record()
    action_change = []  # sum over the 4 commands of |a - previous a|; large = chattering
    infer_ms = []  # wall-clock time of each policy call (one action from one observation)
    for t in range(env.max_steps):
        start = time.perf_counter()
        action = policy(obs, t)
        infer_ms.append(1000 * (time.perf_counter() - start))
        action = np.clip(action, -1, 1)
        action_change.append(np.abs(action - env.prev_action).sum())
        thrust_cmd, rate_cmd = env._scale_action(action)
        log["thrust_cmd"].append(env.quad.m * thrust_cmd)
        log["rate_cmd"].append(rate_cmd)
        obs, r, term, trunc, info = env.step(action)
        total += r
        record()
        if term or trunc:
            break
    # Repeat the last command so commands line up with the states (it is held until the end)
    log["thrust_cmd"].append(log["thrust_cmd"][-1])
    log["rate_cmd"].append(log["rate_cmd"][-1])
    flip = f"  flip_done={info['flip_done']}" if env.task == "flip" else ""
    print(f"steps={t + 1}  total reward={total:.2f}  final dist={info['dist']:.3f}{flip}  "
          f"final upright={env.quad.R[2, 2]:.2f}  final z={env.quad.p[2]:.2f}  crashed={term}  "
          f"mean |a - prev a|={np.mean(action_change):.2f}")
    tail = max(1, round(SETTLE_WINDOW / env.policy_dt))
    print(f"last {SETTLE_WINDOW} s: mean dist={np.mean(log['dist'][-tail:]):.3f}  "
          f"mean speed={np.mean(log['speed'][-tail:]):.2f}  final speed={log['speed'][-1]:.2f}")
    # The first call is excluded: it includes one-off warm-up (lazy init, memory allocation)
    steady = np.array(infer_ms[1:] or infer_ms)
    print(f"inference time per step: median={np.median(steady):.2f} ms  p99={np.percentile(steady, 99):.2f} ms  "
          f"first call={infer_ms[0]:.1f} ms  (control period {1000 * env.policy_dt:.0f} ms)")
    return log


def animate(log, policy_dt, save=None, start_pause=0.0):
    """start_pause: seconds to hold the first frame (t = 0) before playing."""
    P = np.array(log["p"])
    # Arm tips in body frame (X configuration) and the body "up" axis
    arms = ARM * np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) / np.sqrt(2)

    fig = plt.figure(figsize=(9, 11))
    gs = fig.add_gridspec(5, 2, height_ratios=[4, 1, 1, 1, 1], hspace=0.25, wspace=0.3)
    ax = fig.add_subplot(gs[0, :], projection="3d")
    c, half = P.mean(0), max(np.ptp(P, 0).max() / 2 + 0.5, 1.0)
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(max(0, c[2] - half), c[2] + half)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    arm1, = ax.plot([], [], [], "k-", lw=3)
    arm2, = ax.plot([], [], [], "k-", lw=3)
    front, = ax.plot([], [], [], "ro", ms=3)          # marks one arm so you can see rotation
    trail, = ax.plot([], [], [], "-", color="gray", alpha=0.5)
    text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    arrow, = ax.plot([], [], [], "b-o", lw=2, ms=3, markevery=[1])  # thrust arrow (a line, so it can be blitted)

    hover_force = log["mass"] * G
    ARROW_PER_HOVER = 0.3                             # arrow length [m] at hover thrust (m*g)

    # Under the 3D view, drawn up to the current time. Left column: commands (dashed) vs actual (solid) for
    # thrust and the body rates. Right column: height and the attitude angles. All share one time axis.
    T = np.arange(len(P)) * policy_dt
    series = {"thrust_cmd": np.array(log["thrust_cmd"]), "thrust": np.array(log["thrust"]),
              "rate_cmd": np.array(log["rate_cmd"]), "w": np.array(log["w"])}
    # ZYX Euler angles, unwrapped so a flip reads as a smooth 0 -> 360 deg instead of jumping at +-180
    Rs = np.array(log["R"])
    euler = np.stack([np.arctan2(Rs[:, 2, 1], Rs[:, 2, 2]),
                      -np.arcsin(np.clip(Rs[:, 2, 0], -1, 1)),
                      np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0])], 1)
    series["angle"] = np.degrees(np.unwrap(euler, axis=0))
    # (grid cell, y label, colour, [(series, column, style, legend label)])
    series["z"] = P[:, 2]
    panels = [(gs[1, 0], "thrust [N]", "b", [("thrust_cmd", None, "--", "cmd"), ("thrust", None, "-", "actual")]),
              (gs[1, 1], "height z [m]", "k", [("z", None, "-", None)])]
    for k, (name, col) in enumerate(zip(("roll", "pitch", "yaw"), ("C3", "C2", "C0"))):
        panels += [(gs[k + 2, 0], f"{name} rate [rad/s]", col, [("rate_cmd", k, "--", "cmd"), ("w", k, "-", "actual")]),
                   (gs[k + 2, 1], f"{name} [deg]", col, [("angle", k, "-", None)])]
    lines = []  # (line, series name, column or None)
    for cell, ylabel, col, curves in panels:
        a = fig.add_subplot(cell)
        ys = []
        for key, c, style, label in curves:
            # Commands are held over each step, so draw them as steps
            draw = "steps-post" if style == "--" else "default"
            lines.append((a.plot([], [], style, color=col, lw=1 if style == "--" else 1.5, label=label,
                                 drawstyle=draw)[0], key, c))
            ys.append(series[key] if c is None else series[key][:, c])
        lo, hi = min(y.min() for y in ys), max(y.max() for y in ys)
        pad = 0.1 * (hi - lo) + 1e-3
        a.set_xlim(0, T[-1]); a.set_ylim(lo - pad, hi + pad); a.set_ylabel(ylabel, fontsize=8); a.grid(alpha=0.3)
        a.tick_params(labelsize=7)
        if "yaw" in ylabel:
            a.set_xlabel("time [s]")
        else:
            a.tick_params(labelbottom=False)
        if "thrust" in ylabel:
            a.axhline(hover_force, color="gray", lw=0.5)
            a.legend(loc="upper right", fontsize=7, ncol=2)

    def update(i):
        p, R = P[i], log["R"][i]
        tips = p + arms @ R.T
        arm1.set_data_3d(*tips[[0, 1]].T)
        arm2.set_data_3d(*tips[[2, 3]].T)
        front.set_data_3d(*tips[[0]].T)

        # Thrust arrow: along body z, length proportional to thrust force
        F = log["thrust"][i]
        d = R @ [0, 0, ARROW_PER_HOVER * F / hover_force]
        arrow.set_data_3d(*np.array([p, p + d]).T)

        trail.set_data_3d(*P[: i + 1].T)
        flip = f"  revs={log['revs'][i]:.2f}  flip_done={log['done'][i]}" if log["task"] == "flip" else ""
        text.set_text(f"t={i * policy_dt:.2f}s  dist to target={log['dist'][i]:.2f} m{flip}\n"
                      f"thrust={F:.1f} N ({F / hover_force:.2f}x hover)  reward={log['r'][i]:.1f}")
        for line, key, col in lines:
            y = series[key][: i + 1]
            line.set_data(T[: i + 1], y if col is None else y[:, col])
        # Everything that changes, so blitting redraws only these instead of the whole figure
        return arm1, arm2, front, trail, text, arrow, *(line for line, _, _ in lines)

    hold = [0] * round(start_pause / FRAME_DT)  # repeat t = 0 so the starting position is easy to see
    frames = hold + list(range(0, len(P), max(1, round(FRAME_DT / policy_dt))))
    anim = FuncAnimation(fig, update, frames=frames, interval=FRAME_DT * 1000, blit=True)
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        anim.save(save, writer="pillow", fps=round(1 / FRAME_DT))
        print(f"saved {save}")
    plt.show()


if __name__ == "__main__":
    cfg = load_config("animate_rollout")
    if cfg.policy not in BASELINES + trained_models():
        raise ValueError(f"policy must be a baseline {BASELINES} or a model in models/ {trained_models()}, "
                         f"got {cfg.policy!r}")
    env = QuadEnv(cfg=env_config_for(cfg.policy))
    log = rollout(env, make_policy(cfg.policy, env, plan=cfg.plan, scripted=cfg.scripted), cfg.seed)
    animate(log, env.policy_dt, cfg.save, cfg.start_pause)
