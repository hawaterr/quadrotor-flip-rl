"""Quick 3D visualisation of QuadEnv.

The policy name also picks the task: <name>_flip runs the flip env, everything else the hover env.

    python visualize.py                           # hand-coded hover controller (tests env + physics, no RL)
    python visualize.py --policy scripted_flip    # hand-coded flip
    python visualize.py --policy zero             # thrust off: should fall
    python visualize.py --policy hover            # constant hover thrust, no feedback: drifts from the perturbed start
    python visualize.py --policy random
    python visualize.py --policy ppo_hover        # trained model from train_ppo.py (ppo_hover.zip)
    python visualize.py --policy ppo_flip         # trained model from train_ppo.py --task flip (ppo_flip.zip)
    python visualize.py --policy tdmpc2_flip      # trained agent from train_tdmpc2.py --task flip (tdmpc2_flip.pt), with MPPI
    python visualize.py --save hover.gif          # also save the animation
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from quad_env import TASKS, QuadEnv
from quadrotor import G

ARM = 0.15  # visual arm length [m]
POLICIES = ["zero", "hover", "random"] + [f"{kind}_{task}" for kind in ("scripted", "ppo", "tdmpc2") for task in TASKS]


def task_of(policy):
    """scripted_flip / ppo_flip / tdmpc2_flip -> "flip"; open-loop baselines run in the hover env."""
    suffix = policy.rsplit("_", 1)[-1]
    return suffix if suffix in TASKS else "hover"


def make_policy(name, env):
    if name == "zero":
        return lambda obs, t: np.array([-1.0, 0, 0, 0])          # thrust off
    if name == "hover":
        return lambda obs, t: np.array([0.0, 0, 0, 0])           # thrust = g
    if name == "random":
        return lambda obs, t: env.action_space.sample()
    if name.startswith("ppo_"):
        from stable_baselines3 import PPO
        model = PPO.load(name)  # ppo_<task>.zip, as saved by train_ppo.py
        return lambda obs, t: model.predict(obs, deterministic=True)[0]
    if name.startswith("tdmpc2_"):
        from tdmpc2 import TDMPC2
        agent, _ = TDMPC2.load(f"{name}.pt")  # tdmpc2_<task>.pt, as saved by train_tdmpc2.py
        return lambda obs, t: agent.act(obs, t0=t == 0, eval_mode=True)

    # scripted_hover: cascaded PD controller. Desired acceleration from position/velocity error sets the
    # desired thrust direction; body rates rotate body z towards it.
    def pd(obs, t):
        q = env.quad
        a_des = -4.0 * (q.p - env.target) - 3.0 * q.v + np.array([0, 0, G])
        z_des = a_des / np.linalg.norm(a_des)
        rates = 8.0 * np.cross([0, 0, 1], q.R.T @ z_des)
        thrust = a_des @ q.R[:, 2]
        a0 = (thrust - G) / (q.max_thrust_acc - G) if thrust >= G else thrust / G - 1.0
        return np.clip(np.array([a0, *(rates / q.max_rate)]), -1, 1)
    if name == "scripted_hover":
        return pd

    # scripted_flip: climb, roll until one revolution, then recover with the PD controller
    def flip(obs, t):
        if t < 25:
            return np.array([1.0, 0, 0, 0])
        if not env.flip_done:
            return np.array([-0.2, 1.0, 0, 0])
        return pd(obs, t)
    return flip


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    log = {"p": [], "R": [], "r": [], "dist": [], "revs": [], "done": [], "thrust": [],
           "thrust_cmd": [], "rate_cmd": [], "w": [], "mass": env.quad.m, "task": env.task}
    total = 0.0
    for t in range(env.max_steps):
        action = np.clip(policy(obs, t), -1, 1)
        obs, r, term, trunc, info = env.step(action)
        thrust_cmd, rate_cmd = env._scale_action(action)
        total += r
        log["p"].append(env.quad.p.copy())
        log["R"].append(env.quad.R.copy())
        log["r"].append(total)
        log["dist"].append(info["dist"])
        log["revs"].append(info["roll_progress"] / (2 * np.pi))
        log["done"].append(info["flip_done"])
        log["thrust"].append(env.quad.m * env.quad.thrust_acc)  # actual thrust force [N] (after motor lag)
        log["thrust_cmd"].append(env.quad.m * thrust_cmd)
        log["rate_cmd"].append(rate_cmd)
        log["w"].append(env.quad.w.copy())
        if term or trunc:
            break
    flip = f"  flip_done={info['flip_done']}" if env.task == "flip" else ""
    print(f"steps={t + 1}  total reward={total:.2f}  final dist={info['dist']:.3f}{flip}  "
          f"final upright={env.quad.R[2, 2]:.2f}  final z={env.quad.p[2]:.2f}  crashed={term}")
    return log


def animate(log, ctrl_dt, save=None):
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
    arrow = [None]                                    # quiver can't be updated in place, so we redraw it

    hover_force = log["mass"] * G
    ARROW_PER_HOVER = 0.3                             # arrow length [m] at hover thrust (m*g)

    # Under the 3D view, drawn up to the current time: thrust across the top, then one row per axis
    # with the body rate (dashed = commanded, solid = actual) on the left and the attitude angle on the right
    T = np.arange(1, len(P) + 1) * ctrl_dt
    series = {"thrust_cmd": np.array(log["thrust_cmd"]), "thrust": np.array(log["thrust"]),
              "rate_cmd": np.array(log["rate_cmd"]), "w": np.array(log["w"])}
    # ZYX Euler angles, unwrapped so a flip reads as a smooth 0 -> 360 deg instead of jumping at +-180
    Rs = np.array(log["R"])
    euler = np.stack([np.arctan2(Rs[:, 2, 1], Rs[:, 2, 2]),
                      -np.arcsin(np.clip(Rs[:, 2, 0], -1, 1)),
                      np.arctan2(Rs[:, 1, 0], Rs[:, 0, 0])], 1)
    series["angle"] = np.degrees(np.unwrap(euler, axis=0))
    # (grid cell, y label, colour, [(series, column, style, legend label)])
    panels = [(gs[1, :], "thrust [N]", "b", [("thrust_cmd", None, "--", "cmd"), ("thrust", None, "-", "actual")])]
    for k, (name, col) in enumerate(zip(("roll", "pitch", "yaw"), ("C3", "C2", "C0"))):
        panels += [(gs[k + 2, 0], f"{name} rate [rad/s]", col, [("rate_cmd", k, "--", "cmd"), ("w", k, "-", "actual")]),
                   (gs[k + 2, 1], f"{name} [deg]", col, [("angle", k, "-", None)])]
    lines = []  # (line, series name, column or None)
    for cell, ylabel, col, curves in panels:
        a = fig.add_subplot(cell)
        ys = []
        for key, c, style, label in curves:
            lines.append((a.plot([], [], style, color=col, lw=1 if style == "--" else 1.5, label=label)[0], key, c))
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
        if arrow[0] is not None:
            arrow[0].remove()
        arrow[0] = ax.quiver(*p, *d, color="b", lw=2, arrow_length_ratio=0.25)

        trail.set_data_3d(*P[: i + 1].T)
        flip = f"  revs={log['revs'][i]:.2f}  flip_done={log['done'][i]}" if log["task"] == "flip" else ""
        text.set_text(f"t={i * ctrl_dt:.2f}s  dist to target={log['dist'][i]:.2f} m{flip}\n"
                      f"thrust={F:.1f} N ({F / hover_force:.2f}x hover)  reward={log['r'][i]:.1f}")
        for line, key, col in lines:
            y = series[key][: i + 1]
            line.set_data(T[: i + 1], y if col is None else y[:, col])
        return arm1, arm2, front, trail, text

    frames = range(0, len(P), max(1, round(0.02 / ctrl_dt)))  # ~real time at 20 ms/frame
    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=False)
    if save:
        anim.save(save, writer="pillow", fps=50)
        print(f"saved {save}")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="scripted_hover", choices=POLICIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="e.g. hover.gif")
    args = ap.parse_args()

    ctrl_dt = 0.01
    if args.policy.startswith("tdmpc2_"):
        import torch
        ctrl_dt = torch.load(f"{args.policy}.pt", weights_only=False)["ctrl_dt"]  # it was trained at this step
    env = QuadEnv(task=task_of(args.policy), ctrl_dt=ctrl_dt)
    log = rollout(env, make_policy(args.policy, env), args.seed)
    animate(log, env.ctrl_dt, args.save)
