"""Quick 3D visualisation of FlipEnv.

    python visualize.py                      # hand-coded flip (tests env + physics, no RL)
    python visualize.py --policy zero        # no action: should fall
    python visualize.py --policy hover       # constant hover thrust: should stay put
    python visualize.py --policy random
    python visualize.py --policy ppo         # trained model from train_ppo.py (ppo_flip.zip)
    python visualize.py --save flip.gif      # also save the animation
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from flip_env import FlipEnv
from quadrotor import G

ARM = 0.15  # visual arm length [m]


def make_policy(name, env):
    if name == "zero":
        return lambda obs, t: np.array([-1.0, 0, 0, 0])          # thrust off
    if name == "hover":
        return lambda obs, t: np.array([-0.5, 0, 0, 0])          # thrust = g (with max 4g)
    if name == "random":
        return lambda obs, t: env.action_space.sample()
    if name == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load("ppo_flip")
        return lambda obs, t: model.predict(obs, deterministic=True)[0]

    # "scripted": climb, roll until one revolution, then hover and stop rotating
    def scripted(obs, t):
        if t < 25:
            return np.array([1.0, 0, 0, 0])
        if not env.flip_done:
            return np.array([-0.6, 1.0, 0, 0])
        return np.array([-0.5, 0, 0, 0])
    return scripted


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    log = {"p": [], "R": [], "r": [], "revs": [], "done": [], "thrust": [], "mass": env.quad.m}
    total = 0.0
    for t in range(env.max_steps):
        obs, r, term, trunc, info = env.step(policy(obs, t))
        total += r
        log["p"].append(env.quad.p.copy())
        log["R"].append(env.quad.R.copy())
        log["r"].append(total)
        log["revs"].append(info["roll_progress"] / (2 * np.pi))
        log["done"].append(info["flip_done"])
        log["thrust"].append(env.quad.m * env.quad.thrust_acc)  # actual thrust force [N] (after motor lag)
        if term or trunc:
            break
    print(f"steps={t + 1}  total reward={total:.2f}  flip_done={info['flip_done']}  "
          f"final upright={env.quad.R[2, 2]:.2f}  final z={env.quad.p[2]:.2f}  crashed={term}")
    return log


def animate(log, ctrl_dt, save=None):
    P = np.array(log["p"])
    # Arm tips in body frame (X configuration) and the body "up" axis
    arms = ARM * np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) / np.sqrt(2)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
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
        text.set_text(f"t={i * ctrl_dt:.2f}s  revs={log['revs'][i]:.2f}  flip_done={log['done'][i]}\n"
                      f"thrust={F:.1f} N ({F / hover_force:.2f}x hover)  reward={log['r'][i]:.1f}")
        return arm1, arm2, front, trail, text

    frames = range(0, len(P), 2)  # every 2nd step -> ~real time at 20 ms/frame
    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=False)
    if save:
        anim.save(save, writer="pillow", fps=50)
        print(f"saved {save}")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="scripted", choices=["scripted", "zero", "hover", "random", "ppo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="e.g. flip.gif")
    args = ap.parse_args()

    env = FlipEnv()
    log = rollout(env, make_policy(args.policy, env), args.seed)
    animate(log, env.ctrl_dt, args.save)
