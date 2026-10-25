"""Roll out one QuadEnv episode with any policy and plot it (shared by train_ppo.py and train_tdmpc2.py)."""
import os

import matplotlib.pyplot as plt
import numpy as np


def evaluate_and_plot(env, act, out_png, seed=0):
    """act(obs, t) -> action. Saves the plot to out_png."""
    obs, _ = env.reset(seed=seed, options={"exact": True})
    log = {"t": [], "pos_err": [], "dist": [], "upright": [], "w": [], "revs": [], "thrust": []}
    done, action_change = False, []
    while not done:
        action = np.clip(act(obs, env.t), -1, 1)
        action_change.append(np.abs(action - env.prev_action).sum())
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        q = env.quad
        log["t"].append(env.t * env.policy_dt)
        log["pos_err"].append(q.p - env.target)
        log["dist"].append(info["dist"])
        log["upright"].append(q.R[2, 2])
        log["w"].append(np.linalg.norm(q.w))
        log["revs"].append(info["roll_progress"] / (2 * np.pi))
        log["thrust"].append(q.thrust_acc)
    task = env.task
    if task == "flip":
        print(f"flip done: {info['flip_done']}")
    print(f"crashed: {term}, final dist: {log['dist'][-1]:.3f} m, final upright: {log['upright'][-1]:.2f}, "
          f"mean |a - prev a|: {np.mean(action_change):.2f}")

    err = np.array(log["pos_err"])
    n = 5 if task == "flip" else 4
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(7, 2.25 * n))
    for i, name in enumerate("xyz"):
        axs[0].plot(log["t"], err[:, i], label=name)
    axs[0].legend(); axs[0].set_ylabel("pos error [m]")
    axs[1].plot(log["t"], log["upright"]); axs[1].set_ylabel("R[2,2] (1=level)")
    axs[2].plot(log["t"], log["w"]); axs[2].set_ylabel("|body rate| [rad/s]")
    axs[3].plot(log["t"], log["thrust"]); axs[3].set_ylabel("thrust acc [m/s²]")
    if task == "flip":
        axs[4].plot(log["t"], log["revs"]); axs[4].set_ylabel("roll revs")
    axs[-1].set_xlabel("time [s]")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png); plt.show()
