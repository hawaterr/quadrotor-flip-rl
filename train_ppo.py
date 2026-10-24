"""Train PPO on QuadEnv, or evaluate and plot a trained policy.

    python train_ppo.py                         # train hover; saves ppo_hover.zip
    python train_ppo.py --task flip             # train flip; saves ppo_flip.zip
    python train_ppo.py --task flip --eval      # roll out saved policy and plot
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from quad_env import TASKS, QuadEnv


def train(task, steps):
    flip = task == "flip"
    # Flip: half the episodes start mid-roll, and a little entropy keeps it exploring rolls
    env = make_vec_env(QuadEnv, n_envs=8, env_kwargs=dict(task=task, mid_flip_prob=0.5 if flip else 0.0))
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=2048, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.005 if flip else 0.0, verbose=1,
                policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="runs")
    try:
        model.learn(total_timesteps=steps)
    except KeyboardInterrupt:
        print(f"\ninterrupted at {model.num_timesteps} steps, saving what we have")
    finally:
        env.close()
    model.save(f"ppo_{task}")
    print(f"saved ppo_{task}.zip")


def evaluate(task):
    env = QuadEnv(task=task)
    model = PPO.load(f"ppo_{task}")
    obs, _ = env.reset(seed=0)
    log = {"t": [], "pos_err": [], "dist": [], "upright": [], "w": [], "revs": [], "thrust": []}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        q = env.quad
        log["t"].append(env.t * env.ctrl_dt)
        log["pos_err"].append(q.p - env.target)
        log["dist"].append(info["dist"])
        log["upright"].append(q.R[2, 2])
        log["w"].append(np.linalg.norm(q.w))
        log["revs"].append(info["roll_progress"] / (2 * np.pi))
        log["thrust"].append(q.thrust_acc)
    if task == "flip":
        print(f"flip done: {info['flip_done']}")
    print(f"crashed: {term}, final dist: {log['dist'][-1]:.3f} m, final upright: {log['upright'][-1]:.2f}")

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
    plt.tight_layout(); plt.savefig(f"{task}_eval.png"); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hover", choices=TASKS)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--steps", type=int, default=1_000_000)
    args = ap.parse_args()
    evaluate(args.task) if args.eval else train(args.task, args.steps)
