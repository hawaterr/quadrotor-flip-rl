"""Train PPO on HoverEnv, or evaluate and plot a trained policy.

    python train_ppo.py                 # train
    python train_ppo.py --eval          # roll out saved policy and plot
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from hover_env import HoverEnv


def train(steps):
    env = make_vec_env(HoverEnv, n_envs=8)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=2048, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.0, verbose=1,
                policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="runs")
    model.learn(total_timesteps=steps)
    model.save("ppo_hover")


def evaluate():
    env = HoverEnv()
    model = PPO.load("ppo_hover")
    obs, _ = env.reset(seed=0)
    log = {"t": [], "pos_err": [], "dist": [], "upright": [], "w": [], "thrust": []}
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
        log["thrust"].append(q.thrust_acc)
    print(f"crashed: {term}, final dist: {log['dist'][-1]:.3f} m, final upright: {log['upright'][-1]:.2f}")

    err = np.array(log["pos_err"])
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(7, 9))
    for i, name in enumerate("xyz"):
        axs[0].plot(log["t"], err[:, i], label=name)
    axs[0].legend(); axs[0].set_ylabel("pos error [m]")
    axs[1].plot(log["t"], log["upright"]); axs[1].set_ylabel("R[2,2] (1=level)")
    axs[2].plot(log["t"], log["w"]); axs[2].set_ylabel("|body rate| [rad/s]")
    axs[3].plot(log["t"], log["thrust"]); axs[3].set_ylabel("thrust acc [m/s²]")
    axs[3].set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig("hover_eval.png"); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--steps", type=int, default=1_000_000)
    args = ap.parse_args()
    evaluate() if args.eval else train(args.steps)
