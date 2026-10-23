"""Train PPO on FlipEnv, or evaluate and plot a trained policy.

    python train_ppo.py                 # train
    python train_ppo.py --eval          # roll out saved policy and plot
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from flip_env import FlipEnv


def train(steps):
    env = make_vec_env(FlipEnv, n_envs=8)
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=2048, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.0, verbose=1,
                policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="runs")
    model.learn(total_timesteps=steps)
    model.save("ppo_flip")


def evaluate():
    env = FlipEnv()
    model = PPO.load("ppo_flip")
    obs, _ = env.reset(seed=0)
    log = {"t": [], "z": [], "upright": [], "roll_progress": [], "wx": [], "thrust": []}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        q = env.quad
        log["t"].append(env.t * env.ctrl_dt)
        log["z"].append(q.p[2])
        log["upright"].append(q.R[2, 2])
        log["roll_progress"].append(info["roll_progress"] / (2 * np.pi))
        log["wx"].append(q.w[0])
        log["thrust"].append(q.thrust_acc)
    print(f"flip done: {info['flip_done']}, final upright: {log['upright'][-1]:.2f}, final z: {log['z'][-1]:.2f}")

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(7, 9))
    axs[0].plot(log["t"], log["z"]); axs[0].set_ylabel("height [m]")
    axs[1].plot(log["t"], log["upright"]); axs[1].set_ylabel("R[2,2] (1=level)")
    axs[2].plot(log["t"], log["roll_progress"]); axs[2].set_ylabel("roll revs")
    axs[3].plot(log["t"], log["wx"], label="roll rate")
    axs[3].plot(log["t"], log["thrust"], label="thrust acc")
    axs[3].legend(); axs[3].set_xlabel("time [s]")
    plt.tight_layout(); plt.savefig("flip_eval.png"); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--steps", type=int, default=3_000_000)
    args = ap.parse_args()
    evaluate() if args.eval else train(args.steps)
