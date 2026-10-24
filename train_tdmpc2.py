"""Train the minimal TD-MPC2 (tdmpc2.py) on QuadEnv, or evaluate and plot a trained agent.

    python train_tdmpc2.py --task pendulum --steps 20000   # sanity check on Pendulum-v1 first
    python train_tdmpc2.py                                 # train hover; saves tdmpc2_hover.pt
    python train_tdmpc2.py --task flip                     # train flip; saves tdmpc2_flip.pt
    python train_tdmpc2.py --task flip --eval              # roll out saved agent (with MPPI) and plot
"""
import argparse
import time

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from quad_env import TASKS, QuadEnv
from tdmpc2 import TDMPC2, Config


def make_env(task, ctrl_dt, train=True):
    if task == "pendulum":
        return gym.wrappers.RescaleAction(gym.make("Pendulum-v1"), -1.0, 1.0)
    # Flip: half the training episodes start mid-roll, as for PPO
    return QuadEnv(task=task, ctrl_dt=ctrl_dt, mid_flip_prob=0.5 if (train and task == "flip") else 0.0)


class Buffer:
    """Flat transition storage; samples length-H sub-trajectories that stay inside one episode."""

    def __init__(self, obs_dim, act_dim, capacity, horizon):
        self.H = horizon
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros((capacity, act_dim), np.float32)
        self.reward = np.zeros((capacity, 1), np.float32)
        self.terminated = np.zeros((capacity, 1), np.float32)
        self.n, self.ep_start, self.starts = 0, 0, []

    def add(self, obs, action, reward, terminated, next_obs):
        i = self.n
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.terminated[i], self.next_obs[i] = terminated, next_obs
        self.n += 1

    def end_episode(self):
        self.starts.extend(range(self.ep_start, self.n - self.H + 1))
        self.ep_start = self.n

    def sample(self, batch_size):
        s = np.random.choice(self.starts, batch_size)[None] + np.arange(self.H)[:, None]  # (H, B)
        obs = np.concatenate([self.obs[s[:1]], self.next_obs[s]])  # (H+1, B, O)
        t = torch.as_tensor
        return t(obs), t(self.action[s]), t(self.reward[s]), t(self.terminated[s])


def train(task, steps, ctrl_dt, plan_after, cfg):
    env = make_env(task, ctrl_dt)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = TDMPC2(obs_dim, act_dim, cfg)
    buf = Buffer(obs_dim, act_dim, steps, cfg.horizon)
    ep_len = env.spec.max_episode_steps if task == "pendulum" else env.max_steps
    seed_steps = max(1000, 5 * ep_len)
    print(f"{agent.n_params() / 1e6:.2f}M params, episode length {ep_len}, discount {cfg.discount:.3f}, "
          f"{seed_steps} random seed steps")
    writer = SummaryWriter(f"runs/tdmpc2_{task}_{time.strftime('%m%d-%H%M%S')}")

    obs, _ = env.reset(seed=0)
    t0, ep_ret, ep_len_now, ep, tic, info = True, 0.0, 0, 0, time.time(), {}
    step = 0
    try:
        for step in range(steps):
            if step < seed_steps:
                a = env.action_space.sample()
            else:
                a = agent.act(obs, t0=t0, plan=step >= plan_after)
            next_obs, r, term, trunc, env_info = env.step(a)
            buf.add(obs, a, r, term, next_obs)
            obs, t0, ep_ret, ep_len_now = next_obs, False, ep_ret + r, ep_len_now + 1
            if term or trunc:
                buf.end_episode()
                writer.add_scalar("episode/return", ep_ret, step)
                writer.add_scalar("episode/length", ep_len_now, step)
                if task == "flip":
                    writer.add_scalar("episode/flip_done", float(env_info["flip_done"]), step)
                    writer.add_scalar("episode/crashed", float(term), step)
                ep += 1
                if ep % 10 == 0:
                    sps = step / (time.time() - tic)
                    print(f"step {step}  ep {ep}  return {ep_ret:.1f}  {sps:.1f} steps/s  "
                          + "  ".join(f"{k} {v:.3f}" for k, v in info.items()))
                obs, _ = env.reset()
                t0, ep_ret, ep_len_now = True, 0.0, 0
            if step >= seed_steps:
                # Pretrain on the seed data once, as in the paper, then one update per env step
                for _ in range(seed_steps if step == seed_steps else 1):
                    info = agent.update(*buf.sample(cfg.batch_size))
                if step % 1000 == 0:
                    for k, v in info.items():
                        writer.add_scalar(f"train/{k}", v, step)
    except KeyboardInterrupt:
        print(f"\ninterrupted at {step} steps, saving what we have")
    finally:
        writer.close()
    agent.save(f"tdmpc2_{task}.pt", task=task, ctrl_dt=ctrl_dt)
    print(f"saved tdmpc2_{task}.pt")


def evaluate(task, plan=True):
    agent, ckpt = TDMPC2.load(f"tdmpc2_{task}.pt")
    if task == "pendulum":
        env = make_env(task, None, train=False)
        obs, _ = env.reset(seed=0)
        total, done, t = 0.0, False, 0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.act(obs, t0=t == 0, eval_mode=True, plan=plan))
            total, done, t = total + r, term or trunc, t + 1
        print(f"pendulum return: {total:.1f}")
        return
    from eval_plot import evaluate_and_plot
    evaluate_and_plot(make_env(task, ckpt["ctrl_dt"], train=False),
                      lambda obs, t: agent.act(obs, t0=t == 0, eval_mode=True, plan=plan),
                      f"{task}_tdmpc2_eval.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hover", choices=TASKS + ("pendulum",))
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--no-plan", action="store_true", help="eval: act with the policy prior instead of MPPI")
    ap.add_argument("--steps", type=int, default=200_000)
    # A flip takes ~0.6 s; a 3-step plan at 0.01 s only sees 30 ms, so act at 0.02 s by default
    ap.add_argument("--ctrl-dt", type=float, default=0.02)
    ap.add_argument("--plan-after", type=int, default=0,
                    help="act with the policy prior (fast) until this step, then with MPPI")
    ap.add_argument("--samples", type=int, default=512, help="MPPI samples (fewer = faster)")
    ap.add_argument("--iterations", type=int, default=6, help="MPPI iterations (fewer = faster)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.eval:
        evaluate(args.task, plan=not args.no_plan)
    else:
        # Same effective horizon as PPO's gamma=0.99 at 0.01 s steps (~1 s)
        discount = 0.99 if args.task == "pendulum" else 0.99 ** (args.ctrl_dt / 0.01)
        cfg = Config(discount=discount, num_samples=args.samples, iterations=args.iterations)
        train(args.task, args.steps, args.ctrl_dt, args.plan_after, cfg)
