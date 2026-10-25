"""Train the minimal TD-MPC2 (tdmpc2/agent.py) on QuadEnv.

Settings: configs/tdmpc2.yaml (+ configs/env.yaml as env.*); override any of them as key=value.
Watch a trained agent with animate_rollout.py; the learning curve is logged to results/ (plot_curves.py).
Run from the repo root:
    python -m tdmpc2.train_tdmpc2                               # train hover; saves models/tdmpc2_hover.pt
    python -m tdmpc2.train_tdmpc2 task=flip                     # train flip; saves models/tdmpc2_flip.pt
"""
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from configs import load_config
from envs.evaluation import EvalLog, evaluate as evaluate_policy
from envs.quad_env import QuadEnv
from tdmpc2.agent import TDMPC2


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


def train(cfg):
    task = cfg.task
    env = QuadEnv(cfg=cfg.env, training=True)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = TDMPC2(obs_dim, act_dim, cfg.agent)
    buf = Buffer(obs_dim, act_dim, cfg.steps, cfg.agent.horizon)
    seed_steps = max(cfg.min_seed_steps, cfg.seed_episodes * env.max_steps)
    print(f"{agent.n_params() / 1e6:.2f}M params, episode length {env.max_steps}, discount {cfg.agent.discount:.3f}, "
          f"{seed_steps} random seed steps")
    writer = SummaryWriter(f"runs/tdmpc2_{task}_{time.strftime('%m%d-%H%M%S')}")

    def save():
        # The config goes with the model, so evaluation rebuilds the env it was trained in
        agent.save(f"models/tdmpc2_{task}.pt", task=task, config=cfg.to_dict(), env_cfg=cfg.env.to_dict())

    torch.manual_seed(cfg.train_seed)
    np.random.seed(cfg.train_seed)
    env.action_space.seed(cfg.train_seed)
    # Learning curve: evaluation starts from rest, without the mid-roll curriculum
    eval_log, eval_env = EvalLog(f"tdmpc2_{task}"), QuadEnv(cfg=cfg.env)
    obs, _ = env.reset(seed=cfg.train_seed)
    t0, ep_ret, ep_len_now, ep, tic, info = True, 0.0, 0, 0, time.time(), {}
    step = 0
    try:
        for step in range(cfg.steps):
            if step < seed_steps:
                a = env.action_space.sample()
            else:
                a = agent.act(obs, t0=t0, plan=step >= cfg.plan_after)
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
                if ep % cfg.log_every_episodes == 0:
                    sps = step / (time.time() - tic)
                    print(f"step {step}  ep {ep}  return {ep_ret:.1f}  {sps:.1f} steps/s  "
                          + "  ".join(f"{k} {v:.3f}" for k, v in info.items()))
                obs, _ = env.reset()
                t0, ep_ret, ep_len_now = True, 0.0, 0
            if step >= seed_steps:
                # Pretrain on the seed data once, as in the paper, then one update per env step
                for _ in range(seed_steps if step == seed_steps else 1):
                    info = agent.update(*buf.sample(cfg.agent.batch_size))
                if step % cfg.log_every_steps == 0:
                    for k, v in info.items():
                        writer.add_scalar(f"train/{k}", v, step)
            if step > 0 and step % cfg.save_every == 0:
                save()
                print(f"checkpoint saved at step {step}")
            if step % cfg.eval_every == 0:
                act = lambda o, t: agent.act(o, t0=t == 0, eval_mode=True)  # with MPPI, the actual algorithm
                eval_log.write(step, time.time() - tic, evaluate_policy(eval_env, act, cfg.eval_seeds))
                t0 = True  # evaluation used the MPPI warm start; restart it for the training episode
    except KeyboardInterrupt:
        print(f"\ninterrupted at {step} steps, saving what we have")
    finally:
        writer.close()
    save()
    print(f"saved models/tdmpc2_{task}.pt")


if __name__ == "__main__":
    cfg = load_config("tdmpc2")
    torch.set_num_threads(cfg.threads)
    train(cfg)
