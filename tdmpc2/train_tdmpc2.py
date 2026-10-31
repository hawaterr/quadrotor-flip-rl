"""Train the minimal TD-MPC2 (tdmpc2/agent.py) on QuadEnv.

Settings: configs/tdmpc2.yaml (+ configs/env.yaml as env.*); override any of them as key=value.
Watch a trained agent with animate_rollout.py; the learning curve is logged to results/ (plot_curves.py).
Run from the repo root:
    python -m tdmpc2.train_tdmpc2                               # train hover; saves models/tdmpc2_hover.pt
    python -m tdmpc2.train_tdmpc2 task=flip                     # train flip; saves models/tdmpc2_flip.pt
    python -m tdmpc2.train_tdmpc2 task=flip name=test           # an experiment; saves models/test.pt
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
        self.n, self.ep_start, self.starts = 0, 0, np.zeros(0, np.int64)

    def add(self, obs, action, reward, terminated, next_obs):
        i = self.n
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.terminated[i], self.next_obs[i] = terminated, next_obs
        self.n += 1

    def end_episode(self):
        # An array, not a list: np.random.choice would convert a list of up to `capacity` starts on every sample
        self.starts = np.concatenate([self.starts, np.arange(self.ep_start, self.n - self.H + 1)])
        self.ep_start = self.n

    def sample(self, batch_size):
        s = np.random.choice(self.starts, batch_size)[None] + np.arange(self.H)[:, None]  # (H, B)
        obs = np.concatenate([self.obs[s[:1]], self.next_obs[s]])  # (H+1, B, O)
        t = torch.as_tensor
        return t(obs), t(self.action[s]), t(self.reward[s]), t(self.terminated[s])


def _build(cfg):
    """The training env, the agent and its replay buffer, seeded with cfg.train_seed. Also returns the number of
    random-action seed steps before learning starts."""
    env = QuadEnv(cfg=cfg.env, training=True)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    # Seeded before the agent is built, so train_seed also fixes the network initialisation
    torch.manual_seed(cfg.train_seed)
    np.random.seed(cfg.train_seed)
    env.action_space.seed(cfg.train_seed)
    agent = TDMPC2(obs_dim, act_dim, cfg.agent)
    buf = Buffer(obs_dim, act_dim, cfg.steps, cfg.agent.horizon)
    seed_steps = max(cfg.min_seed_steps, cfg.seed_episodes * env.max_steps)
    print(f"{agent.n_params() / 1e6:.2f}M params, episode length {env.max_steps}, discount {cfg.agent.discount:.3f}, "
          f"{seed_steps} random seed steps")
    return env, agent, buf, seed_steps


def _log_episode(writer, step, ep_ret, ep_len, task, env_info, crashed):
    """TensorBoard scalars of a finished training episode."""
    writer.add_scalar("episode/return", ep_ret, step)
    writer.add_scalar("episode/length", ep_len, step)
    if task == "flip":
        writer.add_scalar("episode/flip_done", float(env_info["flip_done"]), step)
        writer.add_scalar("episode/crashed", float(crashed), step)


def _learn(agent, buf, cfg, step, seed_steps, writer, info):
    """The gradient updates of this env step (none during the seed steps), and the losses to TensorBoard every
    log_every_steps. Returns the latest losses (`info` itself if there was no update)."""
    if step >= seed_steps and (step - seed_steps) % cfg.update_every == 0:
        # Pretrain on the seed data once, as in the paper, then one update every update_every env steps
        for _ in range(seed_steps if step == seed_steps else 1):
            info = agent.update(*buf.sample(cfg.agent.batch_size))
    if step >= seed_steps and step % cfg.log_every_steps == 0:
        for k, v in info.items():
            writer.add_scalar(f"train/{k}", v, step)
    return info


def _evaluate(agent, eval_env, eval_log, seeds, step, wall_time):
    """Evaluate with MPPI (the actual algorithm) and with the learned policy alone (no planning); log both."""
    mppi = evaluate_policy(eval_env, lambda o, t: agent.act(o, t0=t == 0, eval_mode=True), seeds)
    policy = evaluate_policy(eval_env, lambda o, t: agent.act(o, t0=t == 0, eval_mode=True, plan=False), seeds)
    eval_log.write(step, wall_time, {**mppi, **{f"policy_{k}": v for k, v in policy.items()}})


def train(cfg):
    task = cfg.task
    name = cfg.name or f"tdmpc2_{task}"
    env, agent, buf, seed_steps = _build(cfg)
    writer = SummaryWriter(f"runs/{name}_{time.strftime('%m%d-%H%M%S')}")

    def save():
        # The config goes with the model, so evaluation rebuilds the env it was trained in
        agent.save(f"models/{name}.pt", task=task, config=cfg.to_dict(), env_cfg=cfg.env.to_dict())

    # Learning curve: evaluation starts from rest, without the mid-roll curriculum
    eval_log, eval_env = EvalLog(name), QuadEnv(cfg=cfg.env)
    obs, _ = env.reset(seed=cfg.train_seed)
    t0, ep_ret, ep_len_now, ep, tic, info = True, 0.0, 0, 0, time.time(), {}
    step = 0
    try:
        for step in range(cfg.steps):
            if step < seed_steps:
                a = env.action_space.sample()
            else:
                a = agent.act(obs, t0=t0, plan=cfg.plan_after is not None and step >= cfg.plan_after)
            next_obs, r, term, trunc, env_info = env.step(a)
            buf.add(obs, a, r, term, next_obs)
            obs, t0, ep_ret, ep_len_now = next_obs, False, ep_ret + r, ep_len_now + 1
            if term or trunc:
                buf.end_episode()
                _log_episode(writer, step, ep_ret, ep_len_now, task, env_info, term)
                ep += 1
                if ep % cfg.log_every_episodes == 0:
                    sps = step / (time.time() - tic)
                    print(f"step {step}  ep {ep}  return {ep_ret:.1f}  {sps:.1f} steps/s  "
                          + "  ".join(f"{k} {v:.3f}" for k, v in info.items()))
                obs, _ = env.reset()
                t0, ep_ret, ep_len_now = True, 0.0, 0
            info = _learn(agent, buf, cfg, step, seed_steps, writer, info)
            if step > 0 and step % cfg.save_every == 0:
                save()
                print(f"checkpoint saved at step {step}")
            if step % cfg.eval_every == 0:
                _evaluate(agent, eval_env, eval_log, cfg.eval_seeds, step, time.time() - tic)
                t0 = True  # evaluation used the MPPI warm start; restart it for the training episode
            if cfg.max_minutes is not None and time.time() - tic > 60 * cfg.max_minutes:
                print(f"time limit of {cfg.max_minutes} min reached at {step} steps")
                break
    except KeyboardInterrupt:
        print(f"\ninterrupted at {step} steps, saving what we have")
    finally:
        writer.close()
    save()
    print(f"saved models/{name}.pt")


if __name__ == "__main__":
    cfg = load_config("tdmpc2")
    torch.set_num_threads(cfg.threads)
    train(cfg)
