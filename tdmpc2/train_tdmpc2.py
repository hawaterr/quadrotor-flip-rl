"""Train the minimal TD-MPC2 (tdmpc2/agent.py) on QuadEnv, or evaluate and plot a trained agent.

Settings: configs/tdmpc2.yaml (+ configs/env.yaml as env.*); override any of them as key=value.
Run from the repo root:
    python -m tdmpc2.train_tdmpc2 task=pendulum                 # sanity check on Pendulum-v1 first
    python -m tdmpc2.train_tdmpc2                               # train hover; saves models/tdmpc2_hover.pt
    python -m tdmpc2.train_tdmpc2 task=flip                     # train flip; saves models/tdmpc2_flip.pt
    python -m tdmpc2.train_tdmpc2 task=flip eval=true           # roll out saved agent (with MPPI) and plot
    python -m tdmpc2.train_tdmpc2 task=flip eval=true plan=false seed=2   # policy prior alone, another start
"""
import time

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from configs import env_from_checkpoint, load_config
from envs.quad_env import QuadEnv
from tdmpc2.agent import TDMPC2

# Pendulum-v1 facts (fixed by Gymnasium, not settings)
PENDULUM_DT = 0.05          # s per step
PENDULUM_MAX_TORQUE = 2.0   # torque range is [-2, 2]; actions are rescaled to [-1, 1]


def make_env(task, env_cfg, training=True):
    if task == "pendulum":
        return gym.wrappers.RescaleAction(gym.make("Pendulum-v1"), -1.0, 1.0)
    return QuadEnv(cfg=env_cfg, training=training)


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
    env = make_env(task, cfg.get("env"))  # no env section for pendulum
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = TDMPC2(obs_dim, act_dim, cfg.agent)
    buf = Buffer(obs_dim, act_dim, cfg.steps, cfg.agent.horizon)
    ep_len = env.spec.max_episode_steps if task == "pendulum" else env.max_steps
    seed_steps = max(cfg.min_seed_steps, cfg.seed_episodes * ep_len)
    print(f"{agent.n_params() / 1e6:.2f}M params, episode length {ep_len}, discount {cfg.agent.discount:.3f}, "
          f"{seed_steps} random seed steps")
    writer = SummaryWriter(f"runs/tdmpc2_{task}_{time.strftime('%m%d-%H%M%S')}")

    def save():
        # The config goes with the model, so evaluation rebuilds the env it was trained in
        env_cfg = cfg.get("env")
        agent.save(f"models/tdmpc2_{task}.pt", task=task, config=cfg.to_dict(),
                   env_cfg=env_cfg.to_dict() if env_cfg is not None else None)

    obs, _ = env.reset(seed=0)
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
    except KeyboardInterrupt:
        print(f"\ninterrupted at {step} steps, saving what we have")
    finally:
        writer.close()
    save()
    print(f"saved models/tdmpc2_{task}.pt")


def animate_pendulum(thetas, torques, dt):
    """Swinging rod on the left, torque over time on the right."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    T = np.arange(len(thetas)) * dt
    fig, (ax, ax_u) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")
    rod, = ax.plot([], [], "-o", color="C3", lw=4, ms=8, markevery=[1])  # dot = the free end
    ax.plot(0, 0, "ks", ms=8)  # pivot
    text = ax.text(-1.15, 1.1, "")
    ax_u.set_xlim(0, T[-1]); ax_u.set_ylim(-1.1 * PENDULUM_MAX_TORQUE, 1.1 * PENDULUM_MAX_TORQUE); ax_u.set_ylabel("torque"); ax_u.set_xlabel("time [s]")
    u_line, = ax_u.plot([], [], "C0", drawstyle="steps-post")

    def update(i):
        rod.set_data([0, np.sin(thetas[i])], [0, np.cos(thetas[i])])
        text.set_text(f"t={T[i]:.2f}s")
        u_line.set_data(T[: min(i + 1, len(torques))], torques[: i + 1])
        return rod, text, u_line

    anim = FuncAnimation(fig, update, frames=len(thetas), interval=dt * 1000, blit=False)
    plt.tight_layout(); plt.show()
    return anim


def evaluate(cfg):
    task, plan, seed = cfg.task, cfg.plan, cfg.seed
    agent, ckpt = TDMPC2.load(f"models/tdmpc2_{task}.pt")
    # MPPI and the policy sample with torch/numpy, so seed those too for a repeatable rollout
    torch.manual_seed(seed); np.random.seed(seed)
    if task == "pendulum":
        env = make_env(task, None, training=False)
        obs, _ = env.reset(seed=seed)
        total, done, t = 0.0, False, 0
        thetas, torques = [np.arctan2(obs[1], obs[0])], []  # obs = [cos, sin, angular velocity]; 0 = upright
        while not done:
            a = agent.act(obs, t0=t == 0, eval_mode=True, plan=plan)
            obs, r, term, trunc, _ = env.step(a)
            total, done, t = total + r, term or trunc, t + 1
            thetas.append(np.arctan2(obs[1], obs[0]))
            torques.append(PENDULUM_MAX_TORQUE * a[0])
        print(f"pendulum return: {total:.1f}")
        animate_pendulum(np.array(thetas), np.array(torques), dt=PENDULUM_DT)
        return
    from envs.eval_plot import evaluate_and_plot
    evaluate_and_plot(make_env(task, env_from_checkpoint(ckpt, task), training=False),
                      lambda obs, t: agent.act(obs, t0=t == 0, eval_mode=True, plan=plan),
                      f"plots/{task}_tdmpc2_eval.png", seed=seed)


if __name__ == "__main__":
    cfg = load_config("tdmpc2")
    torch.set_num_threads(cfg.threads)
    evaluate(cfg) if cfg.eval else train(cfg)
