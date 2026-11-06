"""Temporary: compare envs/mujoco_quadrotor.py with the hand-written envs/quadrotor.py. Deleted with the old sim.

    python compare_sims.py
"""
import time

import numpy as np
import torch

from animate_rollout import make_policy
from configs import Config, load_config, load_env
from envs.evaluation import evaluate
from envs.mujoco_quadrotor import MujocoQuadrotor
from envs.quad_env import QuadEnv
from envs.quadrotor import G, Quadrotor

EVAL_SEEDS = [0, 1, 2, 3, 4]


def mujoco_env(cfg, **kwargs):
    env = QuadEnv(cfg=cfg, **kwargs)
    env.quad = MujocoQuadrotor(cfg.dynamics)
    return env


def attitude_error_deg(R_a, R_b):
    c = (np.trace(R_a.T @ R_b) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def record_actions(cfg, policy_name, seed):
    """Action sequence of a policy flown on the old sim (so both sims replay exactly the same one)."""
    env = QuadEnv(cfg=cfg)
    scripted = load_config("animate_rollout", argv=[]).scripted
    if policy_name == "random":
        env.action_space.seed(seed)
    policy = make_policy(policy_name, env, plan=False, scripted=scripted)
    obs, _ = env.reset(seed=seed, options={"exact": True})
    actions = []
    for t in range(env.max_steps):
        a = np.clip(policy(obs, t), -1, 1)
        actions.append(a)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return actions


def replay(cfg, actions, seed):
    """Replay `actions` through both sims from the same start, without stopping at termination. Returns the position
    and attitude error per step, and the two final returns."""
    old, new = QuadEnv(cfg=cfg), mujoco_env(cfg)
    obs_old, _ = old.reset(seed=seed, options={"exact": True})
    obs_new, _ = new.reset(seed=seed, options={"exact": True})
    start_obs_diff = np.abs(obs_old - obs_new).max()
    pos, att, obs_diff, ret_old, ret_new = [], [], [], 0.0, 0.0
    for a in actions:
        obs_old, r_old, *_ = old.step(a)
        obs_new, r_new, *_ = new.step(a)
        ret_old += r_old
        ret_new += r_new
        pos.append(np.linalg.norm(old.quad.p - new.quad.p))
        att.append(attitude_error_deg(old.quad.R, new.quad.R))
        obs_diff.append(np.abs(obs_old - obs_new).max())
    return dict(start_obs_diff=start_obs_diff, pos=np.array(pos), att=np.array(att), obs=np.array(obs_diff),
                ret_old=ret_old, ret_new=ret_new)


def compare_trajectories():
    print("\n== 1. Same actions through both sims (no early stop) ==")
    print(f"{'task':6} {'actions':16} {'steps':>5} {'max pos err [mm]':>17} {'final pos err [mm]':>19} "
          f"{'max att err [deg]':>18} {'max obs diff':>13} {'return old':>11} {'return new':>11}")
    cases = [("flip", "scripted_flip"), ("hover", "scripted_hover"), ("flip", "random"), ("hover", "random"),
             ("flip", "ppo_flip_seed0"), ("flip", "tdmpc2_flip_seed0")]
    for task, name in cases:
        cfg = saved_cfg(name) if "seed" in name else load_env(task)
        for seed in (0, 1):
            actions = record_actions(cfg, name, seed)
            r = replay(cfg, actions, seed)
            assert r["start_obs_diff"] < 1e-6, f"different starts: {r['start_obs_diff']}"
            print(f"{task:6} {name:16} {len(actions):5d} {1000 * r['pos'].max():17.3f} {1000 * r['pos'][-1]:19.3f} "
                  f"{r['att'].max():18.3f} {r['obs'].max():13.5f} {r['ret_old']:11.3f} {r['ret_new']:11.3f}")


def saved_cfg(policy):
    if policy.startswith("tdmpc2_"):
        return Config(torch.load(f"models/{policy}.pt", weights_only=False)["env_cfg"])
    from ppo.train_ppo import saved_env_config
    return saved_env_config(policy)


def compare_evaluations():
    print("\n== 2. Saved models, evaluate() over seeds 0-4 ==")
    print(f"{'model':28} {'old sim':>9} {'MuJoCo':>9} {'diff':>8} {'logged':>9}   metrics on MuJoCo")
    from stable_baselines3 import PPO
    from tdmpc2.agent import TDMPC2

    for seed, ppo_log, mppi_log, pi_log in ((0, 203.9087, 172.5447, 194.2834),):
        ppo_name, td_name = f"ppo_flip_seed{seed}", f"tdmpc2_flip_seed{seed}"
        model = PPO.load(f"models/{ppo_name}")
        agent, _ = TDMPC2.load(f"models/{td_name}.pt")
        rows = [
            (ppo_name, saved_cfg(ppo_name), lambda o, t: model.predict(o, deterministic=True)[0], ppo_log),
            (f"{td_name} (no plan)", saved_cfg(td_name),
             lambda o, t: agent.act(o, t0=t == 0, eval_mode=True, plan=False), pi_log),
            (f"{td_name} (MPPI)", saved_cfg(td_name), lambda o, t: agent.act(o, t0=t == 0, eval_mode=True), mppi_log),
        ]
        for label, cfg, act, logged in rows:
            old = evaluate(QuadEnv(cfg=cfg), act, EVAL_SEEDS)
            new = evaluate(mujoco_env(cfg), act, EVAL_SEEDS)
            print(f"{label:28} {old['return']:9.2f} {new['return']:9.2f} {new['return'] - old['return']:8.2f} "
                  f"{logged:9.2f}   final_dist={new['final_dist']:.3f} settle_speed={new['settle_speed']:.4f} "
                  f"crashed={new['crashed']:.0f}", flush=True)


def compare_speed():
    print("\n== 3. Speed ==")
    cfg = load_env("flip")
    rate = np.array([1.0, 0.5, -0.5])
    for label, cls in (("old sim", Quadrotor), ("MuJoCo", MujocoQuadrotor)):
        quad = cls(cfg.dynamics)
        n = 200_000
        tic = time.perf_counter()
        for _ in range(n):
            quad.step(G, rate)
        print(f"{label:8} physics: {n / (time.perf_counter() - tic):10.0f} substeps/s")
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.05, 0.05, (5000, 4))
    for label, make in (("old sim", lambda: QuadEnv(cfg=cfg)), ("MuJoCo", lambda: mujoco_env(cfg))):
        env = make()
        env.reset(seed=0)
        tic = time.perf_counter()
        for a in actions:
            *_, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset()
        print(f"{label:8} env:     {len(actions) / (time.perf_counter() - tic):10.0f} env steps/s")


def sanity_checks():
    print("\n== 4. Sanity checks on MujocoQuadrotor ==")
    quad = MujocoQuadrotor(load_env("hover").dynamics)
    quad.reset(p=[0, 0, 10])
    quad.thrust_acc = 0.0
    for _ in range(500):  # 1 s free fall
        quad.step(0.0, np.zeros(3))
    print(f"free fall 1s: z={quad.p[2]:.2f} (expect ~{10 - 0.5 * G:.2f} minus a bit of thrust lag)")

    quad.reset(p=[0, 0, 2])
    for _ in range(1000):  # 2 s hover
        quad.step(G, np.zeros(3))
    print(f"hover 2s: z={quad.p[2]:.3f} (expect 2.000)")

    quad.reset(p=[0, 0, 2])
    for _ in range(int(2 * np.pi / 10.0 / quad.dt)):  # roll at 10 rad/s for one revolution
        quad.step(G, np.array([10.0, 0, 0]))
    print(f"one roll rev: body z-axis world z = {quad.R[2, 2]:.2f} (expect ~1, slightly less from controller lag)")


if __name__ == "__main__":
    torch.set_num_threads(4)
    sanity_checks()
    compare_trajectories()
    compare_speed()
    compare_evaluations()
