"""Policy evaluation on fixed seeds, and the CSV log behind the learning curves."""
import csv
import os
from collections.abc import Callable

import numpy as np
import torch

from envs.quad_env import QuadEnv

Policy = Callable[[np.ndarray, int], np.ndarray]  # (obs, step in episode) -> action in [-1, 1]
SETTLE_WINDOW = 0.5  # s at the end of an episode for the settle metric


def evaluate(env: QuadEnv, act: Policy, seeds: list[int]) -> dict[str, float]:
    """Mean metrics over one episode per seed, acting with act(obs, t) -> action."""
    tail = max(1, round(SETTLE_WINDOW / env.policy_dt))
    rows = []
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()  # restored below: called mid-training
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)  # MPPI and policies may sample
        obs, _ = env.reset(seed=seed)
        total, dist, speed = 0.0, [], []
        for t in range(env.max_steps):
            obs, r, term, trunc, info = env.step(act(obs, t))
            total += r
            dist.append(info["dist"])
            speed.append(np.linalg.norm(env.quad.v))
            if term or trunc:
                break
        rows.append({"return": total, "final_dist": dist[-1],
                     "settle_speed": float(np.mean(speed[-tail:])), "crashed": float(info["crashed"])})
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


class EvalLog:
    """Appends one row per evaluation to results/<name>.csv (a fresh file per training run)."""

    def __init__(self, name: str):
        os.makedirs("results", exist_ok=True)
        self.path = f"results/{name}.csv"
        self.file = None

    def write(self, step: int, wall_time: float, metrics: dict[str, float]):
        row = {"step": step, "wall_time": round(wall_time, 1), **{k: round(v, 4) for k, v in metrics.items()}}
        if self.file is None:
            self.file = open(self.path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()  # readable while training
        print(f"eval at step {step}: return {metrics['return']:.1f}  final dist {metrics['final_dist']:.2f} m")
