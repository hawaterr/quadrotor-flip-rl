"""Plot the benchmark learning curves: total episode reward vs env steps, one line per algorithm, the mean over
training seeds with a band of +- 1 std.

Reads the evaluation logs the trainers write during training (results/<run>_seed<i>.csv, every eval_every
steps). Settings: configs/plot_curves.yaml; override any of them as key=value.

    python plot_curves.py                         # -> docs/learning_curves.png (shown in the README)
"""
import csv
import os

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from configs import load_config  # noqa: E402

# Line colours (one per curve, in order), then background, text, secondary text and grid
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def read_seeds(run, column, seeds):
    """(env steps, seeds x evals array) of `column` over the seeds' logs that exist, cut to the shortest log, so
    runs still training can be plotted. None if no seed has a log yet."""
    logs = [read_log(path) for path in (f"results/{run}_seed{s}.csv" for s in seeds) if os.path.exists(path)]
    logs = [log for log in logs if log and column in log]
    if not logs:
        return None
    n = min(len(log["step"]) for log in logs)
    step = logs[0]["step"][:n]
    if any(log["step"][:n] != step for log in logs):  # same eval_every in every run
        raise SystemExit(f"{run}: the seeds' logs were evaluated at different steps")
    return step, np.array([log[column][:n] for log in logs])


def main():
    cfg = load_config("plot_curves")
    curves = {}  # label: (env steps, seeds x evals total episode reward)
    for label, curve in cfg.curves.items():
        data = read_seeds(curve.run, curve.column, cfg.seeds)
        if data:
            curves[label] = data
    if not curves:
        raise SystemExit(f"no evaluation logs found: results/<run>_seed<i>.csv for runs "
                         f"{sorted({c.run for c in cfg.curves.values()})}, seeds {cfg.seeds}")

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (label, (step, y)), color in zip(curves.items(), SERIES):
        mean = y.mean(axis=0)
        ax.plot(step, mean, color=color, lw=2, label=f"{label} (n={len(y)})")
        if len(y) > 1:
            std = y.std(axis=0, ddof=1)  # sample std over seeds
            ax.fill_between(step, mean - std, mean + std, color=color, alpha=0.18, lw=0)
    ax.set_title("Total episode reward", loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel("env steps (0.02 s each)")
    ax.set_ylabel("total reward per episode\n(mean ± std over training seeds)")
    if cfg.log_x:
        ax.set_xscale("symlog", linthresh=1e3)  # symlog: the first evaluation is at step 0
        top = max(max(step) for step, _ in curves.values())
        ax.set_xticks([0] + [10 ** k for k in range(3, 8) if 10 ** k <= top])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    fig.savefig(cfg.out, dpi=150, facecolor=SURFACE)
    print(f"saved {cfg.out}")


if __name__ == "__main__":
    main()
