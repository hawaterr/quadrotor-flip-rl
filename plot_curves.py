"""Plot the benchmark learning curves: total episode reward vs env steps, one line per algorithm.

Reads the evaluation logs the trainers write during training (results/<model name>.csv, every eval_every
steps). Settings: configs/plot_curves.yaml; override any of them as key=value.

    python plot_curves.py                         # -> docs/learning_curves.png (shown in the README)
"""
import csv
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from configs import load_config  # noqa: E402

# Validated reference palette (categorical slots 1-3 in fixed order) and recessive ink, light surface
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def main():
    cfg = load_config("plot_curves")
    curves = {}  # label: (env steps, total episode reward)
    for label, curve in cfg.curves.items():
        log = read_log(curve.file) if os.path.exists(curve.file) else None
        if log and curve.column in log:
            curves[label] = (log["step"], log[curve.column])
    if not curves:
        raise SystemExit(f"no evaluation logs found: {[c.file for c in cfg.curves.values()]}")

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": GRID})
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for (label, (step, y)), color in zip(curves.items(), SERIES):
        ax.plot(step, y, color=color, lw=2, label=label)
    ax.set_title("Total episode reward", loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel("env steps (0.02 s each)")
    ax.set_ylabel("total reward per episode (mean over starts)")
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
