"""Plot the benchmark learning curves: evaluation return and success rate vs env steps, one line per algorithm.

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

from configs import load_config, load_env  # noqa: E402

# Validated reference palette (categorical slots 1, 2 in fixed order) and recessive ink, light surface
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else None


def baseline_return(policy, seeds):
    """Mean evaluation return of a scripted baseline on the current env.yaml, for a reference line."""
    from animate_rollout import make_policy, task_of
    from envs.evaluation import evaluate
    from envs.quad_env import QuadEnv

    env = QuadEnv(cfg=load_env(task_of(policy)))
    scripted = load_config("animate_rollout", []).scripted
    return evaluate(env, make_policy(policy, env, scripted=scripted), seeds)["return"]


def main():
    cfg = load_config("plot_curves")
    logs = {label: read_log(path) for label, path in cfg.runs.items() if os.path.exists(path)}
    logs = {label: log for label, log in logs.items() if log}
    if not logs:
        raise SystemExit(f"no evaluation logs found: {list(cfg.runs.values())}")

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": GRID})
    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax, key, title in ((ax_r, "return", "Evaluation return"), (ax_s, "success", "Success rate")):
        ax.set_facecolor(SURFACE)
        for (label, log), color in zip(logs.items(), SERIES):
            y = [v * 100 for v in log[key]] if key == "success" else log[key]
            ax.plot(log["step"], y, color=color, lw=2, label=label)
            # direct label at the end of each line (identity isn't colour alone; the legend repeats it)
            ax.annotate(label, (log["step"][-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("env steps (0.02 s each)")
        if cfg.log_x:
            ax.set_xscale("symlog", linthresh=1e3)  # symlog: the first evaluation is at step 0
            top = max(max(log["step"]) for log in logs.values())
            ax.set_xticks([0] + [10 ** k for k in range(3, 8) if 10 ** k <= top])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax_s.set_ylabel("% of evaluation starts")
    ax_s.set_ylim(-5, 105)
    ax_r.set_ylabel("return (mean over starts)")
    if cfg.baseline:
        ret = baseline_return(cfg.baseline, cfg.eval_seeds)
        ax_r.axhline(ret, color=TEXT_2, lw=1, ls="--")
        ax_r.annotate(f"{cfg.baseline} baseline", (0.01, ret), xycoords=("axes fraction", "data"),
                      xytext=(0, 4), textcoords="offset points", fontsize=8, color=TEXT_2)
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    fig.savefig(cfg.out, dpi=150, facecolor=SURFACE)
    print(f"saved {cfg.out}")


if __name__ == "__main__":
    main()
