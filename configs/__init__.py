"""Config files. Every tunable number lives in one of these:

    env.yaml              simulator, tasks and reward weights (shared by both algorithms)
    ppo.yaml              PPO settings
    tdmpc2.yaml           TD-MPC2 settings
    animate_rollout.yaml  animate_rollout.py settings and the scripted baseline controllers

load_config() merges them into one config and applies command-line overrides, the same style as the
official TD-MPC2 (Hydra):  python -m tdmpc2.train_tdmpc2 task=flip steps=60000 env.reward.smooth=0.3
"""
import os
import sys

from omegaconf import OmegaConf

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS = ("hover", "flip")  # QuadEnv tasks; env.yaml and the algorithm files have a section per task


class Config(dict):
    """Nested dict with attribute access (cfg.env.reward.smooth). Plain Python, so it is fast in hot loops."""

    def __init__(self, d=()):
        super().__init__({k: Config(v) if isinstance(v, dict) else v for k, v in dict(d).items()})

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r}") from None

    def to_dict(self):
        """Plain nested dicts, for saving with a model."""
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def _load(name):
    return OmegaConf.load(os.path.join(CONFIG_DIR, f"{name}.yaml"))


def _env_section(task):
    """env.yaml's shared settings plus the section for `task`, as an OmegaConf config."""
    raw = _load("env")
    shared = {k: v for k, v in raw.items() if k not in TASKS}
    return OmegaConf.merge(shared, {"mid_flip": None}, raw[task], {"task": task})


def load_env(task, policy_dt=None):
    """The env config for `task` on its own (scripts without an algorithm config). policy_dt overrides env.yaml's."""
    env = _env_section(task)
    if policy_dt is not None:
        env.policy_dt = policy_dt
    return Config(OmegaConf.to_container(env))


def load_config(name, argv=None):
    """configs/<name>.yaml, plus `env` (env.yaml for the chosen task), plus the file's section for that task,
    plus command-line `key=value` overrides (argv, default sys.argv[1:]).

    Overriding a key that doesn't exist (a typo like env.reward.smoth=0.3) raises an error."""
    raw = _load(name)
    cli = OmegaConf.from_cli(sys.argv[1:] if argv is None else argv)
    cfg = OmegaConf.create({k: v for k, v in raw.items() if k not in TASKS})
    task = cli.get("task", cfg.get("task"))
    if task is not None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        cfg.env = _env_section(task)
    OmegaConf.set_struct(cfg, True)  # from here on, merging an unknown key is an error
    task_section = raw.get(task) if task is not None else None
    cfg = OmegaConf.merge(cfg, task_section or {}, cli)
    return Config(OmegaConf.to_container(cfg))


def env_from_checkpoint(meta, task):
    """The env config saved with a model. Models saved before config files existed fall back to env.yaml,
    plus the ctrl_dt (the old name of policy_dt) / smooth_coef they stored."""
    if meta.get("env_cfg") is not None:
        return Config(meta["env_cfg"])
    env = load_env(task, policy_dt=meta.get("ctrl_dt"))
    if meta.get("smooth_coef") is not None:
        env.reward["smooth"] = meta["smooth_coef"]
    return env
