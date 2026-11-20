"""Config loading: the YAML files here, merged, with key=value command-line overrides."""
import os
import sys

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS = ("hover", "flip")  # QuadEnv tasks; configs have a section per task


class Config(dict):
    """Nested dict with attribute access (cfg.env.reward.smooth). Plain Python, so it is fast in hot loops."""

    def __init__(self, d=()):
        super().__init__({k: Config(v) if isinstance(v, dict) else v for k, v in dict(d).items()})

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r}") from None

    def to_dict(self) -> dict:
        """Plain nested dicts, for saving with a model."""
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def _load(name: str) -> DictConfig:
    return OmegaConf.load(os.path.join(CONFIG_DIR, f"{name}.yaml"))


def _env_section(task: str) -> DictConfig:
    """env.yaml's shared settings plus the section for `task`, as an OmegaConf config."""
    raw = _load("env")
    shared = {k: v for k, v in raw.items() if k not in TASKS}
    return OmegaConf.merge(shared, {"mid_flip": None}, raw[task], {"task": task})


def load_env(task: str) -> Config:
    """The env config for `task` on its own (scripts without an algorithm config)."""
    return Config(OmegaConf.to_container(_env_section(task)))


def load_config(name: str, argv: list[str] | None = None) -> Config:
    """configs/<name>.yaml + env.yaml + the task's sections + command-line overrides; unknown keys raise."""
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
