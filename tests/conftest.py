"""Shared fixtures: env configs, optionally for an ideal drone (no sensor noise, randomisation or latency)."""
from collections.abc import Callable

import pytest

from configs import Config, load_env


def make_env_cfg(task: str, ideal: bool = False) -> Config:
    """env.yaml for `task`; ideal: noiseless sensors, nominal dynamics, no latency."""
    cfg = load_env(task)
    if ideal:
        cfg = Config({**cfg.to_dict(), "randomisation": None, "latency": [0.0, 0.0],
                      "sensors": {k: 0.0 for k in cfg.sensors}})
    return cfg


@pytest.fixture
def env_cfg() -> Callable[..., Config]:
    """make_env_cfg(task, ideal=False)."""
    return make_env_cfg
