"""Config loading and a PPO smoke run with the asymmetric policy."""
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from omegaconf.errors import ConfigKeyError
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import Config, load_config
from envs.quad_env import QuadEnv
from ppo.asymmetric import AsymmetricPolicy


def test_configs():
    cfg = load_config("ppo", ["env.reward.smooth=0.3", "task=hover"])
    # Check that overrides are applied
    assert cfg.env.task == "hover"
    assert cfg.env.reward.smooth == 0.3

    # Typos are rejected
    with pytest.raises(ConfigKeyError):
        load_config("ppo", ["env.reward.no_such_key=1"])
    with pytest.raises(ValueError):
        load_config("ppo", ["task=backflip"])


def test_ppo_smoke(env_cfg: Callable[..., Config], tmp_path: Path):
    """Trains a few steps; save/load round-trips."""
    cfg = env_cfg("flip")
    env = DummyVecEnv([lambda: QuadEnv(cfg=cfg, training=True)])
    n_actor = env.envs[0].unwrapped.n_actor_obs  # type: ignore[attr-defined]
    model = PPO(AsymmetricPolicy, env, n_steps=128, batch_size=64, n_epochs=1, seed=0,
                policy_kwargs=dict(net_arch=[16, 16], n_actor=n_actor))
    model.learn(total_timesteps=256)

    obs = np.stack([env.envs[0].reset(seed=s)[0] for s in range(4)])
    action, _ = model.predict(obs, deterministic=True)
    model.save(tmp_path / "model.zip")
    loaded = PPO.load(tmp_path / "model.zip")
    np.testing.assert_array_equal(action, loaded.predict(obs, deterministic=True)[0])
