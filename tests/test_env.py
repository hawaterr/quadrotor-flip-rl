"""QuadEnv: gym contract, determinism, the actor/critic observation split, and a scripted flip."""
from collections.abc import Callable

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from animate_rollout import scripted_flip_policy
from configs import Config, load_config
from envs.quad_env import QuadEnv


@pytest.mark.parametrize("task", ["hover", "flip"])
@pytest.mark.parametrize("ideal", [False, True], ids=["randomised", "ideal"])
def test_gym_contract(env_cfg: Callable[..., Config], task: str, ideal: bool):
    env = QuadEnv(cfg=env_cfg(task, ideal), training=True)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs in env.observation_space
    obs, *_ = env.step(env.action_space.sample())
    assert obs in env.observation_space


def test_determinism(env_cfg: Callable[..., Config]):
    """Same seed and actions -> same episode (evaluate() and results/ rely on it)."""
    actions = np.random.default_rng(0).uniform(-1, 1, (30, 4))

    def run() -> tuple[np.ndarray, list[float]]:
        env = QuadEnv(cfg=env_cfg("flip"), training=True)
        observations, rewards = [env.reset(seed=3)[0]], []
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            observations.append(obs)
            rewards.append(r)
            if term or trunc:
                break
        return np.array(observations), rewards

    (obs1, rew1), (obs2, rew2) = run(), run()
    np.testing.assert_array_equal(obs1, obs2)
    assert rew1 == rew2


def test_actor_sees_no_privileged_state(env_cfg: Callable[..., Config]):
    """Changing the true position, velocity or dynamics params only changes the critic's inputs."""
    env = QuadEnv(cfg=env_cfg("flip"))
    env.reset(seed=0)
    before = env._obs()
    env.quad.p = env.quad.p + 1.0
    env.quad.v = env.quad.v + 1.0
    env.param_obs = env.param_obs + 0.5
    after = env._obs()
    n = env.n_actor_obs
    np.testing.assert_array_equal(before[:n], after[:n])
    assert not np.array_equal(before[n:], after[n:])


def test_scripted_flip(env_cfg: Callable[..., Config]):
    """End to end: the hand-written flip controller completes the flip without crashing."""
    env = QuadEnv(cfg=env_cfg("flip"))
    policy = scripted_flip_policy(env, load_config("animate_rollout", []).scripted)
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        for t in range(env.max_steps):
            obs, _, term, trunc, info = env.step(policy(obs, t))
            if term or trunc:
                break
        assert info["flip_done"] and not info["crashed"], f"seed {seed}"
