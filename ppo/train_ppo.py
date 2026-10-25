"""Train PPO on QuadEnv, or evaluate and plot a trained policy.

Settings: configs/ppo.yaml (+ configs/env.yaml as env.*); override any of them as key=value.
Run from the repo root:
    python -m ppo.train_ppo                          # train hover; saves models/ppo_hover.zip (+ .yaml)
    python -m ppo.train_ppo task=flip steps=300000   # train flip; saves models/ppo_flip.zip (+ .yaml)
    python -m ppo.train_ppo task=flip eval=true      # roll out saved policy and plot to plots/flip_eval.png
"""
import os

from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from configs import ENV_TASKS, Config, load_config, load_env
from envs.eval_plot import evaluate_and_plot
from envs.quad_env import QuadEnv


def saved_env_config(task):
    """The env config saved next to models/ppo_<task>.zip; env.yaml for models saved before configs existed."""
    path = f"models/ppo_{task}.yaml"
    if not os.path.exists(path):
        return load_env(task)
    return Config(OmegaConf.to_container(OmegaConf.load(path)).get("env"))


def train(cfg):
    env = make_vec_env(QuadEnv, n_envs=cfg.n_envs, env_kwargs=dict(cfg=cfg.env, training=True))
    model = PPO("MlpPolicy", env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, gamma=cfg.gamma,
                learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, verbose=1,
                policy_kwargs=dict(net_arch=cfg.net_arch), tensorboard_log="runs")
    try:
        model.learn(total_timesteps=cfg.steps)
    except KeyboardInterrupt:
        print(f"\ninterrupted at {model.num_timesteps} steps, saving what we have")
    finally:
        env.close()
    model.save(f"models/ppo_{cfg.task}")  # SB3 creates the folder and adds .zip
    # Save the config next to the model, so evaluation rebuilds the env it was trained in
    OmegaConf.save(cfg.to_dict(), f"models/ppo_{cfg.task}.yaml")
    print(f"saved models/ppo_{cfg.task}.zip and models/ppo_{cfg.task}.yaml")


def evaluate(cfg):
    model = PPO.load(f"models/ppo_{cfg.task}")
    evaluate_and_plot(QuadEnv(cfg=saved_env_config(cfg.task)),
                      lambda obs, t: model.predict(obs, deterministic=True)[0],
                      f"plots/{cfg.task}_eval.png", seed=cfg.seed)


if __name__ == "__main__":
    cfg = load_config("ppo")
    if cfg.task not in ENV_TASKS:
        raise ValueError(f"PPO task must be one of {ENV_TASKS}, got {cfg.task!r}")
    evaluate(cfg) if cfg.eval else train(cfg)
