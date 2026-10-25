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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from configs import ENV_TASKS, Config, load_config, load_env
from envs.eval_plot import evaluate_and_plot
from envs.quad_env import QuadEnv


def model_name(cfg):
    return cfg.name or f"ppo_{cfg.task}"


def saved_env_config(task, name=None):
    """The env config saved next to models/<name>.zip (default ppo_<task>); env.yaml for models saved before
    configs existed."""
    path = f"models/{name or f'ppo_{task}'}.yaml"
    if not os.path.exists(path):
        return load_env(task)
    return Config(OmegaConf.to_container(OmegaConf.load(path)).get("env"))


def save(model, cfg):
    """Save the model and, next to it, its config (so evaluation rebuilds the env it was trained in).
    Written to temporary files first, so an eval started mid-save never reads a half-written file."""
    path = f"models/{model_name(cfg)}"
    os.makedirs("models", exist_ok=True)
    model.save(f"{path}.tmp.zip")
    os.replace(f"{path}.tmp.zip", f"{path}.zip")
    OmegaConf.save(cfg.to_dict(), f"{path}.tmp.yaml")
    os.replace(f"{path}.tmp.yaml", f"{path}.yaml")


class Checkpoint(BaseCallback):
    """Save every cfg.save_every env steps, so progress can be checked while training continues."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.next_save = cfg, cfg.save_every

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            save(self.model, self.cfg)
            print(f"checkpoint saved at step {self.num_timesteps}")
            self.next_save += self.cfg.save_every
        return True


def train(cfg):
    env = make_vec_env(QuadEnv, n_envs=cfg.n_envs, env_kwargs=dict(cfg=cfg.env, training=True))
    if cfg.init_model:  # fine-tune: start from a saved model's weights, with this run's env and settings
        model = PPO.load(cfg.init_model, env=env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, gamma=cfg.gamma,
                         learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, verbose=1, tensorboard_log="runs")
    else:
        model = PPO("MlpPolicy", env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, gamma=cfg.gamma,
                    learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, verbose=1,
                    policy_kwargs=dict(net_arch=cfg.net_arch), tensorboard_log="runs")
    try:
        model.learn(total_timesteps=cfg.steps, callback=Checkpoint(cfg))
    except KeyboardInterrupt:
        print(f"\ninterrupted at {model.num_timesteps} steps, saving what we have")
    finally:
        env.close()
    save(model, cfg)
    print(f"saved models/{model_name(cfg)}.zip and models/{model_name(cfg)}.yaml")


def evaluate(cfg):
    model = PPO.load(f"models/{model_name(cfg)}")
    evaluate_and_plot(QuadEnv(cfg=saved_env_config(cfg.task, model_name(cfg))),
                      lambda obs, t: model.predict(obs, deterministic=True)[0],
                      f"plots/{cfg.task}_eval.png", seed=cfg.seed)


if __name__ == "__main__":
    cfg = load_config("ppo")
    if cfg.task not in ENV_TASKS:
        raise ValueError(f"PPO task must be one of {ENV_TASKS}, got {cfg.task!r}")
    evaluate(cfg) if cfg.eval else train(cfg)
