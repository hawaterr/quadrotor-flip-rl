"""Train PPO on QuadEnv.

Settings: configs/ppo.yaml (+ configs/env.yaml as env.*); override any of them as key=value.
Watch a trained model with animate_rollout.py; the learning curve is logged to results/ (plot_curves.py).
Run from the repo root:
    python -m ppo.train_ppo                          # train hover; saves models/ppo_hover.zip (+ .yaml)
    python -m ppo.train_ppo task=flip steps=300000   # train flip; saves models/ppo_flip.zip (+ .yaml)
"""
import os
import time

from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from configs import Config, load_config, load_env
from envs.evaluation import EvalLog, evaluate as evaluate_policy
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


class PeriodicSaveAndEval(BaseCallback):
    """Every cfg.save_every env steps: save, so progress can be checked while training continues.
    Every cfg.eval_every env steps (and at the start): evaluate, for the learning curve.
    A callback because Stable-Baselines3 runs the training loop itself; TD-MPC2's loop is ours, so it just
    does the same with two checks inside the loop."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.next_save, self.next_eval = cfg, cfg.save_every, 0
        self.log, self.start = EvalLog(model_name(cfg)), time.time()
        self.eval_env = QuadEnv(cfg=cfg.env)  # evaluation starts: from rest, no mid-roll curriculum

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            save(self.model, self.cfg)
            print(f"checkpoint saved at step {self.num_timesteps}")
            self.next_save += self.cfg.save_every
        if self.num_timesteps >= self.next_eval:
            act = lambda obs, t: self.model.predict(obs, deterministic=True)[0]
            self.log.write(self.num_timesteps, time.time() - self.start,
                           evaluate_policy(self.eval_env, act, self.cfg.eval_seeds))
            self.next_eval += self.cfg.eval_every
        return True


def train(cfg):
    env = make_vec_env(QuadEnv, n_envs=cfg.n_envs, seed=cfg.train_seed, env_kwargs=dict(cfg=cfg.env, training=True))
    if cfg.init_model:  # fine-tune: start from a saved model's weights, with this run's env and settings
        model = PPO.load(cfg.init_model, env=env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, gamma=cfg.gamma,
                         learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, verbose=1, tensorboard_log="runs",
                         seed=cfg.train_seed)
    else:
        model = PPO("MlpPolicy", env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, gamma=cfg.gamma,
                    learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, verbose=1,
                    policy_kwargs=dict(net_arch=cfg.net_arch), tensorboard_log="runs", seed=cfg.train_seed)
    try:
        model.learn(total_timesteps=cfg.steps, callback=PeriodicSaveAndEval(cfg))
    except KeyboardInterrupt:
        print(f"\ninterrupted at {model.num_timesteps} steps, saving what we have")
    finally:
        env.close()
    save(model, cfg)
    print(f"saved models/{model_name(cfg)}.zip and models/{model_name(cfg)}.yaml")


if __name__ == "__main__":
    cfg = load_config("ppo")
    train(cfg)
