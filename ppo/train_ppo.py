"""Train PPO on QuadEnv: python -m ppo.train_ppo [key=value ...]."""
import os
import time

from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from configs import Config, load_config
from envs.evaluation import EvalLog, evaluate as evaluate_policy
from envs.quad_env import QuadEnv
from ppo.asymmetric import AsymmetricPolicy


def model_name(cfg):
    return cfg.name or f"ppo_{cfg.task}"


def saved_env_config(name):
    """The env config saved next to models/<name>.zip."""
    return Config(OmegaConf.to_container(OmegaConf.load(f"models/{name}.yaml"))["env"])


def save(model, cfg):
    """Save the model and its config, via temp files so readers never see half-written files."""
    path = f"models/{model_name(cfg)}"
    os.makedirs("models", exist_ok=True)
    model.save(f"{path}.tmp.zip")
    os.replace(f"{path}.tmp.zip", f"{path}.zip")
    OmegaConf.save(cfg.to_dict(), f"{path}.tmp.yaml")
    os.replace(f"{path}.tmp.yaml", f"{path}.yaml")


class PeriodicSaveAndEval(BaseCallback):
    """Save every save_every and evaluate every eval_every env steps."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.next_save, self.next_eval = cfg, cfg.save_every, 0
        self.log, self.start = EvalLog(model_name(cfg)), time.time()
        self.eval_env = QuadEnv(cfg=cfg.env)  # no mid-roll starts

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
    # One process per env: the simulation is the bottleneck
    env = make_vec_env(QuadEnv, n_envs=cfg.n_envs, seed=cfg.train_seed, env_kwargs=dict(cfg=cfg.env, training=True),
                       vec_env_cls=SubprocVecEnv)
    if cfg.init_model:  # fine-tune
        model = PPO.load(cfg.init_model, env=env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, gamma=cfg.gamma,
                         learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, verbose=1, tensorboard_log="runs",
                         seed=cfg.train_seed)
    else:
        # Actor: deployable observations only; critic: all
        n_actor = QuadEnv(cfg=cfg.env).n_actor_obs
        model = PPO(AsymmetricPolicy, env, n_steps=cfg.n_steps, batch_size=cfg.batch_size, gamma=cfg.gamma,
                    learning_rate=cfg.learning_rate, ent_coef=cfg.ent_coef, verbose=1,
                    policy_kwargs=dict(net_arch=cfg.net_arch, log_std_init=cfg.log_std_init, n_actor=n_actor),
                    tensorboard_log="runs", seed=cfg.train_seed)
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
