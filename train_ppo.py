"""Train PPO on QuadEnv, or evaluate and plot a trained policy.

    python train_ppo.py                         # train hover; saves ppo_hover.zip
    python train_ppo.py --task flip             # train flip; saves ppo_flip.zip
    python train_ppo.py --task flip --eval      # roll out saved policy and plot
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from eval_plot import evaluate_and_plot
from quad_env import TASKS, QuadEnv


def train(task, steps):
    flip = task == "flip"
    # Flip: half the episodes start mid-roll, and a little entropy keeps it exploring rolls
    env = make_vec_env(QuadEnv, n_envs=8, env_kwargs=dict(task=task, mid_flip_prob=0.5 if flip else 0.0))
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=2048, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.005 if flip else 0.0, verbose=1,
                policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="runs")
    try:
        model.learn(total_timesteps=steps)
    except KeyboardInterrupt:
        print(f"\ninterrupted at {model.num_timesteps} steps, saving what we have")
    finally:
        env.close()
    model.save(f"ppo_{task}")
    print(f"saved ppo_{task}.zip")


def evaluate(task, seed=0):
    model = PPO.load(f"ppo_{task}")
    evaluate_and_plot(QuadEnv(task=task), lambda obs, t: model.predict(obs, deterministic=True)[0],
                      f"{task}_eval.png", seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hover", choices=TASKS)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="eval: seed of the episode's starting state")
    ap.add_argument("--steps", type=int, default=1_000_000)
    args = ap.parse_args()
    evaluate(args.task, args.seed) if args.eval else train(args.task, args.steps)
