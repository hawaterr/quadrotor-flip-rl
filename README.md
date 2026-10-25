# quadrotor-flip-rl

Teaching a simulated quadrotor to do a flip with reinforcement learning: one roll flip, then recover to level and stop rotating. PPO (Stable-Baselines3) does the flip; now trying a minimal single-task TD-MPC2 on the same env.

**Current step: hover.** Before flipping, the agent learns to hold a fixed point from a perturbed start (random offset, velocity, tilt and body rates). Both tasks live in one env (`QuadEnv(task="hover" | "flip")`), which share dynamics and actions but have their own reward and termination.

## Files

```
envs/                   simulator and Gym env, shared by both algorithms
  quadrotor.py          quadrotor dynamics
  quad_env.py           Gymnasium environment, task="hover" or task="flip"
  eval_plot.py          roll out one episode with any policy and plot it
ppo/
  train_ppo.py          train and evaluate PPO
tdmpc2/
  agent.py              minimal single-task TD-MPC2 (world model, MPPI planner, ~1M params), CPU friendly
  train_tdmpc2.py       train and evaluate TD-MPC2 (--task pendulum for a quick sanity check)
animate_rollout.py      run one episode of any policy and animate it
configs/                every tunable number: env.yaml, ppo.yaml, tdmpc2.yaml (see Configuration)
models/  plots/  runs/  outputs: trained models, eval plots, TensorBoard logs (not tracked by git)
```

Run every command from the repo root.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch (much smaller than the default CUDA build)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# check that everything imports
python -c "import gymnasium, stable_baselines3, torch, matplotlib; print('ok', torch.__version__)"
```

## Configuration

All settings live in YAML files in `configs/`, not in the code. Each script loads them with one function (`configs.load_config`) into a single config, and any value can be overridden on the command line as `key=value`, the same style as the official TD-MPC2 (Hydra):

Every model is saved with the full config it was trained with (`models/ppo_<task>.yaml`, inside `models/tdmpc2_<task>.pt`), so evaluation and `animate_rollout.py` rebuild the same environment even after you edit the YAML.

## Checkpoints and fine-tuning

Training saves the model every `save_every` env steps (`configs/ppo.yaml`: 100k, `configs/tdmpc2.yaml`: 5k), not only at the end or on Ctrl-C. It writes the usual model file, so you can check progress while training continues, with the usual commands:

```bash
python animate_rollout.py policy=tdmpc2_flip plan=false     # latest checkpoint of a running TD-MPC2 flip run
python -m ppo.train_ppo task=flip eval=true
```

Because checkpoints overwrite the model file, a new run replaces your current model, early (bad) checkpoints first. For PPO:

```bash
python -m ppo.train_ppo task=flip name=ppo_flip_test                    # experiment: saves models/ppo_flip_test.zip, main model untouched
python -m ppo.train_ppo task=flip init_model=models/ppo_flip.zip steps=500000 env.reward.speed=0.3
                                                                        # fine-tune: continue from a saved model with changed settings
```

Fine-tuning is useful when a reward change only needs to polish existing behaviour (e.g. remove a small oscillation): it takes minutes instead of a full run, and it can work where training from scratch with the same reward fails.

## Usage

```bash
# PPO: train (steps from configs/ppo.yaml); saves models/ppo_<task>.zip. task is hover (default) or flip
python -m ppo.train_ppo steps=300000
python -m ppo.train_ppo task=flip
python -m ppo.train_ppo task=flip env.reward.smooth=0.3     # an experiment: any config value can be overridden

# watch training curves
tensorboard --logdir runs

# roll out the saved policy and plot it to plots/<task>_eval.png (seed picks the start)
python -m ppo.train_ppo eval=true
python -m ppo.train_ppo task=flip eval=true seed=2

# TD-MPC2: sanity check on Pendulum-v1 first, then hover / flip; saves models/tdmpc2_<task>.pt
# Control period, MPPI size and the rest are in configs/tdmpc2.yaml (about 10 steps/s on a laptop CPU).
python -m tdmpc2.train_tdmpc2 task=pendulum
python -m tdmpc2.train_tdmpc2 task=flip
python -m tdmpc2.train_tdmpc2 task=flip eval=true             # plan=false: the policy prior alone, without MPPI

# animate a rollout: policy is one of zero, hover, random, scripted_hover, scripted_flip, ppo_hover, ppo_flip, tdmpc2_hover, tdmpc2_flip
# (scripted_* are hand-written controllers, a baseline; ppo_<task> / tdmpc2_<task> load from models/; plan=false for tdmpc2 without MPPI)
python animate_rollout.py policy=ppo_hover
python animate_rollout.py policy=ppo_flip save=plots/flip.gif
```
