# quadrotor-flip-rl

Teaching a simulated quadrotor to do a flip with reinforcement learning: one roll flip, then recover to level. Comparison between PPO and TD-MPC2.

## Results

One training run each, from scratch, with the same reward and the same control period (0.02 s per env step). TD-MPC2 is shown twice: acting with MPPI planning (the actual algorithm), and with its learned policy alone (no planning, same model).

<table>
  <tr><th>PPO</th><th>TD-MPC2 (MPPI)</th><th>TD-MPC2 (no plan)</th></tr>
  <tr>
    <td><img src="docs/ppo_flip.gif" alt="PPO: flip, fly back to the start point and hover"></td>
    <td><img src="docs/tdmpc2_flip.gif" alt="TD-MPC2 with MPPI: flip, then drift near the start point"></td>
    <td><img src="docs/tdmpc2_flip_no_plan.gif" alt="TD-MPC2 policy alone: flip, fly back towards the start point and hover"></td>
  </tr>
</table>

![Evaluation return vs env steps](docs/learning_curves.png)

Final models, evaluated on 5 starts from rest (`envs/evaluation.py`):

| | PPO | TD-MPC2 (MPPI) | TD-MPC2 (no plan) |
|---|---|---|---|
| Return | 203 | 94 | 165 |
| Final distance to start point | 0.07 m | 0.55 m | 0.32 m |
| Speed, last 0.5 s | 0.001 m/s | 1.5 m/s | 0.09 m/s |
| Crashes | 0/5 | 0/5 | 0/5 |
| Env steps to first reach return 150 | 475k | 20k | 30k |
| Training steps / wall-clock time | 1M / 21 min | 60k / 71 min | (same run) |

TD-MPC2 learns the flip with about 20× fewer env steps, but its evaluation return swings a lot from one evaluation to the next (e.g. 10 at 47.5k steps, 190 at 52.5k), and the final 60k model happens to sit in a dip with MPPI. PPO needs far more env steps but less wall-clock time, and ends steadier: it hovers back at the start point.

Regenerate: `python animate_rollout.py policy=ppo_flip save=docs/ppo_flip.gif` (same for `tdmpc2_flip`, plus `plan=false save=docs/tdmpc2_flip_no_plan.gif`) and `python plot_curves.py`.

## Files

```
envs/                   simulator and Gym env, shared by both algorithms
  quadrotor.py          quadrotor dynamics
  quad_env.py           Gymnasium environment, task="hover" or task="flip"
  evaluation.py         evaluate a policy over fixed starts; logs the learning curve during training
ppo/
  train_ppo.py          train PPO
tdmpc2/
  agent.py              minimal single-task TD-MPC2 (world model, MPPI planner, 0.33M params), CPU friendly
  train_tdmpc2.py       train TD-MPC2
animate_rollout.py      run one episode of any policy and animate it
plot_curves.py          plot the learning curves (results/*.csv) to docs/learning_curves.png
configs/                every tunable number: env.yaml, ppo.yaml, tdmpc2.yaml, ... (see Configuration)
results/  docs/         learning-curve logs and README figures (tracked by git)
models/  plots/  runs/  outputs: trained models, plots, TensorBoard logs (not tracked by git)
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
python plot_curves.py                                       # learning curves so far (results/ is written every eval_every steps)
```

Because checkpoints overwrite the model file, a new run replaces your current model, early (bad) checkpoints first. For PPO:

```bash
python -m ppo.train_ppo task=flip name=ppo_flip_test # experiment: saves models/ppo_flip_test.zip, main model untouched
python -m ppo.train_ppo task=flip init_model=models/ppo_flip.zip steps=500000 env.reward.speed=0.3 # fine-tune: continue from a saved model with changed settings
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

# TD-MPC2: train hover / flip; saves models/tdmpc2_<task>.pt
# MPPI size and the rest are in configs/tdmpc2.yaml (about 14 steps/s on a laptop CPU).
python -m tdmpc2.train_tdmpc2 task=flip

# learning curves: evaluation return vs env steps, from the logs both trainers write
python plot_curves.py

# watch a model: policy is zero, hover, random, scripted_hover, scripted_flip, or any model in models/ (ppo_*, tdmpc2_*)
# (scripted_* are hand-written controllers, a baseline; seed picks the start; plan=false for tdmpc2 without MPPI)
python animate_rollout.py policy=ppo_hover
python animate_rollout.py policy=ppo_flip save=plots/flip.gif
```
