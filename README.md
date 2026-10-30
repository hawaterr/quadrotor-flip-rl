# quadrotor-flip-rl

Teaching a simulated quadrotor to do a flip with reinforcement learning: one roll flip, then recover to level. Comparison between PPO and TD-MPC2.

## Results

TD-MPC2 is shown twice: acting with MPPI planning (the actual algorithm), and with its learned policy alone (no planning, same model).

**TD-MPC2 (MPPI)**

![TD-MPC2 with MPPI: flip, then drift near the start point](docs/tdmpc2_flip.gif)

**TD-MPC2 (no plan)**

![TD-MPC2 policy alone: flip, fly back towards the start point and hover](docs/tdmpc2_flip_no_plan.gif)

**PPO**

![PPO: flip, fly back to the start point and hover](docs/ppo_flip.gif)

![Evaluation return vs env steps](docs/learning_curves.png)



| | PPO | TD-MPC2 (MPPI) | TD-MPC2 (no plan) |
|---|---|---|---|
| Return | 203 | 94 | 165 |
| Final distance to start point | 0.07 m | 0.55 m | 0.32 m |
| Speed, last 0.5 s | 0.001 m/s | 1.5 m/s | 0.09 m/s |
| Crashes | 0/5 | 0/5 | 0/5 |
| Env steps to first reach return 150 | 475k | 20k | 30k |
| Training steps / wall-clock time | 1M / 21 min | 60k / 71 min | (same run) |
| Model size (trained parameters) | 40k | 331k | (same model) |
| Inference time per step, median (p99) | 0.2 ms (0.6 ms) | 18 ms (25 ms) | 0.5 ms (1.4 ms) |

TD-MPC2 learns the flip with about 20× fewer env steps, but its evaluation return swings a lot from one evaluation to the next. PPO needs far more env steps but less wall-clock time, and ends steadier: it hovers back at the start point.

Inference time is the wall-clock time to compute one action from one observation, on a laptop CPU (Intel i7-1065G7, PyTorch CPU build, batch size 1).

Regenerate: `python animate_rollout.py policy=ppo_flip save=docs/ppo_flip.gif` (same for `tdmpc2_flip`, plus `plan=false save=docs/tdmpc2_flip_no_plan.gif`) and `python plot_curves.py`.

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

## Checkpoints and fine-tuning

Training saves the model every `save_every` env steps (`configs/ppo.yaml`: 100k, `configs/tdmpc2.yaml`: 5k), not only at the end or on Ctrl-C. It writes the usual model file, so you can check progress while training continues, with the usual commands:

```bash
python animate_rollout.py policy=tdmpc2_flip plan=false     # latest checkpoint of a running TD-MPC2 flip run
python plot_curves.py                                       # learning curves so far (results/ is written every eval_every steps)
```

Fine tuning an existing model is supported for PPO:

```bash
python -m ppo.train_ppo task=flip name=ppo_flip_test # experiment: saves models/ppo_flip_test.zip, main model untouched
python -m ppo.train_ppo task=flip init_model=models/ppo_flip.zip steps=500000 env.reward.speed=0.3 # fine-tune: continue from a saved model with changed settings
```

## Usage

```bash
# PPO: train (steps from configs/ppo.yaml); saves models/ppo_<task>.zip. task is hover (default) or flip
python -m ppo.train_ppo steps=300000
python -m ppo.train_ppo task=flip
python -m ppo.train_ppo task=flip env.reward.smooth=0.3     # an experiment: any config value can be overridden

# watch training curves
tensorboard --logdir runs

# TD-MPC2: train hover / flip; saves models/tdmpc2_<task>.pt
# MPPI size and the rest are in configs/tdmpc2.yaml.
python -m tdmpc2.train_tdmpc2 task=flip

# learning curves: evaluation return vs env steps, from the logs both trainers write
python plot_curves.py

# watch a model: policy is zero, hover, random, scripted_hover, scripted_flip, or any model in models/ (ppo_*, tdmpc2_*)
python animate_rollout.py policy=ppo_hover
python animate_rollout.py policy=ppo_flip save=plots/flip.gif
```
