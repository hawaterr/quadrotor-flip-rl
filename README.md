# quadrotor-flip-rl

Teaching a simulated quadrotor to do a flip with reinforcement learning: one roll flip, then recover to level and stop rotating. Starting with PPO (Stable-Baselines3), and maybe trying TD-MPC2 later.

**Current step: hover.** Before flipping, the agent learns to hold a fixed point from a perturbed start (random offset, velocity, tilt and body rates). Both tasks live in one env (`QuadEnv(task="hover" | "flip")`), which share dynamics and actions but have their own reward and termination.

## Files

- `quadrotor.py`: quadrotor dynamics
- `quad_env.py`: Gymnasium environment, `task="hover"` or `task="flip"`
- `train_ppo.py`: train and evaluate PPO
- `visualize.py`: animate a rollout

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

## Usage

```bash
# train (default 1M steps); saves ppo_<task>.zip. --task is hover (default) or flip
python train_ppo.py --steps 300000
python train_ppo.py --task flip

# watch training curves
tensorboard --logdir runs

# roll out the saved policy and plot it to <task>_eval.png
python train_ppo.py --eval
python train_ppo.py --task flip --eval

# animate a rollout: --policy is one of zero, hover, random, scripted_hover, scripted_flip, ppo_hover, ppo_flip
# (scripted_* are hand-written controllers, a baseline to compare PPO against; ppo_<task> loads ppo_<task>.zip)
python visualize.py --policy ppo_hover
python visualize.py --policy ppo_flip --save flip.gif
```
