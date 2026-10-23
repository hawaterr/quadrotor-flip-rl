# quadrotor-flip-rl

Teaching a simulated quadrotor to do a flip with reinforcement learning: one roll flip, then recover to level and stop rotating. Starting with PPO (Stable-Baselines3), and maybe trying TD-MPC2 later.

**Current step: hover.** Before flipping, the agent learns to hold a fixed point from a perturbed start (random offset, velocity, tilt and body rates). The flip env is kept in `flip_env.py` for later.

## Files

- `quadrotor.py`: quadrotor dynamics
- `hover_env.py`: Gymnasium hover environment (current)
- `flip_env.py`: Gymnasium flip environment (parked for now)
- `train_ppo.py`: train and evaluate PPO on the hover env
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
# train (default 1M steps); saves ppo_hover.zip
python train_ppo.py --steps 300000

# watch training curves
tensorboard --logdir runs

# roll out the saved policy and plot it to hover_eval.png
python train_ppo.py --eval

# animate a rollout: --policy is one of pd, zero, hover, random, ppo
# (pd is a hand-written PD controller, a baseline to compare PPO against)
python visualize.py --policy ppo
python visualize.py --policy ppo --save hover.gif
```
