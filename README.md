# quadrotor-flip-rl

Teaching a simulated quadrotor to do a flip with reinforcement learning: one roll flip, then recover to level and stop rotating. Starting with PPO (Stable-Baselines3), and maybe trying TD-MPC2 later.

## Files

- `quadrotor.py`: quadrotor dynamics
- `flip_env.py`: Gymnasium flip environment
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
# train (default 3M steps); saves ppo_flip.zip
python train_ppo.py --steps 200000

# watch training curves
tensorboard --logdir runs

# roll out the saved policy and plot it to flip_eval.png
python train_ppo.py --eval

# animate a rollout: --policy is one of scripted, zero, hover, random, ppo
python visualize.py --policy ppo
python visualize.py --policy ppo --save flip.gif
```
