# quadrotor-flip-rl

Teaching a simulated quadrotor to do a flip with reinforcement learning: one roll flip, then recover to level and stop rotating. PPO (Stable-Baselines3) does the flip; now trying a minimal single-task TD-MPC2 on the same env.

**Current step: hover.** Before flipping, the agent learns to hold a fixed point from a perturbed start (random offset, velocity, tilt and body rates). Both tasks live in one env (`QuadEnv(task="hover" | "flip")`), which share dynamics and actions but have their own reward and termination.

## Files

- `quadrotor.py`: quadrotor dynamics
- `quad_env.py`: Gymnasium environment, `task="hover"` or `task="flip"`
- `train_ppo.py`: train and evaluate PPO
- `tdmpc2.py`: minimal single-task TD-MPC2 (world model, MPPI planner, ~1M params), CPU friendly
- `train_tdmpc2.py`: train and evaluate TD-MPC2 (`--task pendulum` for a quick sanity check)
- `eval_plot.py`: roll out one episode with any policy and plot it
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

# TD-MPC2: sanity check on Pendulum-v1 first, then hover / flip; saves tdmpc2_<task>.pt
# It acts every 0.02 s (--ctrl-dt) so its 3-step plan sees further ahead. On CPU it is slow per step,
# so --samples/--iterations shrink MPPI and --plan-after N acts with the policy prior (fast) until step N.
python train_tdmpc2.py --task pendulum --steps 15000 --samples 256 --iterations 4
python train_tdmpc2.py --task flip --samples 256 --iterations 4
python train_tdmpc2.py --task flip --eval          # add --no-plan to roll out the policy prior alone

# animate a rollout: --policy is one of zero, hover, random, scripted_hover, scripted_flip, ppo_hover, ppo_flip, tdmpc2_hover, tdmpc2_flip
# (scripted_* are hand-written controllers, a baseline to compare PPO against; ppo_<task> loads ppo_<task>.zip)
python visualize.py --policy ppo_hover
python visualize.py --policy ppo_flip --save flip.gif
```
