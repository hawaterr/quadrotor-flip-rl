#!/bin/bash
# Reproduce the README results: train both algorithms on 3 seeds, then the learning curves and the GIFs (seed 0).
# Run from anywhere, with the venv from the README's Setup active.
#
#   ./reproduce.sh                   # everything: ~1 h of PPO + ~10 h of TD-MPC2 on a laptop CPU
#   ./reproduce.sh --skip-training   # only the figure and GIFs, from the models already in models/
#
# OUT=<dir> writes the figure and GIFs there instead of docs/.
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(0 1 2)
OUT=${OUT:-docs}
export MPLBACKEND=Agg  # save the GIFs without opening a window

if [[ "${1:-}" != "--skip-training" ]]; then
  for i in "${SEEDS[@]}"; do
    python -m ppo.train_ppo train_seed="$i" name="ppo_flip_seed$i"            # ~23 min each
    python -m tdmpc2.train_tdmpc2 train_seed="$i" name="tdmpc2_flip_seed$i"   # ~3.4 h each
  done
fi

python plot_curves.py out="$OUT/learning_curves.png"
python animate_rollout.py policy=ppo_flip_seed0 plots=false save="$OUT/ppo_flip.gif"
python animate_rollout.py policy=tdmpc2_flip_seed0 plots=false save="$OUT/tdmpc2_flip.gif"
python animate_rollout.py policy=tdmpc2_flip_seed0 plan=false plots=false save="$OUT/tdmpc2_flip_no_plan.gif"
