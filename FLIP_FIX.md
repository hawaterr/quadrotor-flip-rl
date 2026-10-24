# Flip fix: why PPO only flew straight up, and what changed

## The problem

Training on the flip task stalled after about 50k steps. Episodes reached the full 200 steps, the reward flattened at about 2, and the policy just climbed.

Returns under the old flip reward:

| behaviour | return |
|---|---|
| full thrust straight up (ends at z = 60 m) | -0.01 |
| hover thrust | -1.42 |
| roll at max rate without climbing first (crashes) | -34.5 |
| scripted flip (climb, flip, PD recovery) | +37.7 |

A good flip existed, but PPO never found it:

1. **Climbing was free.** The reward only penalised dropping *below* the start height, and the flip task had no distance limit. Climbing forever was a safe return of about 0. It also pushed the position-error observation to about 28, far outside what the network was trained on.
2. **Trying to roll was punished hard.** Rolling without climbing first loses height and crashes, so the policy quickly learned not to roll.
3. **It almost never saw a full roll.** A flip needs about 0.4 s of steady full roll rate. Random exploration around zero averages out, and with `ent_coef=0` the exploration noise shrank over time.
4. **The flip reward was weak.** Summed `w_x * dt` over one revolution is only 2π ≈ 6.3, and the +10 bonus came only at 95% of a revolution, which was never reached.

## What changed

### `quad_env.py` (flip task only; hover is unchanged)

- **Distance limit.** The episode ends more than 6 m from the start point. The new per-task `MAX_DIST` keeps hover at 3 m.
- **Distance penalty with a free zone.** The first 3 m are free, then each extra metre costs 0.5 per step. Penalising distance from 0 m didn't work: a good flip climbs first and ends about 3.5 m up, so it scored worse than doing nothing.
- **Reward only for new roll progress.** The agent earns `3.0 × (gain in clip(roll_progress, 0, 2π))` over the furthest point reached so far. Rolling back and forth earns nothing. The +10 completion bonus stays.
- **Mid-flip starts (`mid_flip_prob`).** This share of episodes starts already rolling at a random angle, at about 10 rad/s with some upward speed. The agent practises finishing a flip and recovering before it can do a whole flip itself. It defaults to 0, so evaluation and `visualize.py` always start from rest.

### `train_ppo.py`

- For `--task flip` only: `mid_flip_prob=0.5` and `ent_coef=0.005`, so it keeps exploring. Hover training is unchanged.