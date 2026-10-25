# TD-MPC2: the 5 Q-functions and Q̄

Think of it as **5 pairs** of networks, 10 in total. All of them answer the same question: "starting in state z, taking action a, and then acting well, how much total reward will I get?"

## The 10 networks

```
     trained by gradient descent        slow EMA copies
     ──────────────────────────         ───────────────
pair 1:      Q₁          ──follows──▶       Q̄₁
pair 2:      Q₂          ──follows──▶       Q̄₂
pair 3:      Q₃          ──follows──▶       Q̄₃
pair 4:      Q₄          ──follows──▶       Q̄₄
pair 5:      Q₅          ──follows──▶       Q̄₅
```

- **Q₁…Q₅** are the live Q-functions, the ones that learn.
- **Q̄₁…Q̄₅** are their lagging shadows. Each Q̄ⱼ is a slowly moving average of its own partner Qⱼ, as if it were Qⱼ from roughly 100 updates ago.

All five live Q's are trained on the same data toward the same target. They still end up giving somewhat different answers, because each starts from different random weights. That disagreement is useful, as you'll see below.

## The four places Q is used

**1. Building the TD target (uses the Q̄'s)**

To create the target qₜ = rₜ + γ · Q̄(next state, policy's action):

- pick **2 of the 5 Q̄'s** at random,
- ask both for the value of the next state,
- take the **smaller** of the two answers.

This produces **one** target number per transition.

**2. Training the Q's (all 5 live Q's)**

Each of Q₁…Q₅ predicts the value of (zₜ, aₜ). Each is pushed toward that **same shared target**. All five learn at every update.

**3. Training the policy (2 live Q's)**

To judge the policy's proposed action:

- pick **2 of the 5 live Q's** at random,
- **average** their answers.

The policy changes itself to make this average higher. The Q's themselves are not updated here.

**4. Planning (2 live Q's)**

The planner imagines 3 steps ahead, then needs to know "how good is it to be here after those 3 steps?"

- pick **2 of the 5 live Q's** at random,
- **average** them,
- add that to the imagined rewards.

## Summary

| Purpose | Which networks | How many | Combine by |
|---|---|---|---|
| TD target | Q̄ (shadows) | 2 random | **min** |
| Value regression | Q (live) | all 5 | each trained separately |
| Policy update | Q (live) | 2 random | average |
| Planning bootstrap | Q (live) | 2 random | average |

## Why it's arranged this way

**Why shadows for the target?** The target should stay still while the live Q's chase it. If the target came from the live networks, it would shift every time they learned.

**Why min for the target?** Value estimates tend to come out too high. An error that randomly overshoots gets fed back into the target, then learned, then fed back again, so the overestimation compounds. Taking the more pessimistic of two independent opinions cancels much of that bias.

**Why average, not min, for policy and planning?** Here nothing is fed back into training targets, so the compounding problem doesn't arise. Using min would make the agent overly cautious. The average gives a fair estimate.

**Why 5 networks but only ever compare 2?** This is the trick from REDQ (Chen et al., 2021), cited in Section 3.1 of the paper. Comparing a random pair each time gives different pessimistic estimates on different updates, which spreads out the errors. Taking the minimum of all 5 every time would be too pessimistic. The paper's Q-function ablation (Figure 9) shows 5 works noticeably better than 2.
