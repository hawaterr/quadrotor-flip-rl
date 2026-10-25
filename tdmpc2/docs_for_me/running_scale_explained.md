# TD-MPC2: RunningScale

## When it's used

Only in **place 3: training the policy** (see `q_functions_explained.md`). Never in the other three.

## What the policy is trying to do

Two live Q's judge the policy's action, and their answers are averaged. The policy wants that number to be **high**.

The policy also has a second, smaller goal: **stay a bit random**, so it keeps exploring. This is the entropy bonus, with a small fixed weight.

So the policy balances two things: a high Q, and staying a bit random.

## The problem

How big Q is depends on the task. One task might give Q-values around 1, another around 1000.

- If Q is around 1000, it overwhelms "stay random", and the policy stops exploring too early.
- If Q is very small, "stay random" wins, and the policy never commits.

The weight on "stay random" is fixed, so the balance would depend on the task.

## What RunningScale does

It **shrinks the Q-values to a standard size** before the policy compares them with "stay random".

1. It keeps **one number**: a running estimate of how spread out the Q-values usually are.
2. Each time, it looks at the current Q-values, measures their spread, and moves its number a little (1%) toward it.
3. It divides the Q-values by that number. If the number is below 1, it divides by 1 instead, so values are never made larger.

After this, Q-values sit around 1 on every task, so the balance with "stay random" is always the same.

## What it doesn't change

Every Q-value is divided by the same number, so the best action is still the best action. RunningScale only sets how strongly "pick the best action" weighs against "stay a bit random".
