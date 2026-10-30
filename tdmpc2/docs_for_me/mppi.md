# MPPI

Suppose you have a nominal control sequence u₀, u₁, …, u_{T−1} over a horizon of T steps. This is typically last iteration's plan, shifted forward one step.

1. **Sample perturbations.** Draw K noise sequences (often thousands) from a Gaussian:
   εₖ,ₜ ~ N(0, Σ)
2. **Roll out.** For each sample k, simulate the system forward using controls uₜ + εₖ,ₜ:
   xₜ₊₁ = f(xₜ, uₜ + εₖ,ₜ)
3. **Compute costs.** Sum running costs plus a terminal cost for each rollout to get Sₖ. There is also a small control-cost term that comes out of the theory.
4. **Weight the samples.** Use exponentiated costs:
   wₖ = exp(−(Sₖ − S_min) / λ) / η
   Here η normalizes the weights so they sum to 1. Subtracting S_min is a numerical-stability trick.
5. **Update the plan.**
   uₜ ← uₜ + Σₖ wₖ · εₖ,ₜ
6. **Act and shift.** Send u₀ to the robot, shift the sequence left by one, initialize the last entry, and repeat at the next control step.

## Updating σ

- **Vanilla MPPI:** updates only the mean. σ is a fixed hyperparameter you choose by hand.
- **TD-MPC2:** updates both. The new σ is the weighted spread of the elites, so the search automatically narrows as it becomes confident and stays wide when the good samples disagree.

How TD-MPC2 updates σ, at each of its planning iterations:

1. Keep the top-k samples (the elites) by score. Call their action sequences aᵢ and their weights wᵢ, computed as in step 4 but over the elites only.
2. Update the mean as the weighted average of the elites:
   μₜ ← Σᵢ wᵢ · aᵢ,ₜ
3. Update σ as the weighted standard deviation of the elites around that new mean, per timestep and action dimension:
   σₜ ← √( Σᵢ wᵢ · (aᵢ,ₜ − μₜ)² )
4. Clamp σ to a range [σ_min, σ_max], about [0.05, 2]. This stops it from collapsing to zero or blowing up.

## Note

σ is reset to σ_max at the start of each new control step.
To accelerate convergence of planning, a fraction of action sequences originate from the policy prior p.