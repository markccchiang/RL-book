# Module II — Modeling Financial Applications

[← Notes index](../book-summary.md) · [← Module I](01-processes-and-planning.md) · [Module III →](03-rl-algorithms.md)

## Chapter 6 — Utility Theory

```
x_{CE} = U^{-1}(E[U(x)]), π_A = E[x] - x_{CE}, π_R = π_A / E[x]
```

Second-order Taylor expansion gives the risk premium in terms of curvature:

```
π_A ≈ 1/2 A(x̄)σ_x^2, A(x) = -(U''(x))/(U'(x)), R(x) = -(U''(x)x)/(U'(x))
```

| Family | Utility | Risk aversion | Optimal risky allocation |
|---|---|---|---|
| **CARA** | `U(x) = (1 - e^{-ax})/a` | A(x) = a | `π^* = (μ - r)/aσ^2` |
| **CRRA** | `U(x) = (x^{1-γ} - 1)/(1-γ)` | R(x) = γ | `π^* = (μ - r)/γσ^2` |

(Both limits: a → 0 gives U(x) = x; γ → 1 gives U(x) = log x.)

For CARA with `x ~ N(μ, σ^2)`: `x_{CE} = μ - aσ^2/2`, so
`π_A = aσ^2/2` exactly. This closed form (via the Gaussian MGF, appendix 1) is what makes
the CARA cases of chapters 7, 8 and 9 analytically solvable.

## Chapter 7 — Dynamic Asset-Allocation and Consumption (Merton's Problem)

**Setup.** Risky asset `dS_t = μ S_t dt + σ S_t dz_t`, riskless `dR_t = rR_t dt`, CRRA
utility, horizon T. Choose fraction `π_t` and consumption rate `c_t` to maximize

```
E[∫_t^T (e^{-ρ(s-t)} c_s^{1-γ})/(1-γ)ds + (e^{-ρ(T-t)}ε^γ W_T^{1-γ})/(1-γ) | W_t]
```

subject to `dW_t = ((r + π_t(μ - r))W_t - c_t)dt + π_t σ W_tdz_t`.

**Solution method** — the book's flagship worked derivation:

1. Write the **HJB equation** (appendix 4) and expand `dV^*` with **Itô's Lemma** (appendix 3).
2. Take `∂/∂π_t` and `∂/∂c_t` of the resulting expression and set to zero:
   ```
   π_t^* = (-∂V^*/∂W_t · (μ - r))/(∂^2 V^*/∂W_t^2 · σ^2 W_t), c_t^* = ((∂V^*)/(∂W_t))^{-1/γ}
   ```
3. Substitute back to get a non-linear PDE in `V^*`.
4. **Guess** `V^*(t, W_t) = f(t)^γ(W_t^{1-γ})/(1-γ)`; the PDE reduces to the ODE f'(t) = ν f(t) - 1 with `ν = (ρ - (1-γ)(((μ-r)^2)/2σ^2γ + r))/γ`.
5. Solve the ODE with terminal condition:
   ```
   f(t) = (1 + (νε - 1)e^{-ν(T-t)})/ν (ν ≠ 0), f(t) = T - t + ε (ν = 0)
   ```

**Results.** `π^*(t, W_t) = (μ - r)/σ^2γ` — *constant*, independent of both
time and wealth — and `c^*(t, W_t) = W_t / f(t)`.

**Discrete-time analogue** (§ *A Discrete-Time Asset-Allocation Example*), CARA utility, single
risky asset with `Y_t ~ N(μ, σ^2)`, `W_{t+1} = x_t(Y_t - r) + W_t(1+r)`:

```
V^*_t(W_t) = -b_t e^{-c_t W_t}, b_t = (e^{-((μ-r)^2(T-t))/2σ^2})/a, c_t = a(1+r)^{T-t}
```
```
x^*_t = (μ - r)/(σ^2 a (1+r)^{T-t-1})
```

> **Method:** backward induction on the Bellman optimality equation, with the Gaussian MGF used
> to evaluate `E[e^{-cW_{t+1}}]` in closed form at each step.

The chapter then *re-solves the same problem* by ADP with the feature set
`φ = (1, W_t, x_t, x_t^2)` for `Q^*_t` — recovering the analytic answer numerically. This
"known answer, then learn it" pattern is the book's standard validation technique
(`rl/chapter7/`).

## Chapter 8 — Derivatives Pricing and Hedging

**Single-period setting.** m+1 assets, n states of the world Ω, real-world measure
μ, portfolio θ.

**Risk-neutral measure** π: any measure equivalent to μ (same null sets) under which
every asset is a discounted martingale,
`S_j^{(0)} = 1/(1+r)Σ_i π(ω_i) S_j^{(i)}`. It follows immediately that *every*
portfolio satisfies `V_θ^{(0)} = 1/(1+r)Σ_i π(ω_i) V_θ^{(i)}`.

- **1st FTAP:** no arbitrage ⟺ a risk-neutral measure exists.
- **2nd FTAP:** the market is complete (every derivative is replicable) ⟺ the risk-neutral measure is *unique*.

**Two pricing routes, and they agree in a complete market:**

```
V_D^{(0)} = Σ_j θ_j S_j^{(0)} (replication) and V_D^{(0)} = 1/(1+r)Σ_i π(ω_i) V_D^{(i)} (risk-neutral expectation)
```

**Incomplete markets.** The price is no longer unique; it lies in the interval between the
super-hedging and sub-hedging bounds, obtained as a linear program whose dual is exactly the
sup/inf of discounted expectation over the set of risk-neutral measures:

```
SP = sup_πΣ_i (π(ω_i))/(1+r)V_D^{(i)}, SB = ∈f_πΣ_i (π(ω_i))/(1+r)V_D^{(i)}
```

To pick a single price, use **utility indifference**: solve
`max_θ Σ_i μ(ω_i) U(V_D^{(i)} + Σ_j θ_j S_j^{(i)})` with and without the
derivative and equate. For CARA + Gaussian this is closed-form
(`α^* = (μ - (1+r)S_0)/aσ^2`); otherwise it is cast as an MDP.

**Multi-period / American options.** Under the risk-neutral measure the underlying follows
`dS_t = rS_t dt + σ S_t dz_t` — note the drift is r, not μ. Optimal exercise is an
**optimal stopping** problem:

```
V^*(X_t) = max(H(X_t), E[V^*(X_{t+1}) | X_t])
```

with payoff H. Cast as a finite MDP over a **binomial tree**: with n steps to expiry
T, `u = e^{σ√(T/n)}` and risk-neutral up-probability
`q = (e^{rT/n + σ√(T/n)} - 1)/(e^{2σ√(T/n)} - 1)`, node prices
`S_{i, j} = S_{0, 0}e^{(2j-i)σ√(T/n)}`.

> **Methods:** backward induction on the tree (chapter 8), then LSPI and deep Q-learning on the
> same problem in chapter 12 — the direct comparison of a DP answer against an RL answer.
> `rl/chapter8/`, `rl/chapter12/optimal_exercise_rl.py`.

## Chapter 9 — Order-Book Trading Algorithms

**Order book dynamics.** Bids `[(P_i^{(b)}, N_i^{(b)})]` descending, asks
`[(P_i^{(a)}, N_i^{(a)})]` ascending; the chapter gives the exact set arithmetic for how a limit
or market order removes and adds levels, and the resulting proceeds.

**Optimal execution.** Sell N shares over T steps; `R_t` shares remaining,
`R_{t+1} = R_t - N_t`, `R_T = 0`. Price dynamics with permanent impact α and temporary
impact β: `P_{t+1} = P_t - α N_t + ε_t`, execution price
`Q_t = P_t - β N_t`. Maximize `E[Σ_t γ^t U(N_t Q_t)]`.

> **Method:** backward induction. With a linear price-impact model and risk-neutral utility the
> value function stays quadratic, giving the closed form
> ```
> N_t^* = R_t/(T-t), V_t^*((P_t, R_t)) = R_t P_t - R_t^2/2((2β + α(T-t-1))/(T-t))
> ```
> — i.e. **uniform (TWAP) execution is optimal** under these assumptions. Adding a mean-reverting
> signal `X_{t+1} = ρ X_t + η_t` perturbs it to
> `N_t^* = R_t/(T-t) + h(t, β, θ, ρ)· X_t`. Beyond these tractable cases: RL.

**Optimal market-making (Avellaneda–Stoikov).** State `(t, S_t, W_t, I_t)` — time, OB mid,
trading PnL, inventory. Hits arrive as Poisson processes whose intensities depend on the spread:
`λ_t^{(b)} = f^{(b)}(δ_t^{(b)})`. Maximize `E[U(W_T + I_T S_T)]` with CARA.

> **Method:** HJB, then the substitution `V^* = -e^{-γ(W_t + θ(t, S_t, I_t))}` to strip out
> wealth, then an expansion `θ ≈ θ^{(0)} + I_tθ^{(1)} + I_t^2/2θ^{(2)}`
> which decouples the PDE into three solvable ones.

With exponential intensity `f(δ) = ce^{-kδ}`, the closed-form solution:

```
Optimal bid-ask spread δ_t^{(b)*} + δ_t^{(a)*} = γσ^2(T-t) + 2/γlog(1 + γ/k)
```
```
Optimal pseudo-mid Q_t^{(m)} = S_t - I_t γ σ^2 (T-t)
```

The pseudo-mid is skewed *away* from the mid in proportion to inventory — the mechanism by
which a market maker manages inventory risk.

---

[← Notes index](../book-summary.md) · [← Module I](01-processes-and-planning.md) · [Module III →](03-rl-algorithms.md)
