# Module I — Processes and Planning Algorithms

[← Notes index](../book-summary.md) · [← Notation and Variable Glossary](00-notation.md) · [Module II →](02-financial-applications.md)

## Chapter 2 — Markov Processes and Markov Reward Processes

**Markov Property.** The defining assumption of everything that follows:

```
Pr[S_{t+1}|S_t, S_{t-1}, ..., S_0] = Pr[S_{t+1}| S_t]
```

The chapter's stock-price examples show the real content of this: a non-Markov process becomes
Markov by *enlarging the state*. Process 2 tracks `(X_t, X_t - X_{t-1})`; process 3 tracks the
running counts `(U_t, D_t)` of up- and down-moves.

**Stationary distribution.** `π(s') = Σ_s π(s)P(s, s')`, i.e.
`π^T = π^T · P`, equivalently
`P^T · π = π`.

> **Method:** eigenvector of `P^T` for eigenvalue 1, normalized to sum to 1.
> `FiniteMarkovProcess.get_stationary_distribution` in `rl/markov_process.py`.

**Bellman Equation for an MRP.** Expanding `V(s) = E[G_t | S_t = s]` and using the
Markov property:

```
V(s) = R(s) + γ Σ_{s' ∈ N} P(s, s') · V(s')
```

In matrix form, with `V, R ∈ ℝ^m` and P the
m × m non-terminal transition matrix:

```
V = R + γ P · V ⟹ V = (I_m - γ P)^{-1} · R
```

> **Method:** direct linear solve, `O(m^3)`. This is the only exact, non-iterative solution in
> the book, and it is why the finite/tabular case matters pedagogically.
> `FiniteMarkovRewardProcess.get_value_function_vec`.

**Running example.** `SimpleInventoryMRP`: state (α, β) = (on-hand, on-order),
Poisson(λ) demand, capacity C, holding cost h, stockout cost p. Reward
-hα - p max(i - (α+β), 0) for demand i. This example recurs through chapters
2, 3, 4, 10 and 11 as the ground truth against which learning algorithms are checked.

## Chapter 3 — Markov Decision Processes

**MDP → MRP collapse.** Fixing a policy π reduces an MDP to an MRP:

```
P_R^π(s, r, s') = Σ_a π(s, a)P_R(s, a, r, s'), R^π(s) = Σ_a π(s, a)R(s, a)
```

This is the structural trick that lets all MRP machinery be reused for policy evaluation
(`MarkovDecisionProcess.apply_policy`).

**Bellman Policy Equations** (four interlocking forms):

```
V^π(s) = Σ_a π(s, a) · Q^π(s, a)
```
```
Q^π(s, a) = R(s, a) + γ Σ_{s' ∈ N} P(s, a, s') · V^π(s')
```
```
V^π(s) = Σ_a π(s, a) (R(s, a) + γ Σ_{s'} P(s, a, s') V^π(s'))
```
```
Q^π(s, a) = R(s, a) + γ Σ_{s'} P(s, a, s') Σ_{a'} π(s', a') Q^π(s', a')
```

**Bellman Optimality Equations:**

```
V^*(s) = max_a Q^*(s, a)
```
```
Q^*(s, a) = R(s, a) + γ Σ_{s'} P(s, a, s') · V^*(s')
```
```
V^*(s) = max_a { R(s, a) + γ Σ_{s'} P(s, a, s') V^*(s') }
```
```
Q^*(s, a) = R(s, a) + γ Σ_{s'} P(s, a, s') max_{a'} Q^*(s', a')
```

**Optimal policy extraction.** `π_D^*(s) = argmax_a Q^*(s, a)`.

The key theorem: for a discounted MDP with finite S and A there always
exists an optimal *deterministic* policy `π_D^*` with `V^{π_D^*} = V^*`. Searching over
deterministic policies loses nothing.

> **Method:** the policy equations are linear in `V^π` (solvable exactly, as in chapter 2);
> the optimality equations are **non-linear** because of the max, and need iteration —
> which is what chapter 4 is about.

**Variants covered:** finite-horizon (§ chapter 4), continuous state/action, POMDPs (belief
state `b(h)_t` as the sufficient statistic replacing the unobservable state).

## Chapter 4 — Dynamic Programming Algorithms

**Fixed-point theory (Banach).** If f: X → X is a contraction with
modulus L < 1 on a complete metric space, it has a unique fixed point `x^*` and
`lim_{i → ∞} f^i(x_0) = x^*` for *any* `x_0`, with `d(x^*, x_i) ≤ L^i/(1-L) · d(x_1, x_0)`.
Every DP algorithm in the book is an instance of iterating a contraction.

**Bellman Policy Operator** `B^π: ℝ^m → ℝ^m`:

```
B^π(V) = R^π + γ P^π · V
```

It is a γ-contraction under the `L^∞` norm
`‖X - Y‖_∞ = max_s|(X-Y)(s)|`, and its unique
fixed point is `V^π`.

> **Method (Policy Evaluation):** iterate `V_{i+1} = B^π(V_i)` to convergence.
> `evaluate_mrp` / `evaluate_mrp_result` in `rl/dynamic_programming.py`.

**Greedy policy operator:**

```
G(V)(s) = argmax_a { R(s, a) + γ Σ_{s'} P(s, a, s') V(s') }
```

**Policy Improvement Theorem.** `V^{G(V^π)} ≥ V^π` componentwise. Proved
from `B^{π_D'}(V^π) ≥ V^π` plus the monotonicity of `B^π` and
induction on repeated application.

> **Method (Policy Iteration):** alternate
> `π_{j+1} = G(V_j)` and `V_{j+1} = lim_i (B^{π_{j+1}})^i(V_j)`.
> Terminates finitely (finitely many deterministic policies). `policy_iteration`.

**Bellman Optimality Operator:**

```
B^*(V)(s) = max_a { R(s, a) + γ Σ_{s'} P(s, a, s') V(s') }
```

Also a γ-contraction — proved from two properties: **monotonicity**
(`X ≥ Y ⟹ B^*(X) ≥ B^*(Y)`) and **constant shift**
(`B^*(X + c) = B^*(X) + γ c`). Fixed point is `V^*`.

The bridge between the two operators: `B^{G(V)}(V) = B^*(V)` for all
V — "greedy improvement *is* the optimality operator".

> **Method (Value Iteration):** iterate `V_{i+1} = B^*(V_i)`, then extract
> `π^* = G(V^*)`. `value_iteration` / `value_iteration_result`.

**Generalized Policy Iteration (GPI).** Policy Iteration and Value Iteration are the two
endpoints of a spectrum: any interleaving of *partial* evaluation with *partial* improvement
converges. This is the conceptual frame the whole rest of the book hangs on — MC Control and
SARSA are GPI with sampling in the evaluation step.

**Asynchronous DP.** In-place updates, prioritized sweeping (order states by Bellman error
`g(s) =|V(s) - max_a{...}|`), real-time DP (update only visited states).

**Finite-horizon / backward induction.** Wrap states as `(t, s_t)` to make a non-stationary
problem stationary, then unwrap by time step:

```
V^*_t(s_t) = max_{a_t} { Σ_{s_{t+1}, r_{t+1}} (P_R)_t(s_t, a_t, r_{t+1}, s_{t+1}) · (r_{t+1} + γ · W^*_{t+1}(s_{t+1})) }
```

> **Method:** a single backward sweep t = T-1, ..., 0 — no iteration to a fixed point
> needed, because the horizon terminates. `rl/finite_horizon.py`
> (`finite_horizon_MDP`, `unwrap_finite_horizon_MDP`, `optimal_vf_and_policy`).

**Complexity, and why RL exists.** Exact DP is `O(m^2|A|)` per sweep and
requires the model `P_R` *and* enumeration of N. The two curses —
dimensionality (too many states) and modeling (no known `P_R`) — motivate chapters 5
and 10 onward.

## Chapter 5 — Function Approximation and Approximate DP

**The `FunctionApprox` abstraction.** f(x; w) models the conditional distribution
`Pr[y | x]`; predictions are `E_M[y | x]`. Fitting is maximum likelihood:

```
w^* = argmax_w Σ_{i=1}^n log f(x_i; w)(y_i)
```

For the exponential family with the canonical link, the loss gradient collapses to the same
shape for every model in the book:

```
∇_w Obj(x_i, y_i) = (E_M[y | x_i] - y_i) · ∇_w Out(x_i)
```

i.e. *prediction error times the gradient of the output*. Everything downstream (MC, TD,
policy gradient) is a special case.

**Linear function approximation.** `E_M[y | x] = φ(x)^T · w`, with L2
regularization:

```
L(w) = 1/2nΣ_{i=1}^n (φ(x_i)^T w - y_i)^2 + λ/2|w|^2
```

> **Two methods, both implemented:**
> - **Direct solve** (normal equations): `w^* = (Φ^T Φ + nλ I_m)^{-1} Φ^T Y` — `LinearFunctionApprox.solve`.
> - **SGD:** `w_{t+1} = w_t - α_t G_{(x_t, y_t)}(w_t)` — `update`.

**Neural networks.** Backpropagation as the recursion
`P_l = (w_{l+1}^T · P_{l+1}) ∘ g_l'(s_l)` with
`∇_{w_l}L = P_l · i_l^T`, and the clean result that with a
canonical link the output-layer gradient is just `P_L = (o_L - y)/d(τ)`. Adam is the default
optimizer (`AdamGradient`).

**Tabular as a `FunctionApprox`.** `Tabular` is linear FA with indicator features and a
count-based learning rate f(n) = 1/n; this unifies tabular and approximate algorithms under
one interface, which is why the same `td_prediction` runs in both modes.

**Approximate DP.** Replace the exhaustive sweep over N with:
1. a *sample* of non-terminal states from a distribution μ (`NTStateDistribution`), and
2. a function approximation of the value function.

Gives approximate policy evaluation, approximate value iteration, and approximate
backward induction (`rl/approximate_dynamic_programming.py`:
`evaluate_mrp`, `value_iteration`, `backward_evaluate`, `back_opt_vf_and_policy`,
`back_opt_qvf`). Type aliases fixed here and used everywhere after:
`ValueFunctionApprox[S] = FunctionApprox[NonTerminal[S]]`,
`QValueFunctionApprox[S,A] = FunctionApprox[Tuple[NonTerminal[S], A]]`.

---

[← Notes index](../book-summary.md) · [← Notation and Variable Glossary](00-notation.md) · [Module II →](02-financial-applications.md)
