# Notation and Variable Glossary

[← Notes index](../book-summary.md) · [Module I →](01-processes-and-planning.md)

The book's own table is `book/notation/notation.md`. What follows is the domain-specific
vocabulary that recurs across chapters.

## States, processes, rewards

| Symbol | Meaning |
|---|---|
| S | State space (all states) |
| T | Terminal states; N = S - T is the **non-terminal** states |
| A | Action space |
| D | Set of reward values |
| `S_t, A_t, R_t` | State, action, reward random variables at time t (reward `R_{t+1}` accompanies the transition out of `S_t`) |
| P(s, s') | Transition probability, `Pr[S_{t+1}=s' \| S_t=s]` |
| `P_R(s, r, s')` | Joint transition/reward probability `Pr[(R_{t+1}=r, S_{t+1}=s') \| S_t=s]` — the primitive the code actually models |
| `R_T(s, s')` | Expected reward *conditional on the transition* s → s' |
| R(s) | Expected reward from s: `R(s) = Σ_{s'} P(s, s')R_T(s, s')` |
| γ | Discount factor, `γ ∈ [0, 1]` |
| `G_t` | Return: `G_t = Σ_{i=t+1}^∞ γ^{i-t-1} R_i = R_{t+1} + γ G_{t+1}` |
| m | \|N\|, the number of non-terminal states |

For MDPs the same symbols take an extra action argument: P(s, a, s'),
`P_R(s, a, r, s')`, R(s, a).

## Policies and value functions

| Symbol | Meaning |
|---|---|
| π(s, a) | Stochastic policy, `Pr[A_t = a \| S_t = s]` |
| `π_D(s)` | Deterministic policy (a function N → A) |
| Π | The set of all policies |
| `V^π, Q^π` | State- and action-value functions for a fixed π |
| `V^*, Q^*` | Optimal value functions; `π^*` / `π_D^*` an optimal policy |
| `A^π(s, a)` | Advantage, `Q^π(s, a) - V^π(s)` |
| `B^π` | Bellman Policy Operator |
| `B^*` | Bellman Optimality Operator |
| G(V) | Greedy-policy operator: maps a value function to the deterministic policy greedy w.r.t. it |
| W(s') | Value function extended to terminal states: V(s') if s' ∈ N, else 0 |
| `ρ^π(s)` | Discounted state-visitation measure, `Σ_{S_0} Σ_t γ^t p_0(S_0) p(S_0 → s, t, π)` |

## Approximation and learning

| Symbol | Meaning |
|---|---|
| w | Parameters of a function approximation; f(x;w) the parameterized function |
| φ(x) | Feature vector `(φ_1(x), ..., φ_m(x))`; Φ the n × m feature matrix |
| α | Learning rate; `α_n` a schedule; λ either the TD(λ) parameter or an L2 regularization coefficient (context disambiguates) |
| `δ_t` | TD error, `R_{t+1} + γ V(S_{t+1}) - V(S_t)` |
| `E_t` | Eligibility trace |
| ε | Exploration probability in ε-greedy |
| θ | Policy parameters (policy gradient); D the diagonal matrix of `μ_π`, the on-policy state distribution |

## Finance

| Symbol | Meaning |
|---|---|
| U(·) | Utility function; `x_{CE}` certainty-equivalent; `π_A, π_R` absolute/relative risk premia |
| A(x), R(x) | Absolute / relative risk-aversion; a the CARA coefficient, γ the CRRA coefficient |
| `W_t` | Wealth; `c_t` consumption rate; `π_t` the *fraction* of wealth in the risky asset (chapter 7 only — not a policy) |
| μ, σ, r | Risky-asset drift, volatility; riskless rate |
| `z_t` | Standard Brownian motion; `dz_t ~ N(0, dt)` |
| ρ | Utility discount rate (chapter 7 / HJB) |

---

[← Notes index](../book-summary.md) · [Module I →](01-processes-and-planning.md)
