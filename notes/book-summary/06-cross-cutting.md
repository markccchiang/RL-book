# Cross-Cutting Reference

[← Notes index](../book-summary.md) · [← Appendices](05-appendices.md)

## Which method solves which equation

| Equation | Structure | Method | Where |
|---|---|---|---|
| MRP Bellman | linear | matrix inverse $(\boldsymbol{I}-\gamma\boldsymbol{\mathcal{P}})^{-1}\boldsymbol{\mathcal{R}}$ | `markov_process.py` |
| Stationary distribution | eigenproblem | eigenvector of $\boldsymbol{\mathcal{P}}^T$ at $\lambda=1$ | `markov_process.py` |
| Bellman Policy Eq. | linear | iterate $\boldsymbol{B}^{\pi}$ (contraction) | `dynamic_programming.evaluate_mrp` |
| Bellman Optimality Eq. | non-linear ($\max$) | iterate $\boldsymbol{B}^*$, or Policy Iteration | `value_iteration`, `policy_iteration` |
| Finite-horizon Bellman | non-linear, acyclic | one backward sweep | `finite_horizon.py` |
| Same, large state space | intractable | ADP: sample states + function approx | `approximate_dynamic_programming.py` |
| Same, no model | unknown $\mathcal{P}_R$ | MC / TD / SARSA / Q-learning | `monte_carlo.py`, `td.py` |
| Linear FA normal equations | linear | direct solve or SGD | `function_approx.py` |
| PBE fixed point | linear in $\boldsymbol{w}$ | LSTD: accumulate $\boldsymbol{A},\boldsymbol{b}$; $\boldsymbol{w}=\boldsymbol{A}^{-1}\boldsymbol{b}$ | `td.least_squares_td` |
| PBE, off-policy | needs true gradient | Gradient TD / TDC | chapter 12 |
| $\nabla_{\boldsymbol{\theta}}J$ | expectation | Policy Gradient Theorem + sampling | `policy_gradient.py` |
| HJB PDE | non-linear PDE | guess the functional form → ODE | chapters 7, 9 |
| Optimal stopping | recursive max | backward induction on a tree, or LSPI/DQN | chapters 8, 12 |
| Efficient frontier | constrained QP | Lagrangian | appendix 2 |
| Black–Scholes PDE | linear PDE | change of variables → heat equation | appendix 5 |

## Recurring examples

| Example | Introduced | Reused in |
|---|---|---|
| `SimpleInventory` (MP → MRP → MDP) | ch. 2–3 | ch. 4 (DP), ch. 10 (prediction), ch. 11 (control) |
| Random-walk MRP | ch. 10 | ch. 12 (LSTD) |
| Asset allocation (CARA, discrete) | ch. 7 | ch. 13 (policy gradient) |
| American option exercise | ch. 8 (binomial DP) | ch. 12 (LSPI, DQN) |
| Windy Grid | ch. 11 | — |

## Library map

| Layer | Module | Role |
|---|---|---|
| 1 | `distribution.py` | `Distribution[A]` (`sample`) vs `FiniteDistribution` (`table`, exact `expectation`) |
| 2 | `markov_process.py` | `State = NonTerminal \| Terminal`; MP → MRP → finite variants |
| 3 | `markov_decision_process.py`, `policy.py` | `MarkovDecisionProcess`, `apply_policy` collapsing MDP+policy → MRP |
| 4 | `dynamic_programming.py`, `finite_horizon.py` | exact tabular PE / PI / VI / backward induction |
| 5 | `function_approx.py` | `Tabular`, `Dynamic`, `LinearFunctionApprox`, `DNNApprox` — immutable, gradient-based |
| 5 | `approximate_dynamic_programming.py` | ADP + the `ValueFunctionApprox` / `QValueFunctionApprox` aliases |
| 6 | `monte_carlo.py`, `td.py`, `td_lambda.py`, `policy_gradient.py`, `experience_replay.py`, `returns.py` | the learning algorithms |
| — | `iterate.py` | `iterate`, `last`, `converge`, `converged`, `accumulate` — how callers stop infinite iterators |

---

[← Notes index](../book-summary.md) · [← Appendices](05-appendices.md)
