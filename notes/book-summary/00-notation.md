# Notation and Variable Glossary

[← Notes index](../book-summary.md) · [Module I →](01-processes-and-planning.md)

The book's own table is `book/notation/notation.md`. What follows is the domain-specific
vocabulary that recurs across chapters.

## States, processes, rewards

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | State space (all states) |
| $\mathcal{T}$ | Terminal states; $\mathcal{N} = \mathcal{S} - \mathcal{T}$ is the **non-terminal** states |
| $\mathcal{A}$ | Action space |
| $\mathcal{D}$ | Set of reward values |
| $S_t, A_t, R_t$ | State, action, reward random variables at time $t$ (reward $R_{t+1}$ accompanies the transition out of $S_t$) |
| $\mathcal{P}(s,s')$ | Transition probability, $\mathbb{P}[S_{t+1}=s' \mid S_t=s]$ |
| $\mathcal{P}_R(s,r,s')$ | Joint transition/reward probability $\mathbb{P}[(R_{t+1}=r, S_{t+1}=s') \mid S_t=s]$ — the primitive the code actually models |
| $\mathcal{R}_T(s,s')$ | Expected reward *conditional on the transition* $s \to s'$ |
| $\mathcal{R}(s)$ | Expected reward from $s$: $\mathcal{R}(s) = \sum_{s'} \mathcal{P}(s,s')\,\mathcal{R}_T(s,s')$ |
| $\gamma$ | Discount factor, $\gamma \in [0,1]$ |
| $G_t$ | Return: $G_t = \sum_{i=t+1}^{\infty} \gamma^{i-t-1} R_i = R_{t+1} + \gamma G_{t+1}$ |
| $m$ | $\lvert\mathcal{N}\rvert$, the number of non-terminal states |

For MDPs the same symbols take an extra action argument: $\mathcal{P}(s,a,s')$,
$\mathcal{P}_R(s,a,r,s')$, $\mathcal{R}(s,a)$.

## Policies and value functions

| Symbol | Meaning |
|---|---|
| $\pi(s,a)$ | Stochastic policy, $\mathbb{P}[A_t = a \mid S_t = s]$ |
| $\pi_D(s)$ | Deterministic policy (a function $\mathcal{N} \to \mathcal{A}$) |
| $\Pi$ | The set of all policies |
| $V^{\pi}, Q^{\pi}$ | State- and action-value functions for a fixed $\pi$ |
| $V^*, Q^*$ | Optimal value functions; $\pi^*$ / $\pi_D^*$ an optimal policy |
| $A^{\pi}(s,a)$ | Advantage, $Q^{\pi}(s,a) - V^{\pi}(s)$ |
| $\boldsymbol{B}^{\pi}$ | Bellman Policy Operator |
| $\boldsymbol{B}^*$ | Bellman Optimality Operator |
| $G(\boldsymbol{V})$ | Greedy-policy operator: maps a value function to the deterministic policy greedy w.r.t. it |
| $\boldsymbol{W}(s')$ | Value function extended to terminal states: $\boldsymbol{V}(s')$ if $s' \in \mathcal{N}$, else $0$ |
| $\rho^{\pi}(s)$ | Discounted state-visitation measure, $\sum_{S_0} \sum_{t} \gamma^t p_0(S_0)\, p(S_0 \to s, t, \pi)$ |

## Approximation and learning

| Symbol | Meaning |
|---|---|
| $\boldsymbol{w}$ | Parameters of a function approximation; $f(x;\boldsymbol{w})$ the parameterized function |
| $\boldsymbol{\phi}(x)$ | Feature vector $(\phi_1(x),\ldots,\phi_m(x))$; $\boldsymbol{\Phi}$ the $n \times m$ feature matrix |
| $\alpha$ | Learning rate; $\alpha_n$ a schedule; $\lambda$ either the TD($\lambda$) parameter or an L2 regularization coefficient (context disambiguates) |
| $\delta_t$ | TD error, $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ |
| $\boldsymbol{E}_t$ | Eligibility trace |
| $\epsilon$ | Exploration probability in $\epsilon$-greedy |
| $\boldsymbol{\theta}$ | Policy parameters (policy gradient); $\boldsymbol{D}$ the diagonal matrix of $\boldsymbol{\mu_{\pi}}$, the on-policy state distribution |

## Finance

| Symbol | Meaning |
|---|---|
| $U(\cdot)$ | Utility function; $x_{CE}$ certainty-equivalent; $\pi_A, \pi_R$ absolute/relative risk premia |
| $A(x), R(x)$ | Absolute / relative risk-aversion; $a$ the CARA coefficient, $\gamma$ the CRRA coefficient |
| $W_t$ | Wealth; $c_t$ consumption rate; $\pi_t$ the *fraction* of wealth in the risky asset (chapter 7 only — not a policy) |
| $\mu, \sigma, r$ | Risky-asset drift, volatility; riskless rate |
| $z_t$ | Standard Brownian motion; $dz_t \sim \mathcal{N}(0, dt)$ |
| $\rho$ | Utility discount rate (chapter 7 / HJB) |

---

[← Notes index](../book-summary.md) · [Module I →](01-processes-and-planning.md)
