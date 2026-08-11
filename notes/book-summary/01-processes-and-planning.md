# Module I — Processes and Planning Algorithms

[← Notes index](../book-summary.md) · [← Notation and Variable Glossary](00-notation.md) · [Module II →](02-financial-applications.md)

## Chapter 2 — Markov Processes and Markov Reward Processes

**Markov Property.** The defining assumption of everything that follows:

$$\mathbb{P}[S_{t+1} \mid S_t, S_{t-1}, \ldots, S_0] = \mathbb{P}[S_{t+1} \mid S_t]$$

The chapter's stock-price examples show the real content of this: a non-Markov process becomes
Markov by *enlarging the state*. Process 2 tracks $(X_t, X_t - X_{t-1})$; process 3 tracks the
running counts $(U_t, D_t)$ of up- and down-moves.

**Stationary distribution.** $\pi(s') = \sum_{s} \pi(s)\mathcal{P}(s,s')$, i.e.
$\boldsymbol{\pi}^T = \boldsymbol{\pi}^T \cdot \boldsymbol{\mathcal{P}}$, equivalently
$\boldsymbol{\mathcal{P}}^T \cdot \boldsymbol{\pi} = \boldsymbol{\pi}$.

> **Method:** eigenvector of $\boldsymbol{\mathcal{P}}^T$ for eigenvalue 1, normalized to sum to 1.
> `FiniteMarkovProcess.get_stationary_distribution` in `rl/markov_process.py`.

**Bellman Equation for an MRP.** Expanding $V(s) = \mathbb{E}[G_t \mid S_t = s]$ and using the
Markov property:

$$V(s) = \mathcal{R}(s) + \gamma \sum_{s' \in \mathcal{N}} \mathcal{P}(s,s') \cdot V(s')$$

In matrix form, with $\boldsymbol{V}, \boldsymbol{\mathcal{R}} \in \mathbb{R}^m$ and $\boldsymbol{\mathcal{P}}$ the
$m \times m$ non-terminal transition matrix:

$$\boldsymbol{V} = \boldsymbol{\mathcal{R}} + \gamma \boldsymbol{\mathcal{P}} \cdot \boldsymbol{V} \quad\Longrightarrow\quad \boldsymbol{V} = (\boldsymbol{I_m} - \gamma \boldsymbol{\mathcal{P}})^{-1} \cdot \boldsymbol{\mathcal{R}}$$

> **Method:** direct linear solve, $O(m^3)$. This is the only exact, non-iterative solution in
> the book, and it is why the finite/tabular case matters pedagogically.
> `FiniteMarkovRewardProcess.get_value_function_vec`.

**Running example.** `SimpleInventoryMRP`: state $(\alpha, \beta)$ = (on-hand, on-order),
Poisson($\lambda$) demand, capacity $C$, holding cost $h$, stockout cost $p$. Reward
$-h\alpha - p\max(i - (\alpha+\beta), 0)$ for demand $i$. This example recurs through chapters
2, 3, 4, 10 and 11 as the ground truth against which learning algorithms are checked.

## Chapter 3 — Markov Decision Processes

**MDP → MRP collapse.** Fixing a policy $\pi$ reduces an MDP to an MRP:

$$\mathcal{P}_R^{\pi}(s,r,s') = \sum_{a} \pi(s,a)\,\mathcal{P}_R(s,a,r,s'), \qquad \mathcal{R}^{\pi}(s) = \sum_{a} \pi(s,a)\,\mathcal{R}(s,a)$$

This is the structural trick that lets all MRP machinery be reused for policy evaluation
(`MarkovDecisionProcess.apply_policy`).

**Bellman Policy Equations** (four interlocking forms):

$$V^{\pi}(s) = \sum_{a} \pi(s,a) \cdot Q^{\pi}(s,a)$$
$$Q^{\pi}(s,a) = \mathcal{R}(s,a) + \gamma \sum_{s' \in \mathcal{N}} \mathcal{P}(s,a,s') \cdot V^{\pi}(s')$$
$$V^{\pi}(s) = \sum_{a} \pi(s,a) \Big( \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') V^{\pi}(s') \Big)$$
$$Q^{\pi}(s,a) = \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') \sum_{a'} \pi(s',a')\, Q^{\pi}(s',a')$$

**Bellman Optimality Equations:**

$$V^*(s) = \max_{a} Q^*(s,a)$$
$$Q^*(s,a) = \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') \cdot V^*(s')$$
$$V^*(s) = \max_{a} \Big\{ \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') V^*(s') \Big\}$$
$$Q^*(s,a) = \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') \max_{a'} Q^*(s',a')$$

**Optimal policy extraction.** $\pi_D^*(s) = \arg\max_{a} Q^*(s,a)$.

The key theorem: for a discounted MDP with finite $\mathcal{S}$ and $\mathcal{A}$ there always
exists an optimal *deterministic* policy $\pi_D^*$ with $V^{\pi_D^*} = V^*$. Searching over
deterministic policies loses nothing.

> **Method:** the policy equations are linear in $V^{\pi}$ (solvable exactly, as in chapter 2);
> the optimality equations are **non-linear** because of the $\max$, and need iteration —
> which is what chapter 4 is about.

**Variants covered:** finite-horizon (§ chapter 4), continuous state/action, POMDPs (belief
state $b(h)_t$ as the sufficient statistic replacing the unobservable state).

## Chapter 4 — Dynamic Programming Algorithms

**Fixed-point theory (Banach).** If $f: \mathcal{X} \to \mathcal{X}$ is a contraction with
modulus $L < 1$ on a complete metric space, it has a unique fixed point $x^*$ and
$\lim_{i \to \infty} f^i(x_0) = x^*$ for *any* $x_0$, with $d(x^*, x_i) \le L^i/(1-L) \cdot d(x_1, x_0)$.
Every DP algorithm in the book is an instance of iterating a contraction.

**Bellman Policy Operator** $\boldsymbol{B}^{\pi}: \mathbb{R}^m \to \mathbb{R}^m$:

$$\boldsymbol{B}^{\pi}(\boldsymbol{V}) = \boldsymbol{\mathcal{R}}^{\pi} + \gamma \boldsymbol{\mathcal{P}}^{\pi} \cdot \boldsymbol{V}$$

It is a $\gamma$-contraction under the $L^{\infty}$ norm
$\lVert \boldsymbol{X} - \boldsymbol{Y}\rVert_{\infty} = \max_s \lvert (\boldsymbol{X}-\boldsymbol{Y})(s) \rvert$, and its unique
fixed point is $\boldsymbol{V}^{\pi}$.

> **Method (Policy Evaluation):** iterate $\boldsymbol{V_{i+1}} = \boldsymbol{B}^{\pi}(\boldsymbol{V_i})$ to convergence.
> `evaluate_mrp` / `evaluate_mrp_result` in `rl/dynamic_programming.py`.

**Greedy policy operator:**

$$G(\boldsymbol{V})(s) = \arg\max_{a} \Big\{ \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') \boldsymbol{V}(s') \Big\}$$

**Policy Improvement Theorem.** $\boldsymbol{V}^{G(\boldsymbol{V}^{\pi})} \geq \boldsymbol{V}^{\pi}$ componentwise. Proved
from $\boldsymbol{B}^{\pi_D'}(\boldsymbol{V}^{\pi}) \geq \boldsymbol{V}^{\pi}$ plus the monotonicity of $\boldsymbol{B}^{\pi}$ and
induction on repeated application.

> **Method (Policy Iteration):** alternate
> $\pi_{j+1} = G(\boldsymbol{V_j})$ and $\boldsymbol{V_{j+1}} = \lim_i (\boldsymbol{B}^{\pi_{j+1}})^i(\boldsymbol{V_j})$.
> Terminates finitely (finitely many deterministic policies). `policy_iteration`.

**Bellman Optimality Operator:**

$$\boldsymbol{B}^*(\boldsymbol{V})(s) = \max_{a} \Big\{ \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') \boldsymbol{V}(s') \Big\}$$

Also a $\gamma$-contraction — proved from two properties: **monotonicity**
($\boldsymbol{X} \geq \boldsymbol{Y} \Rightarrow \boldsymbol{B}^*(\boldsymbol{X}) \geq \boldsymbol{B}^*(\boldsymbol{Y})$) and **constant shift**
($\boldsymbol{B}^*(\boldsymbol{X} + c) = \boldsymbol{B}^*(\boldsymbol{X}) + \gamma c$). Fixed point is $\boldsymbol{V}^*$.

The bridge between the two operators: $\boldsymbol{B}^{G(\boldsymbol{V})}(\boldsymbol{V}) = \boldsymbol{B}^*(\boldsymbol{V})$ for all
$\boldsymbol{V}$ — "greedy improvement *is* the optimality operator".

> **Method (Value Iteration):** iterate $\boldsymbol{V_{i+1}} = \boldsymbol{B}^*(\boldsymbol{V_i})$, then extract
> $\pi^* = G(\boldsymbol{V}^*)$. `value_iteration` / `value_iteration_result`.

**Generalized Policy Iteration (GPI).** Policy Iteration and Value Iteration are the two
endpoints of a spectrum: any interleaving of *partial* evaluation with *partial* improvement
converges. This is the conceptual frame the whole rest of the book hangs on — MC Control and
SARSA are GPI with sampling in the evaluation step.

**Asynchronous DP.** In-place updates, prioritized sweeping (order states by Bellman error
$g(s) = \lvert V(s) - \max_a\{\ldots\}\rvert$), real-time DP (update only visited states).

**Finite-horizon / backward induction.** Wrap states as $(t, s_t)$ to make a non-stationary
problem stationary, then unwrap by time step:

$$V^*_t(s_t) = \max_{a_t} \Big\{ \sum_{s_{t+1}, r_{t+1}} (\mathcal{P}_R)_t(s_t,a_t,r_{t+1},s_{t+1}) \cdot (r_{t+1} + \gamma \cdot W^*_{t+1}(s_{t+1})) \Big\}$$

> **Method:** a single backward sweep $t = T-1, \ldots, 0$ — no iteration to a fixed point
> needed, because the horizon terminates. `rl/finite_horizon.py`
> (`finite_horizon_MDP`, `unwrap_finite_horizon_MDP`, `optimal_vf_and_policy`).

**Complexity, and why RL exists.** Exact DP is $O(m^2 \lvert\mathcal{A}\rvert)$ per sweep and
requires the model $\mathcal{P}_R$ *and* enumeration of $\mathcal{N}$. The two curses —
dimensionality (too many states) and modeling (no known $\mathcal{P}_R$) — motivate chapters 5
and 10 onward.

## Chapter 5 — Function Approximation and Approximate DP

**The `FunctionApprox` abstraction.** $f(x; \boldsymbol{w})$ models the conditional distribution
$\mathbb{P}[y \mid x]$; predictions are $\mathbb{E}_M[y \mid x]$. Fitting is maximum likelihood:

$$\boldsymbol{w}^* = \arg\max_{\boldsymbol{w}} \sum_{i=1}^n \log f(x_i; \boldsymbol{w})(y_i)$$

For the exponential family with the canonical link, the loss gradient collapses to the same
shape for every model in the book:

$$\nabla_{\boldsymbol{w}} Obj(x_i, y_i) = (\mathbb{E}_M[y \mid x_i] - y_i) \cdot \nabla_{\boldsymbol{w}} Out(x_i)$$

i.e. *prediction error times the gradient of the output*. Everything downstream (MC, TD,
policy gradient) is a special case.

**Linear function approximation.** $\mathbb{E}_M[y\mid x] = \boldsymbol{\phi}(x)^T \cdot \boldsymbol{w}$, with L2
regularization:

$$\mathcal{L}(\boldsymbol{w}) = \frac{1}{2n}\sum_{i=1}^n (\boldsymbol{\phi}(x_i)^T \boldsymbol{w} - y_i)^2 + \frac{\lambda}{2}\lvert \boldsymbol{w}\rvert^2$$

> **Two methods, both implemented:**
> - **Direct solve** (normal equations): $\boldsymbol{w}^* = (\boldsymbol{\Phi}^T \boldsymbol{\Phi} + n\lambda \boldsymbol{I_m})^{-1} \boldsymbol{\Phi}^T \boldsymbol{Y}$ — `LinearFunctionApprox.solve`.
> - **SGD:** $\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \alpha_t \mathcal{G}_{(x_t,y_t)}(\boldsymbol{w}_t)$ — `update`.

**Neural networks.** Backpropagation as the recursion
$\boldsymbol{P_l} = (\boldsymbol{w_{l+1}}^T \cdot \boldsymbol{P_{l+1}}) \circ g_l'(\boldsymbol{s_l})$ with
$\nabla_{\boldsymbol{w_l}}\mathcal{L} = \boldsymbol{P_l} \cdot \boldsymbol{i_l}^T$, and the clean result that with a
canonical link the output-layer gradient is just $P_L = (o_L - y)/d(\tau)$. Adam is the default
optimizer (`AdamGradient`).

**Tabular as a `FunctionApprox`.** `Tabular` is linear FA with indicator features and a
count-based learning rate $f(n) = 1/n$; this unifies tabular and approximate algorithms under
one interface, which is why the same `td_prediction` runs in both modes.

**Approximate DP.** Replace the exhaustive sweep over $\mathcal{N}$ with:
1. a *sample* of non-terminal states from a distribution $\mu$ (`NTStateDistribution`), and
2. a function approximation of the value function.

Gives approximate policy evaluation, approximate value iteration, and approximate
backward induction (`rl/approximate_dynamic_programming.py`:
`evaluate_mrp`, `value_iteration`, `backward_evaluate`, `back_opt_vf_and_policy`,
`back_opt_qvf`). Type aliases fixed here and used everywhere after:
`ValueFunctionApprox[S] = FunctionApprox[NonTerminal[S]]`,
`QValueFunctionApprox[S,A] = FunctionApprox[Tuple[NonTerminal[S], A]]`.

---

[← Notes index](../book-summary.md) · [← Notation and Variable Glossary](00-notation.md) · [Module II →](02-financial-applications.md)
