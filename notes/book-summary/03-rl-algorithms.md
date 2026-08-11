# Module III — Reinforcement Learning Algorithms

[← Notes index](../book-summary.md) · [← Module II](02-financial-applications.md) · [Module IV →](04-finishing-touches.md)

The unifying signature: every algorithm takes an `Iterable` of experience plus a starting
`FunctionApprox` and returns an **infinite `Iterator` of improving approximations**. The caller
stops it (`rl/iterate.py`: `iterate`, `last`, `converge`, `converged`).

## Chapter 10 — MC and TD for Prediction

**Monte-Carlo.** Supervised learning on $(S_t, G_t)$ pairs:

$$\mathcal{L}_{(S_t,G_t)}(\boldsymbol{w}) = \tfrac{1}{2}(V(S_t;\boldsymbol{w}) - G_t)^2 \;\Longrightarrow\; \Delta\boldsymbol{w} = \alpha (G_t - V(S_t;\boldsymbol{w}))\nabla_{\boldsymbol{w}}V(S_t;\boldsymbol{w})$$

Tabular form $V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$; with $\alpha = 1/n$ this is
exactly the running mean, and with fixed $\alpha$ it is an exponentially-weighted mean
$V_n = \sum_j \alpha(1-\alpha)^{n-j}Y^{(j)}$.

**Temporal-Difference.** Replace the return with the bootstrapped estimate:

$$V(S_t) \leftarrow V(S_t) + \alpha \big( \underbrace{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}_{\delta_t} \big)$$
$$\Delta\boldsymbol{w} = \alpha\,(R_{t+1} + \gamma V(S_{t+1};\boldsymbol{w}) - V(S_t;\boldsymbol{w}))\,\nabla_{\boldsymbol{w}}V(S_t;\boldsymbol{w})$$

This is a **semi-gradient**: the target $R_{t+1} + \gamma V(S_{t+1};\boldsymbol{w})$ depends on $\boldsymbol{w}$
but is treated as a constant. It is not the gradient of any objective — the point chapter 12
returns to.

**Learning-rate schedule** (`learning_rate_schedule`), and the Robbins–Monro conditions
$\sum \alpha_n = \infty$, $\sum \alpha_n^2 < \infty$ required for convergence:

$$\alpha_n = \frac{\alpha}{1 + \left(\frac{n-1}{H}\right)^{\beta}}$$

**TD vs MC — the substantive comparison:**

| | MC | TD |
|---|---|---|
| Bias / variance | unbiased, high variance | biased, low variance |
| Episodes | needs complete, terminating episodes | works on continuing tasks, learns online |
| Markov property | doesn't exploit it | exploits it |
| Batch convergence | minimizes MSE against observed returns | converges to the VF of the **MLE MRP** fitted from the data, $\mathcal{P}_R(s,r,s') = \frac{\sum_i \mathbb{I}_{S_i=s,R_{i+1}=r,S_{i+1}=s'}}{\sum_i \mathbb{I}_{S_i=s}}$ |

Empirically (`rl/chapter10/random_walk_mrp.py`) TD reaches a low RMSE faster at a comparable
constant $\alpha$, and MC's error curve is visibly choppier.

**$n$-step bootstrapping and the $\lambda$-return.** Interpolating between the two:

$$G_{t,n} = \sum_{i=t+1}^{t+n}\gamma^{i-t-1}R_i + \gamma^n V(S_{t+n}), \qquad G_t^{(\lambda)} = (1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t,n} + \lambda^{T-t-1}G_t$$

$\lambda = 0$ is TD, $\lambda = 1$ is MC.

**Eligibility traces** turn the *forward view* (which needs the whole episode) into an online
*backward view*:

$$\boldsymbol{E}_t = \gamma\lambda\boldsymbol{E}_{t-1} + \nabla_{\boldsymbol{w}}V(S_t;\boldsymbol{w}), \qquad \Delta\boldsymbol{w} = \alpha\,\delta_t\,\boldsymbol{E}_t$$

The chapter proves the two views are equivalent in the offline, episode-summed sense:
$\sum_t \alpha \delta_t E_t(s) = \sum_t \alpha (G_t^{(\lambda)} - V(S_t))\mathbb{I}_{S_t=s}$.

> **Code:** `rl/monte_carlo.py` (`mc_prediction`), `rl/td.py` (`td_prediction`),
> `rl/td_lambda.py` (`lambda_return_prediction`, `td_lambda_prediction`), `rl/returns.py`.

## Chapter 11 — MC and TD for Control

Control = GPI with a *sampled* evaluation step. Since we no longer have $\mathcal{P}$, improvement
must be greedy w.r.t. $Q$, not $V$: $\pi_D'(s) = \arg\max_a Q^{\pi}(s,a)$.

**GLIE MC Control.** Tabular update on first/every visit:

$$Count(S_t,A_t) \mathrel{+}= 1, \qquad Q(S_t,A_t) \mathrel{+}= \frac{1}{Count(S_t,A_t)}(G_t - Q(S_t,A_t))$$

**GLIE** (Greedy in the Limit with Infinite Exploration) requires
$\lim_k Count_k(s,a) = \infty$ and $\lim_k \pi_k(s,a) = \mathbb{I}_{a = \arg\max_b Q(s,b)}$;
$\epsilon_k = 1/k$ satisfies both. Under GLIE + Robbins–Monro, $Q \to Q^*$.

**SARSA** (on-policy TD control):

$$\Delta\boldsymbol{w} = \alpha\,(R_{t+1} + \gamma Q(S_{t+1},A_{t+1};\boldsymbol{w}) - Q(S_t,A_t;\boldsymbol{w}))\,\nabla_{\boldsymbol{w}}Q(S_t,A_t;\boldsymbol{w})$$

with $n$-step and SARSA($\lambda$) variants built the same way as in chapter 10.

**Q-Learning** (off-policy TD control) — the target uses the max, not the taken action:

$$\delta_t = R_{t+1} + \gamma\max_{a}Q(S_{t+1},a;\boldsymbol{w}) - Q(S_t,A_t;\boldsymbol{w})$$

**Off-policy via importance sampling.** For behavior policy $\mu$ and target $\pi$,
$\mathbb{E}_{X\sim P}[f(X)] = \mathbb{E}_{X\sim Q}[\frac{P(X)}{Q(X)}f(X)]$. MC needs the full
trajectory ratio $\rho_t = \prod_{k=t}^{T-1}\frac{\pi(S_k,A_k)}{\mu(S_k,A_k)}$ (very high
variance); TD needs only the one-step ratio $\frac{\pi(S_t,A_t)}{\mu(S_t,A_t)}$. Q-Learning
avoids ratios entirely, which is why it is the practical off-policy method.

**Convergence.** The root cause of every failure below is that the semi-gradient TD update
"does not follow the gradient of *any* objective function".

*Prediction* (✓ = converges), including Gradient TD from chapter 12:

| Policy | Algorithm | Tabular | Linear | Non-Linear |
|---|---|:--:|:--:|:--:|
| On | MC | ✓ | ✓ | ✓ |
| On | TD / TD($\lambda$) | ✓ | ✓ | ✗ |
| On | **Gradient TD** | ✓ | ✓ | ✓ |
| Off | MC | ✓ | ✓ | ✓ |
| Off | TD / TD($\lambda$) | ✓ | ✗ | ✗ |
| Off | **Gradient TD** | ✓ | ✓ | ✓ |

*Control* — (✓) means it doesn't reach $V^*$ but bounces around near it:

| Algorithm | Tabular | Linear | Non-Linear |
|---|:--:|:--:|:--:|
| MC Control | ✓ | (✓) | ✗ |
| SARSA | ✓ | (✓) | ✗ |
| Q-Learning | ✓ | ✗ | ✗ |
| **Gradient Q-Learning** | ✓ | ✓ | ✗ |

**The Deadly Triad** — [bootstrapping, off-policy, function approximation]. Not a theorem, a rule
of thumb: when all three combine, expect divergence, so drop one. Function approximation is
non-negotiable at real-world scale, so the escape routes are a high $\lambda$ (mitigation),
Gradient TD (chapter 12), or DQN (chapter 12). Note the triad survives Gradient TD in the
*control* setting with non-linear FA.

> **Code:** `glie_mc_control` (`rl/monte_carlo.py`); `glie_sarsa`, `q_learning`
> (`rl/td.py`); comparison drivers in `rl/chapter11/control_utils.py`.

## Chapter 12 — Batch RL, Experience-Replay, DQN, LSPI, Gradient TD

**Batch RL.** Given a *fixed*, finite dataset, use it repeatedly rather than once:

$$\Delta\boldsymbol{w} = \alpha\cdot\frac{1}{n}\sum_{i=1}^n (R_i + \gamma V(S'_i;\boldsymbol{w}) - V(S_i;\boldsymbol{w}))\nabla_{\boldsymbol{w}}V(S_i;\boldsymbol{w})$$

**Experience-replay** samples uniformly from a growing memory, decorrelating updates
(`rl/experience_replay.py`).

**Least-Squares TD (LSTD).** With linear FA the semi-gradient equation is *linear in $\boldsymbol{w}$*,
so the fixed point can be solved directly instead of iterated:

$$\sum_i \boldsymbol{\phi}(S_i)\big(\boldsymbol{\phi}(S_i)^T\boldsymbol{w}^* - (R_i + \gamma\boldsymbol{\phi}(S'_i)^T\boldsymbol{w}^*)\big) = 0$$

> **Method:** accumulate $\boldsymbol{A} \mathrel{+}= \boldsymbol{\phi}(S_i)(\boldsymbol{\phi}(S_i) - \gamma\boldsymbol{\phi}(S'_i))^T$
> (an outer product) and $\boldsymbol{b} \mathrel{+}= \boldsymbol{\phi}(S_i)R_i$, then $\boldsymbol{w}^* = \boldsymbol{A}^{-1}\boldsymbol{b}$.
> No learning rate at all. Sherman–Morrison gives an incremental inverse.
> `least_squares_td` in `rl/td.py`. LSTD($\lambda$) replaces $\boldsymbol{\phi}(S_i)$ with $\boldsymbol{E}_{i,t}$.

**LSPI.** LSTDQ (the $Q$-version of LSTD, with features $\boldsymbol{\phi}(s,a)$) inside a policy
iteration loop — an off-policy, experience-reusing, learning-rate-free control algorithm.
`least_squares_tdq`, `least_squares_policy_iteration`.

**DQN.** Experience replay + a frozen target network $\boldsymbol{w}^-$:

$$\Delta\boldsymbol{w} = \alpha\sum_i \big(r_i + \gamma\max_{a'}Q(s'_i,a';\boldsymbol{w}^-) - Q(s_i,a_i;\boldsymbol{w})\big)\nabla_{\boldsymbol{w}}Q(s_i,a_i;\boldsymbol{w})$$

**Value Function Geometry.** The most conceptually dense section. Work in the $n$-dimensional
space of value functions with the $\boldsymbol{\mu_{\pi}}$-weighted norm
$d(\boldsymbol{V_1},\boldsymbol{V_2}) = (\boldsymbol{V_1}-\boldsymbol{V_2})^T\boldsymbol{D}(\boldsymbol{V_1}-\boldsymbol{V_2})$. Two operators act on it:

- $\boldsymbol{B}^{\pi}\cdot\boldsymbol{V} = \boldsymbol{\mathcal{R}}^{\pi} + \gamma\boldsymbol{\mathcal{P}}^{\pi}\boldsymbol{V}$ — the Bellman operator, whose fixed point $\boldsymbol{V}^{\pi}$ generally lies *outside* the representable subspace $\{\boldsymbol{\Phi}\boldsymbol{w}\}$;
- $\boldsymbol{\Pi_{\Phi}} = \boldsymbol{\Phi}(\boldsymbol{\Phi}^T\boldsymbol{D}\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\boldsymbol{D}$ — orthogonal projection *onto* that subspace.

Three different objectives, three different answers:

| Objective | Minimizes | Solution |
|---|---|---|
| **BE** (Bellman Error) | $d(\boldsymbol{B}^{\pi}\boldsymbol{V_w}, \boldsymbol{V_w})$ | $\boldsymbol{w}_{BE} = ((\boldsymbol{\Phi} - \gamma\boldsymbol{\mathcal{P}}^{\pi}\boldsymbol{\Phi})^T\boldsymbol{D}(\boldsymbol{\Phi} - \gamma\boldsymbol{\mathcal{P}}^{\pi}\boldsymbol{\Phi}))^{-1}(\boldsymbol{\Phi}-\gamma\boldsymbol{\mathcal{P}}^{\pi}\boldsymbol{\Phi})^T\boldsymbol{D}\boldsymbol{\mathcal{R}}^{\pi}$ |
| **TDE** (TD Error) | $\mathbb{E}[\delta^2]$ | naive residual gradient; converges to the wrong place |
| **PBE** (Projected Bellman Error) | $d(\boldsymbol{\Pi_{\Phi}}\boldsymbol{B}^{\pi}\boldsymbol{V_w}, \boldsymbol{V_w})$ | $\boldsymbol{w}_{PBE} = \boldsymbol{A}^{-1}\boldsymbol{b}$, $\boldsymbol{A} = \boldsymbol{\Phi}^T\boldsymbol{D}(\boldsymbol{\Phi} - \gamma\boldsymbol{\mathcal{P}}^{\pi}\boldsymbol{\Phi})$, $\boldsymbol{b} = \boldsymbol{\Phi}^T\boldsymbol{D}\boldsymbol{\mathcal{R}}^{\pi}$ |

$\boldsymbol{w}_{PBE}$ is exactly the LSTD solution, and the semi-gradient TD update is a stochastic
approximation of it — which explains both what TD converges to and why it can diverge off-policy.

**Gradient TD (TDC).** A *true* gradient of the PBE, using a second weight vector $\boldsymbol{\theta}$
to estimate $(\mathbb{E}[\boldsymbol{\phi}\boldsymbol{\phi}^T])^{-1}\mathbb{E}[\delta\boldsymbol{\phi}]$:

$$\Delta\boldsymbol{w} = \alpha\,\delta\,\boldsymbol{\phi}(s) - \alpha\gamma\,\boldsymbol{\phi}(s')(\boldsymbol{\phi}(s)^T\boldsymbol{\theta}), \qquad \Delta\boldsymbol{\theta} = \beta(\delta - \boldsymbol{\phi}(s)^T\boldsymbol{\theta})\boldsymbol{\phi}(s)$$

For *prediction* this converges in every cell of the table above, including off-policy with
non-linear FA. For *control*, Gradient Q-Learning converges with linear FA but still diverges
with non-linear FA — the triad is defused, not eliminated.

## Chapter 13 — Policy Gradient Algorithms

Parameterize the policy directly as $\pi(s,a;\boldsymbol{\theta})$ and ascend
$J(\boldsymbol{\theta}) = \mathbb{E}_{\pi}[\sum_t \gamma^t R_{t+1}]$. Motivation: continuous/large action
spaces, stochastic optimal policies, smoother convergence than $\arg\max$-based methods.

**Policy Gradient Theorem.** The central result — the gradient contains **no**
$\nabla_{\boldsymbol{\theta}}$ of the environment dynamics:

$$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}) = \sum_{s}\rho^{\pi}(s)\sum_{a}\nabla_{\boldsymbol{\theta}}\pi(s,a;\boldsymbol{\theta})\,Q^{\pi}(s,a) = \mathbb{E}_{s\sim\rho^{\pi}, a\sim\pi}\big[\nabla_{\boldsymbol{\theta}}\log\pi(s,a;\boldsymbol{\theta})\cdot Q^{\pi}(s,a)\big]$$

(Proved by unrolling $V^{\pi}(S_0)$ through the Bellman policy equation and collecting terms into
$\rho^{\pi}$.) The identity that makes it work is the **likelihood-ratio trick**
$\nabla_{\boldsymbol{\theta}}\pi = \pi\nabla_{\boldsymbol{\theta}}\log\pi$.

**Score functions** $\nabla_{\boldsymbol{\theta}}\log\pi(s,a;\boldsymbol{\theta})$ for the two canonical policies:

| Policy | Form | Score |
|---|---|---|
| Softmax | $\pi \propto e^{\boldsymbol{\phi}(s,a)^T\boldsymbol{\theta}}$ | $\boldsymbol{\phi}(s,a) - \mathbb{E}_{\pi}[\boldsymbol{\phi}(s,\cdot)]$ |
| Gaussian | $a \sim \mathcal{N}(\boldsymbol{\phi}(s)^T\boldsymbol{\theta}, \sigma^2)$ | $\dfrac{(a - \boldsymbol{\phi}(s)^T\boldsymbol{\theta})\boldsymbol{\phi}(s)}{\sigma^2}$ |

**The algorithm family**, each replacing $Q^{\pi}$ with a cheaper estimate:

| Algorithm | $\boldsymbol{\theta}$-update uses | Note |
|---|---|---|
| **REINFORCE** | $G_t$ | MC, unbiased, high variance |
| **Actor-Critic** | $Q(S_t,A_t;\boldsymbol{w})$ | critic learned by TD; biased |
| **with baseline** | $Q(S_t,A_t;\boldsymbol{w}) - B(S_t)$ | any $B(s)$ leaves the gradient unbiased, since $\sum_a \nabla_{\boldsymbol{\theta}}\pi(s,a) B(s) = B(s)\nabla_{\boldsymbol{\theta}}1 = 0$ |
| **Advantage AC** | $A = Q(s,a;\boldsymbol{w}) - V(s;\boldsymbol{v})$ | two critics |
| **TD-error AC** | $\delta = R_{t+1} + \gamma V(S_{t+1};\boldsymbol{v}) - V(S_t;\boldsymbol{v})$ | since $\mathbb{E}_{\pi}[\delta^{\pi}\mid s,a] = A^{\pi}(s,a)$, one critic suffices |

All carry the $\gamma^t$ factor: $\Delta\boldsymbol{\theta} = \alpha\gamma^t\nabla_{\boldsymbol{\theta}}\log\pi(S_t,A_t;\boldsymbol{\theta})\cdot(\ldots)$.

**Compatible Function Approximation Theorem.** If (1)
$\nabla_{\boldsymbol{w}}Q(s,a;\boldsymbol{w}^*) = \nabla_{\boldsymbol{\theta}}\log\pi(s,a;\boldsymbol{\theta})$ and (2) $\boldsymbol{w}^*$
minimizes the expected squared critic error, then substituting the critic for $Q^{\pi}$ introduces
**no bias**. So a linear critic on the score features is exactly compatible.

**Natural Policy Gradient.** Under the Fisher metric,
$\nabla^{nat}_{\boldsymbol{\theta}}J = \boldsymbol{FIM}^{-1}\nabla_{\boldsymbol{\theta}}J$; combined with the compatible-FA
result $\nabla_{\boldsymbol{\theta}}J = \boldsymbol{FIM}\cdot\boldsymbol{w}^*_{\theta}$, this collapses to the strikingly
simple $\nabla^{nat}_{\boldsymbol{\theta}}J(\boldsymbol{\theta}) = \boldsymbol{w}^*_{\theta}$, i.e.
$\Delta\boldsymbol{\theta} = \alpha_{\boldsymbol{\theta}}\boldsymbol{w}$.

**Deterministic Policy Gradient (DPG).**
$\nabla_{\boldsymbol{\theta}}J = \mathbb{E}_{s\sim\rho^{\pi_D}}[\nabla_{\boldsymbol{\theta}}\pi_D(s;\boldsymbol{\theta})\nabla_a Q^{\pi_D}(s,a)\big|_{a=\pi_D(s)}]$ — the chain rule through the action.

**Evolutionary Strategies.** Not RL at all — a black-box gradient estimate
$\frac{1}{\sigma}\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\boldsymbol{I})}[\boldsymbol{\epsilon}F(\boldsymbol{\theta}+\sigma\boldsymbol{\epsilon})]$,
included as a competitive, embarrassingly parallel alternative.

> **Code:** `rl/policy_gradient.py` — `reinforce_gaussian`, `actor_critic_gaussian`,
> `actor_critic_advantage_gaussian`, `actor_critic_td_error_gaussian`. Applied back to the
> asset-allocation problem of chapter 7 in `rl/chapter13/`.

---

[← Notes index](../book-summary.md) · [← Module II](02-financial-applications.md) · [Module IV →](04-finishing-touches.md)
