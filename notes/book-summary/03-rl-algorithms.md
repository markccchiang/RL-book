# Module III — Reinforcement Learning Algorithms

[← Notes index](../book-summary.md) · [← Module II](02-financial-applications.md) · [Module IV →](04-finishing-touches.md)

The unifying signature: every algorithm takes an `Iterable` of experience plus a starting
`FunctionApprox` and returns an **infinite `Iterator` of improving approximations**. The caller
stops it (`rl/iterate.py`: `iterate`, `last`, `converge`, `converged`).

## Chapter 10 — MC and TD for Prediction

**Monte-Carlo.** Supervised learning on `(S_t, G_t)` pairs:

```
L_{(S_t, G_t)}(w) = 1/2(V(S_t;w) - G_t)^2 ⟹ Δw = α (G_t - V(S_t;w))∇_wV(S_t;w)
```

Tabular form `V(S_t) ← V(S_t) + α(G_t - V(S_t))`; with α = 1/n this is
exactly the running mean, and with fixed α it is an exponentially-weighted mean
`V_n = Σ_j α(1-α)^{n-j}Y^{(j)}`.

**Temporal-Difference.** Replace the return with the bootstrapped estimate:

```
V(S_t) ← V(S_t) + α·δ_t     where the TD error is
δ_t     = R_{t+1} + γ·V(S_{t+1}) - V(S_t)
```
```
Δw = α(R_{t+1} + γ V(S_{t+1};w) - V(S_t;w))∇_wV(S_t;w)
```

This is a **semi-gradient**: the target `R_{t+1} + γ V(S_{t+1};w)` depends on w
but is treated as a constant. It is not the gradient of any objective — the point chapter 12
returns to.

**Learning-rate schedule** (`learning_rate_schedule`), and the Robbins–Monro conditions
`Σ α_n = ∞`, `Σ α_n^2 < ∞` required for convergence:

```
α_n = α/(1 + ≤ft((n-1)/H)^β)
```

**TD vs MC — the substantive comparison:**

| | MC | TD |
|---|---|---|
| Bias / variance | unbiased, high variance | biased, low variance |
| Episodes | needs complete, terminating episodes | works on continuing tasks, learns online |
| Markov property | doesn't exploit it | exploits it |
| Batch convergence | minimizes MSE against observed returns | converges to the VF of the **MLE MRP** fitted from the data, `P_R(s, r, s') = (Σ_i 1_{S_i=s, R_{i+1}=r, S_{i+1}=s'})/(Σ_i 1[S_i=s])` |

Empirically (`rl/chapter10/random_walk_mrp.py`) TD reaches a low RMSE faster at a comparable
constant α, and MC's error curve is visibly choppier.

**n-step bootstrapping and the λ-return.** Interpolating between the two:

```
G_{t, n} = Σ_{i=t+1}^{t+n}γ^{i-t-1}R_i + γ^n V(S_{t+n}), G_t^{(λ)} = (1-λ)Σ_{n=1}^{T-t-1}λ^{n-1}G_{t, n} + λ^{T-t-1}G_t
```

λ = 0 is TD, λ = 1 is MC.

**Eligibility traces** turn the *forward view* (which needs the whole episode) into an online
*backward view*:

```
E_t = γλE_{t-1} + ∇_wV(S_t;w), Δw = αδ_tE_t
```

The chapter proves the two views are equivalent in the offline, episode-summed sense:
`Σ_t α δ_t E_t(s) = Σ_t α (G_t^{(λ)} - V(S_t))1[S_t=s]`.

> **Code:** `rl/monte_carlo.py` (`mc_prediction`), `rl/td.py` (`td_prediction`),
> `rl/td_lambda.py` (`lambda_return_prediction`, `td_lambda_prediction`), `rl/returns.py`.

## Chapter 11 — MC and TD for Control

Control = GPI with a *sampled* evaluation step. Since we no longer have P, improvement
must be greedy w.r.t. Q, not V: `π_D'(s) = argmax_a Q^π(s, a)`.

**GLIE MC Control.** Tabular update on first/every visit:

```
Count(S_t, A_t) += 1, Q(S_t, A_t) += 1/(Count(S_t, A_t))(G_t - Q(S_t, A_t))
```

**GLIE** (Greedy in the Limit with Infinite Exploration) requires
`lim_k Count_k(s, a) = ∞` and `lim_k π_k(s, a) = 1[a = argmax_b Q(s, b)]`;
`ε_k = 1/k` satisfies both. Under GLIE + Robbins–Monro, `Q → Q^*`.

**SARSA** (on-policy TD control):

```
Δw = α(R_{t+1} + γ Q(S_{t+1}, A_{t+1};w) - Q(S_t, A_t;w))∇_wQ(S_t, A_t;w)
```

with n-step and SARSA(λ) variants built the same way as in chapter 10.

**Q-Learning** (off-policy TD control) — the target uses the max, not the taken action:

```
δ_t = R_{t+1} + γmax_aQ(S_{t+1}, a;w) - Q(S_t, A_t;w)
```

**Off-policy via importance sampling.** For behavior policy μ and target π,
`E_{X~ P}[f(X)] = E_{X~ Q}[(P(X))/(Q(X))f(X)]`. MC needs the full
trajectory ratio `ρ_t = ∏_{k=t}^{T-1}(π(S_k, A_k))/(μ(S_k, A_k))` (very high
variance); TD needs only the one-step ratio `(π(S_t, A_t))/(μ(S_t, A_t))`. Q-Learning
avoids ratios entirely, which is why it is the practical off-policy method.

**Convergence.** The root cause of every failure below is that the semi-gradient TD update
"does not follow the gradient of *any* objective function".

*Prediction* (✓ = converges), including Gradient TD from chapter 12:

| Policy | Algorithm | Tabular | Linear | Non-Linear |
|---|---|:--:|:--:|:--:|
| On | MC | ✓ | ✓ | ✓ |
| On | TD / TD(λ) | ✓ | ✓ | ✗ |
| On | **Gradient TD** | ✓ | ✓ | ✓ |
| Off | MC | ✓ | ✓ | ✓ |
| Off | TD / TD(λ) | ✓ | ✗ | ✗ |
| Off | **Gradient TD** | ✓ | ✓ | ✓ |

*Control* — (✓) means it doesn't reach `V^*` but bounces around near it:

| Algorithm | Tabular | Linear | Non-Linear |
|---|:--:|:--:|:--:|
| MC Control | ✓ | (✓) | ✗ |
| SARSA | ✓ | (✓) | ✗ |
| Q-Learning | ✓ | ✗ | ✗ |
| **Gradient Q-Learning** | ✓ | ✓ | ✗ |

**The Deadly Triad** — [bootstrapping, off-policy, function approximation]. Not a theorem, a rule
of thumb: when all three combine, expect divergence, so drop one. Function approximation is
non-negotiable at real-world scale, so the escape routes are a high λ (mitigation),
Gradient TD (chapter 12), or DQN (chapter 12). Note the triad survives Gradient TD in the
*control* setting with non-linear FA.

> **Code:** `glie_mc_control` (`rl/monte_carlo.py`); `glie_sarsa`, `q_learning`
> (`rl/td.py`); comparison drivers in `rl/chapter11/control_utils.py`.

## Chapter 12 — Batch RL, Experience-Replay, DQN, LSPI, Gradient TD

**Batch RL.** Given a *fixed*, finite dataset, use it repeatedly rather than once:

```
Δw = α·1/nΣ_{i=1}^n (R_i + γ V(S'_i;w) - V(S_i;w))∇_wV(S_i;w)
```

**Experience-replay** samples uniformly from a growing memory, decorrelating updates
(`rl/experience_replay.py`).

**Least-Squares TD (LSTD).** With linear FA the semi-gradient equation is *linear in w*,
so the fixed point can be solved directly instead of iterated:

```
Σ_i φ(S_i)(φ(S_i)^Tw^* - (R_i + γφ(S'_i)^Tw^*)) = 0
```

> **Method:** accumulate `A += φ(S_i)(φ(S_i) - γφ(S'_i))^T`
> (an outer product) and `b += φ(S_i)R_i`, then `w^* = A^{-1}b`.
> No learning rate at all. Sherman–Morrison gives an incremental inverse.
> `least_squares_td` in `rl/td.py`. LSTD(λ) replaces `φ(S_i)` with `E_{i, t}`.

**LSPI.** LSTDQ (the Q-version of LSTD, with features φ(s, a)) inside a policy
iteration loop — an off-policy, experience-reusing, learning-rate-free control algorithm.
`least_squares_tdq`, `least_squares_policy_iteration`.

**DQN.** Experience replay + a frozen target network `w^-`:

```
Δw = αΣ_i (r_i + γmax_{a'}Q(s'_i, a';w^-) - Q(s_i, a_i;w))∇_wQ(s_i, a_i;w)
```

**Value Function Geometry.** The most conceptually dense section. Work in the n-dimensional
space of value functions with the `μ_π`-weighted norm
`d(V_1, V_2) = (V_1-V_2)^TD(V_1-V_2)`. Two operators act on it:

- `B^π·V = R^π + γP^πV` — the Bellman operator, whose fixed point `V^π` generally lies *outside* the representable subspace {Φw};
- `Π_Φ = Φ(Φ^TDΦ)^{-1}Φ^TD` — orthogonal projection *onto* that subspace.

Three different objectives, three different answers:

| Objective | Minimizes | Solution |
|---|---|---|
| **BE** (Bellman Error) | `d(B^πV_w, V_w)` | `w_{BE} = ((Φ - γP^πΦ)^TD(Φ - γP^πΦ))^{-1}(Φ-γP^πΦ)^TDR^π` |
| **TDE** (TD Error) | `E[δ^2]` | naive residual gradient; converges to the wrong place |
| **PBE** (Projected Bellman Error) | `d(Π_ΦB^πV_w, V_w)` | `w_{PBE} = A^{-1}b`, `A = Φ^TD(Φ - γP^πΦ)`, `b = Φ^TDR^π` |

`w_{PBE}` is exactly the LSTD solution, and the semi-gradient TD update is a stochastic
approximation of it — which explains both what TD converges to and why it can diverge off-policy.

**Gradient TD (TDC).** A *true* gradient of the PBE, using a second weight vector θ
to estimate `(E[φφ^T])^{-1}E[δφ]`:

```
Δw = αδφ(s) - αγφ(s')(φ(s)^Tθ), Δθ = β(δ - φ(s)^Tθ)φ(s)
```

For *prediction* this converges in every cell of the table above, including off-policy with
non-linear FA. For *control*, Gradient Q-Learning converges with linear FA but still diverges
with non-linear FA — the triad is defused, not eliminated.

## Chapter 13 — Policy Gradient Algorithms

Parameterize the policy directly as π(s, a;θ) and ascend
`J(θ) = E_π[Σ_t γ^t R_{t+1}]`. Motivation: continuous/large action
spaces, stochastic optimal policies, smoother convergence than argmax-based methods.

**Policy Gradient Theorem.** The central result — the gradient contains **no**
`∇_θ` of the environment dynamics:

```
∇_θJ(θ) = Σ_sρ^π(s)Σ_a∇_θπ(s, a;θ)Q^π(s, a) = E_{s~ρ^π, a~π}[∇_θlogπ(s, a;θ)· Q^π(s, a)]
```

(Proved by unrolling `V^π(S_0)` through the Bellman policy equation and collecting terms into
`ρ^π`.) The identity that makes it work is the **likelihood-ratio trick**
`∇_θπ = π∇_θlogπ`.

**Score functions** `∇_θlogπ(s, a;θ)` for the two canonical policies:

| Policy | Form | Score |
|---|---|---|
| Softmax | `π ∝ e^{φ(s, a)^Tθ}` | `φ(s, a) - E_π[φ(s, ·)]` |
| Gaussian | `a ~ N(φ(s)^Tθ, σ^2)` | `((a - φ(s)^Tθ)φ(s))/σ^2` |

**The algorithm family**, each replacing `Q^π` with a cheaper estimate:

| Algorithm | θ-update uses | Note |
|---|---|---|
| **REINFORCE** | `G_t` | MC, unbiased, high variance |
| **Actor-Critic** | `Q(S_t, A_t;w)` | critic learned by TD; biased |
| **with baseline** | `Q(S_t, A_t;w) - B(S_t)` | any B(s) leaves the gradient unbiased, since `Σ_a ∇_θπ(s, a) B(s) = B(s)∇_θ1 = 0` |
| **Advantage AC** | A = Q(s, a;w) - V(s;v) | two critics |
| **TD-error AC** | `δ = R_{t+1} + γ V(S_{t+1};v) - V(S_t;v)` | since `E_π[δ^π \| s, a] = A^π(s, a)`, one critic suffices |

All carry the `γ^t` factor: `Δθ = αγ^t∇_θlogπ(S_t, A_t;θ)·(...)`.

**Compatible Function Approximation Theorem.** If (1)
`∇_wQ(s, a;w^*) = ∇_θlogπ(s, a;θ)` and (2) `w^*`
minimizes the expected squared critic error, then substituting the critic for `Q^π` introduces
**no bias**. So a linear critic on the score features is exactly compatible.

**Natural Policy Gradient.** Under the Fisher metric,
`∇^{nat}_θJ = FIM^{-1}∇_θJ`; combined with the compatible-FA
result `∇_θJ = FIM·w^*_θ`, this collapses to the strikingly
simple `∇^{nat}_θJ(θ) = w^*_θ`, i.e.
`Δθ = α_θw`.

**Deterministic Policy Gradient (DPG).**
`∇_θJ = E_{s~ρ^{π_D}}[∇_θπ_D(s;θ)∇_a Q^{π_D}(s, a)|_{a=π_D(s)}]` — the chain rule through the action.

**Evolutionary Strategies.** Not RL at all — a black-box gradient estimate
`1/σE_{ε~N(0, I)}[εF(θ+σε)]`,
included as a competitive, embarrassingly parallel alternative.

> **Code:** `rl/policy_gradient.py` — `reinforce_gaussian`, `actor_critic_gaussian`,
> `actor_critic_advantage_gaussian`, `actor_critic_td_error_gaussian`. Applied back to the
> asset-allocation problem of chapter 7 in `rl/chapter13/`.

---

[← Notes index](../book-summary.md) · [← Module II](02-financial-applications.md) · [Module IV →](04-finishing-touches.md)
