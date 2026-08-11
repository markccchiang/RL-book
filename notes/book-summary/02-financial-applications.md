# Module II — Modeling Financial Applications

[← Notes index](../book-summary.md) · [← Module I](01-processes-and-planning.md) · [Module III →](03-rl-algorithms.md)

## Chapter 6 — Utility Theory

$$x_{CE} = U^{-1}(\mathbb{E}[U(x)]), \qquad \pi_A = \mathbb{E}[x] - x_{CE}, \qquad \pi_R = \pi_A / \mathbb{E}[x]$$

Second-order Taylor expansion gives the risk premium in terms of curvature:

$$\pi_A \approx \tfrac{1}{2} A(\bar{x})\,\sigma_x^2, \qquad A(x) = -\frac{U''(x)}{U'(x)}, \qquad R(x) = -\frac{U''(x)\,x}{U'(x)}$$

| Family | Utility | Risk aversion | Optimal risky allocation |
|---|---|---|---|
| **CARA** | $U(x) = \dfrac{1 - e^{-ax}}{a}$ | $A(x) = a$ | $\pi^* = \dfrac{\mu - r}{a\sigma^2}$ |
| **CRRA** | $U(x) = \dfrac{x^{1-\gamma} - 1}{1-\gamma}$ | $R(x) = \gamma$ | $\pi^* = \dfrac{\mu - r}{\gamma\sigma^2}$ |

(Both limits: $a \to 0$ gives $U(x) = x$; $\gamma \to 1$ gives $U(x) = \log x$.)

For CARA with $x \sim \mathcal{N}(\mu,\sigma^2)$: $x_{CE} = \mu - a\sigma^2/2$, so
$\pi_A = a\sigma^2/2$ exactly. This closed form (via the Gaussian MGF, appendix 1) is what makes
the CARA cases of chapters 7, 8 and 9 analytically solvable.

## Chapter 7 — Dynamic Asset-Allocation and Consumption (Merton's Problem)

**Setup.** Risky asset $dS_t = \mu S_t dt + \sigma S_t dz_t$, riskless $dR_t = rR_t dt$, CRRA
utility, horizon $T$. Choose fraction $\pi_t$ and consumption rate $c_t$ to maximize

$$\mathbb{E}\Big[\int_t^T \frac{e^{-\rho(s-t)} c_s^{1-\gamma}}{1-\gamma}\,ds + \frac{e^{-\rho(T-t)}\epsilon^{\gamma} W_T^{1-\gamma}}{1-\gamma} \,\Big|\, W_t\Big]$$

subject to $dW_t = ((r + \pi_t(\mu - r))W_t - c_t)\,dt + \pi_t \sigma W_t\,dz_t$.

**Solution method** — the book's flagship worked derivation:

1. Write the **HJB equation** (appendix 4) and expand $dV^*$ with **Itô's Lemma** (appendix 3).
2. Take $\partial/\partial \pi_t$ and $\partial/\partial c_t$ of the resulting expression and set to zero:
   $$\pi_t^* = \frac{-\partial V^*/\partial W_t \cdot (\mu - r)}{\partial^2 V^*/\partial W_t^2 \cdot \sigma^2 W_t}, \qquad c_t^* = \Big(\frac{\partial V^*}{\partial W_t}\Big)^{-1/\gamma}$$
3. Substitute back to get a non-linear PDE in $V^*$.
4. **Guess** $V^*(t, W_t) = f(t)^{\gamma}\dfrac{W_t^{1-\gamma}}{1-\gamma}$; the PDE reduces to the ODE $f'(t) = \nu f(t) - 1$ with $\nu = \dfrac{\rho - (1-\gamma)\big(\frac{(\mu-r)^2}{2\sigma^2\gamma} + r\big)}{\gamma}$.
5. Solve the ODE with terminal condition:
   $$f(t) = \frac{1 + (\nu\epsilon - 1)e^{-\nu(T-t)}}{\nu} \quad (\nu \neq 0), \qquad f(t) = T - t + \epsilon \quad (\nu = 0)$$

**Results.** $\pi^*(t, W_t) = \dfrac{\mu - r}{\sigma^2\gamma}$ — *constant*, independent of both
time and wealth — and $c^*(t, W_t) = W_t / f(t)$.

**Discrete-time analogue** (§ *A Discrete-Time Asset-Allocation Example*), CARA utility, single
risky asset with $Y_t \sim \mathcal{N}(\mu,\sigma^2)$, $W_{t+1} = x_t(Y_t - r) + W_t(1+r)$:

$$V^*_t(W_t) = -b_t e^{-c_t W_t}, \qquad b_t = \frac{e^{-\frac{(\mu-r)^2(T-t)}{2\sigma^2}}}{a}, \qquad c_t = a(1+r)^{T-t}$$
$$x^*_t = \frac{\mu - r}{\sigma^2 a (1+r)^{T-t-1}}$$

> **Method:** backward induction on the Bellman optimality equation, with the Gaussian MGF used
> to evaluate $\mathbb{E}[e^{-cW_{t+1}}]$ in closed form at each step.

The chapter then *re-solves the same problem* by ADP with the feature set
$\phi = (1, W_t, x_t, x_t^2)$ for $Q^*_t$ — recovering the analytic answer numerically. This
"known answer, then learn it" pattern is the book's standard validation technique
(`rl/chapter7/`).

## Chapter 8 — Derivatives Pricing and Hedging

**Single-period setting.** $m+1$ assets, $n$ states of the world $\Omega$, real-world measure
$\mu$, portfolio $\theta$.

**Risk-neutral measure** $\pi$: any measure equivalent to $\mu$ (same null sets) under which
every asset is a discounted martingale,
$S_j^{(0)} = \frac{1}{1+r}\sum_i \pi(\omega_i) S_j^{(i)}$. It follows immediately that *every*
portfolio satisfies $V_{\theta}^{(0)} = \frac{1}{1+r}\sum_i \pi(\omega_i) V_{\theta}^{(i)}$.

- **1st FTAP:** no arbitrage $\iff$ a risk-neutral measure exists.
- **2nd FTAP:** the market is complete (every derivative is replicable) $\iff$ the risk-neutral measure is *unique*.

**Two pricing routes, and they agree in a complete market:**

$$V_D^{(0)} = \sum_{j} \theta_j S_j^{(0)} \quad \text{(replication)} \qquad\text{and}\qquad V_D^{(0)} = \frac{1}{1+r}\sum_i \pi(\omega_i) V_D^{(i)} \quad \text{(risk-neutral expectation)}$$

**Incomplete markets.** The price is no longer unique; it lies in the interval between the
super-hedging and sub-hedging bounds, obtained as a linear program whose dual is exactly the
sup/inf of discounted expectation over the set of risk-neutral measures:

$$SP = \sup_{\pi}\sum_i \frac{\pi(\omega_i)}{1+r}V_D^{(i)}, \qquad SB = \inf_{\pi}\sum_i \frac{\pi(\omega_i)}{1+r}V_D^{(i)}$$

To pick a single price, use **utility indifference**: solve
$\max_{\theta} \sum_i \mu(\omega_i) U(V_D^{(i)} + \sum_j \theta_j S_j^{(i)})$ with and without the
derivative and equate. For CARA + Gaussian this is closed-form
($\alpha^* = \frac{\mu - (1+r)S_0}{a\sigma^2}$); otherwise it is cast as an MDP.

**Multi-period / American options.** Under the risk-neutral measure the underlying follows
$dS_t = rS_t dt + \sigma S_t dz_t$ — note the drift is $r$, not $\mu$. Optimal exercise is an
**optimal stopping** problem:

$$V^*(X_t) = \max\big(H(X_t),\; \mathbb{E}[V^*(X_{t+1}) \mid X_t]\big)$$

with payoff $H$. Cast as a finite MDP over a **binomial tree**: with $n$ steps to expiry
$T$, $u = e^{\sigma\sqrt{T/n}}$ and risk-neutral up-probability
$q = \dfrac{e^{rT/n + \sigma\sqrt{T/n}} - 1}{e^{2\sigma\sqrt{T/n}} - 1}$, node prices
$S_{i,j} = S_{0,0}e^{(2j-i)\sigma\sqrt{T/n}}$.

> **Methods:** backward induction on the tree (chapter 8), then LSPI and deep Q-learning on the
> same problem in chapter 12 — the direct comparison of a DP answer against an RL answer.
> `rl/chapter8/`, `rl/chapter12/optimal_exercise_rl.py`.

## Chapter 9 — Order-Book Trading Algorithms

**Order book dynamics.** Bids $[(P_i^{(b)}, N_i^{(b)})]$ descending, asks
$[(P_i^{(a)}, N_i^{(a)})]$ ascending; the chapter gives the exact set arithmetic for how a limit
or market order removes and adds levels, and the resulting proceeds.

**Optimal execution.** Sell $N$ shares over $T$ steps; $R_t$ shares remaining,
$R_{t+1} = R_t - N_t$, $R_T = 0$. Price dynamics with permanent impact $\alpha$ and temporary
impact $\beta$: $P_{t+1} = P_t - \alpha N_t + \epsilon_t$, execution price
$Q_t = P_t - \beta N_t$. Maximize $\mathbb{E}[\sum_t \gamma^t U(N_t Q_t)]$.

> **Method:** backward induction. With a linear price-impact model and risk-neutral utility the
> value function stays quadratic, giving the closed form
> $$N_t^* = \frac{R_t}{T-t}, \qquad V_t^*((P_t,R_t)) = R_t P_t - \frac{R_t^2}{2}\Big(\frac{2\beta + \alpha(T-t-1)}{T-t}\Big)$$
> — i.e. **uniform (TWAP) execution is optimal** under these assumptions. Adding a mean-reverting
> signal $X_{t+1} = \rho X_t + \eta_t$ perturbs it to
> $N_t^* = \frac{R_t}{T-t} + h(t,\beta,\theta,\rho)\cdot X_t$. Beyond these tractable cases: RL.

**Optimal market-making (Avellaneda–Stoikov).** State $(t, S_t, W_t, I_t)$ — time, OB mid,
trading PnL, inventory. Hits arrive as Poisson processes whose intensities depend on the spread:
$\lambda_t^{(b)} = f^{(b)}(\delta_t^{(b)})$. Maximize $\mathbb{E}[U(W_T + I_T S_T)]$ with CARA.

> **Method:** HJB, then the substitution $V^* = -e^{-\gamma(W_t + \theta(t,S_t,I_t))}$ to strip out
> wealth, then an expansion $\theta \approx \theta^{(0)} + I_t\theta^{(1)} + \frac{I_t^2}{2}\theta^{(2)}$
> which decouples the PDE into three solvable ones.

With exponential intensity $f(\delta) = ce^{-k\delta}$, the closed-form solution:

$$\text{Optimal bid-ask spread} \quad \delta_t^{(b)*} + \delta_t^{(a)*} = \gamma\sigma^2(T-t) + \frac{2}{\gamma}\log\Big(1 + \frac{\gamma}{k}\Big)$$
$$\text{Optimal pseudo-mid} \quad Q_t^{(m)} = S_t - I_t \gamma \sigma^2 (T-t)$$

The pseudo-mid is skewed *away* from the mid in proportion to inventory — the mechanism by
which a market maker manages inventory risk.

---

[← Notes index](../book-summary.md) · [← Module I](01-processes-and-planning.md) · [Module III →](03-rl-algorithms.md)
