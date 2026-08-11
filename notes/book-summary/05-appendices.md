# Appendices

[← Notes index](../book-summary.md) · [← Module IV](04-finishing-touches.md) · [Cross-Cutting Reference →](06-cross-cutting.md)

| # | Topic | Key content |
|---|---|---|
| **1** | Moment Generating Function | $f_x(t) = \mathbb{E}[e^{tx}]$, $f_x^{(n)}(0) = \mathbb{E}[x^n]$. For $x \sim \mathcal{N}(\mu,\sigma^2)$: $f_x(t) = e^{\mu t + \sigma^2t^2/2}$; minimized at $t^* = -\mu/\sigma^2$ with value $e^{-\mu^2/(2\sigma^2)}$. This is the identity behind every CARA closed form in chapters 6–9. |
| **2** | Portfolio Theory | Efficient frontier $\sigma_p^2 = \frac{a - 2br_p + cr_p^2}{ac - b^2}$ with $a = R^TV^{-1}R$, $b = R^TV^{-1}1_n$, $c = 1_n^TV^{-1}1_n$. Derived by Lagrangian minimization of $X^TVX$ s.t. $X^T1_n = 1$, $X^TR = r_p$. GMVP, two-fund theorem, CAPM $R = r_z1_n + (r_p - r_z)\beta_p$. |
| **3** | Stochastic Calculus | Brownian motion as a scaled random walk; $(dz_t)^2 = dt$; sample paths have infinite variation but finite quadratic variation. **Itô's Lemma:** $df(t,Y_t) = (\frac{\partial f}{\partial t} + \mu_t\frac{\partial f}{\partial Y_t} + \frac{\sigma_t^2}{2}\frac{\partial^2 f}{\partial Y_t^2})dt + \sigma_t\frac{\partial f}{\partial Y_t}dz_t$. Lognormal and Ornstein–Uhlenbeck processes solved by the substitutions $y_t = \log x_t$ and $y_t = x_te^{-\int\mu}$. |
| **4** | HJB Equation | The continuous-time Bellman optimality equation: $\rho V^*(t,s_t)dt = \max_{a_t}\{\mathbb{E}[dV^*] + \mathcal{R}\,dt\}$; expanded via Itô for an Itô state process. Used in chapters 7 and 9. |
| **5** | Black–Scholes | Delta-hedged portfolio $\Pi_t = -V + \frac{\partial V}{\partial S_t}S_t$ has no $dz_t$ term, so must earn $r$; this gives $\frac{\partial V}{\partial t} + \frac{\sigma^2}{2}S_t^2\frac{\partial^2 V}{\partial S_t^2} + rS_t\frac{\partial V}{\partial S_t} + rV = 0$. Change of variables reduces it to the heat equation, giving $C = S_tN(d_1) - Ke^{-r(T-t)}N(d_2)$. |
| **6** | Function Approximations as Affine Spaces | Formalizes why the `FunctionApprox` API is shaped as it is: gradients live in a vector space, parameters in an affine space over it, and an SGD update is $\boldsymbol{p} \oplus (\alpha * (e(\boldsymbol{p}) * G(x)(\boldsymbol{p})))$. |
| **7** | Conjugate Priors | Gaussian: $\mu \mid x_{1:n} \sim \mathcal{N}(\theta_n, \sigma^2/n)$, $\sigma^2 \mid x_{1:n} \sim IG(\alpha_n,\beta_n)$, with incremental updates. Bernoulli: $p \mid x_{1:n} \sim Beta(\alpha_n,\beta_n)$, updated by $\alpha \mathrel{+}= \mathbb{I}_{x=1}$, $\beta \mathrel{+}= \mathbb{I}_{x=0}$. Powers Bayesian UCB and Thompson sampling in chapter 14. |

---

[← Notes index](../book-summary.md) · [← Module IV](04-finishing-touches.md) · [Cross-Cutting Reference →](06-cross-cutting.md)
