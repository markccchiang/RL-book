# Appendices

[← Notes index](../book-summary.md) · [← Module IV](04-finishing-touches.md) · [Cross-Cutting Reference →](06-cross-cutting.md)

| # | Topic | Key content |
|---|---|---|
| **1** | Moment Generating Function | `f_x(t) = E[e^{tx}]`, `f_x^{(n)}(0) = E[x^n]`. For `x ~ N(μ, σ^2)`: `f_x(t) = e^{μ t + σ^2t^2/2}`; minimized at `t^* = -μ/σ^2` with value `e^{-μ^2/(2σ^2)}`. This is the identity behind every CARA closed form in chapters 6–9. |
| **2** | Portfolio Theory | Efficient frontier `σ_p^2 = (a - 2br_p + cr_p^2)/(ac - b^2)` with `a = R^TV^{-1}R`, `b = R^TV^{-1}1_n`, `c = 1_n^TV^{-1}1_n`. Derived by Lagrangian minimization of `X^TVX` s.t. `X^T1_n = 1`, `X^TR = r_p`. GMVP, two-fund theorem, CAPM `R = r_z1_n + (r_p - r_z)β_p`. |
| **3** | Stochastic Calculus | Brownian motion as a scaled random walk; `(dz_t)^2 = dt`; sample paths have infinite variation but finite quadratic variation. **Itô's Lemma:** `df(t, Y_t) = ((∂f)/(∂t) + μ_t(∂f)/(∂Y_t) + σ_t^2/2(∂^2 f)/(∂Y_t^2))dt + σ_t(∂f)/(∂Y_t)dz_t`. Lognormal and Ornstein–Uhlenbeck processes solved by the substitutions `y_t = log x_t` and `y_t = x_te^{-∫μ}`. |
| **4** | HJB Equation | The continuous-time Bellman optimality equation: `ρ V^*(t, s_t)dt = max_{a_t}{E[dV^*] + Rdt}`; expanded via Itô for an Itô state process. Used in chapters 7 and 9. |
| **5** | Black–Scholes | Delta-hedged portfolio `Π_t = -V + (∂V)/(∂S_t)S_t` has no `dz_t` term, so must earn r; this gives `(∂V)/(∂t) + σ^2/2S_t^2(∂^2 V)/(∂S_t^2) + rS_t(∂V)/(∂S_t) + rV = 0`. Change of variables reduces it to the heat equation, giving `C = S_tN(d_1) - Ke^{-r(T-t)}N(d_2)`. |
| **6** | Function Approximations as Affine Spaces | Formalizes why the `FunctionApprox` API is shaped as it is: gradients live in a vector space, parameters in an affine space over it, and an SGD update is `p ⊕ (α * (e(p) * G(x)(p)))`. |
| **7** | Conjugate Priors | Gaussian: `μ \| x_{1:n} ~ N(θ_n, σ^2/n)`, `σ^2 \| x_{1:n} ~ IG(α_n, β_n)`, with incremental updates. Bernoulli: `p \| x_{1:n} ~ Beta(α_n, β_n)`, updated by `α += 1[x=1]`, `β += 1[x=0]`. Powers Bayesian UCB and Thompson sampling in chapter 14. |

---

[← Notes index](../book-summary.md) · [← Module IV](04-finishing-touches.md) · [Cross-Cutting Reference →](06-cross-cutting.md)
