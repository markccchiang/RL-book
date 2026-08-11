'''Moment Generating Functions and their applications.

The code here follows book/appendix1 (Appendix A of the book).

The MGF of a random variable x is

    f_x(t) = E[e^{tx}]  for all t in R

and its derivatives generate the moments of x:

    f_x^{(n)}(t) = E[x^n e^{tx}],   so   f_x^{(n)}(0) = E[x^n]

This module states each of the appendix's closed forms as code and then checks
it three ways: against a generic numerical derivative of the MGF, against a
Monte-Carlo estimate of the defining expectation, and (for the minimization
results) against a numerical optimizer.  Every closed form in the appendix is
therefore executable and self-verifying.

Run it with:

    python -m rl.appendix1.moment_generating_function
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import comb, exp, log, sqrt
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from rl.distribution import Categorical, Distribution, Gaussian


@dataclass(frozen=True)
class MomentGeneratingFunction(ABC):
    '''The MGF f_x(t) = E[e^{tx}] of a random variable x.

    Only mgf() and distribution() are specific to a distribution; everything
    else here follows from the definition and so works for any of them.
    '''

    @abstractmethod
    def mgf(self, t: float) -> float:
        '''f_x(t) = E[e^{tx}], in closed form.'''

    @abstractmethod
    def distribution(self) -> Distribution[float]:
        '''The distribution of x, used to check the MGF by sampling.'''

    def derivative(self, t: float, n: int = 1, h: float = 1e-3) -> float:
        '''f_x^{(n)}(t), by an n-th order central difference.

        This is the generic route to the moments: it needs nothing but the
        MGF itself, which is what makes f^{(n)}(0) = E[x^n] useful.
        '''
        return sum((-1) ** k * comb(n, k) * self.mgf(t + (n / 2 - k) * h)
                   for k in range(n + 1)) / h ** n

    def moment(self, n: int) -> float:
        '''E[x^n] = f_x^{(n)}(0).'''
        return self.derivative(0., n)

    def mean(self) -> float:
        '''E[x] = f_x'(0).'''
        return self.moment(1)

    def variance(self) -> float:
        '''Var[x] = f_x''(0) - (f_x'(0))^2.'''
        return self.moment(2) - self.moment(1) ** 2

    def sampled_mgf(self, t: float) -> float:
        '''E[e^{tx}] estimated from the distribution, not from the formula.'''
        return self.distribution().expectation(lambda x: exp(t * x))

    def sampled_moment(self, n: int) -> float:
        '''E[x^n] estimated from the distribution.'''
        return self.distribution().expectation(lambda x: x ** n)

    @abstractmethod
    def minimizing_argument(self) -> float:
        '''t* = argmin_t f_x(t), in closed form.'''

    def minimum(self) -> float:
        '''min_t f_x(t), evaluated at the closed-form t*.'''
        return self.mgf(self.minimizing_argument())

    def numerical_minimizing_argument(self) -> float:
        '''t*, found by an optimizer instead of by calculus.'''
        return float(minimize_scalar(self.mgf).x)


@dataclass(frozen=True)
class Normal(MomentGeneratingFunction):
    '''x ~ N(μ, σ^2), whose MGF is e^{μt + σ^2 t^2 / 2}.'''

    μ: float
    σ: float

    def mgf(self, t: float) -> float:
        return exp(self.μ * t + self.σ ** 2 * t ** 2 / 2)

    def distribution(self) -> Distribution[float]:
        return Gaussian(μ=self.μ, σ=self.σ, expectation_samples=1000000)

    def analytic_derivative(self, t: float, n: int = 1) -> float:
        '''f'(t) = (μ + σ^2 t) f(t);  f''(t) = ((μ + σ^2 t)^2 + σ^2) f(t).'''
        scaled = self.μ + self.σ ** 2 * t
        if n == 1:
            return scaled * self.mgf(t)
        if n == 2:
            return (scaled ** 2 + self.σ ** 2) * self.mgf(t)
        raise ValueError(f'no closed form here for n = {n}')

    def minimizing_argument(self) -> float:
        '''t* = -μ / σ^2.'''
        return -self.μ / self.σ ** 2

    def analytic_minimum(self) -> float:
        '''min_t f_x(t) = e^{-μ^2 / (2σ^2)}.'''
        return exp(-self.μ ** 2 / (2 * self.σ ** 2))

    def certainty_equivalent(self, γ: float) -> float:
        '''The CARA application: U(y) = (1 - e^{-γy}) / γ.

        E[U(x)] = (1 - f_x(-γ)) / γ, so the certainty-equivalent wealth is
        μ - γσ^2/2 — the mean penalised by risk-aversion times variance.
        '''
        return -log(1 - γ * self.expected_utility(γ)) / γ

    def expected_utility(self, γ: float) -> float:
        '''E[(1 - e^{-γx}) / γ], written in terms of the MGF at t = -γ.'''
        return (1 - self.mgf(-γ)) / γ


@dataclass(frozen=True)
class SymmetricBinary(MomentGeneratingFunction):
    '''x ~ B(μ + σ, μ - σ): either outcome with probability 0.5.

    Mean μ and variance σ^2, matching Normal(μ, σ) in its first two moments
    but not in its MGF.
    '''

    μ: float
    σ: float

    def mgf(self, t: float) -> float:
        return 0.5 * (exp((self.μ + self.σ) * t) + exp((self.μ - self.σ) * t))

    def distribution(self) -> Distribution[float]:
        return Categorical({self.μ + self.σ: 0.5, self.μ - self.σ: 0.5})

    def analytic_derivative(self, t: float, n: int = 1) -> float:
        '''f^{(n)}(t) = 0.5 ((μ+σ)^n e^{(μ+σ)t} + (μ-σ)^n e^{(μ-σ)t}).'''
        up, down = self.μ + self.σ, self.μ - self.σ
        return 0.5 * (up ** n * exp(up * t) + down ** n * exp(down * t))

    def minimizing_argument(self) -> float:
        '''t* = ln((σ - μ) / (μ + σ)) / (2σ), which needs |μ| < σ.'''
        if not -self.σ < self.μ < self.σ:
            raise ValueError(
                f'μ = {self.μ} must lie in (-σ, σ) = ({-self.σ}, {self.σ}) '
                'for the MGF to have a minimum'
            )
        return log((self.σ - self.μ) / (self.μ + self.σ)) / (2 * self.σ)

    def analytic_minimum(self) -> float:
        '''0.5 (r^{(μ+σ)/(2σ)} + r^{(μ-σ)/(2σ)}) for r = (σ-μ)/(μ+σ).'''
        r = (self.σ - self.μ) / (self.μ + self.σ)
        return 0.5 * (r ** ((self.μ + self.σ) / (2 * self.σ)) +
                      r ** ((self.μ - self.σ) / (2 * self.σ)))


def linear_combination_mgf(
        α_0: float,
        terms: Sequence[Tuple[float, MomentGeneratingFunction]],
        t: float
) -> float:
    '''MGF of x = α_0 + Σ α_i x_i for independent x_i, at t.

    f_x(t) = e^{α_0 t} Π f_{x_i}(α_i t) — the appendix's point that a linear
    combination is easy through MGFs and painful through convolutions.
    '''
    product = 1.
    for α, f in terms:
        product *= f.mgf(α * t)
    return exp(α_0 * t) * product


def normal_combination(
        α_0: float,
        terms: Sequence[Tuple[float, Normal]]
) -> Normal:
    '''The same combination of independent normals, done in closed form.'''
    return Normal(
        μ=α_0 + sum(α * n.μ for α, n in terms),
        σ=sqrt(sum((α * n.σ) ** 2 for α, n in terms))
    )


def row(label: str, *values: Optional[float]) -> None:
    '''Print a closed form beside the values that should match it.'''
    print(f'  {label:34s}' +
          ''.join(f'{"—":>15s}' if v is None else f'{v:15.6f}'
                  for v in values))


if __name__ == '__main__':

    np.random.seed(1729)

    μ, σ, γ = 0.5, 2.0, 0.3
    normal = Normal(μ=μ, σ=σ)
    binary = SymmetricBinary(μ=μ, σ=σ)

    print(f'\nx ~ N(μ={μ}, σ^2={σ ** 2})       f(t) = e^(μt + σ^2t^2/2)')
    print(f'  {"quantity":34s}{"closed form":>15s}{"d/dt of f":>15s}'
          f'{"sampled":>15s}')
    row('f(1)   = E[e^x]', normal.mgf(1.), None, normal.sampled_mgf(1.))
    row("f'(0)  = E[x] = μ", normal.analytic_derivative(0., 1),
        normal.derivative(0., 1), normal.sampled_moment(1))
    row("f''(0) = E[x^2] = μ^2 + σ^2", normal.analytic_derivative(0., 2),
        normal.derivative(0., 2), normal.sampled_moment(2))
    row("f'(1)  = E[x e^x]", normal.analytic_derivative(1., 1),
        normal.derivative(1., 1),
        normal.distribution().expectation(lambda x: x * exp(x)))
    row("f''(1) = E[x^2 e^x]", normal.analytic_derivative(1., 2),
        normal.derivative(1., 2),
        normal.distribution().expectation(lambda x: x ** 2 * exp(x)))
    row("variance = f''(0) - f'(0)^2", σ ** 2, normal.variance(), None)

    print('\nMinimizing the MGF')
    print('  distribution                       t* (calculus)   t* (optimizer)'
          '   min f(t*)   closed form')
    for name, f, closed in [
            ('N(μ, σ^2):  t* = -μ/σ^2', normal, normal.analytic_minimum()),
            ('B(μ+σ, μ-σ): t* = ln(r)/2σ', binary, binary.analytic_minimum())]:
        print(f'  {name:34s} {f.minimizing_argument():13.6f} '
              f'{f.numerical_minimizing_argument():16.6f} '
              f'{f.minimum():11.6f} {closed:13.6f}')

    print('\nSame mean and variance, different MGF')
    row('N(μ, σ^2) at t = 1', normal.mgf(1.))
    row('B(μ+σ, μ-σ) at t = 1', binary.mgf(1.))

    print('\nLinear combination: x = α_0 + Σ α_i x_i, independent x_i')
    α_0 = 1.5
    terms: Sequence[Tuple[float, Normal]] = [
        (0.4, Normal(μ=1.0, σ=0.5)),
        (-0.7, Normal(μ=-2.0, σ=1.5)),
        (2.0, Normal(μ=0.25, σ=0.75)),
    ]
    combined = normal_combination(α_0, terms)
    print(f'  x ~ N({combined.μ:.4f}, {combined.σ ** 2:.4f})')
    print('  t      e^(α_0 t) Π f_i(α_i t)      MGF of the combined normal'
          '     sampled')
    for t in [-1.0, -0.25, 0.5, 1.0]:
        sampled = sum(exp(t * (α_0 + sum(α * n.distribution().sample()
                                         for α, n in terms)))
                      for _ in range(200000)) / 200000
        print(f'  {t:5.2f} {linear_combination_mgf(α_0, terms, t):22.6f}'
              f'{combined.mgf(t):28.6f} {sampled:14.6f}')

    print(f'\nCARA utility U(y) = (1 - e^(-γy))/γ with γ = {γ}')
    print('  E[U(x)] is just the MGF at t = -γ:')
    print(f'  {"quantity":34s}{"via the MGF":>15s}{"sampled":>15s}')
    row('E[U(x)] = (1 - f(-γ))/γ', normal.expected_utility(γ),
        normal.distribution().expectation(lambda x: (1 - exp(-γ * x)) / γ))
    row('certainty equivalent', normal.certainty_equivalent(γ), None)
    row('  which equals μ - γσ^2/2', μ - γ * σ ** 2 / 2, None)
    print()
