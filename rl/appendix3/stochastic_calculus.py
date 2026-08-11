'''Basics of Stochastic Calculus.

The code here follows book/appendix3 (Appendix C of the book): the simple
random walk, Brownian motion as its scaled limit, the Ito integral, Ito's
lemma, and the two Ito processes the book leans on later — the lognormal
(geometric Brownian motion) process and the mean-reverting
(Ornstein-Uhlenbeck) process.

The appendix's results are stated as code and then checked by simulation.
Two of those checks are worth calling out, because they are the places where
stochastic calculus stops behaving like ordinary calculus:

  * Quadratic variation is a statement of *certainty*, not of expectation.
    [z]_T = T holds on every single sample trace, while the total variation
    of the same trace diverges as the time step shrinks.
  * The Ito integral is the left-endpoint sum, and that choice is not a
    detail: for X = z it converges to (z_T^2 - T)/2, whereas the midpoint
    (Stratonovich) sum of the same trace converges to z_T^2 / 2.

Run it with:

    python -m rl.appendix3.stochastic_calculus
'''

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# --- the simple random walk -------------------------------------------------


@dataclass(frozen=True)
class SimpleRandomWalk:
    '''Z_{t+1} = Z_t + Y_t, with Y_t = +1 or -1 with probability 0.5 each.'''

    steps: int

    def traces(self, num_traces: int, rng: np.random.Generator) -> np.ndarray:
        '''num_traces sample traces, each of shape (steps + 1,).'''
        increments = rng.choice([-1., 1.], size=(num_traces, self.steps))
        return np.concatenate(
            [np.zeros((num_traces, 1)), np.cumsum(increments, axis=1)], axis=1
        )

    def scaled_traces(
            self,
            n: int,
            num_traces: int,
            rng: np.random.Generator
    ) -> np.ndarray:
        '''z^(n)_t = Z_{nt} / sqrt(n): speed up time, scale down the steps.

        As n grows this converges to standard Brownian motion, so the value
        at t = 1 should look more and more like a draw from N(0, 1).
        '''
        return SimpleRandomWalk(self.steps * n).traces(num_traces, rng) \
            / np.sqrt(n)


# --- path functionals -------------------------------------------------------


def quadratic_variation(path: np.ndarray) -> float:
    '''[X]_T = sum of squared increments along one trace.'''
    return float(np.sum(np.diff(path) ** 2))


def total_variation(path: np.ndarray) -> float:
    '''The sum of absolute increments, which diverges for Brownian motion.'''
    return float(np.sum(np.abs(np.diff(path))))


def brownian_paths(
        T: float,
        steps: int,
        num_paths: int,
        rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    '''Standard Brownian motion: increments dz_t ~ N(0, dt).

    Returns the time grid and the paths, of shape (num_paths, steps + 1).
    '''
    dt = T / steps
    increments = rng.normal(0., np.sqrt(dt), size=(num_paths, steps))
    paths = np.concatenate(
        [np.zeros((num_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return np.linspace(0., T, steps + 1), paths


# --- the Ito integral -------------------------------------------------------


def ito_integral(integrand: np.ndarray, path: np.ndarray) -> np.ndarray:
    '''Y_T = int_0^T X_t dz_t, as the left-endpoint sum.

    Evaluating the integrand at the *start* of each interval is what keeps Y
    a martingale: X_t cannot peek at the increment it multiplies.
    '''
    return np.sum(integrand[:, :-1] * np.diff(path, axis=1), axis=1)


def stratonovich_integral(
        integrand: np.ndarray,
        path: np.ndarray
) -> np.ndarray:
    '''The same sum with the integrand at the midpoint of each interval.

    Included only for contrast: it is not the Ito integral and does not obey
    Ito's lemma.
    '''
    midpoints = (integrand[:, :-1] + integrand[:, 1:]) / 2
    return np.sum(midpoints * np.diff(path, axis=1), axis=1)


# --- the two Ito processes the book uses ------------------------------------


@dataclass(frozen=True)
class GeometricBrownianMotion:
    '''dx_t = μ x_t dt + σ x_t dz_t, with constant μ and σ.

    Ito's lemma applied to y = log(x) turns this into a Brownian motion with
    drift: dy_t = (μ - σ^2/2) dt + σ dz_t, so x_T is lognormal.
    '''

    x_0: float
    μ: float
    σ: float

    def log_mean(self, T: float) -> float:
        '''E[log(x_T)] = log(x_0) + (μ - σ^2/2) T.'''
        return np.log(self.x_0) + (self.μ - self.σ ** 2 / 2) * T

    def log_variance(self, T: float) -> float:
        '''Var[log(x_T)] = σ^2 T.'''
        return self.σ ** 2 * T

    def mean(self, T: float) -> float:
        '''E[x_T] = x_0 e^{μT} — the σ^2/2 terms cancel.'''
        return self.x_0 * np.exp(self.μ * T)

    def variance(self, T: float) -> float:
        '''Var[x_T] = x_0^2 e^{2μT} (e^{σ^2 T} - 1).'''
        return self.x_0 ** 2 * np.exp(2 * self.μ * T) \
            * (np.exp(self.σ ** 2 * T) - 1)

    def exact_paths(self, times: np.ndarray, path: np.ndarray) -> np.ndarray:
        '''x_t from the closed-form solution, driven by a given z path.'''
        return self.x_0 * np.exp(
            (self.μ - self.σ ** 2 / 2) * times + self.σ * path
        )

    def euler_paths(
            self,
            times: np.ndarray,
            path: np.ndarray
    ) -> np.ndarray:
        '''x_t stepped through the SDE itself, driven by the same z path.

        Euler-Maruyama: x_{t+dt} = x_t + μ x_t dt + σ x_t dz_t.  Agreeing
        with exact_paths (as dt shrinks) is the check on Ito's lemma.
        '''
        dt = float(times[1] - times[0])
        dz = np.diff(path, axis=1)
        out = np.empty_like(path)
        out[:, 0] = self.x_0
        for i in range(dz.shape[1]):
            x = out[:, i]
            out[:, i + 1] = x + self.μ * x * dt + self.σ * x * dz[:, i]
        return out


@dataclass(frozen=True)
class OrnsteinUhlenbeck:
    '''dx_t = μ x_t dt + σ dz_t, with constant μ and σ.

    Mean-reverting to a baseline of 0 when μ < 0.  Ito's lemma applied to
    y_t = x_t e^{-μt} kills the drift, leaving a martingale, so x_T is
    normal rather than lognormal.
    '''

    x_0: float
    μ: float
    σ: float

    def mean(self, T: float) -> float:
        '''E[x_T] = x_0 e^{μT}.'''
        return self.x_0 * np.exp(self.μ * T)

    def variance(self, T: float) -> float:
        '''Var[x_T] = σ^2 (e^{2μT} - 1) / (2μ).'''
        return self.σ ** 2 * (np.exp(2 * self.μ * T) - 1) / (2 * self.μ)

    def stationary_variance(self) -> float:
        '''The T -> infinity limit, which exists only for μ < 0.'''
        return -self.σ ** 2 / (2 * self.μ)

    def euler_paths(self, times: np.ndarray, path: np.ndarray) -> np.ndarray:
        '''x_{t+dt} = x_t + μ x_t dt + σ dz_t, driven by a given z path.'''
        dt = float(times[1] - times[0])
        dz = np.diff(path, axis=1)
        out = np.empty_like(path)
        out[:, 0] = self.x_0
        for i in range(dz.shape[1]):
            out[:, i + 1] = out[:, i] + self.μ * out[:, i] * dt \
                + self.σ * dz[:, i]
        return out


def row(label: str, *values: float) -> None:
    '''Print a label and a row of numbers, aligned under a header.'''
    print(f'  {label:40s}' + ''.join(f'{v:14.6f}' for v in values))


if __name__ == '__main__':

    rng = np.random.default_rng(1729)

    print('\nSimple random walk: variance of an increment vs quadratic '
          'variation')
    walk_steps = 200
    traces = SimpleRandomWalk(walk_steps).traces(20000, rng)
    qvs = np.array([quadratic_variation(trace) for trace in traces])
    print(f'  {"quantity":40s}{"simulated":>14s}{"appendix":>14s}')
    row('E[Z_T - Z_0]  (martingale)', float(np.mean(traces[:, -1])), 0.)
    row('E[(Z_T - Z_0)^2]  =  T', float(np.mean(traces[:, -1] ** 2)),
        float(walk_steps))
    row('[Z]_T averaged over traces', float(np.mean(qvs)), float(walk_steps))
    print(f'  ...but [Z]_T is identical on every trace (min {qvs.min():.0f}, '
          f'max {qvs.max():.0f}): quadratic variation is certain, not an '
          'average')

    print('\nBrownian motion as the scaled limit z^(n)_t = Z_{nt}/sqrt(n), '
          'at t = 1')
    print(f'  {"n":>8s}{"mean":>14s}{"variance":>14s}{"P[|z_1| < 1.96]":>18s}')
    for n in [1, 4, 25, 400]:
        ends = SimpleRandomWalk(1).scaled_traces(n, 20000, rng)[:, -1]
        print(f'  {n:8d}{np.mean(ends):14.6f}{np.var(ends):14.6f}'
              f'{np.mean(np.abs(ends) < 1.96):18.6f}')
    print(f'  {"N(0, 1)":>8s}{0.:14.6f}{1.:14.6f}{0.95:18.6f}')

    print('\nOne sample trace over [0, 1], refined: total variation diverges,'
          ' quadratic variation does not')
    print(f'  {"steps":>8s}{"total variation":>18s}'
          f'{"quadratic variation":>21s}{"max |dz|/dt":>14s}')
    for steps in [100, 1000, 10000, 100000]:
        _, single = brownian_paths(1., steps, 1, rng)
        print(f'  {steps:8d}{total_variation(single[0]):18.4f}'
              f'{quadratic_variation(single[0]):21.6f}'
              f'{np.max(np.abs(np.diff(single[0]))) * steps:14.1f}')
    print('  [z]_T -> T = 1, while total variation and the difference '
          'quotient both blow up')

    print('\nIto integral of X = z over [0, 1]: the left endpoint matters')
    times, paths = brownian_paths(1., 2000, 20000, rng)
    ito = ito_integral(paths, paths)
    strat = stratonovich_integral(paths, paths)
    ends = paths[:, -1]
    print(f'  {"quantity":40s}{"simulated":>14s}{"appendix":>14s}')
    row('E[int z dz]  (martingale)', float(np.mean(ito)), 0.)
    row('E[(int z dz)^2]  (Ito isometry: T^2/2)',
        float(np.mean(ito ** 2)), 0.5)
    row('mean |int z dz - (z_T^2 - T)/2|',
        float(np.mean(np.abs(ito - (ends ** 2 - 1) / 2))), 0.)
    row('mean |Stratonovich - z_T^2/2|',
        float(np.mean(np.abs(strat - ends ** 2 / 2))), 0.)
    print('  the two integrals of the same trace differ by exactly T/2 = 0.5:',
          f'{float(np.mean(strat - ito)):.6f}')

    print('\nIto lemma: d log(x) = (μ - σ^2/2) dt + σ dz for a GBM x')
    gbm = GeometricBrownianMotion(x_0=100., μ=0.08, σ=0.25)
    print(f'  {"steps over [0, 1]":>18s}{"max |closed form - Euler|":>28s}')
    for steps in [500, 5000, 50000]:
        times, paths = brownian_paths(1., steps, 200, rng)
        gap = np.abs(gbm.exact_paths(times, paths)[:, -1]
                     - gbm.euler_paths(times, paths)[:, -1])
        print(f'  {steps:18d}{float(np.max(gap)):28.6f}')
    print('  the SDE and the closed form Ito\'s lemma gives for it agree as '
          'dt -> 0')

    print('\nGeometric Brownian motion, x_0 = 100, μ = 0.08, σ = 0.25, T = 1')
    z_T = rng.normal(0., 1., size=(500000, 1))
    x_T = gbm.exact_paths(np.array([1.]), z_T)[:, 0]
    print(f'  {"quantity":40s}{"simulated":>14s}{"appendix":>14s}')
    row('E[log(x_T)] = log(x_0) + (μ-σ^2/2)T',
        float(np.mean(np.log(x_T))), gbm.log_mean(1.))
    row('Var[log(x_T)] = σ^2 T',
        float(np.var(np.log(x_T))), gbm.log_variance(1.))
    row('E[x_T] = x_0 e^{μT}', float(np.mean(x_T)), gbm.mean(1.))
    row('Var[x_T]', float(np.var(x_T)), gbm.variance(1.))

    print('\nOrnstein-Uhlenbeck (mean-reverting), x_0 = 5, μ = -1.5, σ = 0.4')
    ou = OrnsteinUhlenbeck(x_0=5., μ=-1.5, σ=0.4)
    times, paths = brownian_paths(4., 4000, 4000, rng)
    ou_paths = ou.euler_paths(times, paths)
    print(f'  {"quantity":40s}{"simulated":>14s}{"appendix":>14s}')
    for T in [0.5, 2.0, 4.0]:
        index = int(T / 4. * 4000)
        row(f'E[x_T] at T = {T}', float(np.mean(ou_paths[:, index])),
            ou.mean(T))
        row(f'Var[x_T] at T = {T}', float(np.var(ou_paths[:, index])),
            ou.variance(T))
    row('Var[x_T] as T -> infinity  (-σ^2/2μ)',
        float(np.var(ou_paths[:, -1])), ou.stationary_variance())
    print()
