'''The Black-Scholes equation and its call/put solution.

The code here follows book/appendix5 (Appendix E of the book).  The appendix
derives the PDE

    dV/dt + (σ^2/2) S^2 d^2V/dS^2 + r S dV/dS - r V = 0

by forming the riskless portfolio Π = -V + (dV/dS) S, and then solves it in
closed form for European calls and puts.  This module exercises both halves:

  * the *equation*, by checking that the closed-form price satisfies it, by
    reproducing the change of variables that turns it into the heat
    equation, and by building the hedged portfolio the derivation is founded
    on and confirming that it really is riskless;
  * the *solution*, by put-call parity and by risk-neutral Monte Carlo.

The drift μ is the thread running through it.  It appears in the underlying's
process and nowhere in the equation, so the third of the appendix's closing
points — that a derivative's price does not depend on the expected return of
its underlying — is a claim simulation can check directly, and does below.

Run it with:

    python -m rl.appendix5.black_scholes
'''

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BlackScholes:
    '''A European option on a lognormal underlying.

    dS_t = μ S_t dt + σ S_t dz_t, riskless rate r, strike K, expiry T.
    '''

    K: float
    T: float
    r: float
    σ: float
    μ: float
    is_call: bool = True

    def d1_d2(self, t: float, spot: np.ndarray) -> Tuple[np.ndarray,
                                                         np.ndarray]:
        '''d1 = (log(S/K) + (r + σ^2/2)τ) / (σ√τ) and d2 = d1 - σ√τ.'''
        τ = max(self.T - t, 1e-12)
        vol = self.σ * np.sqrt(τ)
        d1 = (np.log(spot / self.K) + (self.r + self.σ ** 2 / 2) * τ) / vol
        return d1, d1 - vol

    def price(self, t: float, spot: np.ndarray) -> np.ndarray:
        '''C = S N(d1) - K e^{-rτ} N(d2), or the corresponding put.'''
        τ = max(self.T - t, 0.)
        d1, d2 = self.d1_d2(t, spot)
        discounted = self.K * np.exp(-self.r * τ)
        if self.is_call:
            return spot * norm.cdf(d1) - discounted * norm.cdf(d2)
        return discounted * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def delta(self, t: float, spot: np.ndarray) -> np.ndarray:
        '''dV/dS = N(d1) for a call, N(d1) - 1 for a put.

        This is the number of units of the underlying held in the riskless
        portfolio of the derivation.
        '''
        d1, _ = self.d1_d2(t, spot)
        return norm.cdf(d1) if self.is_call else norm.cdf(d1) - 1

    def payoff(self, spot: np.ndarray) -> np.ndarray:
        '''max(S - K, 0) for a call, max(K - S, 0) for a put.'''
        return np.maximum(spot - self.K, 0.) if self.is_call \
            else np.maximum(self.K - spot, 0.)

    def pde_residual(
            self,
            t: float,
            spot: np.ndarray,
            discount_sign: float = -1.,
            h: float = 1e-4
    ) -> np.ndarray:
        '''V_t + (σ^2/2) S^2 V_SS + r S V_S + discount_sign r V.

        Zero everywhere for the closed-form price with discount_sign = -1.
        The sign is a parameter because the appendix's final displayed
        equation prints "+ r V" while the derivation just above it gives
        "- r V"; main() evaluates both.
        '''
        dS = h * spot
        value = self.price(t, spot)
        v_t = (self.price(t + h, spot) - self.price(t - h, spot)) / (2 * h)
        v_s = (self.price(t, spot + dS)
               - self.price(t, spot - dS)) / (2 * dS)
        v_ss = (self.price(t, spot + dS) - 2 * value
                + self.price(t, spot - dS)) / dS ** 2
        return (v_t + self.σ ** 2 / 2 * spot ** 2 * v_ss
                + self.r * spot * v_s + discount_sign * self.r * value)

    def heat_equation_residual(
            self,
            τ: float,
            x: np.ndarray,
            h: float = 1e-4
    ) -> np.ndarray:
        '''u_τ - (σ^2/2) u_xx, for u(τ, x) = C(t, S) e^{rτ}.

        With τ = T - t and x = log(S/K) + (r - σ^2/2)τ, the appendix's
        change of variables turns Black-Scholes into the heat equation.
        '''
        def u(tau: float, x_value: np.ndarray) -> np.ndarray:
            spot = self.K * np.exp(x_value - (self.r - self.σ ** 2 / 2) * tau)
            return self.price(self.T - tau, spot) * np.exp(self.r * tau)

        u_τ = (u(τ + h, x) - u(τ - h, x)) / (2 * h)
        u_xx = (u(τ, x + h) - 2 * u(τ, x) + u(τ, x - h)) / h ** 2
        return u_τ - self.σ ** 2 / 2 * u_xx

    def transformed_payoff(self, x: np.ndarray) -> np.ndarray:
        '''u(0, x) = K (e^{max(x, 0)} - 1): the heat equation's initial
        condition, which the terminal payoff transforms into.'''
        return self.K * (np.exp(np.maximum(x, 0.)) - 1)

    def paths(
            self,
            spot_0: float,
            steps: int,
            num_paths: int,
            rng: np.random.Generator,
            drift: Optional[float] = None
    ) -> np.ndarray:
        '''Underlying paths, defaulting to the real-world drift μ.

        Pass drift=r to sample under the risk-neutral measure instead.
        '''
        rate = self.μ if drift is None else drift
        dt = self.T / steps
        shocks = rng.normal(0., np.sqrt(dt), size=(num_paths, steps))
        increments = (rate - self.σ ** 2 / 2) * dt + self.σ * shocks
        return spot_0 * np.exp(
            np.concatenate([np.zeros((num_paths, 1)),
                            np.cumsum(increments, axis=1)], axis=1)
        )

    def hedged_pnl(self, spot_paths: np.ndarray) -> np.ndarray:
        '''P&L of selling the option and running the delta hedge to expiry.

        Short one option, hold dV/dS units of the underlying, keep the rest
        in the riskless asset, and rebalance at every step.  The derivation
        says this portfolio is riskless, so the P&L should be 0 on every
        path, up to the error from rebalancing discretely.
        '''
        num_paths, points = spot_paths.shape
        steps = points - 1
        dt = self.T / steps
        growth = np.exp(self.r * dt)

        held = self.delta(0., spot_paths[:, 0])
        cash = self.price(0., spot_paths[:, 0]) - held * spot_paths[:, 0]
        for i in range(1, steps):
            cash *= growth
            wanted = self.delta(i * dt, spot_paths[:, i])
            cash -= (wanted - held) * spot_paths[:, i]
            held = wanted
        cash *= growth
        return cash + held * spot_paths[:, -1] - self.payoff(spot_paths[:, -1])

    def unhedged_pnl(self, spot_paths: np.ndarray) -> np.ndarray:
        '''P&L of selling the option and doing nothing at all.'''
        premium = self.price(0., spot_paths[:, 0]) * np.exp(self.r * self.T)
        return premium - self.payoff(spot_paths[:, -1])


def row(label: str, *values: float) -> None:
    '''Print a label and a row of numbers, aligned under a header.'''
    print(f'  {label:40s}' + ''.join(f'{v:16.8f}' for v in values))


def scan(label: str, spots: np.ndarray,
         quantity: Callable[[np.ndarray], np.ndarray]) -> None:
    '''Print one quantity across a row of underlying prices.'''
    print(f'  {label:40s}' + ''.join(f'{v:16.8f}' for v in quantity(spots)))


if __name__ == '__main__':

    rng = np.random.default_rng(1729)

    call = BlackScholes(K=100., T=1., r=0.03, σ=0.2, μ=0.15, is_call=True)
    put = BlackScholes(K=100., T=1., r=0.03, σ=0.2, μ=0.15, is_call=False)
    spots = np.array([80., 100., 120.])
    spot_0 = 100.

    print('\n1. Does the closed-form price satisfy the Black-Scholes '
          'equation?')
    print(f'  {"S":40s}' + ''.join(f'{s:16.2f}' for s in spots))
    scan('C(0, S)', spots, lambda s: call.price(0., s))
    scan('residual with  - r V  (derivation)', spots,
         lambda s: call.pde_residual(0.5, s, discount_sign=-1.))
    scan('residual with  + r V  (as printed)', spots,
         lambda s: call.pde_residual(0.5, s, discount_sign=+1.))
    scan('  which is exactly 2 r V', spots,
         lambda s: 2 * call.r * call.price(0.5, s))

    print('\n2. The change of variables that solves it')
    x_values = np.array([-0.4, 0., 0.4])
    print(f'  {"x":40s}' + ''.join(f'{x:16.2f}' for x in x_values))
    scan('u_τ - (σ^2/2) u_xx  at τ = 0.5', x_values,
         lambda x: call.heat_equation_residual(0.5, x))
    scan('u(τ, x) near τ = 0', x_values,
         lambda x: call.price(call.T - 1e-8, call.K * np.exp(
             x - (call.r - call.σ ** 2 / 2) * 1e-8)) * np.exp(call.r * 1e-8))
    scan('K (e^{max(x, 0)} - 1)', x_values, call.transformed_payoff)

    print('\n3. Put-call parity: C - P = S - K e^{-rT}')
    print(f'  {"S":40s}' + ''.join(f'{s:16.2f}' for s in spots))
    scan('C(0, S) - P(0, S)', spots,
         lambda s: call.price(0., s) - put.price(0., s))
    scan('S - K e^{-rT}', spots,
         lambda s: s - call.K * np.exp(-call.r * call.T))

    print('\n4. The riskless portfolio the derivation is built on')
    paths = call.paths(spot_0, steps=500, num_paths=50000, rng=rng)
    hedged = call.hedged_pnl(paths)
    unhedged = call.unhedged_pnl(paths)
    print(f'  {"P&L of selling one call":40s}{"mean":>16s}{"std dev":>16s}'
          f'{"5th pct":>16s}')
    row('delta-hedged, 500 rebalances',
        float(np.mean(hedged)), float(np.std(hedged)),
        float(np.percentile(hedged, 5)))
    row('not hedged at all',
        float(np.mean(unhedged)), float(np.std(unhedged)),
        float(np.percentile(unhedged, 5)))
    print(f'\n  {"rebalances":>12s}{"std dev of hedged P&L":>26s}')
    for steps in [10, 50, 250, 1250]:
        pnl = call.hedged_pnl(call.paths(spot_0, steps, 20000, rng))
        print(f'  {steps:12d}{float(np.std(pnl)):26.8f}')
    print('  the hedge error shrinks with the rebalancing interval: the '
          'portfolio is riskless only in the limit')

    print('\n5. μ appears in the underlying, but not in the price')
    print(f'  {"μ":>12s}{"E[S_T]":>16s}{"hedged P&L mean":>18s}'
          f'{"hedged P&L std":>18s}')
    for drift in [-0.10, 0.05, 0.15, 0.40]:
        moved = BlackScholes(K=100., T=1., r=0.03, σ=0.2, μ=drift)
        pnl = moved.hedged_pnl(moved.paths(spot_0, 500, 20000, rng))
        ends = moved.paths(spot_0, 1, 20000, rng)[:, -1]
        print(f'  {drift:12.2f}{float(np.mean(ends)):16.4f}'
              f'{float(np.mean(pnl)):18.8f}{float(np.std(pnl)):18.8f}')
    print('  the same price and the same hedge work whatever the underlying '
          'is expected to return')

    print('\n6. Risk-neutral Monte Carlo: e^{-rT} E[payoff]')
    print(f'  {"sampling measure":40s}{"MC price":>16s}'
          f'{"closed form":>16s}')
    for name, drift in [('risk-neutral (drift = r)', call.r),
                        ('real-world (drift = μ)', call.μ)]:
        ends = call.paths(spot_0, 1, 2000000, rng, drift=drift)[:, -1]
        mc = float(np.exp(-call.r * call.T) * np.mean(call.payoff(ends)))
        row(name, mc, float(call.price(0., np.array(spot_0))))
    print()
