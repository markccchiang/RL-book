'''Portfolio theory: the efficient frontier and the portfolios on it.

The code here follows book/appendix2 (Appendix B of the book), which states
a chain of results about the frontier -- its equation, the GMVP, orthogonal
portfolios, the two-fund theorem, CAPM, the cross-sectional variance of the
betas, and the tangency portfolio once a riskless asset exists -- without
proving them.  Every one of those is computed here and checked against an
independent route: the frontier equation against numerical constrained
minimization, the tangency portfolio against maximizing the Sharpe ratio,
the beta-variance maximizer against a scan of the frontier, and so on.

By default it runs on a small deterministic economy so it works offline and
gives the same numbers every time:

    python -m rl.appendix2.efficient_frontier

The figure in the appendix comes from real market data, which needs the
optional yfinance dependency and a network connection:

    python -m rl.appendix2.efficient_frontier --live
    python -m rl.appendix2.efficient_frontier --live --plot

Everything below works off the mean-return vector R and the covariance
matrix V, which is all portfolio theory needs; where those come from --
16 tickers or 6 made-up assets -- changes nothing about the results.
'''

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar


@dataclass(frozen=True)
class EfficientFrontier:
    '''The efficient frontier of n risky assets with means R, covariance V.'''

    mean_returns: np.ndarray
    covariance: np.ndarray
    names: Sequence[str]

    @cached_property
    def inverse_covariance(self) -> np.ndarray:
        return np.linalg.inv(self.covariance)

    @cached_property
    def abc(self) -> Tuple[float, float, float]:
        '''a = R' V^-1 R, b = R' V^-1 1, c = 1' V^-1 1.'''
        weighted = self.inverse_covariance.dot(self.mean_returns)
        return (float(self.mean_returns.dot(weighted)),
                float(np.sum(weighted)),
                float(np.sum(self.inverse_covariance)))

    def variance(self, mean: np.ndarray) -> np.ndarray:
        '''The frontier: σ_p^2 = (a - 2 b r_p + c r_p^2) / (ac - b^2).'''
        a, b, c = self.abc
        return (a - 2 * b * mean + c * mean ** 2) / (a * c - b ** 2)

    def weights(self, mean: float) -> np.ndarray:
        '''X_p, from the Lagrangian: the minimum-variance weights for r_p.'''
        a, b, c = self.abc
        determinant = a * c - b ** 2
        ones = np.ones(len(self.mean_returns))
        return (self.inverse_covariance.dot(
            ones * (a - b * mean) + self.mean_returns * (c * mean - b)
        ) / determinant)

    def gmvp_weights(self) -> np.ndarray:
        '''X_0 = V^-1 1 / c, the tip of the frontier.'''
        _, _, c = self.abc
        return self.inverse_covariance.dot(np.ones(len(self.mean_returns))) / c

    def sep_weights(self) -> np.ndarray:
        '''X_1 = V^-1 R / b, the Special Efficient Portfolio.'''
        _, b, _ = self.abc
        return self.inverse_covariance.dot(self.mean_returns) / b

    def orthogonal_mean(self, mean: float) -> float:
        '''r_z = (a - b r_p) / (b - c r_p), the portfolio orthogonal to p.'''
        a, b, c = self.abc
        return (a - b * mean) / (b - c * mean)

    def betas(self, mean: float) -> np.ndarray:
        '''β_p = V X_p / σ_p^2, the CAPM slopes against efficient p.'''
        weights = self.weights(mean)
        return self.covariance.dot(weights) / self.portfolio_variance(weights)

    def tangency_mean(self, riskless: float) -> float:
        '''r_T, where the line from (0, r_F) touches the frontier.

        This is the same formula as the orthogonal mean, which is not a
        coincidence: r_z is an involution on the frontier, and the tangency
        portfolio is the one whose orthogonal portfolio has mean r_F.
        '''
        return self.orthogonal_mean(riskless)

    def portfolio_mean(self, weights: np.ndarray) -> float:
        return float(weights.dot(self.mean_returns))

    def portfolio_variance(self, weights: np.ndarray) -> float:
        return float(weights.dot(self.covariance).dot(weights))

    def covariance_between(self, left: np.ndarray,
                           right: np.ndarray) -> float:
        return float(left.dot(self.covariance).dot(right))

    def minimize_variance(self, mean: float) -> np.ndarray:
        '''The same weights by numerical optimization rather than algebra.'''
        size = len(self.mean_returns)
        result = minimize(
            self.portfolio_variance,
            x0=np.ones(size) / size,
            constraints=[
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq',
                 'fun': lambda x: x.dot(self.mean_returns) - mean}
            ],
            tol=1e-14,
            options={'maxiter': 1000}
        )
        return result.x


def factor_model_economy(
        num_assets: int = 6,
        seed: int = 1729
) -> EfficientFrontier:
    '''A small deterministic economy: one factor plus idiosyncratic risk.

    Annualized numbers in the range real equities live in, so the frontier
    below is quantitatively sensible without needing to be downloaded.
    '''
    rng = np.random.default_rng(seed)
    exposures = rng.uniform(0.6, 1.5, num_assets)
    idiosyncratic = rng.uniform(0.10, 0.30, num_assets)
    factor_variance = 0.16 ** 2
    covariance = (factor_variance * np.outer(exposures, exposures)
                  + np.diag(idiosyncratic ** 2))
    mean_returns = 0.02 + 0.05 * exposures + rng.normal(0., 0.02, num_assets)
    return EfficientFrontier(
        mean_returns=mean_returns,
        covariance=covariance,
        names=[f'A{i + 1}' for i in range(num_assets)]
    )


def market_economy(
        tickers: Sequence[str],
        start: datetime,
        end: datetime,
        days: int = 1
) -> EfficientFrontier:
    '''Annualized means and covariance from Yahoo daily closes.

    Needs the optional yfinance dependency, imported here so that the rest
    of this module works without it.
    '''
    import yfinance as yf

    prices = yf.download(tickers, start=start, end=end, auto_adjust=True,
                         progress=False)['Close']
    # yfinance sorts the tickers; restore the caller's order so that the
    # columns line up with the labels used when annotating the plot.
    prices = prices[list(tickers)]
    percent_change = prices.pct_change(periods=days)
    factor = 252. / days
    return EfficientFrontier(
        mean_returns=percent_change.mean().to_numpy() * factor,
        covariance=percent_change.cov().to_numpy() * factor,
        names=list(tickers)
    )


def plot_frontier(frontier: EfficientFrontier) -> None:
    '''The appendix's figure: mean against standard deviation.'''
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    a, b, c = frontier.abc
    gmvp_mean, sep_mean = b / c, a / b
    asset_stdev = np.sqrt(np.diagonal(frontier.covariance))

    top = max(sep_mean, float(np.max(frontier.mean_returns))) * 1.1
    bottom = min(0., float(np.min(frontier.mean_returns))) - 0.05
    means = np.linspace(bottom, top, 500)
    stdevs = np.sqrt(frontier.variance(means))

    _, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    ax.set_xlabel('Standard Deviation of Returns (Annualized)', fontsize=16)
    ax.set_ylabel('Mean Returns (Annualized)', fontsize=16)
    ax.set_title('Historical Returns Mean versus Standard Deviation',
                 fontsize=20)
    ax.tick_params(labelsize=11)
    percent = FuncFormatter(lambda x, _: f'{x * 100:.1f}%')
    ax.xaxis.set_major_formatter(percent)
    ax.yaxis.set_major_formatter(percent)
    ax.grid()
    # limits come from the data, so this works for any set of assets
    ax.set_xlim(left=0., right=max(float(np.max(asset_stdev)),
                                   float(np.max(stdevs))) * 1.05)
    ax.set_ylim(bottom=bottom, top=top)
    ax.scatter(stdevs, means, s=4)
    ax.scatter(asset_stdev, frontier.mean_returns)
    for point, mean, label in [
            (np.sqrt(1 / c), gmvp_mean, 'GMVP'),
            (np.sqrt(a / b ** 2), sep_mean, 'SEP')]:
        ax.scatter(point, mean, marker='x', c='black', s=100)
        ax.annotate(label, xy=(point, mean), fontsize=15)
    for name, x, y in zip(frontier.names, asset_stdev, frontier.mean_returns):
        ax.annotate(name, xy=(x, y))
    plt.show()


def tangency_by_sharpe(frontier: EfficientFrontier, riskless: float) -> float:
    """r_T found by walking the frontier instead of by formula.

    Below the GMVP the tangent from (0, r_F) touches the upper branch and
    maximizes the Sharpe ratio; above it, it touches the lower branch, where
    the steepest line is the most negative one.
    """
    _, b, c = frontier.abc
    gmvp_mean = b / c

    def negative_slope(mean: float) -> float:
        stdev = float(np.sqrt(frontier.variance(np.array(mean))))
        return -abs(mean - riskless) / stdev

    bounds = (gmvp_mean + 1e-9, gmvp_mean + 5.) if riskless < gmvp_mean \
        else (gmvp_mean - 5., gmvp_mean - 1e-9)
    return float(minimize_scalar(negative_slope, bounds=bounds,
                                 method='bounded',
                                 options={'xatol': 1e-13}).x)


def row(label: str, *values: float) -> None:
    '''Print a label and a row of numbers, aligned under a header.'''
    print(f'  {label:46s}' + ''.join(f'{v:16.8f}' for v in values))


def report(frontier: EfficientFrontier) -> None:
    '''Check every result the appendix states, against an independent route.'''
    rng = np.random.default_rng(20200917)
    a, b, c = frontier.abc
    size = len(frontier.mean_returns)
    gmvp, sep = frontier.gmvp_weights(), frontier.sep_weights()
    targets = [0.04, 0.08, 0.12]

    print(f'\n{size} assets, a = {a:.6f}, b = {b:.6f}, c = {c:.6f}')

    print('\n1. The frontier equation, against numerical minimization')
    print(f'  {"target mean":46s}{"σ^2 by formula":>16s}'
          f'{"σ^2 by SLSQP":>16s}{"max |ΔX|":>16s}')
    for target in targets:
        by_algebra = frontier.weights(target)
        by_solver = frontier.minimize_variance(target)
        row(f'r_p = {target}',
            float(frontier.variance(np.array(target))),
            frontier.portfolio_variance(by_solver),
            float(np.max(np.abs(by_algebra - by_solver))))

    print('\n2. GMVP and SEP')
    print(f'  {"quantity":46s}{"computed":>16s}{"appendix":>16s}')
    row('GMVP mean  =  b / c', frontier.portfolio_mean(gmvp), b / c)
    row('GMVP variance  =  1 / c', frontier.portfolio_variance(gmvp), 1 / c)
    row('SEP mean  =  a / b', frontier.portfolio_mean(sep), a / b)
    row('SEP variance  =  a / b^2', frontier.portfolio_variance(sep),
        a / b ** 2)
    random_weights = rng.normal(0., 1., (200, size))
    random_weights /= random_weights.sum(axis=1, keepdims=True)
    with_gmvp = np.array([frontier.covariance_between(gmvp, w)
                          for w in random_weights])
    with_assets = frontier.covariance.dot(gmvp)
    row('cov(GMVP, any portfolio)  =  1 / c',
        float(np.max(np.abs(with_gmvp - 1 / c))), 0.)
    row('cov(GMVP, any asset)  =  1 / c',
        float(np.max(np.abs(with_assets - 1 / c))), 0.)

    print('\n3. Orthogonal efficient portfolios')
    print(f'  {"r_p":>12s}{"r_z":>16s}{"cov(p, z)":>16s}'
          f'{"tangent intercept":>20s}')
    for target in targets:
        orthogonal = frontier.orthogonal_mean(target)
        covariance = frontier.covariance_between(
            frontier.weights(target), frontier.weights(orthogonal))
        # in (σ, r) space the tangent at p meets the mean axis at r_z
        step = 1e-6
        stdev = np.sqrt(frontier.variance(np.array(target)))
        slope = (2 * step) / (np.sqrt(frontier.variance(
            np.array(target + step))) - np.sqrt(frontier.variance(
                np.array(target - step))))
        print(f'  {target:12.4f}{orthogonal:16.8f}{covariance:16.2e}'
              f'{float(target - slope * stdev):20.8f}')

    print('\n4. Two-fund theorem: every efficient portfolio is α GMVP + '
          '(1-α) SEP')
    print(f'  {"target mean":46s}{"α":>16s}{"max |ΔX|":>16s}')
    for target in targets:
        α = (a / b - target) / (a / b - b / c)
        combination = α * gmvp + (1 - α) * sep
        row(f'r_p = {target}', α,
            float(np.max(np.abs(combination - frontier.weights(target)))))

    print('\n5. CAPM: R = r_z 1 + (r_p - r_z) β_p, for efficient p only')
    print(f'  {"portfolio":46s}{"max |R - CAPM prediction|":>28s}')
    for target in targets:
        orthogonal = frontier.orthogonal_mean(target)
        predicted = orthogonal + (target - orthogonal) * frontier.betas(target)
        error = float(np.max(np.abs(frontier.mean_returns - predicted)))
        print(f'  {f"efficient, r_p = {target}":46s}{error:28.2e}')
    inefficient = np.ones(size) / size
    mean = frontier.portfolio_mean(inefficient)
    betas = frontier.covariance.dot(inefficient) \
        / frontier.portfolio_variance(inefficient)
    orthogonal = frontier.orthogonal_mean(mean)
    predicted = orthogonal + (mean - orthogonal) * betas
    print(f'  {"equally weighted (not on the frontier)":46s}'
          f'{float(np.max(np.abs(frontier.mean_returns - predicted))):28.2e}')

    print('\n6. Cross-sectional variance of the betas')
    scan = np.linspace(b / c - 0.6, b / c + 0.6, 240001)
    spread = np.array([np.var(frontier.betas(m)) for m in scan])
    print(f'  {"quantity":46s}{"by scanning":>16s}{"appendix":>16s}')
    row('variance of β at the GMVP', float(np.var(frontier.betas(b / c))), 0.)
    row('maximizing r_p, below the GMVP',
        float(scan[:120000][int(np.argmax(spread[:120000]))]),
        b / c - np.sqrt(a * c - b ** 2) / c)
    row('maximizing r_p, above the GMVP',
        float(scan[120001:][int(np.argmax(spread[120001:]))]),
        b / c + np.sqrt(a * c - b ** 2) / c)
    for sign, side in [(-1, 'below'), (1, 'above')]:
        extreme = b / c + sign * np.sqrt(a * c - b ** 2) / c
        row(f'  variance there ({side})  =  2 / c',
            float(frontier.variance(np.array(extreme))), 2 / c)

    print('\n7. The efficient set once a riskless asset exists')
    print(f'  {"r_F":>12s}{"r_T (formula)":>18s}{"r_T (max Sharpe)":>20s}'
          f'{"r_T vs r_F":>14s}{"correlation":>14s}')
    for riskless in [0.005, 0.02, 0.05]:
        tangency = frontier.tangency_mean(riskless)
        by_sharpe = tangency_by_sharpe(frontier, riskless)
        # every portfolio on the efficient set is riskless plus tangency, so
        # any two of them are perfectly correlated
        tangency_weights = frontier.weights(tangency)
        mixes = [0.3 * tangency_weights, 0.9 * tangency_weights]
        correlation = frontier.covariance_between(*mixes) / np.sqrt(
            frontier.portfolio_variance(mixes[0])
            * frontier.portfolio_variance(mixes[1]))
        print(f'  {riskless:12.4f}{tangency:18.8f}{by_sharpe:20.8f}'
              f'{"above" if tangency > riskless else "below":>14s}'
              f'{correlation:14.8f}')
    print(f'  the GMVP mean is {b / c:.4f}: r_T sits above r_F when r_F is '
          'below it, and below when above')
    print()


if __name__ == '__main__':

    import sys

    live = '--live' in sys.argv
    if live:
        economy = market_economy(
            tickers=['IBM', 'GOOG', 'AAPL', 'TGT', 'GS', 'MS', 'AMZN',
                     'MSFT', 'WMT', 'NKE', 'UNH', 'PG', 'DB', 'C', 'META',
                     'NVDA'],
            start=datetime(2017, 9, 17),
            end=datetime(2020, 9, 17)
        )
    else:
        economy = factor_model_economy()

    report(economy)

    if '--plot' in sys.argv:
        plot_frontier(economy)
