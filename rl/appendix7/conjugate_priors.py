'''Conjugate priors for the Gaussian and Bernoulli distributions.

The code here follows book/appendix7 (Appendix G of the book), which states
the two updates without deriving them:

  * Bernoulli data with a Beta prior on p:
        α_{n+1} = α_n + 1[x_{n+1} = 1],  β_{n+1} = β_n + 1[x_{n+1} = 0]
  * Gaussian data with a Gaussian-Inverse-Gamma prior on (μ, σ^2):
        θ_{n+1} = (n θ_n + x_{n+1}) / (n + 1),   α_{n+1} = α_n + 1/2,
        β_{n+1} = β_n + n (x_{n+1} - θ_n)^2 / (2 (n + 1))

Since the appendix asks the reader to take these on trust, this module earns
that trust three ways, in increasing order of how much they would catch:

  1. against Bayes' theorem itself — prior times likelihood, normalized on a
     grid, must equal the density the update claims;
  2. against the batch formulas the recursions telescope into;
  3. against reality, by checking that the posterior is *calibrated*: draw a
     parameter from the prior, generate data from it, and the 90% credible
     interval should contain that parameter 90% of the time.

The third is the one with teeth, and the last section shows what it looks
like when an update is subtly wrong.

The appendix writes the Gaussian prior with the pseudo-count tied to the
data count n.  Here it is a separate field ν, so that the prior can be
proper before any data arrives (the appendix's form is ν_n = n); this is
also how rl/chapter14/ts_gaussian.py carries it.

Run it with:

    python -m rl.appendix7.conjugate_priors
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import invgamma, norm, t


@dataclass(frozen=True)
class BetaBernoulli:
    '''Beta(α, β) prior for the p of Bernoulli data.'''

    α: float
    β: float

    def updated(self, x: int) -> BetaBernoulli:
        '''α += 1[x = 1], β += 1[x = 0].'''
        return BetaBernoulli(α=self.α + (x == 1), β=self.β + (x == 0))

    def updated_all(self, xs: Iterable[int]) -> BetaBernoulli:
        posterior = self
        for x in xs:
            posterior = posterior.updated(x)
        return posterior

    def mean(self) -> float:
        '''E[p] = α / (α + β).'''
        return self.α / (self.α + self.β)

    def pdf(self, p: np.ndarray) -> np.ndarray:
        return beta_dist.pdf(p, self.α, self.β)

    def interval(self, level: float = 0.9) -> Tuple[float, float]:
        '''Equal-tailed credible interval for p.'''
        tail = (1 - level) / 2
        return (float(beta_dist.ppf(tail, self.α, self.β)),
                float(beta_dist.ppf(1 - tail, self.α, self.β)))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        '''Draws of p, i.e. what Thompson sampling asks the posterior for.'''
        return rng.beta(self.α, self.β, size=n)


@dataclass(frozen=True)
class NormalInverseGamma:
    '''Gaussian-Inverse-Gamma prior for the (μ, σ^2) of Gaussian data.

    μ | σ^2 ~ N(θ, σ^2 / ν) and σ^2 ~ IG(α, β), so 1/σ^2 is Gamma(α, β) --
    the form the appendix states, with its n carried here as ν.
    '''

    θ: float
    ν: float
    α: float
    β: float

    def updated(self, x: float) -> NormalInverseGamma:
        '''The appendix's three updates, with ν in place of n.'''
        return NormalInverseGamma(
            θ=(self.ν * self.θ + x) / (self.ν + 1),
            ν=self.ν + 1,
            α=self.α + 0.5,
            β=self.β + self.ν * (x - self.θ) ** 2 / (2 * (self.ν + 1))
        )

    def updated_all(self, xs: Iterable[float]) -> NormalInverseGamma:
        posterior = self
        for x in xs:
            posterior = posterior.updated(x)
        return posterior

    def mean_of_variance(self) -> float:
        '''E[σ^2] = β / (α - 1), which needs α > 1.'''
        return self.β / (self.α - 1)

    def log_pdf(self, μ: np.ndarray, variance: np.ndarray) -> np.ndarray:
        '''log joint density over (μ, σ^2), up to an additive constant.'''
        return (-(self.α + 1.5) * np.log(variance)
                - (2 * self.β + self.ν * (μ - self.θ) ** 2) / (2 * variance))

    def sample(
            self,
            n: int,
            rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Draws of (μ, σ^2): σ^2 from IG(α, β), then μ from N(θ, σ^2/ν).'''
        variance = self.β / rng.gamma(shape=self.α, scale=1., size=n)
        return rng.normal(self.θ, np.sqrt(variance / self.ν)), variance

    def interval(self, level: float = 0.9) -> Tuple[Tuple[float, float],
                                                    Tuple[float, float]]:
        '''Equal-tailed credible intervals for μ (Student-t) and σ^2 (IG).'''
        tail = (1 - level) / 2
        scale = np.sqrt(self.β / (self.α * self.ν))
        μ_interval = (float(t.ppf(tail, 2 * self.α, self.θ, scale)),
                      float(t.ppf(1 - tail, 2 * self.α, self.θ, scale)))
        var_interval = (float(invgamma.ppf(tail, self.α, scale=self.β)),
                        float(invgamma.ppf(1 - tail, self.α, scale=self.β)))
        return μ_interval, var_interval

    def predictive_cdf(self, x: np.ndarray) -> np.ndarray:
        '''P[x_{n+1} <= x], integrating μ and σ^2 out: a Student-t.'''
        scale = np.sqrt(self.β * (self.ν + 1) / (self.α * self.ν))
        return t.cdf(x, 2 * self.α, self.θ, scale)


def normalized(log_density: np.ndarray) -> np.ndarray:
    '''exp of a log density, scaled to sum to 1 over the grid.'''
    shifted = np.exp(log_density - np.max(log_density))
    return shifted / np.sum(shifted)


def coverage(
        level: float,
        replications: int,
        rng: np.random.Generator,
        broken: bool = False
) -> Tuple[float, float, float]:
    '''Fraction of replications whose credible interval holds the truth.

    Each replication draws the parameters from the prior, generates data
    from them, and updates.  If the update is the correct posterior, the
    fraction has to come out at `level'.
    '''
    beta_prior = BetaBernoulli(α=2., β=3.)
    gauss_prior = NormalInverseGamma(θ=0., ν=1., α=3., β=4.)
    hits_p, hits_μ, hits_var = 0, 0, 0
    for _ in range(replications):
        p = float(rng.beta(beta_prior.α, beta_prior.β))
        draws = (rng.random(30) < p).astype(int)
        if broken:
            # counts the successes but never the failures
            posterior = BetaBernoulli(α=beta_prior.α + float(np.sum(draws)),
                                      β=beta_prior.β)
        else:
            posterior = beta_prior.updated_all(draws)
        low, high = posterior.interval(level)
        hits_p += low <= p <= high

        variance = float(gauss_prior.β / rng.gamma(gauss_prior.α))
        μ = float(rng.normal(gauss_prior.θ,
                             np.sqrt(variance / gauss_prior.ν)))
        gauss_posterior = gauss_prior.updated_all(
            rng.normal(μ, np.sqrt(variance), size=30)
        )
        (μ_low, μ_high), (var_low, var_high) = gauss_posterior.interval(level)
        hits_μ += μ_low <= μ <= μ_high
        hits_var += var_low <= variance <= var_high
    return (hits_p / replications, hits_μ / replications,
            hits_var / replications)


def row(label: str, *values: float) -> None:
    '''Print a label and a row of numbers, aligned under a header.'''
    print(f'  {label:44s}' + ''.join(f'{v:16.8f}' for v in values))


if __name__ == '__main__':

    rng = np.random.default_rng(1729)

    print('\n1. Beta-Bernoulli: is the update Bayes\' theorem?')
    prior = BetaBernoulli(α=2., β=3.)
    data = (rng.random(25) < 0.7).astype(int)
    posterior = prior.updated_all(data)
    grid = np.linspace(1e-6, 1 - 1e-6, 200001)
    successes = int(np.sum(data))
    by_bayes = normalized(
        np.log(prior.pdf(grid))
        + successes * np.log(grid)
        + (len(data) - successes) * np.log(1 - grid)
    )
    by_update = normalized(np.log(posterior.pdf(grid)))
    print(f'  {len(data)} draws, {successes} successes')
    row('prior Beta(α, β)', prior.α, prior.β)
    row('posterior by the appendix\'s update', posterior.α, posterior.β)
    print(f'  {"max |prior x likelihood - posterior| on a grid":44s}'
          f'{float(np.max(np.abs(by_bayes - by_update))):16.2e}')
    shuffled = prior.updated_all(rng.permutation(data))
    reorder_gap = abs(shuffled.α - posterior.α) + abs(shuffled.β - posterior.β)
    print(f'  {"same posterior when the data is reordered":44s}'
          f'{reorder_gap:16.2e}')

    print('\n2. Gaussian-Inverse-Gamma: the recursion against its batch form')
    start = NormalInverseGamma(θ=0., ν=0., α=1., β=0.)
    xs = rng.normal(2.5, 1.5, size=40)
    updated = start.updated_all(xs)
    n = len(xs)
    print(f'  {"quantity":44s}{"recursion":>16s}{"batch formula":>16s}')
    row('θ_n  =  mean of the data', updated.θ, float(np.mean(xs)))
    row('α_n  =  α_0 + n/2', updated.α, start.α + n / 2)
    row('β_n  =  β_0 + Σ(x - mean)^2 / 2', updated.β,
        start.β + float(np.sum((xs - np.mean(xs)) ** 2)) / 2)
    print('  the β recursion is Welford\'s algorithm: it accumulates the '
          'sum of squared')
    print('  deviations without ever storing the data or revisiting the mean')

    print('\n3. Gaussian-Inverse-Gamma: is *that* update Bayes\' theorem?')
    gauss_prior = NormalInverseGamma(θ=0.5, ν=2., α=3., β=4.)
    sample = rng.normal(2.5, 1.5, size=12)
    gauss_posterior = gauss_prior.updated_all(sample)
    μ_axis = np.linspace(-2., 6., 801)
    var_axis = np.linspace(0.05, 12., 801)
    μ_grid, var_grid = np.meshgrid(μ_axis, var_axis)
    likelihood = np.sum(
        [norm.logpdf(x, μ_grid, np.sqrt(var_grid)) for x in sample], axis=0
    )
    joint_by_bayes = normalized(
        gauss_prior.log_pdf(μ_grid, var_grid) + likelihood)
    joint_by_update = normalized(
        gauss_posterior.log_pdf(μ_grid, var_grid))
    difference = float(np.max(np.abs(joint_by_bayes - joint_by_update)))
    print(f'  {"max |prior x likelihood - posterior| on a grid":44s}'
          f'{difference:16.2e}')
    # with only 12 data points the posterior sits between the prior and the
    # data, which is the whole point of carrying a prior
    print(f'  {"12 data points from N(2.5, 1.5^2)":44s}{"prior":>16s}'
          f'{"posterior":>16s}{"the data":>16s}')
    row('E[μ]    (truth 2.5)', gauss_prior.θ, gauss_posterior.θ,
        float(np.mean(sample)))
    row('E[σ^2]  (truth 2.25)', gauss_prior.mean_of_variance(),
        gauss_posterior.mean_of_variance(), float(np.var(sample, ddof=1)))

    print('\n4. The posterior predictive is a Student-t')
    μ_draws, var_draws = gauss_posterior.sample(400000, rng)
    predicted = rng.normal(μ_draws, np.sqrt(var_draws))
    print(f'  {"x":>12s}{"P[next <= x] sampled":>24s}'
          f'{"Student-t cdf":>18s}')
    for x_value in [0., 1.5, 2.5, 4.]:
        exact = float(gauss_posterior.predictive_cdf(np.array(x_value)))
        print(f'  {x_value:12.2f}'
              f'{float(np.mean(predicted <= x_value)):24.6f}{exact:18.6f}')

    print('\n5. Are the posteriors calibrated?')
    print('  draw the parameters from the prior, generate 30 data points, '
          'update,')
    print('  and ask how often the 90% credible interval contains the truth')
    print(f'  {"":44s}{"p":>16s}{"μ":>16s}{"σ^2":>16s}')
    correct = coverage(0.9, 3000, rng)
    row('the appendix\'s updates', *correct)
    row('nominal', 0.9, 0.9, 0.9)
    broken_p, _, _ = coverage(0.9, 3000, rng, broken=True)
    print(f'  {"an update that forgets to count failures":44s}'
          f'{broken_p:16.8f}')
    print('  which is what makes this the check worth running: the density '
          'comparisons')
    print('  above cannot tell a wrong update from a right one unless you '
          'already know')
    print('  which posterior to compare against')
    print()
