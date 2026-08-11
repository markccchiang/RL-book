'''The Hamilton-Jacobi-Bellman equation.

The code here follows book/appendix4 (Appendix D of the book).  The appendix
states the HJB equation for state transitions given by an Ito process,

    ρ V*(t, s) = max_a { dV*/dt + μ(t,s,a) dV*/ds
                         + (σ(t,s,a)^2 / 2) d^2V*/ds^2 + R(t, s, a) }

with terminal condition V*(T, s) = T(s), and leaves the applications to the
chapters.  This module supplies three of them:

  1. HJB as a *test*: the residual ρV - max_a {...} must vanish everywhere
     for a correct V*.  Applied to Merton's closed-form solution from
     chapter 7, this both confirms the solution and pins down its constants.
  2. HJB as an *equation to solve*: an explicit finite-difference solver for
     a scalar problem, checked against the linear-quadratic regulator, where
     the HJB PDE collapses into a Riccati ODE.
  3. HJB as a *limit*: the same problem solved as a discrete-time MDP with
     step h and discount e^{-ρh} — the appendix's own derivation — whose
     value function converges to the HJB solution as h -> 0.

Run it with:

    python -m rl.appendix4.hjb
'''

from dataclasses import dataclass
from math import exp
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import solve_ivp


# --- 1. the HJB residual, and Merton's problem ------------------------------


@dataclass(frozen=True)
class MertonSolution:
    '''The closed-form solution of chapter 7, as a function to be tested.

    V*(t, W) = f(t)^γ W^{1-γ} / (1-γ), where f' = ν f - 1 and f(T) = ε.  The
    constant ν is the whole content of the solution, so it is a field here
    rather than a formula: the HJB residual is what decides whether a given
    value of it is right.
    '''

    μ: float
    σ: float
    r: float
    ρ: float
    γ: float
    T: float
    ν: float
    ε: float = 1e-6

    def f(self, t: float) -> float:
        remaining = self.T - t
        if self.ν == 0:
            return remaining + self.ε
        return (1 + (self.ν * self.ε - 1) * exp(-self.ν * remaining)) / self.ν

    def value(self, t: float, wealth: float) -> float:
        '''V*(t, W).'''
        return self.f(t) ** self.γ * wealth ** (1 - self.γ) / (1 - self.γ)

    def optimal_allocation(self) -> float:
        '''π* = (μ - r) / (γ σ^2), independent of t and W.'''
        return (self.μ - self.r) / (self.γ * self.σ ** 2)

    def optimal_consumption(self, t: float, wealth: float) -> float:
        '''c* = W / f(t).'''
        return wealth / self.f(t)


def merton_nu(sol: MertonSolution, half: bool = True) -> float:
    '''ν = (ρ - (1-γ)(r + (μ-r)^2 / (k σ^2 γ))) / γ, for k = 2 or k = 1.

    The book's text derives this with k = 2; the code listing beside it (and
    rl/chapter7/merton_solution_graph.py) uses the expected portfolio return
    r + (μ-r)^2/(σ^2 γ), i.e. k = 1.  main() feeds both to the residual.
    '''
    squared_sharpe = (sol.μ - sol.r) ** 2 / (sol.σ ** 2 * sol.γ)
    excess = squared_sharpe / 2 if half else squared_sharpe
    return (sol.ρ - (1 - sol.γ) * (excess + sol.r)) / sol.γ


def merton_hjb_residual(
        sol: MertonSolution,
        t: float,
        wealth: float,
        h: float = 1e-5
) -> Tuple[float, float, float]:
    '''ρV - max_{π,c} Φ, with V's derivatives taken numerically.

    Returns the residual, the maximizing π and the maximizing c, where the
    maximization uses the first-order conditions
    π* = -(μ-r) V_W / (σ^2 W V_WW) and c* = V_W^{-1/γ}.
    '''
    value = sol.value(t, wealth)
    v_t = (sol.value(t + h, wealth) - sol.value(t - h, wealth)) / (2 * h)
    dw = h * wealth
    v_w = (sol.value(t, wealth + dw) - sol.value(t, wealth - dw)) / (2 * dw)
    v_ww = (sol.value(t, wealth + dw) - 2 * value
            + sol.value(t, wealth - dw)) / dw ** 2

    π = -(sol.μ - sol.r) * v_w / (sol.σ ** 2 * wealth * v_ww)
    c = v_w ** (-1 / sol.γ)
    Φ = (v_t
         + v_w * ((π * (sol.μ - sol.r) + sol.r) * wealth - c)
         + 0.5 * π ** 2 * sol.σ ** 2 * wealth ** 2 * v_ww
         + c ** (1 - sol.γ) / (1 - sol.γ))
    return sol.ρ * value - Φ, π, c


# --- 2. solving the HJB equation on a grid ----------------------------------


@dataclass(frozen=True)
class ScalarHJB:
    '''A scalar control problem, ready to be solved on a grid.

    ds = μ(t,s,a) dt + σ(t,s,a) dz, reward rate R(t,s,a), terminal reward
    G(s), discount rate ρ, horizon T, actions drawn from a fixed grid.
    '''

    drift: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dispersion: Callable[[np.ndarray, np.ndarray], np.ndarray]
    reward_rate: Callable[[np.ndarray, np.ndarray], np.ndarray]
    terminal_reward: Callable[[np.ndarray], np.ndarray]
    actions: np.ndarray
    ρ: float
    T: float

    def solve(
            self,
            states: np.ndarray,
            time_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''Step the HJB equation backwards from T, explicitly.

        V(t-dt) = V(t) - dt (ρV - max_a {μ V_s + σ^2/2 V_ss + R}), with the
        derivatives in s taken by central differences and zero curvature
        imposed at the two ends of the grid.  Returns V(0, ·) and the
        maximizing action at each state at t = 0.
        '''
        ds = float(states[1] - states[0])
        dt = self.T / time_steps
        s = states[:, None]
        a = self.actions[None, :]
        drift, dispersion = self.drift(s, a), self.dispersion(s, a)
        reward = self.reward_rate(s, a)

        stable = ds ** 2 / max(float(np.max(dispersion ** 2)), 1e-12)
        if dt > stable:
            raise ValueError(
                f'explicit scheme needs dt <= ds^2/max(σ^2) = {stable:.2e}, '
                f'got dt = {dt:.2e}; raise time_steps or coarsen the grid'
            )

        value = self.terminal_reward(states)
        best = np.zeros_like(states)
        for _ in range(time_steps):
            v_s = np.gradient(value, ds)
            v_ss = np.empty_like(value)
            v_ss[1:-1] = (value[2:] - 2 * value[1:-1] + value[:-2]) / ds ** 2
            v_ss[0], v_ss[-1] = 0., 0.
            candidates = (drift * v_s[:, None]
                          + 0.5 * dispersion ** 2 * v_ss[:, None]
                          + reward)
            best_index = np.argmax(candidates, axis=1)
            value = value - dt * (self.ρ * value
                                  - np.max(candidates, axis=1))
            best = self.actions[best_index]
        return value, best


@dataclass(frozen=True)
class LinearQuadratic:
    '''ds = (a s + b u) dt + σ dz, reward rate -(q s^2 + κ u^2).

    Terminal reward -q_T s^2.  Guessing V(t,s) = -P(t) s^2 - k(t) turns the
    HJB PDE into the Riccati ODE P' = (ρ - 2a) P + (b^2/κ) P^2 - q, which is
    the same trick Merton's f' = νf - 1 is an instance of.
    '''

    a: float
    b: float
    σ: float
    q: float
    κ: float
    q_T: float
    ρ: float
    T: float

    def riccati(self, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''P(t) and k(t), by integrating backwards from t = T.'''
        def backwards(_τ: float, y: np.ndarray) -> np.ndarray:
            P, _ = y
            dP = (self.ρ - 2 * self.a) * P + self.b ** 2 / self.κ * P ** 2 \
                - self.q
            dk = self.ρ * y[1] - self.σ ** 2 * P
            return np.array([-dP, -dk])

        τ = self.T - times[::-1]
        solved = solve_ivp(backwards, (0., self.T), np.array([self.q_T, 0.]),
                           t_eval=τ, rtol=1e-10, atol=1e-12)
        return solved.y[0][::-1], solved.y[1][::-1]

    def value(self, times: np.ndarray, states: np.ndarray) -> np.ndarray:
        '''V(t, s) = -P(t) s^2 - k(t), evaluated at times[0].'''
        P, k = self.riccati(times)
        return -P[0] * states ** 2 - k[0]

    def optimal_control(self, times: np.ndarray,
                        states: np.ndarray) -> np.ndarray:
        '''u* = -(b/κ) P(t) s.'''
        P, _ = self.riccati(times)
        return -self.b / self.κ * P[0] * states

    def as_hjb(self, actions: np.ndarray) -> ScalarHJB:
        return ScalarHJB(
            drift=lambda s, u: self.a * s + self.b * u,
            dispersion=lambda s, u: self.σ * np.ones_like(s * u),
            reward_rate=lambda s, u: -(self.q * s ** 2 + self.κ * u ** 2),
            terminal_reward=lambda s: -self.q_T * s ** 2,
            actions=actions,
            ρ=self.ρ,
            T=self.T
        )


# --- 3. the discrete-time MDP the HJB equation is a limit of ----------------


def discrete_time_value(
        problem: LinearQuadratic,
        states: np.ndarray,
        actions: np.ndarray,
        step: float
) -> np.ndarray:
    '''Backward induction on the discrete-time MDP with time step h.

    V(t, s) = max_a { R(s,a) h + e^{-ρh} E[V(t+h, s')] }, with the shock
    approximated the way appendix C builds Brownian motion out of a random
    walk: s' = s + μ h ± σ sqrt(h), each with probability 0.5.  This is the
    equation the appendix takes the dt -> 0 limit of.

    The state grid must resolve the shock: once σ sqrt(h) drops below the
    grid spacing, both branches land inside a single cell and the linear
    interpolation averages the variance away, leaving the *deterministic*
    problem.  Shrinking h on a fixed grid therefore makes the answer worse,
    not better; callers should shrink the grid spacing along with h.
    '''
    s = states[:, None]
    u = actions[None, :]
    drift = problem.a * s + problem.b * u
    reward = -(problem.q * s ** 2 + problem.κ * u ** 2) * step
    discount = exp(-problem.ρ * step)
    up = s + drift * step + problem.σ * np.sqrt(step)
    down = s + drift * step - problem.σ * np.sqrt(step)

    value = -problem.q_T * states ** 2
    for _ in range(int(round(problem.T / step))):
        expected = 0.5 * (np.interp(up, states, value)
                          + np.interp(down, states, value))
        value = np.max(reward + discount * expected, axis=1)
    return value


def row(label: str, *values: float) -> None:
    '''Print a label and a row of numbers, aligned under a header.'''
    print(f'  {label:42s}' + ''.join(f'{v:16.8f}' for v in values))


if __name__ == '__main__':

    print('\n1. HJB as a test: does Merton\'s closed form satisfy it?')
    base = MertonSolution(μ=0.1, σ=0.1, r=0.02, ρ=0.01, γ=2., T=20., ν=0.)
    print(f'  {"ν used in f(t)":42s}{"ν":>16s}{"HJB residual":>16s}')
    for half, name in [(True, "book's text: r + (μ-r)^2/(2σ^2γ)"),
                       (False, 'code listing: r + (μ-r)^2/(σ^2γ)')]:
        ν = merton_nu(base, half=half)
        candidate = MertonSolution(μ=base.μ, σ=base.σ, r=base.r, ρ=base.ρ,
                                   γ=base.γ, T=base.T, ν=ν)
        residuals = [merton_hjb_residual(candidate, t, w)[0]
                     for t in [1., 5., 10., 15.] for w in [0.5, 1., 5., 20.]]
        print(f'  {name:42s}{ν:16.8f}'
              f'{max(abs(r) for r in residuals):16.8f}')

    solution = MertonSolution(μ=base.μ, σ=base.σ, r=base.r, ρ=base.ρ,
                              γ=base.γ, T=base.T, ν=merton_nu(base))
    print('\n  the maximizing action, recovered from the residual\'s '
          'first-order conditions')
    print(f'  {"(t, W)":42s}{"π* found":>16s}{"(μ-r)/(γσ^2)":>16s}'
          f'{"c* found":>16s}{"W/f(t)":>16s}')
    for t, wealth in [(1., 1.), (10., 5.), (18., 20.)]:
        _, π, c = merton_hjb_residual(solution, t, wealth)
        print(f'  {f"t = {t}, W = {wealth}":42s}{π:16.8f}'
              f'{solution.optimal_allocation():16.8f}{c:16.8f}'
              f'{solution.optimal_consumption(t, wealth):16.8f}')

    print('\n2. HJB as an equation to solve: finite differences vs the '
          'Riccati ODE')
    lq = LinearQuadratic(a=-0.5, b=1., σ=0.3, q=1., κ=1., q_T=0.5, ρ=0.05,
                         T=1.)
    states = np.linspace(-3., 3., 241)
    actions = np.linspace(-4., 4., 161)
    times = np.linspace(0., lq.T, 2001)
    numeric, policy = lq.as_hjb(actions).solve(states, time_steps=20000)
    exact = lq.value(times, states)
    exact_policy = lq.optimal_control(times, states)
    inner = np.abs(states) <= 2.
    print(f'  {"quantity":42s}{"max |difference|":>16s}')
    row('V(0, s) on |s| <= 2',
        float(np.max(np.abs(numeric[inner] - exact[inner]))))
    row('u*(0, s) on |s| <= 2',
        float(np.max(np.abs(policy[inner] - exact_policy[inner]))))
    print(f'  {"":42s}{"grid":>16s}{"Riccati":>16s}')
    for s_value in [-2., 0., 1.5]:
        index = int(np.argmin(np.abs(states - s_value)))
        row(f'V(0, {s_value})', float(numeric[index]), float(exact[index]))

    print('\n3. HJB as a limit: the discrete-time MDP as its step h shrinks')
    print(f'  {"h":>10s}{"shock σ√h":>14s}{"grid spacing":>14s}'
          f'{"nodes":>8s}{"max |V_h - V_HJB| on |s| <= 2":>34s}')
    for step in [0.05, 0.01, 0.002, 0.001]:
        # the grid has to shrink with the shock, or the diffusion is lost
        spacing = lq.σ * np.sqrt(step)
        grid = np.arange(-3., 3. + spacing, spacing)
        discrete = discrete_time_value(lq, grid, actions, step)
        reference = lq.value(times, grid)
        near = np.abs(grid) <= 2.
        gap = float(np.max(np.abs(discrete[near] - reference[near])))
        print(f'  {step:10.4f}{spacing:14.5f}{spacing:14.5f}'
              f'{len(grid):8d}{gap:34.8f}')
    print('  the discrete-time Bellman optimality equation converges to the '
          'HJB solution')

    print('\n  on a grid held fixed at ds = 0.025, the same refinement '
          'diverges instead:')
    print(f'  {"h":>10s}{"shock σ√h":>14s}{"grid spacing":>14s}'
          f'{"":8s}{"max |V_h - V_HJB| on |s| <= 2":>34s}')
    for step in [0.01, 0.002, 0.0005]:
        discrete = discrete_time_value(lq, states, actions, step)
        gap = float(np.max(np.abs(discrete[inner] - exact[inner])))
        print(f'  {step:10.4f}{lq.σ * np.sqrt(step):14.5f}{0.025:14.5f}'
              f'{"":8s}{gap:34.8f}')
    print('  once σ√h < ds the two branches share a grid cell, linear '
          'interpolation averages the')
    print('  variance away, and what is left is the deterministic problem')
    print()
