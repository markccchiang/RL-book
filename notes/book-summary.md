# Foundations of Reinforcement Learning with Applications in Finance — Condensed Notes

A working summary of `book/`: the central equations, what each symbol means, and how each
equation is actually solved (analytically, by DP, or by RL), with pointers to the `rl/`
implementation.

Chapter order follows the top-level `structure` file. Numbering is sparse — chapter 0 is the
overview, chapter 1 is Python design, and the substantive material starts at chapter 2.

The notes are split by module, one file per part, under [`book-summary/`](book-summary/).

## Contents

| Part | Covers | Contents |
|---|---|---|
| **[Notation and Variable Glossary](book-summary/00-notation.md)** | — | [States, processes, rewards](book-summary/00-notation.md#states-processes-rewards) · [Policies and value functions](book-summary/00-notation.md#policies-and-value-functions) · [Approximation and learning](book-summary/00-notation.md#approximation-and-learning) · [Finance](book-summary/00-notation.md#finance) |
| **[Module I — Processes and Planning](book-summary/01-processes-and-planning.md)** | ch. 2–5 | [2 Markov Processes / MRPs](book-summary/01-processes-and-planning.md#chapter-2--markov-processes-and-markov-reward-processes) · [3 Markov Decision Processes](book-summary/01-processes-and-planning.md#chapter-3--markov-decision-processes) · [4 Dynamic Programming](book-summary/01-processes-and-planning.md#chapter-4--dynamic-programming-algorithms) · [5 Function Approximation and ADP](book-summary/01-processes-and-planning.md#chapter-5--function-approximation-and-approximate-dp) |
| **[Module II — Financial Applications](book-summary/02-financial-applications.md)** | ch. 6–9 | [6 Utility Theory](book-summary/02-financial-applications.md#chapter-6--utility-theory) · [7 Merton's Problem](book-summary/02-financial-applications.md#chapter-7--dynamic-asset-allocation-and-consumption-mertons-problem) · [8 Derivatives Pricing](book-summary/02-financial-applications.md#chapter-8--derivatives-pricing-and-hedging) · [9 Order-Book Algorithms](book-summary/02-financial-applications.md#chapter-9--order-book-trading-algorithms) |
| **[Module III — RL Algorithms](book-summary/03-rl-algorithms.md)** | ch. 10–13 | [10 MC and TD Prediction](book-summary/03-rl-algorithms.md#chapter-10--mc-and-td-for-prediction) · [11 MC and TD Control](book-summary/03-rl-algorithms.md#chapter-11--mc-and-td-for-control) · [12 Batch RL, LSPI, Gradient TD](book-summary/03-rl-algorithms.md#chapter-12--batch-rl-experience-replay-dqn-lspi-gradient-td) · [13 Policy Gradient](book-summary/03-rl-algorithms.md#chapter-13--policy-gradient-algorithms) |
| **[Module IV — Finishing Touches](book-summary/04-finishing-touches.md)** | ch. 14–16 | [14 Multi-Armed Bandits](book-summary/04-finishing-touches.md#chapter-14--multi-armed-bandits) · [15 Blending Learning and Planning](book-summary/04-finishing-touches.md#chapter-15--blending-learning-and-planning) · [16 Real-World Considerations](book-summary/04-finishing-touches.md#chapter-16--summary-and-real-world-considerations) |
| **[Appendices](book-summary/05-appendices.md)** | app. 1–7 | MGF · Portfolio Theory · Stochastic Calculus · HJB · Black–Scholes · Affine Spaces · Conjugate Priors |
| **[Cross-Cutting Reference](book-summary/06-cross-cutting.md)** | — | [Which method solves which equation](book-summary/06-cross-cutting.md#which-method-solves-which-equation) · [Recurring examples](book-summary/06-cross-cutting.md#recurring-examples) · [Library map](book-summary/06-cross-cutting.md#library-map) |

## How to read these

The book is built bottom-up and so are these notes. The dependency chain that actually matters:

- **[Notation](book-summary/00-notation.md) first** — the symbol conventions are used unchanged
  everywhere, and a few are overloaded ($\gamma$ is the discount factor in RL but the CRRA
  coefficient in chapters 6–7; $\pi_t$ is a wealth fraction in chapter 7, not a policy).
- **[Module I](book-summary/01-processes-and-planning.md)** is the spine. Everything after it is
  either an application of the Bellman equations or a way to solve them without a model.
- **[Module II](book-summary/02-financial-applications.md)** can be read independently of
  III — it is where the closed-form solutions live, and those closed forms are what
  Module III's algorithms are later validated against.
- **[Module III](book-summary/03-rl-algorithms.md)** assumes chapters 4 and 5 (Bellman operators,
  `FunctionApprox`) and nothing from Module II except the worked examples.
- The **[appendices](book-summary/05-appendices.md)** are prerequisites, not afterthoughts:
  appendix 1 (MGF) underpins every CARA closed form, and appendices 3 and 4 (Itô, HJB) are
  required before chapter 7.
- **[Cross-cutting reference](book-summary/06-cross-cutting.md)** is the lookup table —
  start here if you know the equation and want the method or the code.

Notes on rendering: the book's own `.md` sources use LaTeX macros defined in
`templates/latex.template` (`\bvpi`, `\bbs`, `\pdv`, …) that only exist in the XeLaTeX build.
These notes expand all of them and stick to plain MathJax/KaTeX, so the math renders in an
ordinary Markdown viewer.
