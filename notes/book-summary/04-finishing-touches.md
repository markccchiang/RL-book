# Module IV — Finishing Touches

[← Notes index](../book-summary.md) · [← Module III](03-rl-algorithms.md) · [Appendices →](05-appendices.md)

## Chapter 14 — Multi-Armed Bandits

A single-state MDP; the pure explore/exploit problem. $Q(a) = \mathbb{E}[r\mid a]$,
$V^* = \max_a Q(a)$, gap $\Delta_a = V^* - Q(a)$, and **total regret**

$$L_T = \sum_{t=1}^T \mathbb{E}[V^* - Q(A_t)] = \sum_{a} \mathbb{E}[N_T(a)]\cdot\Delta_a$$

Regret is therefore about *how often* you pull each suboptimal arm.

**Lai–Robbins lower bound** — no algorithm can do better than logarithmic:

$$L_T \geq \log T \sum_{a\mid\Delta_a>0}\frac{\Delta_a}{KL(\mathcal{R}^a \Vert \mathcal{R}^{a^*})}$$

| Algorithm | Rule | Regret |
|---|---|---|
| Greedy | $\arg\max_a \hat{Q}_t(a)$ | linear (can lock onto a bad arm forever) |
| $\epsilon$-greedy (fixed $\epsilon$) | explore w.p. $\epsilon$ | linear |
| Decaying $\epsilon_t = \min(1, \frac{c\lvert\mathcal{A}\rvert}{d^2(t+1)})$ | | logarithmic |
| **UCB1** | $\arg\max_a\{\hat{Q}_t(a) + \sqrt{\frac{\alpha\log t}{2N_t(a)}}\}$ | logarithmic: $L_T \leq \sum_{a}\frac{4\alpha\log T}{\Delta_a} + \frac{2\alpha\Delta_a}{\alpha-1}$ |
| Bayesian UCB | $\arg\max_a \mathbb{E}[\mu_a + \frac{c\sigma_a}{\sqrt{N_t(a)}}]$ over the posterior | |
| **Thompson Sampling** | sample $\mathcal{D}_t$ from the posterior, act greedily w.r.t. it | achieves probability matching (eq. `probability-matching`) |
| Gradient Bandits | $s_{t+1}(a) = s_t(a) + \alpha(R_t - \bar{R}_t)(\mathbb{I}_{a=A_t} - \pi_t(a))$ | policy gradient on a softmax over preferences |

The UCB bonus comes from **Hoeffding's inequality**: setting
$e^{-2N_t(a)\hat{U}_t(a)^2} = p$ gives $\hat{U}_t(a) = \sqrt{\frac{-\log p}{2N_t(a)}}$, and
letting $p$ shrink as $t^{-\alpha}$ yields the UCB1 formula.

**Information State Space MDP.** Reformulate the bandit as an MDP whose state is the posterior
(Bayes-adaptive RL); solving it exactly gives the Gittins index. The conjugate-prior updates that
make this tractable are appendix 7.

Extensions: **contextual bandits** ($\mathcal{R}^a_c(r) = \mathbb{P}[r\mid c,a]$) and then full RL
control, where these same exploration strategies replace naive $\epsilon$-greedy.

## Chapter 15 — Blending Learning and Planning

- **Model-based RL:** learn $\mathcal{P}_R$ from experience (supervised learning), then plan with DP. Sample-efficient; limited by model error.
- **Dyna:** interleave real experience and simulated experience from the learned model.
- **Decision-time planning:** don't compute a global policy — plan *from the current state* only, at every step.

**MCTS.** Four steps per round: **Selection** (descend the tree by a tree policy),
**Expansion** (add a node), **Simulation** (roll out to termination with a cheap rollout policy),
**Backpropagation** (back up the return along the traversed path). The tree policy is **UCT**,
i.e. UCB1 applied per node:

$$\hat{Q}_t(s_t,a_t) + \sqrt{\frac{2\log i}{N_t^{s_t,a_t}}}$$

**Adaptive Multi-Stage Sampling (AMS)** — MCTS/UCT's "spiritual origin" and the version with
proofs. For finite-horizon MDPs with large $\mathcal{S}_t$ but small $\mathcal{A}_t$:

$$\hat{Q}_t(s_t,a_t) = \mathcal{R}_t(s_t,a_t) + \gamma\cdot\frac{\sum_{j=1}^{N_t^{s_t,a_t}}\hat{V}_{t+1}^{N_{t+1}}(s_{t+1}^{(s_t,a_t,j)})}{N_t^{s_t,a_t}}, \qquad \hat{V}_t^{N_t}(s_t) = \sum_{a_t}\frac{N_t^{s_t,a_t}}{N_t}\hat{Q}_t(s_t,a_t)$$

Note $\hat{V}$ is an *allocation-weighted* average, not a max. Convergence is proved, with bias
$0 \leq V_0^*(s_0) - \mathbb{E}[\hat{V}_0^{N_0}(s_0)] \leq O(\sum_t \frac{\ln N_t}{N_t})$.

## Chapter 16 — Summary and Real-World Considerations

Recap of the arc, then a genuinely practical closing section. The authors' recommended
**workflow for a new problem**:

1. **Simplify until it is analytically tractable** and derive a closed form (drop transaction
   costs, assume continuous trading, ignore liquidity constraints). Three payoffs: intuition
   about how the optimum depends on the inputs; a special case to test the full model against;
   and guidance on which **features** to use for function approximation later.
2. **Add back frictions and solve with DP / ADP**, which requires estimating $\mathcal{P}_R$ from
   real data. Often blocked by the Curse of Modeling.
3. **Fall back to RL**, almost always against a *simulator* estimated from real data (and
   frequently augmented with human domain knowledge), not the live environment.

Other points worth carrying:

- Real problems are usually **POMDPs**; approximating one as an MDP is a modeling decision that
  needs domain expertise, and controlling state-space explosion is the central difficulty.
- **Defining the reward function is the hard part** — it usually means eliciting the actual
  business objective (often literally a utility function) from a domain owner.
- Start with MC/TD because they are easy to debug, but expect to end up customizing: the set of
  RL algorithms is not a fixed menu. In the authors' own real-world work the ones that have
  worked are **LSPI, Gradient TD, DQN and Natural Policy Gradient** — all chosen with the
  Deadly Triad in mind.
- Since the real world is non-stationary, **blending model-based with model-free RL**
  (chapter 15) matters, because models must be updated continuously.
- The book's code is educational and deliberately not performant; production needs distributed
  storage/compute, plus the surrounding ecosystem (data management, deployment, instrumentation,
  explainability) and success metrics fed back into model iteration. Being *close* to optimal
  is usually good enough.

---

[← Notes index](../book-summary.md) · [← Module III](03-rl-algorithms.md) · [Appendices →](05-appendices.md)
