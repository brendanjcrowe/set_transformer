# Which POMDPs can actually benchmark a belief encoder?

**Question.** The benchmark compares belief encoders — Set Transformer, CGF, DeepSet,
PointNet — against analytic summaries of the same particle set (Gaussian, k-moments). For
that comparison to mean anything, the environment has to *reward* representing the belief
better. Which of our environments do?

**Answer.** Fewer than we assumed. Three of the four environments we examined had a reward
under which a cheap statistic of the belief was provably sufficient, so no encoder could
win however well it represented the posterior. Two were repairable by changing the task
reward; one is still blocked on an unrelated learning failure. **Check this property
before spending compute on an environment, not after.**

## The failure mode, stated once

An encoder can only matter if the optimal action depends on *more* of the belief than a
cheap statistic captures. That is a property of the **reward**, not of the environment's
dynamics or its observation noise — and it is easy to violate by accident, because the
natural reward for a "guess the hidden state" task is squared error, whose Bayes action is
the posterior mean.

Potential-based shaping cannot repair it. Shaping provably preserves the optimal policy
(Ng et al., 1999), so if the optimal policy ignores the belief structure, shaping leaves
it ignoring the belief structure. The *task reward* has to change.

The diagnostic is cheap and does not require training anything:

> Sample real belief states from rollouts. For each, compute the Bayes-optimal action
> under the reward, and the action implied by the cheap statistic (usually round(mean)).
> **If they agree ~always, the environment cannot discriminate encoders.**

## Per-environment verdicts

| env | belief | verdict |
|---|---|---|
| **Car-Flag** | 1 bit, 3 reachable states | **CONTROL.** Usable, cannot discriminate. |
| **Odd-Even** | comb over one parity | **FIXED** — discriminating after two changes. |
| **Ant-Tag** | 2-D multimodal | **BLANK** — belief is fine; the agent cannot learn. |
| **Multimodal Search** | K random Gaussian modes | **PURPOSE-BUILT** — discriminating by construction. |
| Ant-Heaven-Hell, Two-Boxes | not built | unexamined |

### Car-Flag — a validated control, not a benchmark

The latent is a single bit (which side heaven is on), and the filter has exactly **three
reachable states**: 50/50, all +1, all −1. A Gaussian summary of a one-bit belief is a
sufficient statistic by construction, so no encoder can win *in principle*.

Two consequences. It is still worth running — it is a sanity check that the harness
works, and the flat result across methods is the correct answer rather than a null
result. But its belief is too degenerate to pretrain on: an autoencoder would see three
distinct point clouds and a pairwise-EMD matrix with three distinct values. **The 12
pretrained/aligned cells for Car-Flag are deliberately blank**, and the write-up should
say why.

Separately, its stock reward was broken in the general way described above: the priest
detour cost ~27 steps while information was worth 3, so the optimal policy ignored the
latent entirely. Fixed by `envs.CarFlagRewardWrapper` (step −0.01, heaven +1, hell −1),
which is what makes information worth +0.83 over gambling.

### Odd-Even — two independent defects, both now fixed

Hidden integer `true_state ∈ [1,n]`; every observation is an integer of *true_state's own
parity*. Parity is therefore knowable exactly while the value stays uncertain — a
genuinely multimodal belief over a comb (e.g. `{6:0.30, 8:0.40, 10:0.30}`).

**Defect 1 — the reward made the mean sufficient.** Under `-(predicted - true_state)²`
the Bayes action is round(E[s]). Measured over 24,000 real belief states, the optimal
action differed from round(mean) in **0.0%** of them. Under a parity-gated reward it
differs **39.2%** of the time.

*Fix (`envs.OddEvenParityRewardWrapper`):* correct parity → `-(predicted - true_state)²`;
wrong parity → the task's worst reward, `-(n-1)²`. This asks for something the belief
genuinely contains and a mean provably cannot express — **the mean of a comb sits between
its teeth, on a state of the opposite parity**. Expected-reward gap: −1.70 (optimal) vs
−32.65 (round the mean), ~31 per step.

**Defect 2 — 100 observation samples per step.** `OddEvenPOMDPConfig.n_particles` defaults
to 100, and each step hands the agent that many iid draws, collapsing the exact posterior
to a **point mass after one step**. At 1 sample/step it stays multimodal for ~10 steps
(mean support 3.9 states at t=1, 2.4 at t=5, 1.8 at t=10). *Fix:*
`make_odd_even_base_env(n_obs_samples=1)`, now the default.

**Probe gate: PASSES.** `informed` plays the posterior mode, `gambler` rounds the
posterior mean — i.e. literally the Gaussian baseline's decision rule.

| policy | return [95% CI] | success |
|---|---|---|
| oracle | 0.00 | 1.00 |
| informed (mode) | **−18.4** [−20.7, −16.1] | 1.00 |
| gambler (mean) | **−451.0** [−490.2, −413.9] | 0.32 |
| staller | −4573.8 | 0.09 |

A 24× gap with disjoint CIs. A regression test pins the defect: under the stock reward
`gambler` must score **≥** `informed`, and it does (−15.2 vs −18.5) — representing the
comb does not merely fail to pay there, it actively costs.

Also required: `ParityAwareOddEvenParticleFilter`. The previous filter put particles
anywhere in the continuous interval, diffused them with process noise, and scored them
against the sample mean — assigning mass to states the observations had already excluded.
The new one reproduces the environment's own observation model (verified against its
`update_belief` to atol 0.02).

### Ant-Tag — the belief is fine; the agent cannot learn

Unlike the other two, nothing is wrong with Ant-Tag's belief: a fleeing target under
partial visibility is exactly the multimodal 2-D structure the benchmark wants. The
problem is that **no agent learns the task**.

A 2-seed × 8-method × 2M-step pilot returned **exactly −400.0 on every run** — zero tags,
ever. The task is demonstrably feasible: the pretrained locomotion policy tags 4/20 and
moves the ant 4.90 units, where random actions tag 0/20 and move 2.23.

Three rescues were tried (see `ant-tag-rescue-probe` in project memory for the full
table): SAC (flat −400 through 100k), an explicit tag bonus (flat −40, zero tags through
480k), and a behaviour-cloned locomotion warm-start. The warm-start is the informative
one:

- The clone alone tags 5/15 with mean return +2.3 — it walks and tags before any RL.
- With a reduced learning rate it *keeps* tagging early: **+33.2 at 10k, +12.0 at 30k**.
- It then **decays back to the floor** by ~200k.

So PPO progressively destroys a competent policy rather than improving it. Removing the
shaping helps early but does not change the decay. This needs real Ant hyperparameter
work or a joint locomotion+tagging curriculum, not a config flag.

**Ant-Tag and `ant_tag_cdens` are left blank in the matrix**, and may be revisited. The
infrastructure built for the probe is kept and tested: `AntTagRewardWrapper`, the
`ant_tag_bonus` env, `warmstart_locomotion.py`, `train.py --init_policy` / `--no_shaping`.

### Multimodal Search (`msearch`) — built to satisfy the checklist by construction

The other three environments were inherited and audited. This one was designed backwards
from the checklist: a static target hides in one of K ~ U[2,10] random Gaussian modes, the
agent sees perfectly within radius 1.5 and nothing outside, and the optimal policy is to
visit modes in turn.

Two properties make a Gaussian summary provably blind rather than merely disadvantaged:

* **The belief mean is pinned at the origin.** Mode centres are drawn at random then
  translated to zero centroid, so the mean is the *same constant every episode* and
  carries zero information (verified to float32 precision over 100 episodes). After
  eliminations it becomes the centroid of the surviving modes — empty space between them,
  so it is not merely uninformative but actively misleading.
* **The geometry stays random.** Pinning the centroid on a fixed ring would also zero the
  mean, but an agent could then memorise that ring and sweep it without ever consulting
  its belief. Here mode radii range over ~0.3-13.5, so there is no fixed path to learn.

Each mode is a random full Gaussian, so the optimal policy must also weigh each mode's
*sweep cost* against its distance — a diffuse mode takes several looks, a tight one takes
one. A single global covariance reports the spread of the whole configuration and says
nothing about any individual mode.

**The filter is exact.** Because the target is static, the posterior is simply the prior
restricted to the region not yet observed. So the belief is left bit-identical until the
visibility disc crosses particles; those are then explained (found — the episode ends) or
refuted and redrawn from the prior with the swept region rejected. Redrawing beats copying
surviving neighbours: copies accumulate until the cloud degenerates into a handful of
atoms, and the standard remedy of roughening them with noise quietly diffuses a belief
that should never diffuse. The filter's entire memory is the list of observed disc
centres.

**Checklist results.** (1) 93 distinct belief shapes. (2) The mean is a constant, so the
disagreement with the Bayes action is total by construction. (3) Probe **PASSES**:

| policy | return [95% CI] | steps | success |
|---|---|---|---|
| oracle | 224.7 | 25.3 | 1.00 |
| informed (visits modes) | **178.6** [168.7, 186.9] | 66.4 | **0.98** |
| gambler (steers by the mean, then sweeps) | **−100.6** [−122.4, −78.8] | 198.9 | **0.39** |
| staller | −246.7 | 248.3 | 0.01 |

informed beats gambler by **+279** with disjoint CIs. (4) Learnability: unlike Ant-Tag,
PPO shows real movement early (return −250 → −105, episode length 250 → 180 within 60k
steps) — the dynamics are a point mass, so nothing is hidden behind a motor-control
problem.

**The footprint constraint is load-bearing and enforced in code.** The environment
discriminates only while the modes cover a small fraction of the arena. At
`mode_scale_hi = 1.0` (~19% coverage) an informed tour takes 77 steps against 307 for a
full lawnmower — a 4.0x gap; at `mode_scale_hi = 2.0` with K up to 10 that collapses to
**1.4x** and the environment stops discriminating. `MultimodalSearchConfig.__post_init__`
refuses such configurations with an explanatory error. Do not raise the bound without
re-measuring the gap.

## Checklist for a new environment

Run these **before** committing compute. Each is cheap and each has already caught a real
defect.

1. **Is the belief non-degenerate?** Collect PF snapshots and count distinct belief
   shapes. `pretrain/1_collect_pf_dataset.py` reports `unique_spread_values`: Odd-Even
   gives 600/600, Car-Flag gives 2. Fewer than ~10 means there is nothing to encode and
   pretraining/alignment are vacuous.
2. **Does the optimal action need more than the mean?** Run the sufficiency test above.
   Anything near 0% disagreement means the environment cannot discriminate, whatever its
   dynamics look like.
3. **Does the probe gate pass?** Add a `ProbeSpec` to `probe_env.py` with `oracle` /
   `informed` / `gambler` / `staller`, where **`gambler` implements the cheap statistic's
   own decision rule**. Require `informed > gambler` with disjoint CIs. This is the same
   test as (2) but end-to-end, and it catches reward pathologies that the analytic
   argument misses.
4. **Can anything learn it at all?** Ant-Tag's failure was invisible to (1)–(3): a perfect
   belief and a discriminating reward are worthless if no policy reaches the rewarding
   state. Check that *some* policy — hand-coded, pretrained, or oracle-initialized —
   achieves a non-floor return before sweeping 180 runs.

A useful bias, learned the hard way: prefer environments where the *decision* is
discrete and depends on which mode the belief is in. Squared-error-style rewards on a
continuous latent almost always make the mean sufficient.
