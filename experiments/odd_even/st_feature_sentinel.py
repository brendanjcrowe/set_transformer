"""An Odd-Even-local encoder-collapse sentinel for the ST arm.

WHY NOT THE SHARED CALLBACK. `STFeatureLoggingCallback` in
set_transformer/rl/feature_extractors/st.py logs `st/feat_std_mean`: the
per-feature standard deviation across the encoder's LAST CACHED FORWARD
BATCH, averaged over features. Three properties make it unsuitable as the
collapse sentinel on this domain, and none of them is a bug in the Ant-Tag
context it was written for:

1. **It is a 1-sample std at n_envs=1.** The statistic is computed from
   `extractor.last_st_features`, which at rollout end holds whatever the last
   forward pass saw. With one worker that is a single observation, so
   `std(dim=0)` is undefined and the metric logs **NaN** -- measured. The
   sentinel is then silently absent in exactly the cheap configuration people
   use for a first run. This callback aggregates over the whole ROLLOUT
   instead, so the sample size is `n_steps * n_envs` regardless of n_envs.

2. **It is not scale-free, so PITFALLS.md's absolute ~0.01 threshold does not
   transfer.** The ST feature scale here depends on how long the encoder was
   pretrained, over more than three orders of magnitude, while the encoder
   stays perfectly informative. A threshold calibrated on Ant-Tag reads a
   healthy Odd-Even encoder as dead.

3. **Averaging over features can hide a live subspace.** A mean over 64
   features is dragged down by dead ones; the max is not. Both are logged
   here.

WHAT THE NUMBERS ACTUALLY LOOK LIKE ON THIS DOMAIN. Measured on 600 held-out
episodes, exact-support filter beliefs at step 8, oe50 (n=50), 8x8 features;
`R^2` is a 5-fold-CV Ridge probe predicting the true state from the feature
vector -- i.e. "is the state still linearly decodable from this encoding?":

    encoder                 feat_std_mean   feat_std_relative   probe R^2
    random init (control)      8.5e-03           3.7e-02          +0.688 [*]
    Sinkhorn pretrained,  5ep  8.5e-02           6.9e-02          +0.985
    Sinkhorn pretrained,  8ep  1.5e-02           1.6e-02          +0.941
    Sinkhorn pretrained, 30ep  9.0e-05           9.3e-04          -0.015
    any encoder, final Linear zeroed
                               0.0               0.0              -0.015

[*] UNSEEDED, and an outlier. Seeded across 8 torch seeds a random-init
    encoder probes at 0.984..0.992. Do not read +0.688 as the control value;
    see "A LINEAR PROBE CANNOT..." below, which is why no probe number in
    this table should be trusted to rank an encoder.

Read the third and fourth rows together, because they are the whole reason
this file exists:

* **A low absolute reading is genuinely ambiguous.** The 8-epoch encoder
  reads 1.5e-2 and is healthy (R^2 = 0.94); the 30-epoch encoder reads
  9.0e-5 and is *really* collapsed -- its feature vectors are identical
  across samples to within 2.5e-4, and the state is no longer decodable at
  all. So `feat_std_mean` alone cannot distinguish "small features" from
  "no features", and an abort rule on it would fire on the healthy run and
  the dead one in whichever order the scale happened to fall.
* **The RELATIVE spread does separate them**: every live encoder measured
  here sits at 1.6e-02 or above, and both dead ones at 9.3e-04 or below --
  more than an order of magnitude of clear air. That is why the relative
  statistic is the sentinel and the absolute one is kept only as context.

**A LINEAR PROBE CANNOT TELL A LIVE ENCODER FROM A DEAD ONE HERE.** This is
the sharpest limit in this file, and it is why the sentinel exists rather
than a probe. Every encoder in the table above -- spanning three orders of
magnitude of feature spread, from live to exactly constant -- probes within
about 0.10 of the others once the encoder init is seeded. Measured across 8
torch seeds, a random-init encoder probes at R^2 = 0.984..0.992 (mean 0.988);
the `+0.688` in the table above is ONE unlucky unseeded draw and should not
be read as the control value. A collapsed encoder still probes at R^2 ~ 0.99.

The reason: **a ~1e-4 signal riding on an O(1) offset is linearly recoverable
but not learnable by a policy head.** Ridge on z-scored features will happily
amplify a 1e-4 direction by 1e4; PPO's policy head, training on advantages
through a shared trunk, will not. So the probe answers "is the information
present at all?" and says yes for a near-constant encoding, while the
question that decides the run is "is it present at a magnitude a policy can
use?" -- which is what the relative spread measures.

Peak-to-peak spread relative to feature magnitude,
`mean((max - min) / |mean|)` over features, on held-out beliefs, separates
them where the probe does not. Two independent measurements, which agree on
every ordering and on which encoders are constant, and disagree on magnitude
by up to 20x -- so treat the REGIME, not the digit, as the finding:

    encoder                  feat_std_mean   p2p/|mean|      verdict
    8ep (this file's)          1.5e-02      7.0e-02 / 3.4e-02  varies
    30ep                       9.1e-05      4.3e-03 / 3.7e-04  CONSTANT
    a separate 8ep run         1.8e-04      --      / 6.3e-04  CONSTANT
    random init, seeds 0-3     8.2e-03..2.0e-02  2.3e-01..4.8e-01  varies

(Second column of each pair is the reviewer's independent run; the third row
is the reviewer's own 8-epoch checkpoint, which sits in the same regime as
the 30-epoch collapse. Two 8-epoch runs landing on opposite sides is itself
the point: at 8 epochs this pretraining is marginal, and which side a run
lands on is not predictable from the loss.)

**Consequence for how this domain is instrumented.** Use the relative-spread
sentinel to decide whether an encoder is ALIVE. Use the probe only for what
it actually measures -- whether the information is present at all. This is a
harder version of PITFALLS.md section 6's "the probe measures capacity, not
use": there the caveat was that a random ISAB projection probed at 0.997, so
a high score does not imply a good policy. Here the caveat is stronger --
on this domain the probe cannot distinguish live from dead at all, so a
chance-level score no longer even implies a frozen encoder cannot work,
because nothing scores at chance level. `oddeven.md`'s build-order step 4
("probe before you train") must be read with that limit in mind: run the
probe to rule out missing information, then check the spread before
concluding an encoder is usable.

**Long Sinkhorn pretraining collapsed the encoder on this domain.** Thirty
epochs on 2040 snapshots drove it to a constant. That is a reportable
property, not a fluke of one seed, and it is the opposite failure from
Ant-Tag's (where pretraining preserved the cloud and destroyed the
decision-relevant asymmetry). It is also a direct instance of PITFALLS.md
section 4's "watch for convergence-in-epoch-10" and "a flat loss is not
necessarily collapse" -- here the loss stayed flat and the encoder WAS
collapsing. Check a probe, not the epoch count.

NO ABORT IS IMPLEMENTED. This logs and warns; it never stops a run. The
threshold below is a reporting aid drawn from five measured encoders on one
variant, which is not enough evidence to kill a multi-hour run on.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

#: Relative-spread reading below which the encoding is reported as suspect.
#: Placed in the order-of-magnitude gap measured between live encoders
#: (>= 1.6e-02) and collapsed ones (<= 9.3e-04). Deliberately nearer the
#: dead end: a false "suspect" costs a log line, while a false "healthy"
#: costs the run.
COLLAPSE_RELATIVE_SPREAD = 5e-3


class OddEvenSTFeatureSentinel(BaseCallback):
    """Log scale-free ST feature-spread statistics over each whole rollout.

    Metrics, all under ``st/``:

    ``feat_std_relative``
        ``mean(std_across_rollout / (|mean_across_rollout| + eps))``. The
        sentinel. Scale-free, so it is comparable across pretraining lengths
        and across domains in a way ``feat_std_mean`` is not.
    ``feat_std_relative_max``
        The same ratio's maximum over features. A live subspace inside a
        mostly-dead encoding shows here and not in the mean.
    ``feat_std_max``
        Max per-feature absolute std. Kept for the same reason.
    ``feat_std_mean``
        The shared callback's statistic, recomputed over the rollout rather
        than one forward batch. Logged for continuity with Ant-Tag runs and
        with PITFALLS.md, NOT as the sentinel -- see the module docstring.
    ``feat_abs_mean``
        Mean ``|feature|``. Makes the denominator of the ratio visible, so a
        relative reading can be interpreted rather than guessed at.

    Features are collected by a forward hook on the extractor, so nothing
    depends on ``last_st_features`` and nothing in the shared package is
    touched.
    """

    def __init__(self, eps: float = 1e-8, verbose: int = 0):
        super().__init__(verbose)
        self.eps = float(eps)
        self._warned = False
        #: The statistics from the most recent rollout, under their logged
        #: names. Kept so a test or a post-run check can read exactly what
        #: was recorded; SB3's logger flushes name_to_value each dump.
        self.last_stats: dict[str, float] = {}

    def _on_step(self) -> bool:  # required abstract method
        return True

    def _rollout_features(self) -> torch.Tensor | None:
        """Encoder features of EXACTLY the observations in the rollout buffer.

        Until 2026-09-06 this was a forward hook on the extractor, cleared at
        rollout end. That captured every PPO minibatch forward (n_epochs x the
        rollout) and every EvalCallback forward as well, so at n_epochs=10
        about 91% of the "rollout" sample was re-forwards of the PREVIOUS
        rollout through an encoder mid-update, and the first logged point
        was a different population from all later ones (PITFALLS.md section
        8 item 3). Re-encoding the buffer costs one extra pass over
        n_steps x n_envs observations per rollout and is exact.
        """
        from stable_baselines3.common.utils import obs_as_tensor

        extractor = getattr(self.model.policy, "features_extractor", None)
        buffer = getattr(self.model, "rollout_buffer", None)
        if extractor is None or buffer is None or not isinstance(
                buffer.observations, dict):
            return None
        flat = {key: np.asarray(value).reshape(-1, *value.shape[2:])
                for key, value in buffer.observations.items()}
        n = next(iter(flat.values())).shape[0]
        # The extractor returns [base_obs, st_features]; the base observation
        # is a passthrough and would dilute the statistic with the step
        # index's own (large, deterministic) variation.
        obs_dim = int(np.prod(self.model.observation_space["obs"].shape))
        was_training = extractor.training
        extractor.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, n, 1024):
                batch = {key: value[start:start + 1024] for key, value in flat.items()}
                out = extractor(obs_as_tensor(batch, self.model.device))
                chunks.append(out[:, obs_dim:].detach().float().cpu())
        extractor.train(was_training)
        return torch.cat(chunks, dim=0)

    def _on_rollout_end(self) -> None:
        features = self._rollout_features()
        if features is None:
            return
        if features.shape[0] < 2:
            # Cannot form a std from one sample. Say so rather than logging
            # NaN, which is how the shared callback's reading disappears at
            # n_envs=1 without anyone noticing.
            if not self._warned:
                print("OddEvenSTFeatureSentinel: only "
                      f"{features.shape[0]} feature sample(s) in the "
                      "rollout; no spread statistic is computable.")
                self._warned = True
            return

        with torch.no_grad():
            std = features.std(dim=0)
            abs_mean = features.mean(dim=0).abs()
            relative = _relative(std, abs_mean, self.eps)
            std_mean = float(std.mean())
            relative_mean = float(relative.mean())
            relative_max = float(relative.max())

        self.last_stats = {
            "st/feat_std_relative": relative_mean,
            "st/feat_std_relative_max": relative_max,
            "st/feat_std_max": float(std.max()),
            # NOT "st/feat_std_mean": that key belongs to the shared
            # STFeatureLoggingCallback (last-batch definition, comparable with
            # Ant-Tag). SB3's logger is last-write-wins, and this callback runs
            # after the shared one, so reusing the name silently replaced it.
            "st/feat_std_mean_rollout": std_mean,
            "st/feat_abs_mean": float(abs_mean.mean()),
            "st/feat_samples": float(features.shape[0]),
        }
        for key, value in self.last_stats.items():
            self.logger.record(key, value)

        if relative_mean < COLLAPSE_RELATIVE_SPREAD:
            # A warning, never an abort: five measured encoders on one
            # variant is not enough evidence to kill a run on.
            print("OddEvenSTFeatureSentinel: WARNING relative feature "
                  f"spread {relative_mean:.2e} is below "
                  f"{COLLAPSE_RELATIVE_SPREAD:.0e}; the encoder may have "
                  "collapsed to a constant. Probe it (can a linear readout "
                  "of the encoding recover the true state?) before trusting "
                  f"this run. Absolute feat_std_mean is {std_mean:.2e}, "
                  "which on this domain is NOT itself evidence either way.")


def _relative(std, abs_mean, eps: float):
    """``std / |mean|``, guarded so the guard itself cannot set the scale.

    A fixed additive epsilon would quietly reintroduce an absolute scale:
    with ``eps = 1e-8`` and features of order 1e-4 the epsilon is no longer
    negligible against ``|mean|``, so the ratio drifts under a pure
    rescaling -- measured, 3.029e-02 falling to 3.026e-02 under a 1e-4x
    scaling. Since defeating any absolute scale is the whole point of this
    statistic, the floor is made proportional to the batch's own typical
    magnitude instead. A genuinely all-zero encoding (mean and std both
    exactly 0) still yields 0 rather than a division by zero.
    """
    typical = float(abs_mean.mean())
    floor = eps * typical if typical > 0 else eps
    return std / (abs_mean + floor)


def relative_feature_spread(features, eps: float = 1e-8) -> float:
    """The sentinel statistic, for offline use on a feature matrix.

    Exposed so a probe script or a test can compute exactly what the callback
    logs, on a ``[num_samples, num_features]`` array, without running PPO.
    """
    features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(
            "need a [num_samples, num_features] matrix with at least 2 "
            f"samples, got shape {tuple(features.shape)}")
    std = features.std(dim=0)
    abs_mean = features.mean(dim=0).abs()
    return float(_relative(std, abs_mean, eps).mean())
