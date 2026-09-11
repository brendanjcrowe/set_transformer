"""Step-4 gate: can a linear readout of each encoding recover s*, and is the
signal at a magnitude a policy can use?

PITFALLS.md section 6 says to probe before committing to multi-million-step
runs. On Ant-Tag that cost minutes and predicted the frozen arm's failure.

TWO AXES, NOT ONE. On this domain a probe ALONE cannot decide anything:
encoders spanning three orders of magnitude of feature spread -- including
provably near-constant ones -- all probe s* at R^2 within 0.10 of each other.
A ~1e-4 signal riding an O(1) offset is linearly recoverable (Ridge on
z-scored features amplifies that direction ~1e4x) and still not learnable by a
policy head under PPO. So every encoding is scored on BOTH:

    probe R^2        is the information present at all?
    relative spread  is it present at a magnitude a policy can use?

An encoding passes only on both. That is strictly harder than PITFALLS section
6's caveat: there a chance-level probe still implied a frozen encoder cannot
work, while here nothing scores near chance.

WHAT IS PROBED. BOTH s* and parity, and parity is the discriminating one.

An earlier version of this docstring said "parity is free -- probe s*, not
parity", reasoning that observations are restricted to s*'s parity so one
observation reveals it. That conflated the FILTER with the POLICY. It is free
to the filter. The policy never sees an observation: it sees only an encoding
of the belief, and a compressed encoding can lose parity entirely. Measured:
the full posterior probes parity at 1.000 while every compressed encoding sits
near chance.

R^2 of s* is the weaker axis here -- it saturates near 0.99 for anything that
preserves the first moment, so it cannot rank encodings. Report both, and read
the ranking off parity.

Encodings compared, all reading the same exact-support belief:
    EXACT       the posterior itself -- the ceiling, not an encoder
    CGF_1D      the real WeightedCGFFeaturesExtractor at spread_1d init
    CGF_LEGACY  the same at the legacy linspace_first_dim init
    GAUSS2      weighted mean + variance -- the reference-poor encoder
    ST_RANDOM   SetTransformer at random init: the capacity control
    ST_PRE      SetTransformer after Sinkhorn pretraining (--st_checkpoint)

Usage:
    python3 diagnostics/probe_odd_even_belief_separability.py --variant oe50
    python3 diagnostics/probe_odd_even_belief_separability.py --variant oe50 \
        --st_checkpoint experiments/oe50_st_sentinel/.../checkpoint_latest.pt
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ODD_EVEN_DIR = Path(__file__).resolve().parents[1]

import gymnasium as gym
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score

import pdomains  # noqa: F401 - registers the pdomains-odd-even-* envs
from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig
from set_transformer.rl.feature_extractors.cgf import WeightedCGFFeaturesExtractor
from set_transformer.rl.feature_extractors.st import SetTransformerFeaturesExtractor

# Load the Odd-Even registry BY PATH, never by flat name. experiments/ant_tag/
# has a variants.py too, and any test or launcher that leaves that directory on
# sys.path makes `import variants` return the ANT-TAG registry -- which fails
# here as "Unknown variant 'oe50'". That is Gap 12 in domain_mds/oddeven.md,
# and it bit this very file in the full test suite.
sys.path.insert(0, str(_ODD_EVEN_DIR))
import _sibling  # noqa: E402 - path-based sibling loader

variants = _sibling.load("variants")

#: Below this relative spread an encoding is effectively constant, so a policy
#: head has nothing to learn from however well it probes. Matches
#: st_feature_sentinel.COLLAPSE_RELATIVE_SPREAD.
COLLAPSE_RELATIVE_SPREAD = 5e-3


def collect_beliefs(variant_name, n_episodes, max_step, seed):
    """Exact-support beliefs from the real env, with the labels a probe needs.

    Samples one snapshot per (episode, step) up to max_step, so the transient
    is represented rather than drowned by the one-hot steady state. Returns
    the episode index as GROUPS: consecutive steps of one episode share s*, so
    an ungrouped split leaks the label across folds.
    """
    v = variants.resolve(variant_name)
    ns = v.n_dist_size
    states, weights, labels, groups, steps = [], [], [], [], []
    for ep in range(n_episodes):
        env = OddEvenPOMDP(OddEvenPOMDPConfig(n_dist_size=ns, seed=seed + ep))
        obs0, _ = env.reset(seed=seed + ep)
        pf = v.particle_filter(num_particles=ns, initial_env_obs=obs0,
                               n_dist_size=ns, rng_seed=seed + ep)
        for step in range(max_step):
            states.append(pf.particles.ravel().copy())
            weights.append(pf.weights.copy())
            labels.append(env.true_state)
            groups.append(ep)
            steps.append(step)
            emitted = env._draw_observations(1)
            env.update_belief(int(emitted[0]))
            pf.predict(np.array(0))
            pf.update(emitted)
    return (np.array(states, dtype=np.float64),
            np.array(weights, dtype=np.float64),
            np.array(labels), np.array(groups), np.array(steps))


def _dict_space(ns):
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (ns, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (ns,), np.float32),
    })


def _extractor_features(extractor, states, weights, ns, centre, scale):
    """Run a real SB3 extractor, dropping the base-obs passthrough column.

    Particles are centred and fed RAW; the extractor divides by arena_scale
    itself, exactly as it does at RL time. Feeding pre-scaled particles would
    measure a different input distribution than the policy ever sees.
    """
    p = torch.tensor((states - centre)[:, :, None], dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(p), 512):
            batch = {"obs": torch.zeros(len(p[i:i + 512]), 1),
                     "particles": p[i:i + 512], "weights": w[i:i + 512]}
            out.append(extractor(batch)[:, 1:].numpy())
    return np.concatenate(out)


def build_encodings(states, weights, ns, st_checkpoint, torch_seed):
    """Every encoding under test, as (name, features) pairs."""
    centre, scale = (ns + 1) / 2.0, (ns - 1) / 2.0
    space, out = _dict_space(ns), {}

    # The posterior itself. The ceiling: what any encoder is compressing.
    out["EXACT"] = weights.copy()

    # Weighted mean + variance, on centred/scaled states like the real arm.
    x = (states - centre) / scale
    w = weights / weights.sum(axis=1, keepdims=True)
    mean = (w * x).sum(axis=1)
    var = (w * (x - mean[:, None]) ** 2).sum(axis=1)
    out["GAUSS2"] = np.c_[mean, var]

    for label, mode in (("CGF_1D", "spread_1d"),
                        ("CGF_LEGACY", "linspace_first_dim")):
        torch.manual_seed(torch_seed)
        out[label] = _extractor_features(
            WeightedCGFFeaturesExtractor(space, num_cgf_features=64,
                                         arena_scale=scale, t_init_mode=mode),
            states, weights, ns, centre, scale)

    # The capacity control. PITFALLS section 6: a random ISAB projection
    # probed 0.997 on Ant-Tag, so a high score proves nothing on its own.
    torch.manual_seed(torch_seed)
    out["ST_RANDOM"] = _extractor_features(
        SetTransformerFeaturesExtractor(space, num_encodings=8, dim_encoder=8,
                                        arena_scale=scale, weight_channel=True),
        states, weights, ns, centre, scale)

    if st_checkpoint:
        torch.manual_seed(torch_seed)
        out["ST_PRE"] = _extractor_features(
            SetTransformerFeaturesExtractor(
                space, num_encodings=8, dim_encoder=8, arena_scale=scale,
                weight_channel=True, pretrained_st_model_path=st_checkpoint),
            states, weights, ns, centre, scale)
    return out


def relative_spread(features):
    """Across-sample std relative to feature magnitude. Scale-free.

    feat_std_mean alone is scale- and width-dependent, so PITFALLS' absolute
    ~0.01 threshold does not transfer between domains: a healthy encoder here
    reads 1.5e-2 and a dead one 9.0e-5, while on Ant-Tag features were O(1).
    """
    std = features.std(axis=0)
    mag = np.abs(features).mean() + 1e-12
    return float((std / mag).mean())


def probe(features, labels, groups, n_splits):
    """Grouped-CV R^2 of a ridge readout of s*, on z-scored features."""
    x = features - features.mean(axis=0)
    x = x / (x.std(axis=0) + 1e-12)
    scores = cross_val_score(Ridge(alpha=1.0), x, labels,
                             cv=GroupKFold(n_splits=n_splits), groups=groups,
                             scoring="r2")
    return float(scores.mean()), float(scores.std())


def probe_parity(features, labels, groups, n_splits):
    """Grouped-CV balanced accuracy of a PARITY readout. Chance = 0.500.

    THIS is the discriminating axis on this domain, and the reason is the whole
    point of the env. Observations from an even state are always even numbers,
    so the two parities' observation distributions never overlap -- but the
    POLICY never sees an observation, only an encoding of the belief. A
    Gaussian (mean + variance) encoding is therefore blind to parity while the
    belief is still wide: mean and variance are near-identical for odd- and
    even-state episodes, so rounding the belief mean recovers parity at only
    0.357 after 2 observations and 0.512 after 8.

    Regressing s* as a CONTINUOUS value hides this completely -- the posterior
    mean is a near-perfect estimator of s* (corr 0.9965), so every encoding
    that keeps the first moment scores R^2 ~ 0.99 whether or not it can tell
    an odd state from an even one. Probe parity, not just the value.
    """
    x = features - features.mean(axis=0)
    x = x / (x.std(axis=0) + 1e-12)
    scores = cross_val_score(
        LogisticRegression(max_iter=2000), x, labels % 2,
        cv=GroupKFold(n_splits=n_splits), groups=groups,
        scoring="balanced_accuracy")
    return float(scores.mean()), float(scores.std())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    variants.add_variant_argument(p, default="oe50")
    p.add_argument("--n_episodes", type=int, default=300)
    p.add_argument("--max_step", type=int, default=25,
                   help="Snapshots per episode. Keep near the collapse step so "
                        "the transient is represented, not drowned.")
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--torch_seed", type=int, default=0,
                   help="Seeds every encoder init. A single unseeded random "
                        "init is one draw, not a control.")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--st_checkpoint", type=str, default=None)
    args = p.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    v = variants.resolve(args.variant)
    print(f"Variant {args.variant} | env {v.env_id} | n={v.n_dist_size} | "
          f"filter {v.particle_filter.__name__}")
    print(f"Collecting {args.n_episodes} episodes x {args.max_step} steps ...")
    states, weights, labels, groups, steps = collect_beliefs(
        args.variant, args.n_episodes, args.max_step, args.seed)
    print(f"{len(states)} snapshots, {len(set(labels))} distinct hidden states, "
          f"GroupKFold by episode ({args.n_splits} folds)\n")

    encodings = build_encodings(states, weights, v.n_dist_size,
                                args.st_checkpoint, args.torch_seed)
    print(f"{'encoding':12s} {'R2(s*)':>9s} {'PARITY bacc':>13s} "
          f"{'rel spread':>12s}  width")
    print("-" * 62)
    verdicts = {}
    for name, feats in encodings.items():
        r2, _sd = probe(feats, labels, groups, args.n_splits)
        pa, psd = probe_parity(feats, labels, groups, args.n_splits)
        rs = relative_spread(feats)
        verdicts[name] = (r2, pa, rs)
        print(f"{name:12s} {r2:9.3f} {pa:8.3f} +/-{psd:4.3f} {rs:12.2e}  "
              f"{feats.shape[1]}")
    print("-" * 62)
    print(f"collapse threshold: relative spread < {COLLAPSE_RELATIVE_SPREAD:.0e}"
          "   parity chance = 0.500")
    print("\nPARITY is the discriminating axis. R2(s*) saturates for every")
    print("encoding that keeps the first moment, because the posterior mean is")
    print("already a near-perfect estimator of s*.")
    return verdicts


if __name__ == "__main__":
    main()
