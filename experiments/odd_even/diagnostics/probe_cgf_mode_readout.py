"""Can a readout of a 64-feature CGF encoding name the POSTERIOR MODE?

THE QUESTION, and why the two existing probes cannot answer it.

`probe_odd_even_belief_separability.py` scores PARITY (chance 0.500) and
R^2(s*) (saturates at ~0.99 for anything keeping the first moment).
`probe_trained_rl_features.py` scores the same two axes on a trained agent's
extractor. Neither asks the question the RL result raises.

Under the 0/1 exact-match reward the optimal action IS the posterior mode
(`get_optimal_prediction`), so the policy's job reduces to naming one of 50
states. CGF reaches 0.428 steady exact-match against Gaussian's 0.332 and a
Bayes oracle's 0.881 (domain_mds/oddeven.md, 2026-09-03). Two readings fit
that gap and they call for opposite next steps:

    ENCODING-BOUND  the 64 CGF numbers do not determine the mode, so no
                    policy head on top of them can name it. Fix the encoder.
    POLICY-BOUND    the mode IS recoverable from those 64 numbers and PPO
                    did not learn the readout. Fix the optimisation.

So this probe does 50-way classification of the mode (target B) and of the
true state (target A) from each encoding, and reads the answer off where CGF
lands between the Gaussian floor and the exact-posterior ceiling.

TWO THINGS THAT MAKE THIS PROBE HARDER TO OVER-READ THAN THE OTHERS.

1. **Unstandardised is the run that matters.** PITFALLS.md #6 and the
   two-axis gate in the sibling probe both record that z-scoring lets a linear
   readout amplify a ~1e-4 direction riding an O(1) offset by ~1e4x -- a
   signal a policy head under PPO provably cannot use (a random ISAB
   projection probed 0.997 on Ant-Tag). Every classifier here is therefore
   fitted TWICE, with and without per-feature z-scoring, and the relative
   feature spread is reported alongside. Read the unstandardised column for
   the RL question; the standardised one bounds what information is present
   at all.

2. **The ceiling is not 1.0.** Target A is the true state, and the Bayes
   oracle -- which sees the exact posterior and plays its mode -- scores about
   0.638 transient / 0.881 steady on it, because at step 8 the median P(true)
   is 0.594 and even at step 30 only 63% of episodes exceed 0.9. An encoding
   at 0.60 steady on target A is near the achievable ceiling, not at 68% of
   perfect. Target B (the mode itself) has no such ceiling: the exact
   posterior determines it by definition, so EXACT should read ~1.000 there
   and anything below that is compression loss.

FEATURE SETS. All read the SAME exact-support belief, normalized exactly as
the RL arms normalize it -- (s - 25.5) / 24.5 -- and all CGF variants are
computed by the real `WeightedCGFFeaturesExtractor`, never by a reimplemented
formula, so a change to the encoder cannot silently stop being measured here.

    CGF_T2       64 fixed t equally spaced in [-2, 2]. PRIMARY: [-2, 2] is
                 the RL arm's own t_clamp, so this is the range the trained
                 encoder is confined to.
    CGF_T8       the same at [-8, 8], with t_clamp widened to match. Isolates
                 whether the CLAMP is the binding constraint rather than the
                 CGF map itself. (exp_arg_clamp=20 does not bind: the largest
                 |t . x| here is 8.0.)
    CGF_LEARNED{0,1,2}  the t_values that came OUT of each 3M-step PPO run,
                 loaded from the checkpoint. What the policy actually had.
    GAUSS2       weighted mean + variance. The floor: the encoding the RL
                 comparison beat by 0.096 steady.
    EXACT        the 50-vector posterior. The ceiling.

CLASSIFIERS. Multinomial logistic regression and a 128-unit one-hidden-layer
MLP, both GroupKFold by episode -- consecutive steps of one episode share s*,
so an ungrouped split leaks the label straight across folds.

Usage:
    OMP_NUM_THREADS=4 python3 diagnostics/probe_cgf_mode_readout.py \
        --variant oe50_short --n_episodes 300
"""

from __future__ import annotations

import argparse
import json
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# sys.path bootstrap (PITFALLS.md section 7). Depths from THIS file, one level
# below experiments/odd_even/: parents[2] is the set_transformer submodule
# root, parents[3] the repo root. Both are needed and the ORDER matters --
# `set_transformer` is not pip-installed here, and the repo root contains a
# directory of that name with no __init__.py, so with the repo root first the
# import resolves to that bare directory as a namespace package and
# `set_transformer.rl` does not exist. The submodule root must come first so
# the name resolves to the real package. The repo root is still required, for
# `pdomains`, which variants.py imports.
_HERE = Path(__file__).resolve().parent
_OE_DIR = _HERE.parent
_ST_ROOT = _HERE.parents[2]
_REPO_ROOT = _HERE.parents[3]
for _p in (str(_REPO_ROOT), str(_OE_DIR), str(_ST_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from set_transformer.rl.feature_extractors.cgf import (  # noqa: E402
    WeightedCGFFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.st import (  # noqa: E402
    SetTransformerFeaturesExtractor,
)

# The registry and the belief env live in the package (no flat-name sibling to collide
# with experiments/ant_tag/'s -- Gap 12 in domain_mds/oddeven.md).
from set_transformer.rl.domains import odd_even as variants  # noqa: E402
belief_env = variants

#: The transient/steady split. Steps 1..21 are the transient -- the belief is
#: still sharpening and the encodings provably differ there; 22..cap is the
#: steady state. domain_mds/oddeven.md uses this same boundary for every
#: reward number, so a probe on a different one is not comparable to them.
TRANSIENT_MAX_STEP = 21

#: Sinkhorn-pretrained ST checkpoints from the 2026-09-04 alignment entry in
#: domain_mds/oddeven.md, relative to the REPO ROOT. Both live under
#: experiments/ant_tag/ because step 3 (3_train_st.py) is env-generic and is
#: reused unchanged by this domain -- the directory names the script, not the
#: domain. Recorded there: BOTH decoders emit a single point at the weighted
#: mean (per-cloud output spread 0.0000), so neither ever had to represent the
#: multi-modal same-parity comb, and both should land near Gaussian here.
ST_PLAIN15_CHECKPOINT = ("set_transformer/experiments/ant_tag/experiments/"
                         "oe50_short_st_plain_e15/sinkhorn_2026-09-04_01-18-26/"
                         "checkpoints/checkpoint_best.pt")
ST_ALIGN5_CHECKPOINT = ("set_transformer/experiments/ant_tag/experiments/"
                        "oe50_short_st_align_e5/sinkhorn_2026-09-04_01-14-30/"
                        "checkpoints/checkpoint_best.pt")

#: The end-to-end ST agents, relative to experiments/odd_even/. These scored at
#: chance on reward but probed at R^2 0.974 with relative spread 5e+01
#: (oddeven.md 2026-09-03), i.e. NOT collapsed -- so their mode accuracy
#: separates "the features lack the mode" from "PPO never read it".
ST_E2E_AGENT = ("runs/odd_even_st_oe50_short/"
                "20260903_180819_seed{seed}_exact3M/models/st_agent.zip")

#: The Bayes oracle's own accuracy on target A (the true state), from the
#: 150-episode reference measurement in domain_mds/oddeven.md. Target A's
#: ceiling, printed with the tables so 0.60 is not misread as "40% short".
ORACLE_TRUE_STATE_ACC = {"transient": 0.638, "steady": 0.881}


def _dict_space(num_states: int) -> gym.spaces.Dict:
    """The observation space the RL arms' extractors are constructed against."""
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (num_states, 1),
                                    np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (num_states,), np.float32),
    })


def collect_rollouts(variant: str, n_episodes: int, seed: int):
    """Roll the RL arms' own belief env, recording one snapshot per decision.

    Rolls `make_odd_even_belief_env` -- the same factory every arm and the
    eval build -- so the belief distribution probed is the one the policy
    sees, including the float32 cast of the weights and the wrapper's
    centring of the particles.

    THE TIMING, which is the one thing here that is easy to get silently
    wrong. `step()` fixes the prediction BEFORE drawing that step's
    observations, so the belief available when choosing the action for step t
    holds o_0 .. o_{t-1}. The reset observation is therefore the snapshot for
    step 1, and the obs returned by step t is the snapshot for step t+1. The
    obs after the final step is never acted on and is dropped. Recording the
    post-step belief against step t instead would hand every encoding one
    extra observation and inflate every number in the table.

    Labels come from the SAME info dict that produced the snapshot:
        target A  info['true_state']         -- s*, constant within an episode
        target B  info['optimal_prediction'] -- argmax_s P(s | o_0..o_{t-1})
    Both are read off the env's float64 posterior rather than recomputed from
    the float32 weights in the observation, so an argmax tie broken
    differently by the cast cannot mislabel a row.

    Returns particles/weights as the extractors will receive them, plus
    labels, episode groups and 1-based step indices.
    """
    resolved = variants.resolve(variant)
    num_states = resolved.n_dist_size
    cap = variants.episode_cap(variant)

    env = belief_env.make_odd_even_belief_env(
        variant=variant, num_particles=num_states, rank=0, seed=seed)()

    particles, weights, base_obs = [], [], []
    true_states, modes, groups, steps = [], [], [], []

    for episode in range(n_episodes):
        # seed + episode, never a constant seed: the hidden state is drawn at
        # reset, so one seed IS one episode replayed N times (PITFALLS.md #2
        # and the Gap 4 seeding trap).
        obs, info = env.reset(seed=seed + episode)
        for step in range(1, cap + 1):
            particles.append(np.asarray(obs["particles"], dtype=np.float32))
            weights.append(np.asarray(obs["weights"], dtype=np.float32))
            base_obs.append(np.asarray(obs["obs"], dtype=np.float32))
            true_states.append(int(info["true_state"]))
            modes.append(int(info["optimal_prediction"]))
            groups.append(episode)
            steps.append(step)
            # The action is irrelevant to the belief trajectory: it is a
            # prediction and touches neither the hidden state nor the
            # observation model, so the posterior evolves identically under
            # any policy (domain_mds/oddeven.md, 2026-09-03 visualisation).
            obs, _reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
    env.close()

    return {
        "particles": np.stack(particles),
        "weights": np.stack(weights),
        "base_obs": np.stack(base_obs),
        "true_state": np.asarray(true_states, dtype=np.int64),
        "mode": np.asarray(modes, dtype=np.int64),
        "group": np.asarray(groups, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
        "num_states": num_states,
        "cap": cap,
    }


def _run_extractor(extractor, data, batch_size=1024) -> np.ndarray:
    """Features from a real SB3 extractor, dropping the base-obs passthrough.

    Column 0 of every extractor's output is `obs_dict["obs"]` -- here the
    normalized step index, which is not part of the belief encoding and which
    a 50-way readout would otherwise use as a strong prior over the mode
    (later steps concentrate). The probe must measure the ENCODING, so the
    passthrough is dropped.
    """
    extractor.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data["particles"]), batch_size):
            sl = slice(i, i + batch_size)
            out.append(extractor({
                "obs": torch.from_numpy(data["base_obs"][sl]),
                "particles": torch.from_numpy(data["particles"][sl]),
                "weights": torch.from_numpy(data["weights"][sl]),
            })[:, 1:].numpy())
    return np.concatenate(out).astype(np.float64)


def _fixed_t_cgf(space, num_features, arena_scale, t_lo, t_hi):
    """The real CGF extractor with its t_values replaced by a fixed grid.

    Built through `WeightedCGFFeaturesExtractor` and not reimplemented: the
    formula, the weight sanitising, the arena_scale division and both clamps
    then come from the code the RL arms run, so this probe cannot drift away
    from the encoder it claims to measure.

    `t_clamp` is set to cover the grid. It is NOT a free choice -- forward()
    clamps t elementwise, so leaving the default 2.0 in place would silently
    collapse a [-8, 8] grid onto {-2, 2} and 60 of the 64 features would be
    duplicates of four distinct values. The RL arm's clamp is 2.0, which is
    exactly why CGF_T2 is the primary row and CGF_T8 the diagnostic.
    """
    extractor = WeightedCGFFeaturesExtractor(
        space,
        num_cgf_features=num_features,
        arena_scale=arena_scale,
        t_init_mode="spread_1d",
        t_clamp=max(abs(t_lo), abs(t_hi)),
    )
    grid = torch.linspace(float(t_lo), float(t_hi),
                          num_features).reshape(-1, 1)
    with torch.no_grad():
        extractor.t_values.copy_(grid)
    return extractor


def _learned_t_cgf(agent_path: str, space, arena_scale):
    """The features extractor that came OUT of a PPO run.

    Loaded through PPO.load and taken off the policy, as
    probe_trained_rl_features.py does, rather than by reading t_values out of
    the zip by hand -- the extractor carries its own num_cgf_features,
    arena_scale, t_clamp and exp_arg_clamp, and reconstructing those from a
    run_config would let this probe measure a geometry the run never used.
    The saved extractor is used directly, so its clamp (2.0) applies exactly
    as it did under PPO.
    """
    from stable_baselines3 import PPO

    # SB3 pickles the extractor CLASS by module path, so the arm's module
    # must be importable before the zip can be unpickled.
    importlib.import_module("4_train_rl_cgf")   # experiments/odd_even/ is on sys.path
    model = PPO.load(agent_path, device="cpu")
    extractor = model.policy.features_extractor
    if not isinstance(extractor, WeightedCGFFeaturesExtractor):
        raise TypeError(
            f"{agent_path} carries a {type(extractor).__name__}, not a "
            "WeightedCGFFeaturesExtractor; this probe compares CGF geometries")
    if abs(float(extractor.arena_scale) - float(arena_scale)) > 1e-9:
        raise ValueError(
            f"{agent_path} was trained at arena_scale="
            f"{extractor.arena_scale} but this variant normalizes by "
            f"{arena_scale}; the encoder would see inputs of a different "
            "scale than it was trained on (PITFALLS.md section 4)")
    return extractor


def cgf_k_and_kprime(data, arena_scale, t_values, batch_size=1024):
    """K(t) and its derivative K'(t) from ONE shared score tensor.

        score_ij = t_j * x_i + log w_i
        K(t_j)   = logsumexp_i score_ij                     (the CGF itself)
        K'(t_j)  = sum_i softmax_i(score_ij) * x_i          (the TILTED MEAN)

    K' is the mean of the belief after exponentially tilting it toward large
    x by e^{t x}. At t = 0 it is the plain posterior mean; as t grows it walks
    toward the largest-x support point and as t falls toward the smallest. So
    a GRID of K' values traces out where the belief's mass actually sits,
    which is exactly the information a single mean throws away.

    WHY logsumexp AND NOT THE EXTRACTOR. `WeightedCGFFeaturesExtractor.forward`
    computes exp -> weighted sum -> log with an `exp_arg_clamp` of 20.0. At
    |t| = 50 and |x| = 1 the argument reaches 50, so the clamp would bind on
    most of the grid and silently flatten the wide-t rows into each other --
    measuring the clamp instead of the encoding. logsumexp is the numerically
    stable identity for the same quantity with no clamp at all, so the wide
    grids mean what they say. The [-2, 2] rows are ALSO computed through the
    real extractor elsewhere in this script and the two agree (the clamp does
    not bind there), which is what lets the old and new tables join.

    Zero weights give log w = -inf. That is correct and deliberate: softmax
    maps -inf to exactly 0, so a refuted state contributes nothing to K'
    rather than contributing a tiny epsilon. On this domain the exact-support
    filter zeroes the whole opposite parity every step, so this path is the
    common case, not an edge case.

    Returns (K, K_prime), each [n_samples, len(t_values)], float64.
    """
    particles = torch.from_numpy(
        data["particles"].astype(np.float64)[:, :, 0]) / arena_scale
    weights = torch.from_numpy(data["weights"].astype(np.float64))
    weights = weights / weights.sum(dim=1, keepdim=True)
    log_w = torch.log(weights)  # -inf where w == 0, handled by softmax
    t = torch.as_tensor(np.asarray(t_values, dtype=np.float64))

    k_out, kp_out = [], []
    with torch.no_grad():
        for i in range(0, len(particles), batch_size):
            x = particles[i:i + batch_size]                     # [B, N]
            lw = log_w[i:i + batch_size]                        # [B, N]
            score = x[:, :, None] * t[None, None, :] + lw[:, :, None]
            k_out.append(torch.logsumexp(score, dim=1).numpy())  # [B, T]
            tilted = torch.softmax(score, dim=1)                 # [B, N, T]
            kp_out.append((tilted * x[:, :, None]).sum(dim=1).numpy())
    return np.concatenate(k_out), np.concatenate(kp_out)


def _st_extractor(space, arena_scale, torch_seed, checkpoint=None):
    """A SetTransformerFeaturesExtractor built exactly as the sibling probe does.

    Same geometry as the RL arm and as
    `probe_odd_even_belief_separability.build_encodings`: 8 x 8 = 64 features,
    `weight_channel=True` so the encoder reads the PF weights as an input
    channel, and `arena_scale` from the registry so the particles it sees are
    on the scale it was pretrained on (PITFALLS.md section 4 -- pretraining and
    RL frames disagreeing is a bug this domain has already paid for once).

    torch.manual_seed before construction: an unseeded random init is one draw,
    not a control.
    """
    torch.manual_seed(torch_seed)
    kwargs = dict(num_encodings=8, dim_encoder=8, arena_scale=arena_scale,
                  weight_channel=True)
    if checkpoint is not None:
        kwargs["pretrained_st_model_path"] = checkpoint
    return SetTransformerFeaturesExtractor(space, **kwargs)


def _e2e_st_extractor(agent_path: str, arena_scale):
    """The ST extractor that came OUT of an end-to-end PPO run.

    Same pattern as `_learned_t_cgf` and as
    `probe_trained_rl_features.collect`: SB3 pickles the extractor CLASS by
    module path, so `4_train_rl_st` must be importable before the zip can be
    unpickled -- otherwise the load dies on the class lookup, not on anything
    about the weights.
    """
    from stable_baselines3 import PPO

    importlib.import_module("4_train_rl_st")
    model = PPO.load(agent_path, device="cpu")
    extractor = model.policy.features_extractor
    if not isinstance(extractor, SetTransformerFeaturesExtractor):
        raise TypeError(
            f"{agent_path} carries a {type(extractor).__name__}, not a "
            "SetTransformerFeaturesExtractor")
    if abs(float(extractor.arena_scale) - float(arena_scale)) > 1e-9:
        raise ValueError(
            f"{agent_path} was trained at arena_scale={extractor.arena_scale} "
            f"but this variant normalizes by {arena_scale}")
    return extractor


def build_feature_sets(data, agent_paths, variant):
    """Every encoding under test, as name -> [n_samples, width] features."""
    num_states = data["num_states"]
    space = _dict_space(num_states)
    arena_scale = variants.state_scale(variant)
    sets: dict[str, np.ndarray] = {}

    # The ceiling. Not an encoder -- the posterior itself, which determines
    # target B by definition.
    weights = data["weights"].astype(np.float64)
    sets["EXACT"] = weights / weights.sum(axis=1, keepdims=True)

    # The floor. Computed on the same normalized particles the extractors
    # see, so it is the Gaussian ARM's encoding and not a differently scaled
    # cousin. (WeightedGaussianFeaturesExtractor would give the identical two
    # numbers; done in closed form here to keep the 1-D case explicit.)
    x = data["particles"].astype(np.float64)[:, :, 0] / arena_scale
    w = weights / weights.sum(axis=1, keepdims=True)
    mean = (w * x).sum(axis=1)
    var = (w * (x - mean[:, None]) ** 2).sum(axis=1)
    sets["GAUSS2"] = np.c_[mean, var]

    # PRIMARY: the RL arm's own t range.
    sets["CGF_T2"] = _run_extractor(
        _fixed_t_cgf(space, 64, arena_scale, -2.0, 2.0), data)
    # Is the clamp the binding constraint?
    sets["CGF_T8"] = _run_extractor(
        _fixed_t_cgf(space, 64, arena_scale, -8.0, 8.0), data)

    for label, path in agent_paths:
        sets[label] = _run_extractor(
            _learned_t_cgf(path, space, arena_scale), data)

    # ---- the t sweep, with K, K' and both -----------------------------------
    # Why these ranges. Particles are normalized by arena_scale, so adjacent
    # SAME-PARITY states (the ones the belief actually has to tell apart) are
    # 2 / 24.5 = 0.0816 apart. A tilt of e^{t x} separates two points 0.0816
    # apart by a factor e^{0.0816 t}, so it takes |t| ~ 1 / 0.0816 = 12.3
    # before neighbouring states are told apart by a factor of e. That is why
    # [-2, 2] cannot resolve neighbours -- across the whole state range it is
    # nearly its own Taylor expansion t*mean + t^2*var/2, i.e. the Gaussian
    # encoding -- and why the switch is expected between [-8, 8] and [-20, 20].
    for lo, hi in ((2.0, 2.0), (8.0, 8.0), (20.0, 20.0), (50.0, 50.0)):
        grid = np.linspace(-lo, hi, 64)
        k, kp = cgf_k_and_kprime(data, arena_scale, grid)
        tag = f"T{int(hi)}"
        sets[f"K_{tag}"] = k
        sets[f"KP_{tag}"] = kp
        sets[f"KKP_{tag}"] = np.c_[k, kp]

    # ---- Set Transformer encodings -----------------------------------------
    # Built exactly as probe_odd_even_belief_separability builds ST_RANDOM /
    # ST_PRE, so the rows are comparable with that probe's table.
    st_specs = [("ST_RANDOM", None)]
    for label, path in (("ST_PLAIN15", ST_PLAIN15_CHECKPOINT),
                        ("ST_ALIGN5", ST_ALIGN5_CHECKPOINT)):
        full = _REPO_ROOT / path
        if full.exists():
            st_specs.append((label, str(full)))
        else:
            print(f"  WARNING: {label} checkpoint missing, skipping: {full}")
    for label, checkpoint in st_specs:
        sets[label] = _run_extractor(
            _st_extractor(space, arena_scale, 0, checkpoint), data)

    for seed in (0, 1, 2):
        path = _OE_DIR / ST_E2E_AGENT.format(seed=seed)
        if not path.exists():
            print(f"  WARNING: ST_E2E{seed} agent missing, skipping: {path}")
            continue
        sets[f"ST_E2E{seed}"] = _run_extractor(
            _e2e_st_extractor(str(path), arena_scale), data)

    return sets


def relative_spread(features: np.ndarray) -> float:
    """Across-sample std relative to feature magnitude. Scale-free.

    The magnitude axis of the two-axis gate (PITFALLS.md #6). A ~1e-4 signal
    riding an O(1) offset is linearly recoverable after z-scoring and still
    unusable by a policy head, so this number is what tells a standardised
    probe score apart from a usable one. Same definition as
    `probe_odd_even_belief_separability.relative_spread`, so the values are
    comparable with that table.
    """
    std = features.std(axis=0)
    mag = np.abs(features).mean() + 1e-12
    return float((std / mag).mean())


def geometry(features: np.ndarray, posterior_mean: np.ndarray) -> dict:
    """How many DIRECTIONS the encoding actually varies along, and along what.

    This is the mechanism behind the accuracy table and the reason a 64-wide
    encoding can score like a 1-wide one. Three numbers, all scale-free:

        eff_rank      exp(entropy of the normalized singular-value spectrum)
                      of the centred features. 1.0 means one direction
                      carries the variance however many columns there are.
        pc1_var_frac  variance fraction in the leading direction.
        pc1_corr_mean |corr| between that direction and the POSTERIOR MEAN.
                      If this is ~1, the encoding is a reparameterisation of
                      the mean and cannot carry more than the mean does.
        min_pair_corr smallest |corr| between any two features. Near 1 means
                      every column is a monotone restatement of one number;
                      GAUSS2 reads ~0.02 here because mean and variance are
                      genuinely independent, which is the useful contrast.

    Computed in float64 on the same features the classifiers get. Note the
    numerical rank can still be full while eff_rank is 1.0 -- the tail
    directions exist but at a magnitude the standardised/unstandardised
    contrast is exactly about.
    """
    # Absolute per-feature spread, reported ALONGSIDE the relative one because
    # the relative measure divides by the GLOBAL mean magnitude and so cannot
    # separate "small signal" from "large constant offset". ST_E2E is exactly
    # that case: relative spread 9.5e-4 but absolute per-column std 9.1e-5 on
    # features whose mean magnitude is 0.096.
    abs_std = float(features.std(axis=0).mean())

    centred = features - features.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    total = (singular ** 2).sum()
    if total <= 0:
        return {"eff_rank": 1.0, "pc1_var_frac": 1.0, "abs_std": abs_std,
                "pc1_corr_mean": float("nan"), "min_pair_corr": float("nan")}
    spectrum = singular ** 2 / total
    eff_rank = float(np.exp(-(spectrum * np.log(spectrum + 1e-300)).sum()))

    left, values, _ = np.linalg.svd(centred, full_matrices=False)
    pc1 = left[:, 0] * values[0]
    pc1_corr = abs(float(np.corrcoef(pc1, posterior_mean)[0, 1]))

    if features.shape[1] > 1:
        corr = np.corrcoef(features.T)
        off_diagonal = corr[~np.eye(features.shape[1], dtype=bool)]
        min_pair = float(np.abs(off_diagonal).min())
    else:
        min_pair = float("nan")

    return {"eff_rank": eff_rank, "pc1_var_frac": float(spectrum[0]),
            "abs_std": abs_std,
            "pc1_corr_mean": pc1_corr, "min_pair_corr": min_pair}


def _fit_predict(features, labels, groups, n_splits, classifier, standardise,
                 seed):
    """Grouped-CV out-of-fold predictions for one (classifier, scaling) pair.

    GroupKFold by episode: every step of one episode shares s*, so an
    ungrouped split puts the same label on both sides and the score measures
    memorisation. Scaling statistics are fitted on the TRAINING fold only --
    fitting them on everything leaks the test fold's distribution into the
    amplification the whole standardised/unstandardised contrast is about.
    """
    preds = np.empty_like(labels)
    for train, test in GroupKFold(n_splits=n_splits).split(
            features, labels, groups):
        x_train, x_test = features[train], features[test]
        if standardise:
            mu = x_train.mean(axis=0)
            sd = x_train.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            x_train = (x_train - mu) / sd
            x_test = (x_test - mu) / sd
        if classifier == "logreg":
            # Multinomial (softmax over all 50 classes), which is
            # LogisticRegression's only behaviour from sklearn 1.7 -- the
            # `multi_class` argument that used to select it was removed in
            # 1.9, so passing it raises rather than being ignored.
            model = LogisticRegression(max_iter=3000, C=1.0)
        elif classifier == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=600,
                                  random_state=seed, early_stopping=False)
        else:
            raise ValueError(f"unknown classifier {classifier!r}")
        model.fit(x_train, labels[train])
        preds[test] = model.predict(x_test)
    return preds


def _split_accuracy(preds, labels, steps):
    """Top-1 accuracy in the transient, the steady state, and pooled."""
    correct = preds == labels
    transient = steps <= TRANSIENT_MAX_STEP
    steady = ~transient
    return {
        "transient": float(correct[transient].mean()) if transient.any()
        else float("nan"),
        "steady": float(correct[steady].mean()) if steady.any()
        else float("nan"),
        "pooled": float(correct.mean()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    variants.add_variant_argument(parser, default="oe50_short")
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=9000,
                        help="Base rollout seed; episode e uses seed + e.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mlp_seed", type=int, default=0)
    parser.add_argument(
        "--agent", action="append", default=None, metavar="NAME:PATH",
        help="Trained CGF agent whose learned t_values to probe (repeatable). "
             "Defaults to the three exact3M seeds.")
    parser.add_argument("--json_out", type=str, default=None,
                        help="Write the full result table here.")
    args = parser.parse_args(argv)
    if args.list_variants:
        variants.print_variants()
        return None

    if args.agent:
        agent_paths = [tuple(spec.split(":", 1)) for spec in args.agent]
    else:
        base = (_OE_DIR / "runs/odd_even_cgf_oe50_short/"
                "20260903_180819_seed{}_exact3M/models/cgf_agent.zip")
        agent_paths = [(f"CGF_LEARNED{s}", str(base).format(s))
                       for s in (0, 1, 2)]

    resolved = variants.resolve(args.variant)
    cap = variants.episode_cap(args.variant)
    print(f"variant {args.variant} | env {resolved.env_id} | "
          f"n={resolved.n_dist_size} | cap {cap} | "
          f"filter {resolved.particle_filter.__name__}")
    print(f"normalization (s - {variants.state_centre(args.variant)}) / "
          f"{variants.state_scale(args.variant)}")
    print(f"rolling {args.n_episodes} episodes, seed {args.seed} + episode ...")

    data = collect_rollouts(args.variant, args.n_episodes, args.seed)
    n_transient = int((data["step"] <= TRANSIENT_MAX_STEP).sum())
    print(f"{len(data['step'])} snapshots "
          f"({n_transient} transient steps 1-{TRANSIENT_MAX_STEP}, "
          f"{len(data['step']) - n_transient} steady "
          f"{TRANSIENT_MAX_STEP + 1}-{cap}) | "
          f"{len(set(data['true_state'].tolist()))} distinct s* | "
          f"GroupKFold by episode, {args.n_splits} folds")
    mode_is_true = float((data["mode"] == data["true_state"]).mean())
    print(f"sanity: the oracle (play the posterior mode) hits s* on "
          f"{mode_is_true:.3f} of these snapshots")

    # SAMPLE-SIZE GUARD. Target B is a 50-way problem, so a small run does
    # not have the rows to fit 50 decision regions and EVERY encoding --
    # including EXACT, which determines the label by definition -- reads low.
    # Measured: at 25 episodes EXACT scored 0.43 (logreg, unstandardised) on
    # target B, which would place CGF's 0.16 as "37% of the ceiling" when the
    # ceiling itself was the artefact. At 300 episodes EXACT reads 0.975.
    # So report the per-class support and refuse to run a design where a
    # class can be absent from a fold.
    counts = np.bincount(data["mode"], minlength=resolved.n_dist_size + 1)[1:]
    print(f"target-B class support: {int(counts.min())} min / "
          f"{int(counts.max())} max over {resolved.n_dist_size} classes")
    if counts.min() < args.n_splits:
        raise SystemExit(
            f"a posterior-mode class appears only {int(counts.min())} times, "
            f"fewer than the {args.n_splits} folds, so it cannot be present "
            "in every training fold and the CEILING row will read low -- "
            "which makes the whole floor/ceiling comparison meaningless. "
            "Raise --n_episodes (300 gives >=55 per class on oe50_short).")
    print()

    feature_sets = build_feature_sets(data, agent_paths, args.variant)

    # The posterior mean, for the geometry column and the ROUND_MEAN row.
    norm_weights = data["weights"].astype(np.float64)
    norm_weights /= norm_weights.sum(axis=1, keepdims=True)
    norm_particles = (data["particles"].astype(np.float64)[:, :, 0]
                      / variants.state_scale(args.variant))
    posterior_mean = (norm_weights * norm_particles).sum(axis=1)

    print(f"{'encoding':<15} {'width':>6} {'rel.spread':>11} {'abs.std':>10} "
          f"{'eff.rank':>9} {'PC1 var':>8} {'|r(PC1,mean)|':>14} "
          f"{'min pair |r|':>13}")
    print("-" * 94)
    spreads, geometries = {}, {}
    for name, features in feature_sets.items():
        spreads[name] = relative_spread(features)
        geometries[name] = geometry(features, posterior_mean)
        g = geometries[name]
        print(f"{name:<15} {features.shape[1]:>6} {spreads[name]:>11.2e} "
              f"{g['abs_std']:>10.2e} {g['eff_rank']:>9.2f} "
              f"{g['pc1_var_frac']:>8.5f} "
              f"{g['pc1_corr_mean']:>14.5f} {g['min_pair_corr']:>13.5f}")
    print("-" * 94)
    print("eff.rank 1.0 with |r(PC1,mean)| ~ 1.0 means the encoding is a "
          "reparameterisation of\nthe posterior mean, whatever its width -- "
          "so it cannot carry more than the mean does.")
    print()

    # A NON-CLASSIFIER reference for target B: round the posterior mean back
    # to the nearest state and call that the mode. It is the readout the
    # Gaussian encoding admits in closed form, it needs no fitting, and it
    # bounds what any policy reading only the first moment can achieve. Worth
    # printing because logistic regression scores far BELOW it (a 50-way
    # partition of one scalar axis is not linearly expressible) while the MLP
    # roughly matches it -- which is what makes the MLP the right instrument
    # here and the logreg row an underestimate of the encoding rather than of
    # the information.
    rounded = np.clip(
        np.rint(posterior_mean * variants.state_scale(args.variant)
                + variants.state_centre(args.variant)),
        1, resolved.n_dist_size).astype(np.int64)
    for target_label, truth in (("posterior mode", data["mode"]),
                                ("true state s*", data["true_state"])):
        acc = _split_accuracy(rounded, truth, data["step"])
        print(f"reference  round(posterior mean) -> {target_label:<15} "
              f"tr {acc['transient']:.3f} / st {acc['steady']:.3f} / "
              f"pool {acc['pooled']:.3f}")
    print()

    targets = {"A_true_state": data["true_state"], "B_posterior_mode":
               data["mode"]}
    results: dict = {}
    chance = 1.0 / resolved.n_dist_size

    for target_name, labels in targets.items():
        results[target_name] = {}
        print("=" * 96)
        title = ("target A = true state s*" if target_name.startswith("A")
                 else "target B = posterior argmax (the optimal action)")
        print(f"{title}   |   50-way, chance {chance:.3f}")
        print("=" * 96)
        header = (f"{'encoding':<15} "
                  f"{'logreg raw':>21} {'logreg z':>21} "
                  f"{'mlp raw':>21} {'mlp z':>21}")
        print(header)
        print(f"{'':<15} " + " ".join(
            [f"{'tr / st / pool':>21}"] * 4))
        print("-" * 96)
        for name, features in feature_sets.items():
            row = {}
            cells = []
            for classifier in ("logreg", "mlp"):
                for standardise in (False, True):
                    preds = _fit_predict(
                        features, labels, data["group"], args.n_splits,
                        classifier, standardise, args.mlp_seed)
                    acc = _split_accuracy(preds, labels, data["step"])
                    key = f"{classifier}_{'z' if standardise else 'raw'}"
                    row[key] = acc
                    cells.append(f"{acc['transient']:.3f} /"
                                 f"{acc['steady']:.3f} /{acc['pooled']:.3f}")
            row["relative_spread"] = spreads[name]
            row["geometry"] = geometries[name]
            row["width"] = int(features.shape[1])
            results[target_name][name] = row
            print(f"{name:<15} " + " ".join(f"{c:>21}" for c in cells))
        print("-" * 96)
        print(f"chance {chance:.3f}", end="")
        if target_name.startswith("A"):
            print(f" | Bayes oracle on s*: "
                  f"{ORACLE_TRUE_STATE_ACC['transient']:.3f} transient / "
                  f"{ORACLE_TRUE_STATE_ACC['steady']:.3f} steady "
                  "-- the ceiling on THIS target is not 1.0")
        else:
            print(" | EXACT determines this target by definition, so its row "
                  "is the ceiling")
        print()

    print("READING (domain_mds/oddeven.md): the UNSTANDARDISED columns are the "
          "RL question.\n"
          "  CGF tracks GAUSS2  -> the encoding is the bottleneck\n"
          "  CGF tracks EXACT   -> PPO is failing to read what is there\n"
          "  CGF_T8 >> CGF_T2   -> the t_clamp is the bottleneck")

    if args.json_out:
        payload = {
            "variant": args.variant,
            "env_id": resolved.env_id,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "n_splits": args.n_splits,
            "cap": cap,
            "transient_max_step": TRANSIENT_MAX_STEP,
            "chance": chance,
            "oracle_true_state_acc": ORACLE_TRUE_STATE_ACC,
            "oracle_mode_hits_true_state": mode_is_true,
            "round_mean_reference": {
                "posterior_mode": _split_accuracy(rounded, data["mode"],
                                                  data["step"]),
                "true_state": _split_accuracy(rounded, data["true_state"],
                                              data["step"]),
            },
            "agents": [list(a) for a in agent_paths],
            "results": results,
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")
    return results


if __name__ == "__main__":
    main()
