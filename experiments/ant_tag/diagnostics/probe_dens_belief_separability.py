"""
Offline linear-probe separability test for Twin-Den Ant-Tag beliefs.

This is the cheap GATE in front of the two 6M-step PPO runs. Question:
the belief splits into two mirrored den modes, one tight and one loose;
can a LINEAR readout of a given belief encoding recover WHICH den is the
loose one (equivalently `den_tight`, the ordering bit) from the particle
filter's particles+weights alone?

The construction is moment-matched by design: at equal den weights the two
assignments are exact point reflections of one another, so the pooled mean
is 0 and the pooled covariance is identical -- every feature a
mean+covariance encoder can compute is population-invariant to the swap.
Only ODD moments (the third cumulant along the den diagonal) carry the bit.

Three subcommands:

  invariance  executable form of the moment-matching proof; needs no env.
              Builds an idealized two-disc belief and its point reflection
              and checks that GAUSS5's variance/covariance entries are
              bit-identical while CGF_SPREAD64 separates them.
  collect     roll out the training-identical env stack and log belief
              snapshots.
  probe       cross-validated logistic-regression probe on those snapshots.

Feature sets compared (all numpy mirrors of the real encoders, so the
probe measures encoder capacity, not policy quality):

  GAUSS5        weighted mean(2) + var(2) + cov_xy(1) -- exact mirror of
                WeightedGaussianFeaturesExtractor.forward (4_train_rl_gaussian.py)
  CGF_INIT64    the 64 CGF log-sum-exp features at the exact training
                initialization (t_init_mode="linspace_all_dims",
                t_init_scale=0.1) -- what CGF can see BEFORE any learning
  CGF_SPREAD64  64 fixed projections on 8 directions x 8 norms spanning the
                reachable range under the elementwise t_clamp=2.0 -- what CGF
                could see IF its t_j spread out during training. The den
                diagonal (45 deg) is one of the 8 directions exactly.
  ORACLE4       [w_den0, w_den1, weighted spread of den-0 particles, weighted
                spread of den-1 particles] -- NOT an encoder; a reference
                upper bound (the label is the argmin of the two spreads).

Every feature set is concatenated with ant_pos/arena_scale. The den centers
are constants of the env (known to any policy through training) so they are
not inputs.

Usage:
    python3 diagnostics/probe_dens_belief_separability.py invariance
    python3 diagnostics/probe_dens_belief_separability.py collect --n_episodes 300 \
        --out data/dens_probe_dataset.npz
    python3 diagnostics/probe_dens_belief_separability.py probe --data data/dens_probe_dataset.npz
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The 4_train_rl_* pipeline scripts live one level up, in experiments/ant_tag/.
# Only this script's own directory is on sys.path by default, so add theirs.
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
if str(_ANT_TAG_DIR) not in sys.path:
    sys.path.insert(0, str(_ANT_TAG_DIR))

import numpy as np

DEN_POSITIONS = np.array([[-2.7, -2.7], [2.7, 2.7]], dtype=np.float64)
DEN_R_TIGHT = 0.4
DEN_R_LOOSE = 1.4
ARENA_SCALE = 4.5


# ---------------------------------------------------------------------------
# feature builders (numpy mirrors of the SB3 extractors)
# ---------------------------------------------------------------------------

def _norm_weights(w):
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    return w / (w.sum(axis=1, keepdims=True) + 1e-8)


def feats_gauss5(x, w):
    """Exact numpy mirror of WeightedGaussianFeaturesExtractor.forward."""
    wc = w[..., None]
    mean = np.sum(wc * x, axis=1)                       # [B, 2]
    centered = x - mean[:, None, :]
    cov = np.einsum("bni,bnj->bij", wc * centered, centered)
    var = np.clip(np.diagonal(cov, axis1=1, axis2=2), 0.0, None)
    off = cov[:, 0, 1][:, None]
    return np.concatenate([mean, var, off], axis=1)


def _cgf(x, w, t, exp_arg_clamp=20.0):
    """log sum_i w_i exp(<t_j, x_i>), clamped exactly as the extractor does."""
    exp_arg = np.clip(x @ t.T, -exp_arg_clamp, exp_arg_clamp)   # [B, N, M]
    mgf = np.sum(w[..., None] * np.exp(exp_arg), axis=1)
    return np.log(np.clip(mgf, 1e-8, None))


def t_init_linspace_all_dims(num=64, dim=2, scale=0.1):
    """Deterministic mirror of WeightedCGFFeaturesExtractor's
    t_init_mode='linspace_all_dims' (no randn in this mode)."""
    t = np.zeros((num, dim), dtype=np.float64)
    lin = np.linspace(-scale, scale, num)
    assign = np.arange(num) % dim
    for d in range(dim):
        m = assign == d
        t[m, d] = lin[m]
    return t


def t_spread(num_dirs=8, num_norms=8, rho_lo=0.25, rho_hi=2.8, t_clamp=2.0):
    """8 directions x 8 norms. theta_k = 2*pi*k/8 includes the den diagonal
    (45 deg) exactly. The elementwise t_clamp the real extractor applies to
    its own parameters is applied here too, so this stays a true upper bound
    on what the trained encoder could actually realize."""
    thetas = 2 * np.pi * np.arange(num_dirs) / num_dirs
    rhos = np.geomspace(rho_lo, rho_hi, num_norms)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)   # [K, 2]
    t = (rhos[:, None, None] * dirs[None, :, :]).reshape(-1, 2)
    return np.clip(t, -t_clamp, t_clamp)


def _nearest_den(x, dens=DEN_POSITIONS):
    """x: [B, N, 2] -> (den_idx [B, N], dist [B, N])."""
    d = np.linalg.norm(x[:, :, None, :] - dens[None, None, :, :], axis=3)
    idx = np.argmin(d, axis=2)
    return idx, np.min(d, axis=2)


def den_weights(x, w, dens=DEN_POSITIONS):
    """Per-den belief mass. x raw (un-normalized) particles."""
    idx, _ = _nearest_den(x, dens)
    w0 = np.sum(w * (idx == 0), axis=1)
    w1 = np.sum(w * (idx == 1), axis=1)
    return w0, w1


def feats_oracle4(x, w, dens=DEN_POSITIONS):
    """[w_den0, w_den1, weighted spread den0, weighted spread den1].

    'Spread' is the weighted RMS distance to that den's own weighted mean,
    averaged over the two axes -- the label is simply argmin of the two.
    Not an encoder: a reference upper bound on whether the bit is in the
    belief at all.
    """
    idx, _ = _nearest_den(x, dens)
    out = []
    for k in (0, 1):
        m = (idx == k).astype(np.float64)
        wk = w * m
        mass = wk.sum(axis=1)
        wn = wk / np.clip(mass, 1e-12, None)[:, None]
        mean = np.sum(wn[..., None] * x, axis=1)
        d2 = np.sum((x - mean[:, None, :]) ** 2, axis=2)
        var = np.sum(wn * d2, axis=1) / 2.0
        std = np.sqrt(np.clip(var, 0.0, None))
        out.append(np.where(mass > 1e-8, std, 0.0))
    w0, w1 = den_weights(x, w, dens)
    return np.stack([w0, w1, out[0], out[1]], axis=1)


# ---------------------------------------------------------------------------
# invariance: the executable moment-matching proof
# ---------------------------------------------------------------------------

def _sample_disc(center, radius, n, rng):
    """n points uniform on the disc of the given center/radius."""
    theta = rng.uniform(0.0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.stack([center[0] + r * np.cos(theta),
                     center[1] + r * np.sin(theta)], axis=1)


def analytic_diagonal_cgf_gap(rho, den_dist=2.7, r_tight=DEN_R_TIGHT,
                              r_loose=DEN_R_LOOSE, arena_scale=ARENA_SCALE,
                              w0=0.5):
    """Population CGF gap between the two tight/loose assignments, for the
    projection t = rho * (1,1)/sqrt(2) along the den diagonal, in normalized
    (particles / arena_scale) coordinates.

    For a uniform disc of radius R the MGF of its projection on a unit
    direction is 2*I1(sR)/(sR), so with den weights (w0, 1-w0):

        CGF = log[ w0 * e^{-a} * phi(rho, r_den0) + w1 * e^{+a} * phi(rho, r_den1) ]

    with a = rho * den_dist * sqrt(2) / arena_scale. Swapping which den is
    tight swaps the two phi terms. This is the exact number the empirical
    N-particle test estimates, and the reason the achievable gap is bounded:
    the real extractor clamps t elementwise at t_clamp=2.0, so along the
    45-degree den diagonal |t| can never exceed 2*sqrt(2) = 2.83.
    """
    from scipy.special import iv

    def phi(s, R):
        x = s * R
        return 1.0 if x == 0 else float(2 * iv(1, x) / x)

    dn = den_dist / arena_scale
    rt = r_tight / arena_scale
    rl = r_loose / arena_scale
    a = rho * dn * np.sqrt(2.0)
    w1 = 1.0 - w0
    # tight_den = 0 -> tight disc at D0 (the -a side), loose at D1 (+a side)
    m_a = w0 * np.exp(-a) * phi(rho, rt) + w1 * np.exp(a) * phi(rho, rl)
    m_b = w0 * np.exp(-a) * phi(rho, rl) + w1 * np.exp(a) * phi(rho, rt)
    return float(np.log(m_a) - np.log(m_b))


def invariance(args):
    rng = np.random.default_rng(args.seed)
    n_half = args.n_particles // 2

    # tight_den = 0: tight disc at D0, loose disc at D1.
    tight = _sample_disc(DEN_POSITIONS[0], args.r_tight, n_half, rng)
    loose = _sample_disc(DEN_POSITIONS[1], args.r_loose, n_half, rng)
    X = np.concatenate([tight, loose], axis=0)          # [N, 2]
    # The tight/loose swap IS the point reflection x -> -x (D0 <-> D1).
    X_swap = -X

    w = np.full((1, X.shape[0]), 1.0 / X.shape[0])
    xs = (X / ARENA_SCALE)[None, ...]
    xs_swap = (X_swap / ARENA_SCALE)[None, ...]

    g = feats_gauss5(xs, w)[0]
    g_swap = feats_gauss5(xs_swap, w)[0]

    t_sp = t_spread()
    c = _cgf(xs, w, t_sp)[0]
    c_swap = _cgf(xs_swap, w, t_sp)[0]

    mean_norm = float(np.linalg.norm(g[:2]))
    mean_norm_swap = float(np.linalg.norm(g_swap[:2]))
    var_cov_absdiff = float(np.max(np.abs(g[2:] - g_swap[2:])))
    cgf_absdiff = float(np.max(np.abs(c - c_swap)))
    # Where the separation lives: the den diagonal is direction k=1 (45 deg).
    t_norms = np.linalg.norm(t_sp, axis=1)
    j = int(np.argmax(np.abs(c - c_swap)))

    print("=== Twin-Den moment-invariance test (executable form of the "
          "sec.1.3 claim) ===")
    print(f"N = {X.shape[0]} particles: {n_half} uniform on disc(D0, "
          f"{args.r_tight}) [tight] + {n_half} uniform on disc(D1, "
          f"{args.r_loose}) [loose], equal weights.")
    print(f"Swap = point reflection x -> -x (maps D0 <-> D1, tight <-> loose).")
    print("\n-- GAUSS5 (particles normalized by arena_scale=4.5)")
    print(f"  mean            : {np.round(g[:2], 6)}  ->  "
          f"{np.round(g_swap[:2], 6)}   (negates exactly; population value 0)")
    print(f"  ||mean||        : {mean_norm:.6e}  vs  {mean_norm_swap:.6e}  "
          f"(equal in magnitude; sampling-noise bound "
          f"{args.mean_bound:.4f})")
    print(f"  var_x,var_y,cov : {np.round(g[2:], 12)}")
    print(f"           swapped: {np.round(g_swap[2:], 12)}")
    print(f"  max |diff|      : {var_cov_absdiff:.3e}   (criterion: <= 1e-12)")
    analytic = abs(analytic_diagonal_cgf_gap(2.8, r_tight=args.r_tight,
                                             r_loose=args.r_loose))
    print("\n-- CGF_SPREAD64 (64 fixed projections, 8 dirs x 8 norms)")
    print(f"  max |diff|      : {cgf_absdiff:.4f}   (criterion: > "
          f"{args.cgf_min:.2f})")
    print(f"  analytic ref    : {analytic:.4f}  (exact population gap at "
          f"rho=2.8 along the den diagonal; see analytic_diagonal_cgf_gap)")
    print(f"  argmax at t = {np.round(t_sp[j], 3)} (|t|={t_norms[j]:.2f}, "
          f"angle={np.degrees(np.arctan2(t_sp[j, 1], t_sp[j, 0])):.0f} deg), "
          f"cgf {c[j]:.4f} vs {c_swap[j]:.4f}")
    print(f"  mean |diff|     : {np.mean(np.abs(c - c_swap)):.4f}")

    ok = True
    if var_cov_absdiff > 1e-12:
        print("\nFAIL: pooled variance/covariance is NOT invariant to the swap.")
        ok = False
    if not (abs(mean_norm - mean_norm_swap) < 1e-12):
        print("\nFAIL: ||mean|| differs between the two assignments.")
        ok = False
    if mean_norm >= args.mean_bound:
        print(f"\nFAIL: ||mean||={mean_norm:.4f} exceeds the sampling-noise "
              f"bound {args.mean_bound:.4f}; the population mean should be 0.")
        ok = False
    if cgf_absdiff <= args.cgf_min:
        print(f"\nFAIL: CGF_SPREAD64 does not separate the two assignments "
              f"(max |diff| = {cgf_absdiff:.4f} <= {args.cgf_min:.2f}).")
        ok = False
    if abs(cgf_absdiff - analytic) > 0.25 * analytic:
        print(f"\nFAIL: empirical CGF gap {cgf_absdiff:.4f} is far from the "
              f"analytic population value {analytic:.4f} -- the sampler or "
              f"the feature builder disagrees with the closed form.")
        ok = False
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} -- mean+covariance are "
          f"{'blind' if ok else 'NOT blind'} to the tight/loose swap while "
          f"the CGF projections {'see' if ok else 'do not see'} it.")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def build_probe_env(num_particles: int, visibility_radius: float = 3.0):
    """Exactly the training env stack, minus reward shaping and Monitor."""
    import importlib

    import gymnasium as gym
    import pdomains  # noqa: F401 - registers pdomains-ant-tag-dens-v0
    from set_transformer.rl.particle_filters.ant_tag import (
        TwinDenAntTagParticleFilter,
    )

    _train_rl_cgf = importlib.import_module("4_train_rl_cgf")
    PFDictWithWeightsObservationWrapper = (
        _train_rl_cgf.PFDictWithWeightsObservationWrapper)
    CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
    ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
    get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs

    env = gym.make("pdomains-ant-tag-dens-v0", rendering=False)
    env.reset(seed=0)
    pf_kwargs = get_ant_tag_pf_kwargs(env)
    env = CurriculumVisibilityWrapper(env,
                                      initial_visibility_radius=visibility_radius)
    env = PFDictWithWeightsObservationWrapper(
        env=env,
        particle_filter_class=TwinDenAntTagParticleFilter,
        particle_filter_kwargs=pf_kwargs,
        num_particles=num_particles,
        pf_interaction_mapper=ant_tag_pf_interaction_mapper,
        obs_mask_indices=[-2, -1],
    )
    return env


def collect(args):
    """Two drivers, mixed per episode:
      * mean-chaser (70%): walk toward the current weighted PF mean -- i.e.
        toward the empty midpoint between the dens, deliberately visiting the
        deceptive states -- with periodic random-waypoint exploration kicks.
      * den-visitor (30%): walk to one den center, then the other, so the
        dataset also contains partially- and fully-resolved beliefs.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = build_probe_env(args.num_particles)
    raw_env = env.unwrapped
    cage_max = float(raw_env.cage_max_x)
    dens = np.asarray(raw_env.den_positions, dtype=np.float64)

    # Phase-2 locomotion policy: walks toward whatever xy sits in obs[-2:].
    loco_vec = DummyVecEnv(
        [lambda: gym.make("pdomains-ant-tag-dens-v0", rendering=False)])
    loco_vec = VecNormalize.load(args.locomotion_vecnorm, loco_vec)
    loco_vec.training = False
    loco_vec.norm_reward = False
    model = PPO.load(args.locomotion_model, device="cpu")

    rng = np.random.default_rng(args.driver_seed)

    P, W, A, T, V, EP, TT, DT, DC = [], [], [], [], [], [], [], [], []
    n_visitor = 0

    for ep in range(args.n_episodes):
        env.reset(seed=2000 + ep)
        den_visitor = rng.random() < args.visitor_frac
        n_visitor += int(den_visitor)
        visit_order = [0, 1] if rng.random() < 0.5 else [1, 0]
        visit_idx = 0
        steps_on_den = 0
        random_wp = None
        random_wp_left = 0

        for t in range(args.max_steps):
            pf = env.particle_filter
            ant_now = np.asarray(raw_env.data.qpos[:2], dtype=np.float64)

            if den_visitor:
                waypoint = dens[visit_order[visit_idx]]
                steps_on_den += 1
                reached = np.linalg.norm(ant_now - waypoint) < 1.0
                if (reached or steps_on_den >= args.visitor_leg_steps) \
                        and visit_idx == 0:
                    visit_idx = 1
                    steps_on_den = 0
            else:
                pf_mean = np.average(pf.particles, weights=pf.weights, axis=0)
                # Exploration kick: every 30 steps, 30% chance of switching to
                # a uniform-random waypoint for the next 30 steps.
                if t % 30 == 0:
                    if rng.random() < 0.3:
                        random_wp = rng.uniform(-cage_max, cage_max, size=2)
                        random_wp_left = 30
                    else:
                        random_wp = None
                        random_wp_left = 0
                waypoint = random_wp if random_wp_left > 0 else pf_mean
                random_wp_left = max(0, random_wp_left - 1)

            raw_obs = np.concatenate(
                [raw_env.data.qpos, raw_env.data.qvel, waypoint]
            ).astype(np.float32)
            action, _ = model.predict(loco_vec.normalize_obs(raw_obs[None, :]),
                                      deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action[0])

            ant_pos = np.asarray(raw_env.data.qpos[:2], dtype=np.float32)
            true_target = np.asarray(raw_env.get_target_pos(), dtype=np.float32)
            visible = bool(np.linalg.norm(ant_pos - true_target) < 3.0)

            P.append(env.particle_filter.particles.astype(np.float32))
            W.append(env.particle_filter.weights.astype(np.float32))
            A.append(ant_pos)
            T.append(true_target)
            V.append(visible)
            EP.append(ep)
            TT.append(t)
            DT.append(int(info["den_tight"]))
            DC.append(int(info["den_committed"]))

            if terminated or truncated:
                break

        if (ep + 1) % 25 == 0:
            print(f"  ... {ep + 1}/{args.n_episodes} episodes, {len(P)} snapshots")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        particles=np.asarray(P, dtype=np.float32),
        weights=np.asarray(W, dtype=np.float32),
        ant_pos=np.asarray(A, dtype=np.float32),
        true_target=np.asarray(T, dtype=np.float32),
        visible=np.asarray(V, dtype=bool),
        episode=np.asarray(EP, dtype=np.int32),
        t=np.asarray(TT, dtype=np.int32),
        den_tight=np.asarray(DT, dtype=np.int32),
        den_committed=np.asarray(DC, dtype=np.int32),
        den_positions=dens.astype(np.float32),
        den_radius_tight=np.float32(raw_env.den_radius_tight),
        den_radius_loose=np.float32(raw_env.den_radius_loose),
        arena_scale=np.float32(cage_max),
    )
    print(f"Saved {len(P)} snapshots from {args.n_episodes} episodes "
          f"({n_visitor} den-visitor) to {args.out}")
    env.close()
    loco_vec.close()


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

IMBALANCE_BINS = {
    "bal[0,.05)": (0.0, 0.05),
    "mid[.05,.15)": (0.05, 0.15),
    "imb[.15,.35]": (0.15, 0.351),
    "overall": (0.0, 1.0),
}
_BIN_ORDER = ["overall", "bal[0,.05)", "mid[.05,.15)", "imb[.15,.35]"]


def probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.data)
    arena_scale = float(d["arena_scale"])
    dens = np.asarray(d["den_positions"], dtype=np.float64)
    r_tight = float(d["den_radius_tight"])
    r_loose = float(d["den_radius_loose"])

    particles_all = d["particles"].astype(np.float64)
    weights_all = _norm_weights(d["weights"].astype(np.float64))
    ant_all = d["ant_pos"].astype(np.float64)
    vis_all = d["visible"]
    den_tight_all = d["den_tight"].astype(int)

    n_total = len(vis_all)

    # ---------------- clean-stratum row selection (sec. 4) ----------------
    idx, dist = _nearest_den(particles_all, dens)          # [B, N]
    # per-particle leash radius, keyed by this row's tight/loose assignment
    r_per = np.where(idx == den_tight_all[:, None], r_tight, r_loose)
    settled_mass = np.sum(weights_all * (dist <= r_per + args.settle_slack), axis=1)
    w0 = np.sum(weights_all * (idx == 0), axis=1)
    w1 = np.sum(weights_all * (idx == 1), axis=1)

    ant_to_dens = np.linalg.norm(ant_all[:, None, :] - dens[None, :, :], axis=2)
    urgency_free = np.all(ant_to_dens > args.urgency_margin, axis=1)

    c_invis = ~vis_all
    c_settled = settled_mass >= args.settle_frac
    c_alive = (w0 >= args.min_den_weight) & (w1 >= args.min_den_weight)
    sel = c_invis & c_settled & c_alive & urgency_free

    print(f"Loaded {n_total} snapshots.")
    print("Clean-stratum filters (cumulative):")
    print(f"  not visible                        : {int(c_invis.sum())}")
    print(f"  + settled (>={args.settle_frac:.2f} weight within r_den+"
          f"{args.settle_slack}) : {int((c_invis & c_settled).sum())}")
    print(f"  + both dens alive (w >= {args.min_den_weight})      : "
          f"{int((c_invis & c_settled & c_alive).sum())}")
    print(f"  + ant > {args.urgency_margin} from BOTH den centers   : "
          f"{int(sel.sum())}")
    n_sel = int(sel.sum())
    if n_sel < 100:
        raise SystemExit("Too few selected snapshots to probe; collect more "
                         "episodes.")

    particles = particles_all[sel]
    weights = weights_all[sel]
    ant = ant_all[sel] / arena_scale
    groups = d["episode"][sel]
    y = den_tight_all[sel]
    imb = np.abs(w0[sel] - 0.5)

    xs = particles / arena_scale          # encoder-space particles
    print(f"\nLabel base rate (den_tight == 1): {y.mean():.3f}  "
          f"[{int(y.sum())} / {len(y) - int(y.sum())}]")
    print(f"Episodes represented: {len(np.unique(groups))}")
    bin_counts = {}
    for b in _BIN_ORDER:
        lo, hi = IMBALANCE_BINS[b]
        m = (imb >= lo) & (imb < hi)
        bin_counts[b] = int(m.sum())
    print("Rows per weight-imbalance bin |w0-0.5|: " +
          ", ".join(f"{b}={bin_counts[b]}" for b in _BIN_ORDER))

    # ---------------- feature sets ----------------
    t_init = t_init_linspace_all_dims(args.num_cgf_features, xs.shape[2],
                                      args.t_init_scale)
    t_init = np.clip(t_init, -args.t_clamp, args.t_clamp)
    t_sp = t_spread(t_clamp=args.t_clamp)

    feature_sets = {
        "GAUSS5": feats_gauss5(xs, weights),
        "CGF_INIT64": _cgf(xs, weights, t_init),
        "CGF_SPREAD64": _cgf(xs, weights, t_sp),
        "ORACLE4": feats_oracle4(particles, weights, dens),
    }
    # Every feature set also gets the ant position (the policy has it via obs).
    feature_sets = {k: np.concatenate([v, ant], axis=1)
                    for k, v in feature_sets.items()}

    results = {}
    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(gkf.split(xs, y, groups))

    for name, X in feature_sets.items():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        per_bin = {b: {"auc": [], "bacc": []} for b in IMBALANCE_BINS}
        for tr, te in splits:
            if len(np.unique(y[tr])) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            yhat = (p >= 0.5).astype(int)
            for b, (lo, hi) in IMBALANCE_BINS.items():
                m = (imb[te] >= lo) & (imb[te] < hi)
                if m.sum() < 10 or len(np.unique(y[te][m])) < 2:
                    continue
                per_bin[b]["auc"].append(roc_auc_score(y[te][m], p[m]))
                per_bin[b]["bacc"].append(
                    balanced_accuracy_score(y[te][m], yhat[m]))
        results[name] = per_bin

    print(f"\n=== Linear probe: 'which den is the loose one?' (label = "
          f"den_tight, GroupKFold k={args.n_splits} by episode) ===")
    print(f"{'feature set':<14}" + "".join(f"{b:>22}" for b in _BIN_ORDER))
    for metric in ("bacc", "auc"):
        print(f"-- {'balanced accuracy' if metric == 'bacc' else 'ROC-AUC'}")
        for name in feature_sets:
            row = f"{name:<14}"
            for b in _BIN_ORDER:
                v = results[name][b][metric]
                row += (f"{np.mean(v):>15.3f} +-{np.std(v):.3f}" if v
                        else f"{'n/a':>22}")
            print(row)

    # ---------------- pre-registered gate (plan sec. 6.1) ----------------
    def _bacc(name, b):
        v = results[name][b]["bacc"]
        return float(np.mean(v)) if v else float("nan")

    bal = "bal[0,.05)"
    print("\n=== Pre-registered gate (plan sec. 6.1) ===")
    oracle = _bacc("ORACLE4", "overall")
    n_bal = bin_counts[bal]
    sanity_ok = (oracle >= 0.90) and (n_sel >= 500) and (n_bal >= 150)
    print(f"belief sanity  : bacc(ORACLE4)={oracle:.3f} (>=0.90), rows="
          f"{n_sel} (>=500), balanced-bin rows={n_bal} (>=150)  -> "
          f"{'PASS' if sanity_ok else 'FAIL'}")

    g_bal = _bacc("GAUSS5", bal)
    c_bal = _bacc("CGF_SPREAD64", bal)
    g_all = _bacc("GAUSS5", "overall")
    c_all = _bacc("CGF_SPREAD64", "overall")
    gap = c_all - g_all
    confirms = (g_bal <= 0.58) and (c_bal >= 0.72) and (gap >= 0.10)
    null = g_bal > 0.65
    print(f"CONFIRMS       : balanced-bin GAUSS5={g_bal:.3f} (<=0.58) and "
          f"CGF_SPREAD64={c_bal:.3f} (>=0.72) and overall gap={gap:+.3f} "
          f"(>=0.10)  -> {'YES' if confirms else 'no'}")
    print(f"NULL           : balanced-bin GAUSS5={g_bal:.3f} (>0.65)  -> "
          f"{'YES' if null else 'no'}")
    print(f"secondary      : bacc(CGF_INIT64) overall="
          f"{_bacc('CGF_INIT64', 'overall'):.3f} (expected ~GAUSS5="
          f"{g_all:.3f}); GAUSS5 imbalance trend " +
          " -> ".join(f"{_bacc('GAUSS5', b):.3f}"
                      for b in _BIN_ORDER[1:]))
    if not sanity_ok:
        verdict = "BELIEF-SANITY-FAIL"
    elif confirms:
        verdict = "CONFIRMS"
    elif null:
        verdict = "NULL"
    else:
        verdict = "INCONCLUSIVE (neither CONFIRMS nor NULL thresholds met)"
    print(f"\nVERDICT: {verdict}")


# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("invariance",
                       help="executable moment-matching proof (no env needed)")
    i.add_argument("--n_particles", type=int, default=2000)
    i.add_argument("--r_tight", type=float, default=DEN_R_TIGHT)
    i.add_argument("--r_loose", type=float, default=DEN_R_LOOSE)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--mean_bound", type=float, default=None,
                   help="Sampling-noise bound on ||mean||. Default "
                        "4.5*3/sqrt(N) in normalized units, i.e. 3/sqrt(N).")
    # The plan pre-registered "> 0.1", but that threshold is not achievable
    # at the plan's OWN committed geometry + t_spread grid: the extractor
    # clamps t elementwise at 2.0, so along the 45-degree den diagonal the
    # largest realizable |t| is 2*sqrt(2) = 2.83, and the exact population
    # gap there is 0.084 nats (see analytic_diagonal_cgf_gap; the plan's
    # "0.2+ nats" would need rho ~ 4.4, which the clamp forbids). 0.05 is
    # comfortably above N=2000 sampling noise and below the analytic value,
    # and the test additionally checks the empirical value matches the
    # closed form -- which is the real correctness assertion.
    i.add_argument("--cgf_min", type=float, default=0.05)
    i.set_defaults(func=invariance)

    c = sub.add_parser("collect", help="gather belief snapshots from the dens env")
    c.add_argument("--n_episodes", type=int, default=300)
    c.add_argument("--max_steps", type=int, default=200)
    c.add_argument("--num_particles", type=int, default=100)
    c.add_argument("--visitor_frac", type=float, default=0.3,
                   help="Fraction of episodes driven by the den-visitor "
                        "driver instead of the mean-chaser.")
    c.add_argument("--visitor_leg_steps", type=int, default=90)
    c.add_argument("--locomotion_model", type=str,
                   default="models/ant_locomotion_policy.zip")
    c.add_argument("--locomotion_vecnorm", type=str,
                   default="models/locomotion_vecnorm.pkl")
    c.add_argument("--driver_seed", type=int, default=0)
    c.add_argument("--out", type=str, default="data/dens_probe_dataset.npz")
    c.set_defaults(func=collect)

    q = sub.add_parser("probe", help="run the linear separability probe")
    q.add_argument("--data", type=str, default="data/dens_probe_dataset.npz")
    q.add_argument("--num_cgf_features", type=int, default=64)
    q.add_argument("--t_init_scale", type=float, default=0.1)
    q.add_argument("--t_clamp", type=float, default=2.0)
    q.add_argument("--n_splits", type=int, default=5)
    q.add_argument("--settle_frac", type=float, default=0.80)
    q.add_argument("--settle_slack", type=float, default=0.30)
    q.add_argument("--min_den_weight", type=float, default=0.15)
    q.add_argument("--urgency_margin", type=float, default=3.0 + DEN_R_LOOSE,
                   help="Ant must be farther than this from BOTH den centers "
                        "(visible_radius + r_loose) so no den's particles are "
                        "urgency-deformed.")
    q.set_defaults(func=probe)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "invariance" and args.mean_bound is None:
        args.mean_bound = 3.0 / np.sqrt(args.n_particles)
    args.func(args)
