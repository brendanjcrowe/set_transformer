"""
Offline linear-probe separability test for Ghost Ant-Tag beliefs.

This is the cheap GATE in front of the two 6M-step PPO runs. Question:
after an unreliable long-range ping creates a second belief mode, can a
LINEAR readout of a given belief encoding recover the operationally
decisive bit -- "is the ping-mode the real target?" -- from the particle
filter's particles+weights alone?

Feature sets compared (all numpy mirrors of the real encoders, so the
probe measures encoder capacity, not policy quality):

  GAUSS5       weighted mean(2) + var(2) + cov_xy(1)   -- exact mirror of
               WeightedGaussianFeaturesExtractor.forward (4_train_rl_gaussian.py)
  CGF_INIT64   the 64 CGF log-sum-exp features at the exact training
               initialization (t_init_mode="linspace_all_dims",
               t_init_scale=0.1) -- what CGF can see BEFORE any learning
  CGF_SPREAD64 64 fixed projections on 8 directions x 8 norms spanning the
               reachable range under the elementwise t_clamp=2.0 -- what CGF
               could see IF its t_j spread out during training
  ORACLE3      [weight within 1.5 of the ping, weighted mean distance of
               those particles to the ping, steps_since_ping] -- NOT an
               encoder; a reference upper bound establishing whether the bit
               is present in the belief at all

Every feature set is concatenated with ant_pos/arena_scale. The raw ping
location is deliberately NOT an input to GAUSS5/CGF_*: neither policy ever
sees the ping directly, only its effect on the belief.

Usage:
    python3 diagnostics/probe_ghost_belief_separability.py collect --n_episodes 300 \
        --out data/ghost_probe_dataset.npz
    python3 diagnostics/probe_ghost_belief_separability.py probe --data data/ghost_probe_dataset.npz
"""

import argparse
import importlib
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

import gymnasium as gym
import numpy as np

import pdomains  # noqa: F401 - registers pdomains-ant-tag-ghost-v0
from set_transformer.rl.particle_filters.ant_tag import GhostAntTagParticleFilter

_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
PFDictWithWeightsObservationWrapper = _train_rl_cgf.PFDictWithWeightsObservationWrapper
CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def build_probe_env(num_particles: int, target_speed_scale: float,
                    visibility_radius: float = 3.0):
    """Exactly the training env stack, minus reward shaping and Monitor."""
    env = gym.make("pdomains-ant-tag-ghost-v0", rendering=False,
                   target_speed_scale=target_speed_scale)
    env.reset(seed=0)
    pf_kwargs = get_ant_tag_pf_kwargs(env)
    env = CurriculumVisibilityWrapper(env, initial_visibility_radius=visibility_radius)
    env = PFDictWithWeightsObservationWrapper(
        env=env,
        particle_filter_class=GhostAntTagParticleFilter,
        particle_filter_kwargs=pf_kwargs,
        num_particles=num_particles,
        pf_interaction_mapper=ant_tag_pf_interaction_mapper,
        obs_mask_indices=[-2, -1],
    )
    return env


def collect(args):
    """Mean-chaser driver: walk toward the current weighted PF mean, with
    periodic random-waypoint kicks for state-distribution coverage."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = build_probe_env(args.num_particles, args.target_speed_scale)
    raw_env = env.unwrapped
    cage_max = float(raw_env.cage_max_x)

    # Phase-2 locomotion policy: walks toward whatever xy sits in obs[-2:].
    # VecNormalize stats are loaded via a throwaway vec env purely so we can
    # call normalize_obs on hand-built observations (same trick as
    # gate_dumb_patrol_baseline.py).
    loco_vec = DummyVecEnv([lambda: gym.make("pdomains-ant-tag-ghost-v0", rendering=False)])
    loco_vec = VecNormalize.load(args.locomotion_vecnorm, loco_vec)
    loco_vec.training = False
    loco_vec.norm_reward = False
    model = PPO.load(args.locomotion_model, device="cpu")

    rng = np.random.default_rng(args.driver_seed)

    P, W, A, T, V, EP, TT, SSP, LPP, LPT = [], [], [], [], [], [], [], [], [], []

    for ep in range(args.n_episodes):
        env.reset(seed=1000 + ep)
        steps_since_ping = -1
        last_ping_pos = np.zeros(2, dtype=np.float32)
        last_ping_is_true = False
        random_wp = None
        random_wp_left = 0

        for t in range(args.max_steps):
            pf = env.particle_filter
            pf_mean = np.average(pf.particles, weights=pf.weights, axis=0)

            # Exploration kick: every 30 steps, 30% chance of switching to a
            # uniform-random waypoint for the next 30 steps.
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

            # Convention: steps_since_ping == 1 on the very step the ping was
            # received and consumed by the PF (so this snapshot IS the fresh
            # post-ping posterior), incrementing thereafter; -1 before any
            # ping. This makes the probe's 1<=ssp<=15 window include the
            # freshest, most informative belief.
            ping = info.get("ghost_ping")
            if ping is not None:
                steps_since_ping = 1
                last_ping_pos = np.asarray(ping, dtype=np.float32)
                last_ping_is_true = bool(info["ghost_ping_is_true"])
            elif steps_since_ping >= 0:
                steps_since_ping += 1

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
            SSP.append(steps_since_ping)
            LPP.append(last_ping_pos.copy())
            LPT.append(last_ping_is_true)

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
        steps_since_ping=np.asarray(SSP, dtype=np.int32),
        last_ping_pos=np.asarray(LPP, dtype=np.float32),
        last_ping_is_true=np.asarray(LPT, dtype=bool),
        arena_scale=np.float32(cage_max),
    )
    print(f"Saved {len(P)} snapshots from {args.n_episodes} episodes to {args.out}")
    env.close()
    loco_vec.close()


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
    """8 directions x 8 norms. The elementwise t_clamp the real extractor
    applies to its own parameters is applied here too, so this stays a true
    upper bound on what the trained encoder could actually realize (it only
    bites for the largest norms on near-axis-aligned directions)."""
    thetas = 2 * np.pi * np.arange(num_dirs) / num_dirs
    rhos = np.geomspace(rho_lo, rho_hi, num_norms)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)   # [K, 2]
    t = (rhos[:, None, None] * dirs[None, :, :]).reshape(-1, 2)
    return np.clip(t, -t_clamp, t_clamp)


def feats_oracle3(x, w, ping, ssp):
    d = np.linalg.norm(x - ping[:, None, :], axis=2)            # [B, N]
    near = d < 1.5
    mass = np.sum(w * near, axis=1)
    mean_d = np.where(
        mass > 1e-6,
        np.sum(w * near * d, axis=1) / np.clip(mass, 1e-8, None),
        1.5,  # no mass near the ping -> report the cap, not a spurious 0
    )
    return np.stack([mass, mean_d, ssp.astype(np.float64)], axis=1)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.data)
    arena_scale = float(d["arena_scale"])
    ssp_all = d["steps_since_ping"]
    vis_all = d["visible"]

    sel = (ssp_all >= 1) & (ssp_all <= 15) & (~vis_all)
    n_sel = int(sel.sum())
    print(f"Loaded {len(ssp_all)} snapshots; selected {n_sel} "
          f"(1<=steps_since_ping<=15 and not visible)")
    if n_sel < 200:
        raise SystemExit("Too few selected snapshots to probe; collect more episodes.")

    particles = d["particles"][sel].astype(np.float64)
    weights = _norm_weights(d["weights"][sel].astype(np.float64))
    ant = d["ant_pos"][sel].astype(np.float64) / arena_scale
    ping = d["last_ping_pos"][sel].astype(np.float64)
    ssp = ssp_all[sel]
    groups = d["episode"][sel]
    y = d["last_ping_is_true"][sel].astype(int)

    xs = particles / arena_scale          # encoder-space particles
    print(f"Label base rate (ping was true): {y.mean():.3f}  "
          f"[{y.sum()} true / {len(y) - y.sum()} false]")
    print(f"Episodes represented: {len(np.unique(groups))}")

    # ---------------- belief sanity ----------------
    ping_sigma = args.ping_sigma
    dist_to_ping = np.linalg.norm(particles - ping[:, None, :], axis=2)
    mass_2sig = np.sum(weights * (dist_to_ping < 2.0 * ping_sigma), axis=1)
    frac_bimodal = float(np.mean((mass_2sig >= 0.10) & (mass_2sig <= 0.80)))
    print("\n=== Belief sanity ===")
    print(f"ghost-mode weight (within 2*ping_sigma={2*ping_sigma:.1f} of ping):")
    print(f"  fraction in [0.10, 0.80]           : {frac_bimodal:.3f}   (criterion: >= 0.40)")
    for k in (1, 5, 10):
        m = ssp == k
        if m.any():
            print(f"  mean at steps_since_ping = {k:<2d}      : {mass_2sig[m].mean():.3f}  (n={m.sum()})")
    for k in (1, 5, 10):
        m = ssp == k
        if m.any():
            print(f"    ... split by label, ssp={k:<2d}: true-ping {mass_2sig[m & (y==1)].mean():.3f} "
                  f"/ false-ping {mass_2sig[m & (y==0)].mean():.3f}")

    # ---------------- feature sets ----------------
    t_init = t_init_linspace_all_dims(args.num_cgf_features, xs.shape[2], args.t_init_scale)
    t_init = np.clip(t_init, -args.t_clamp, args.t_clamp)
    t_sp = t_spread()

    feature_sets = {
        "GAUSS5": feats_gauss5(xs, weights),
        "CGF_INIT64": _cgf(xs, weights, t_init),
        "CGF_SPREAD64": _cgf(xs, weights, t_sp),
        "ORACLE3": feats_oracle3(particles, weights, ping, ssp),
    }
    # Every feature set also gets the ant position (the policy has it via obs).
    feature_sets = {k: np.concatenate([v, ant], axis=1) for k, v in feature_sets.items()}

    bins = {"1-3": (1, 3), "4-8": (4, 8), "9-15": (9, 15), "overall": (1, 15)}
    results = {}

    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(gkf.split(xs, y, groups))

    for name, X in feature_sets.items():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        per_bin = {b: {"auc": [], "bacc": []} for b in bins}
        for tr, te in splits:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
            if len(np.unique(y[tr])) < 2:
                continue
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            yhat = (p >= 0.5).astype(int)
            for b, (lo, hi) in bins.items():
                m = (ssp[te] >= lo) & (ssp[te] <= hi)
                if m.sum() < 10 or len(np.unique(y[te][m])) < 2:
                    continue
                per_bin[b]["auc"].append(roc_auc_score(y[te][m], p[m]))
                per_bin[b]["bacc"].append(balanced_accuracy_score(y[te][m], yhat[m]))
        results[name] = per_bin

    print(f"\n=== Linear probe: 'is the ping-mode the real target?' "
          f"(GroupKFold k={args.n_splits} by episode) ===")
    header = f"{'feature set':<14}" + "".join(f"{b:>22}" for b in
                                              ["overall", "1-3", "4-8", "9-15"])
    print(header)
    for metric in ("auc", "bacc"):
        print(f"-- {'ROC-AUC' if metric == 'auc' else 'balanced accuracy'}")
        for name in feature_sets:
            row = f"{name:<14}"
            for b in ["overall", "1-3", "4-8", "9-15"]:
                v = results[name][b][metric]
                row += (f"{np.mean(v):>15.3f} +-{np.std(v):.3f}" if v
                        else f"{'n/a':>22}")
            print(row)

    # ---------------- pre-registered gate (plan 5.1) ----------------
    auc = {n: float(np.mean(results[n]["overall"]["auc"])) for n in feature_sets}
    print("\n=== Pre-registered gate (plan 5.1) ===")
    sanity_ok = auc["ORACLE3"] >= 0.75 and frac_bimodal >= 0.40
    print(f"belief sanity  : AUC(ORACLE3)={auc['ORACLE3']:.3f} (>=0.75) and "
          f"frac_bimodal={frac_bimodal:.3f} (>=0.40)  -> "
          f"{'PASS' if sanity_ok else 'FAIL'}")
    gap = auc["CGF_SPREAD64"] - auc["GAUSS5"]
    confirms = gap >= 0.10 and auc["GAUSS5"] <= 0.65
    null = auc["GAUSS5"] > 0.70
    print(f"CONFIRMS       : AUC(CGF_SPREAD64)-AUC(GAUSS5)={gap:.3f} (>=0.10) and "
          f"AUC(GAUSS5)={auc['GAUSS5']:.3f} (<=0.65)  -> "
          f"{'YES' if confirms else 'no'}")
    print(f"NULL           : AUC(GAUSS5)={auc['GAUSS5']:.3f} (>0.70)  -> "
          f"{'YES' if null else 'no'}")
    print(f"secondary      : AUC(CGF_INIT64)={auc['CGF_INIT64']:.3f} "
          f"(expected ~AUC(GAUSS5)); INIT->SPREAD gain "
          f"{auc['CGF_SPREAD64'] - auc['CGF_INIT64']:+.3f}")
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="gather belief snapshots from the ghost env")
    c.add_argument("--n_episodes", type=int, default=300)
    c.add_argument("--max_steps", type=int, default=400)
    c.add_argument("--num_particles", type=int, default=100)
    c.add_argument("--target_speed_scale", type=float, default=0.0)
    c.add_argument("--locomotion_model", type=str,
                   default="models/ant_locomotion_policy.zip")
    c.add_argument("--locomotion_vecnorm", type=str,
                   default="models/locomotion_vecnorm.pkl")
    c.add_argument("--driver_seed", type=int, default=0)
    c.add_argument("--out", type=str, default="data/ghost_probe_dataset.npz")
    c.set_defaults(func=collect)

    q = sub.add_parser("probe", help="run the linear separability probe")
    q.add_argument("--data", type=str, default="data/ghost_probe_dataset.npz")
    q.add_argument("--num_cgf_features", type=int, default=64)
    q.add_argument("--t_init_scale", type=float, default=0.1)
    q.add_argument("--t_clamp", type=float, default=2.0)
    q.add_argument("--ping_sigma", type=float, default=0.8,
                   help="Must match the env/PF ping_sigma used during collect.")
    q.add_argument("--n_splits", type=int, default=5)
    q.set_defaults(func=probe)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
