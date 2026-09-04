"""
Offline linear-probe separability test for Counterweighted-Den Ant-Tag beliefs.

Successor to probe_dens_belief_separability.py (Twin-Den). The construction
changed: instead of a tight/loose RADIUS asymmetry, the mirror bit
`cden_heavy_side` selects which diagonal side hosts the HEAVY-NEAR den
(distance h, prior occupancy w) versus the LIGHT-FAR den (distance f, prior
occupancy 1-w), with

    w_heavy = f / (h + f)        =>   w*h = (1-w)*f

so the pooled belief mean is pinned at 0 BY CONSTRUCTION (an identity, not a
tuned number) and the covariance is invariant under the point reflection
x -> -x that IS the mirror-bit swap. Only ODD moments carry the bit.

Four subcommands:

  pregate     THE Rule-1 pre-implementation gate. Dependency-free (numpy +
              sklearn only; no repo imports, no MuJoCo). Simulates the
              intended belief distribution directly from the generative
              model and asks whether GAUSS5 can read the bit. Abort the
              whole plan if it can.
  invariance  executable form of the moment-matching proof; needs no env.
  collect     roll out the training-identical env stack and log belief
              snapshots.
  probe       cross-validated logistic-regression probe on those snapshots.

Feature sets compared (all numpy mirrors of the real encoders, so the
probe measures encoder capacity, not policy quality):

  GAUSS5        weighted mean(2) + var(2) + cov_xy(1) -- exact mirror of
                WeightedGaussianFeaturesExtractor.forward
  CGF_INIT64    the 64 CGF log-sum-exp features at the legacy training
                initialization (t_init_mode="linspace_all_dims",
                t_init_scale=0.1) -- arm (b)'s t=0 representation
  CGF_SPREAD64  8 directions x 8 log-spaced norms spanning the reachable
                range under the elementwise t_clamp=2.0 -- literally arm
                (c)/(d)'s t=0 representation
  ORACLE3       [mass near heavy candidate 1, near heavy candidate 2, near
                the two light candidates summed] -- NOT an encoder; a
                reference upper bound that the bit is in the belief at all.

Usage:
    python3 diagnostics/probe_cdens_belief_separability.py pregate
    python3 diagnostics/probe_cdens_belief_separability.py invariance
    python3 diagnostics/probe_cdens_belief_separability.py collect --n_episodes 300 \
        --out data/cdens_probe_dataset.npz
    python3 diagnostics/probe_cdens_belief_separability.py probe --data data/cdens_probe_dataset.npz
"""

import argparse
import os
import re
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

# --- committed geometry (plan sec. 1.2) ---
CDEN_H = 2.4
CDEN_F = 6.75
CDEN_R = 0.4
ARENA_SCALE = 7.0
W_HEAVY = CDEN_F / (CDEN_H + CDEN_F)          # 0.7377... (derived, not tuned)
DIAG = np.array([1.0, 1.0]) / np.sqrt(2.0)

# The four STATIC candidate centers: [-h, +h, -f, +f] along the diagonal.
CDEN_CANDIDATES = np.stack([s * d * DIAG
                            for d in (CDEN_H, CDEN_F)
                            for s in (-1.0, 1.0)])


def cden_centers(s: int):
    """(heavy_pos, light_pos) for mirror bit s in {0, 1}."""
    sign = -1.0 if s == 0 else 1.0
    return sign * CDEN_H * DIAG, -sign * CDEN_F * DIAG


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
    """8 directions x 8 log-spaced norms -- the exact geometry of the new
    t_init_mode='spread' (plan sec. 1.7). theta_k = 2*pi*k/8 includes the den
    diagonal (45 deg) exactly. The elementwise t_clamp the real extractor
    applies to its own parameters is applied here too, so this stays a true
    statement about what the trained encoder can realize."""
    thetas = 2 * np.pi * np.arange(num_dirs) / num_dirs
    rhos = np.geomspace(rho_lo, rho_hi, num_norms)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)   # [K, 2]
    t = (rhos[:, None, None] * dirs[None, :, :]).reshape(-1, 2)
    return np.clip(t, -t_clamp, t_clamp)


def _sample_disc(center, radius, n, rng):
    """n points uniform on the disc of the given center/radius."""
    theta = rng.uniform(0.0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.stack([center[0] + r * np.cos(theta),
                     center[1] + r * np.sin(theta)], axis=1)


def _nearest_candidate(x, cands=CDEN_CANDIDATES):
    """x: [B, N, 2] -> (candidate_idx [B, N], dist [B, N])."""
    d = np.linalg.norm(x[:, :, None, :] - cands[None, None, :, :], axis=3)
    return np.argmin(d, axis=2), np.min(d, axis=2)


def feats_st(x, w, checkpoint, source="pretrain", weight_channel=True,
              num_heads=4, batch=4096):
    """Run a REAL Set Transformer encoder over the beliefs and return its
    features, so the probe measures the ST arm's actual capacity.

    Unlike the numpy mirrors above, this loads trained weights. Two sources:

      "pretrain"  a 3_train_st.py checkpoint (PFSetTransformer state_dict).
                  Answers "can a linear readout of the PRETRAINED encoding
                  recover the bit?" -- i.e. is a frozen/finetuned RL arm
                  being handed a representation that contains the bit at all?
      "policy"    an SB3 .zip from 4_train_rl_st.py. Pulls the encoder out of
                  the saved policy, so a collapsed encoder can be MEASURED
                  rather than inferred from feat_std_mean.

    `x` must already be normalized the way the RL extractor normalizes
    (particles / arena_scale), because that is the input range the encoder
    was trained on.
    """
    import torch
    from set_transformer.models.pf_set_transformer import PFSetTransformer

    if source == "pretrain":
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = PFSetTransformer(
            num_particles=cfg.num_particles,
            dim_particles=cfg.dim_particles + (1 if cfg.weighted_particles else 0),
            num_encodings=cfg.num_encodings, dim_encoder=cfg.dim_encoder,
            num_inds=cfg.num_inds, dim_hidden=cfg.dim_hidden,
            num_heads=cfg.num_heads, ln=cfg.use_layer_norm,
            dim_output_particles=cfg.dim_particles)
        model.load_state_dict(ck["model_state_dict"])
        encoder = model.set_transformer
        weight_channel = bool(cfg.weighted_particles)
    elif source == "policy":
        from stable_baselines3.common.save_util import load_from_zip_file
        _, params, _ = load_from_zip_file(checkpoint, load_data=False,
                                          device="cpu")
        sd = params["policy"]
        # SetTransformerFeaturesExtractor holds the encoder as `self.encoder`;
        # 3_train_st.py's PFSetTransformer calls the same module
        # `set_transformer`. Accept either so both sources work.
        for prefix in ("features_extractor.encoder.",
                       "features_extractor.set_transformer."):
            enc_sd = {k[len(prefix):]: v for k, v in sd.items()
                      if k.startswith(prefix)}
            if enc_sd:
                break
        if not enc_sd:
            raise SystemExit(f"No encoder keys in {checkpoint}; is this an "
                             "ST-arm policy?")
        # Shapes recover the geometry, so no config file is needed.
        # ISAB's mab0 attends inducing points (queries) over the input set
        # (keys/values), so fc_q maps hidden->hidden and only fc_k/fc_v carry
        # the true per-particle input dim. Reading fc_q here yields dim_hidden
        # and silently builds the wrong encoder.
        dim_in = enc_sd["enc.0.mab0.fc_k.weight"].shape[1]
        from set_transformer.models.set_transformer import SetTransformer
        # The final decoder Linear index differs between builds (the RL arm's
        # SetTransformer has one fewer SAB than 3_train_st.py's), so find it
        # rather than hardcoding dec.N.
        dec_linears = sorted(
            int(k.split(".")[1]) for k in enc_sd
            if re.fullmatch(r"dec\.\d+\.weight", k))
        if not dec_linears:
            raise SystemExit(f"No dec.N.weight in {checkpoint}")
        n_out = enc_sd["dec.0.S"].shape[1]
        dim_out = enc_sd[f"dec.{dec_linears[-1]}.weight"].shape[0]
        num_heads_hint = num_heads
        dim_hidden = enc_sd["enc.0.mab0.fc_q.weight"].shape[0]
        num_inds = enc_sd["enc.0.I"].shape[1]
        # num_heads is not recoverable from shapes (MAB splits dim_hidden
        # across heads), so take it from the RL arm's default. A wrong value
        # would change the attention split silently, so assert the load is
        # strict below rather than tolerating a mismatch.
        encoder = SetTransformer(dim_in, num_outputs=n_out, dim_output=dim_out,
                                 num_inds=num_inds, dim_hidden=dim_hidden,
                                 num_heads=num_heads_hint, ln=True)
        encoder.load_state_dict(enc_sd)
        weight_channel = dim_in > x.shape[2]
    else:
        raise ValueError(source)

    encoder.eval()
    xs = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if weight_channel:
        # Identical sanitization to SetTransformerFeaturesExtractor.forward
        # and Trainer._model_input: clamp, renormalize, scale by N.
        wt = torch.from_numpy(np.asarray(w, dtype=np.float32))
        wt = torch.clamp(torch.nan_to_num(wt), min=0.0)
        wt = wt / (wt.sum(dim=1, keepdim=True) + 1e-8)
        xs = torch.cat([xs, (wt * wt.shape[1]).unsqueeze(-1)], dim=-1)

    out = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            out.append(encoder(xs[i:i + batch]).flatten(1).numpy())
    return np.concatenate(out, axis=0)


def feats_oracle3(x, w, slack=0.3):
    """[mass within r+slack of heavy candidate 0, of heavy candidate 1,
    of the two light candidates summed].

    Not an encoder: a reference upper bound on whether the bit is present in
    the belief at all. Candidate order is CDEN_CANDIDATES = [-h, +h, -f, +f],
    so indices 0/1 are the heavy candidates and 2/3 the light ones.
    """
    idx, dist = _nearest_candidate(x)
    near = dist <= (CDEN_R + slack)
    m0 = np.sum(w * near * (idx == 0), axis=1)
    m1 = np.sum(w * near * (idx == 1), axis=1)
    ml = np.sum(w * near * ((idx == 2) | (idx == 3)), axis=1)
    return np.stack([m0, m1, ml], axis=1)


# ---------------------------------------------------------------------------
# pregate: the Rule-1 pre-implementation gate (plan sec. 4.0)
# ---------------------------------------------------------------------------

def _sample_synthetic_beliefs(s, M, n_particles, w_heavy, rng):
    """M synthetic belief snapshots for mirror bit s, drawn directly from the
    intended generative model: n_heavy ~ Binomial(n_particles, w_heavy)
    particles uniform in the heavy disc, the rest uniform in the light disc,
    uniform weights. Returns (X [M, N, 2] normalized, W [M, N], w_hat [M])."""
    heavy, light = cden_centers(s)
    n_heavy = rng.binomial(n_particles, w_heavy, size=M)
    X = np.empty((M, n_particles, 2), dtype=np.float64)
    for m in range(M):
        nh = int(n_heavy[m])
        if nh > 0:
            X[m, :nh] = _sample_disc(heavy, CDEN_R, nh, rng)
        if nh < n_particles:
            X[m, nh:] = _sample_disc(light, CDEN_R, n_particles - nh, rng)
        # particle ORDER must not carry the label (the encoders are
        # permutation invariant, but a linear probe on raw order would not be)
        rng.shuffle(X[m])
    W = np.full((M, n_particles), 1.0 / n_particles)
    return X / ARENA_SCALE, W, n_heavy / float(n_particles)


def _cv_bacc_auc(X, y, n_splits, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    baccs, aucs = [], []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        baccs.append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(baccs)), float(np.std(baccs)), float(np.mean(aucs))


def _smd(F, y):
    """Per-feature standardized mean difference between the two classes."""
    a, b = F[y == 0], F[y == 1]
    pooled = np.sqrt(0.5 * (a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)))
    return (b.mean(axis=0) - a.mean(axis=0)) / np.clip(pooled, 1e-12, None)


def _pregate_one(w_heavy, args, rng, label):
    X0, W0, wh0 = _sample_synthetic_beliefs(0, args.M, args.n_particles,
                                            w_heavy, rng)
    X1, W1, wh1 = _sample_synthetic_beliefs(1, args.M, args.n_particles,
                                            w_heavy, rng)
    X = np.concatenate([X0, X1], axis=0)
    W = np.concatenate([W0, W1], axis=0)
    y = np.concatenate([np.zeros(args.M, int), np.ones(args.M, int)])

    G = feats_gauss5(X, W)
    C = _cgf(X, W, t_spread())
    g_bacc, g_std, g_auc = _cv_bacc_auc(G, y, args.n_splits, args.seed)
    c_bacc, c_std, c_auc = _cv_bacc_auc(C, y, args.n_splits, args.seed)
    smd = _smd(G, y)

    names = ["mean_x", "mean_y", "var_x", "var_y", "cov_xy"]
    print(f"\n--- {label} (w_heavy = {w_heavy:.4f}) ---")
    print(f"  GAUSS5       : bacc {g_bacc:.4f} +-{g_std:.4f}   auc {g_auc:.4f}")
    print(f"  CGF_SPREAD64 : bacc {c_bacc:.4f} +-{c_std:.4f}   auc {c_auc:.4f}")
    print("  GAUSS5 per-feature standardized mean difference (|SMD|):")
    for n, v in zip(names, smd):
        print(f"      {n:<8}: {v:+.5f}")
    print(f"  max |SMD|    : {np.max(np.abs(smd)):.5f}")
    return g_bacc, c_bacc, float(np.max(np.abs(smd)))


def pregate(args):
    rng = np.random.default_rng(args.seed)
    heavy0, light0 = cden_centers(0)
    heavy1, light1 = cden_centers(1)

    print("=== Counterweighted-Den PREGATE (plan sec. 4.0) ===")
    print("Dependency-free simulation of the intended belief distribution.")
    print(f"  h = {CDEN_H}, f = {CDEN_F}, r = {CDEN_R}, arena_scale = "
          f"{ARENA_SCALE}")
    print(f"  w_heavy = f/(h+f) = {W_HEAVY:.6f}   "
          f"mean-pinning residual w*h-(1-w)*f = "
          f"{W_HEAVY * CDEN_H - (1 - W_HEAVY) * CDEN_F:+.3e}")
    print(f"  s=0: heavy {np.round(heavy0, 4)}  light {np.round(light0, 4)}")
    print(f"  s=1: heavy {np.round(heavy1, 4)}  light {np.round(light1, 4)}")
    print(f"  M = {args.M} beliefs per class, N = {args.n_particles} "
          f"particles, {args.n_splits}-fold stratified CV")

    g_bacc, c_bacc, max_smd = _pregate_one(W_HEAVY, args, rng,
                                           "MAIN (exact mean-pinned w)")

    ok_g = g_bacc <= args.abort_bacc
    ok_smd = max_smd <= args.abort_smd
    ok_c = c_bacc >= args.min_cgf_bacc

    print("\n--- informative (non-gating): covariance-leak curve ---")
    print("    what carving/silence-driven weight shifts will later produce")
    for dw in (-args.leak_dw, args.leak_dw):
        _pregate_one(W_HEAVY + dw, args, rng, f"perturbed w {dw:+.2f}")

    print("\n=== PREGATE verdict (plan sec. 4.0 / 6) ===")
    print(f"  GAUSS5 bacc       {g_bacc:.4f}  <= {args.abort_bacc}   -> "
          f"{'PASS' if ok_g else 'ABORT'}")
    print(f"  max |SMD|         {max_smd:.4f}  <= {args.abort_smd}    -> "
          f"{'PASS' if ok_smd else 'ABORT'}")
    print(f"  CGF_SPREAD64 bacc {c_bacc:.4f}  >= {args.min_cgf_bacc}   -> "
          f"{'PASS' if ok_c else 'FAIL'}")
    if ok_g and ok_smd and ok_c:
        print("\nRESULT: GREENLIGHT -- the mirror bit is invisible to the "
              "moment map and readable by the CGF projections. Proceed to "
              "Step 1.")
        return
    if not (ok_g and ok_smd):
        print("\nRESULT: ABORT -- the design itself leaks the bit into the "
              "moment map (the Ghost-Ping failure mode). Fix the geometry on "
              "paper before writing any repo code.")
    else:
        print("\nRESULT: FAIL -- the moment map is clean but the CGF "
              "projections cannot read the bit either; there is no signal to "
              "learn.")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# invariance: the executable exact proof (plan sec. 4.1)
# ---------------------------------------------------------------------------

def analytic_diagonal_cgf_gap(rho, h=CDEN_H, f=CDEN_F, w=W_HEAVY,
                              arena_scale=ARENA_SCALE):
    """Population CGF gap between the two mirror-bit configurations for the
    projection t = rho * (1,1)/sqrt(2) along the den diagonal, in normalized
    (particles / arena_scale) coordinates.

    Both dens share the SAME radius r, so the uniform-disc form factor
    2*I1(rho*r)/(rho*r) is identical for both terms and cancels EXACTLY in
    the difference. With a = rho*h/A and b = rho*f/A:

        CGF(s=0) = log[ w e^{-a} + (1-w) e^{+b} ] + log phi
        CGF(s=1) = log[ w e^{+a} + (1-w) e^{-b} ] + log phi

    Unlike Twin-Den (whose signal was a fine WIDTH difference bounded to
    ~0.08 nats at the clamp), here the signal is a coarse support-extent
    difference and the achievable gap is an order of magnitude larger.
    """
    a = rho * h / arena_scale
    b = rho * f / arena_scale
    return float(np.log(w * np.exp(-a) + (1 - w) * np.exp(b))
                 - np.log(w * np.exp(a) + (1 - w) * np.exp(-b)))


def invariance(args):
    rng = np.random.default_rng(args.seed)
    n = args.n_particles
    # Exact-w split, so the empirical weight imbalance is zero and the ONLY
    # thing distinguishing the two configurations is the reflection itself.
    n_heavy = int(round(W_HEAVY * n))
    heavy, light = cden_centers(0)
    X = np.concatenate([_sample_disc(heavy, CDEN_R, n_heavy, rng),
                        _sample_disc(light, CDEN_R, n - n_heavy, rng)], axis=0)
    # The mirror-bit swap IS the point reflection x -> -x (heavy <-> light
    # sides swap together with their weights, which is why the mean stays 0).
    X_swap = -X

    w = np.full((1, n), 1.0 / n)
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
    t_norms = np.linalg.norm(t_sp, axis=1)
    j = int(np.argmax(np.abs(c - c_swap)))

    print("=== Counterweighted-Den moment-invariance test (executable form "
          "of the sec. 1.6 claim) ===")
    print(f"N = {n} particles: {n_heavy} uniform on disc(heavy, {CDEN_R}) at "
          f"{np.round(heavy, 4)} + {n - n_heavy} uniform on disc(light, "
          f"{CDEN_R}) at {np.round(light, 4)}, equal weights.")
    print(f"Exact-w split: {n_heavy}/{n} = {n_heavy / n:.6f} vs w_heavy = "
          f"{W_HEAVY:.6f}")
    print("Swap = point reflection x -> -x (heavy/light sides exchange).")
    print(f"\n-- GAUSS5 (particles normalized by arena_scale={ARENA_SCALE})")
    print(f"  mean            : {np.round(g[:2], 6)}  ->  "
          f"{np.round(g_swap[:2], 6)}   (negates exactly; population value 0)")
    print(f"  ||mean||        : {mean_norm:.6e}  vs  {mean_norm_swap:.6e}  "
          f"(equal in magnitude; sampling-noise bound {args.mean_bound:.4f})")
    print(f"  var_x,var_y,cov : {np.round(g[2:], 14)}")
    print(f"           swapped: {np.round(g_swap[2:], 14)}")
    print(f"  max |diff|      : {var_cov_absdiff:.3e}   (criterion: <= 1e-12)")
    analytic = abs(analytic_diagonal_cgf_gap(2.8))
    print("\n-- CGF_SPREAD64 (64 fixed projections, 8 dirs x 8 norms)")
    print(f"  max |diff|      : {cgf_absdiff:.4f}   (criterion: > "
          f"{args.cgf_min:.2f})")
    print(f"  analytic ref    : {analytic:.4f}  (exact population gap at "
          f"rho=2.8 along the den diagonal)")
    print(f"  argmax at t = {np.round(t_sp[j], 3)} (|t|={t_norms[j]:.2f}, "
          f"angle={np.degrees(np.arctan2(t_sp[j, 1], t_sp[j, 0])):.0f} deg), "
          f"cgf {c[j]:.4f} vs {c_swap[j]:.4f}")
    print(f"  mean |diff|     : {np.mean(np.abs(c - c_swap)):.4f}")

    ok = True
    if var_cov_absdiff > 1e-12:
        print("\nFAIL: pooled variance/covariance is NOT invariant to the swap.")
        ok = False
    if abs(mean_norm - mean_norm_swap) > 1e-12:
        print("\nFAIL: ||mean|| differs between the two configurations.")
        ok = False
    if mean_norm >= args.mean_bound:
        print(f"\nFAIL: ||mean||={mean_norm:.4f} exceeds the sampling-noise "
              f"bound {args.mean_bound:.4f}; the population mean should be 0.")
        ok = False
    if cgf_absdiff <= args.cgf_min:
        print(f"\nFAIL: CGF_SPREAD64 does not separate the two configurations "
              f"(max |diff| = {cgf_absdiff:.4f} <= {args.cgf_min:.2f}).")
        ok = False
    if abs(cgf_absdiff - analytic) > 0.25 * analytic:
        print(f"\nFAIL: empirical CGF gap {cgf_absdiff:.4f} is far from the "
              f"analytic population value {analytic:.4f} -- the sampler or "
              f"the feature builder disagrees with the closed form.")
        ok = False
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} -- mean+covariance are "
          f"{'blind' if ok else 'NOT blind'} to the mirror-bit swap while "
          f"the CGF projections {'see' if ok else 'do not see'} it.")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# collect (plan sec. 4.2)
# ---------------------------------------------------------------------------

def build_probe_env(num_particles: int, env_id: str,
                    visibility_radius: float | None = None):
    """Exactly the training env stack, minus reward shaping and Monitor."""
    import importlib

    import gymnasium as gym
    import pdomains  # noqa: F401 - registers pdomains-ant-tag-cdens-v0
    from set_transformer.rl.particle_filters.ant_tag import (
        CounterweightedDenAntTagParticleFilter,
    )

    _train_rl_cgf = importlib.import_module("4_train_rl_cgf")
    PFDictWithWeightsObservationWrapper = (
        _train_rl_cgf.PFDictWithWeightsObservationWrapper)
    CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
    ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
    get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs

    env = gym.make(env_id, rendering=False)
    env.reset(seed=0)
    pf_kwargs = get_ant_tag_pf_kwargs(env)
    if visibility_radius is None:
        visibility_radius = float(env.unwrapped.visible_radius)
    env = CurriculumVisibilityWrapper(
        env, initial_visibility_radius=visibility_radius)
    env = PFDictWithWeightsObservationWrapper(
        env=env,
        particle_filter_class=CounterweightedDenAntTagParticleFilter,
        particle_filter_kwargs=pf_kwargs,
        num_particles=num_particles,
        pf_interaction_mapper=ant_tag_pf_interaction_mapper,
        obs_mask_indices=[-2, -1],
    )
    return env


def collect(args):
    """Two drivers, mixed per episode:
      * mean-chaser (70%): walk toward the current weighted PF mean -- which,
        because the mean is PINNED AT ZERO by the counterweight, is the empty
        midfield: deliberately the most deceptive place to stand.
      * den-visitor (30%): walk to one den center, then the other, so the
        dataset also contains partially- and fully-resolved beliefs.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = build_probe_env(args.num_particles, args.env_id)
    raw_env = env.unwrapped
    cage_max = float(raw_env.cage_max_x)
    vis_radius = float(raw_env.visible_radius)

    loco_vec = DummyVecEnv(
        [lambda: gym.make(args.env_id, rendering=False)])
    loco_vec = VecNormalize.load(args.locomotion_vecnorm, loco_vec)
    loco_vec.training = False
    loco_vec.norm_reward = False
    model = PPO.load(args.locomotion_model, device="cpu")

    rng = np.random.default_rng(args.driver_seed)

    P, W, A, T, V, EP, TT = [], [], [], [], [], [], []
    HS, OCC, SPK, HP, LP = [], [], [], [], []
    n_visitor = 0

    for ep in range(args.n_episodes):
        env.reset(seed=args.env_seed + ep)
        den_visitor = rng.random() < args.visitor_frac
        n_visitor += int(den_visitor)
        # The visitor deliberately does NOT know which den is heavy.
        first_heavy = rng.random() < 0.5
        visit_idx = 0
        steps_on_den = 0
        random_wp = None
        random_wp_left = 0

        for t in range(args.max_steps):
            pf = env.particle_filter
            ant_now = np.asarray(raw_env.data.qpos[:2], dtype=np.float64)
            heavy = np.asarray(raw_env.cden_heavy_pos, dtype=np.float64)
            light = np.asarray(raw_env.cden_light_pos, dtype=np.float64)
            order = [heavy, light] if first_heavy else [light, heavy]

            if den_visitor:
                waypoint = order[visit_idx]
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
            visible = bool(np.linalg.norm(ant_pos - true_target) < vis_radius)

            P.append(env.particle_filter.particles.astype(np.float32))
            W.append(env.particle_filter.weights.astype(np.float32))
            A.append(ant_pos)
            T.append(true_target)
            V.append(visible)
            EP.append(ep)
            TT.append(t)
            HS.append(int(info["cden_heavy_side"]))
            OCC.append(1 if info["cden_occupied"] == "heavy" else 0)
            SPK.append(bool(info["cden_spooked"]))
            HP.append(np.asarray(raw_env.cden_heavy_pos, dtype=np.float32))
            LP.append(np.asarray(raw_env.cden_light_pos, dtype=np.float32))

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
        cden_heavy_side=np.asarray(HS, dtype=np.int32),
        cden_occupied_heavy=np.asarray(OCC, dtype=np.int32),
        cden_spooked=np.asarray(SPK, dtype=bool),
        heavy_pos=np.asarray(HP, dtype=np.float32),
        light_pos=np.asarray(LP, dtype=np.float32),
        cden_h=np.float32(raw_env.cden_h),
        cden_f=np.float32(raw_env.cden_f),
        cden_r=np.float32(raw_env.cden_r),
        cden_w_heavy=np.float32(raw_env.cden_w_heavy),
        spook_radius=np.float32(raw_env.cden_spook_radius),
        arena_scale=np.float32(cage_max),
    )
    print(f"Saved {len(P)} snapshots from {args.n_episodes} episodes "
          f"({n_visitor} den-visitor) to {args.out}")
    env.close()
    loco_vec.close()


# ---------------------------------------------------------------------------
# probe (plan sec. 4.3)
# ---------------------------------------------------------------------------

_STRATA = ["clean", "leak[.05,.15)", "leak>=.15", "overall"]


def probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.data)
    arena_scale = float(d["arena_scale"])
    w_heavy = float(d["cden_w_heavy"])
    r = float(d["cden_r"])
    spook_radius = float(d["spook_radius"])

    particles_all = d["particles"].astype(np.float64)
    weights_all = _norm_weights(d["weights"].astype(np.float64))
    ant_all = d["ant_pos"].astype(np.float64)
    vis_all = d["visible"]
    spooked_all = d["cden_spooked"]
    y_all = d["cden_heavy_side"].astype(int)
    heavy_all = d["heavy_pos"].astype(np.float64)
    light_all = d["light_pos"].astype(np.float64)

    n_total = len(vis_all)

    # -- empirical heavy mass w_hat: belief mass nearest the ACTUAL heavy den
    d_h = np.linalg.norm(particles_all - heavy_all[:, None, :], axis=2)
    d_l = np.linalg.norm(particles_all - light_all[:, None, :], axis=2)
    w_hat = np.sum(weights_all * (d_h <= d_l), axis=1)
    dw = np.abs(w_hat - w_heavy)

    ant_to_h = np.linalg.norm(ant_all - heavy_all, axis=1)
    ant_to_l = np.linalg.norm(ant_all - light_all, axis=1)
    far_from_dens = (np.minimum(ant_to_h, ant_to_l)
                     >= spook_radius + args.den_margin)

    # -- clean stratum (plan sec. 4.3) --
    c_invis = ~vis_all
    c_noalarm = ~spooked_all
    c_far = far_from_dens
    base = c_invis & c_noalarm & c_far
    sel = base & (dw < args.clean_dw)

    print(f"Loaded {n_total} snapshots.")
    print("Clean-stratum filters (cumulative):")
    print(f"  not visible                         : {int(c_invis.sum())}")
    print(f"  + no alarm yet                      : "
          f"{int((c_invis & c_noalarm).sum())}")
    print(f"  + ant >= spook_radius+{args.den_margin} from both dens : "
          f"{int(base.sum())}")
    print(f"  + |w_hat - w| < {args.clean_dw}                 : "
          f"{int(sel.sum())}")
    n_sel = int(sel.sum())
    if n_sel < 100:
        raise SystemExit("Too few clean snapshots to probe; collect more "
                         "episodes.")

    # every row that passes the base filters, stratified by weight leak
    rows = base
    particles = particles_all[rows]
    weights = weights_all[rows]
    ant = ant_all[rows] / arena_scale
    groups = d["episode"][rows]
    y = y_all[rows]
    dw_r = dw[rows]
    heavy_r = heavy_all[rows]
    light_r = light_all[rows]

    strat = np.full(len(y), "leak>=.15", dtype=object)
    strat[dw_r < 0.15] = "leak[.05,.15)"
    strat[dw_r < args.clean_dw] = "clean"

    xs = particles / arena_scale
    print(f"\nLabel base rate (cden_heavy_side == 1): {y.mean():.3f}  "
          f"[{int(y.sum())} / {len(y) - int(y.sum())}]")
    print(f"Episodes represented: {len(np.unique(groups))}")
    counts = {s: int((strat == s).sum()) for s in _STRATA[:-1]}
    counts["overall"] = len(y)
    print("Rows per stratum: " + ", ".join(f"{s}={counts[s]}" for s in _STRATA))

    # -- feature sets --
    t_init = t_init_linspace_all_dims(args.num_cgf_features, xs.shape[2],
                                      args.t_init_scale)
    t_init = np.clip(t_init, -args.t_clamp, args.t_clamp)
    t_sp = t_spread(t_clamp=args.t_clamp)

    feature_sets = {
        "GAUSS5": feats_gauss5(xs, weights),
        "CGF_INIT64": _cgf(xs, weights, t_init),
        "CGF_SPREAD64": _cgf(xs, weights, t_sp),
        "ORACLE3": feats_oracle3(particles, weights),
    }
    if args.st_checkpoint:
        feature_sets["ST64"] = feats_st(
            xs, weights, args.st_checkpoint, source=args.st_source)
        print(f"  ST64 from {args.st_source}: {args.st_checkpoint}")
    # NOTE (differs from the Twin-Den probe, deliberately): ant_pos is NOT
    # concatenated by default here.
    #
    # In Twin-Den the den POSITIONS were static constants (+-2.7, +-2.7) and
    # only the tight/loose LABEL varied, so the ant's own position could not
    # encode the label and the concat was harmless. In this construction the
    # den positions MOVE with the mirror bit (heavy at +-h, light at -+f), so
    # any driver that navigates toward a den necessarily writes the bit into
    # its own coordinates. Including ant_pos therefore credits every encoder
    # with information the DRIVER injected rather than information the encoder
    # extracted -- measured here at +0.09 balanced accuracy for GAUSS5
    # (0.616 -> 0.701), with ant_pos alone scoring 0.603.
    # --include_ant restores the old behavior for comparison.
    if args.include_ant:
        feature_sets = {k: np.concatenate([v, ant], axis=1)
                        for k, v in feature_sets.items()}
        feature_sets["ANT_ONLY"] = ant

    results = {}
    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(gkf.split(xs, y, groups))

    for name, X in feature_sets.items():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        per = {s: {"auc": [], "bacc": []} for s in _STRATA}
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
            for s in _STRATA:
                m = (np.ones(len(te), bool) if s == "overall"
                     else strat[te] == s)
                if m.sum() < 10 or len(np.unique(y[te][m])) < 2:
                    continue
                per[s]["auc"].append(roc_auc_score(y[te][m], p[m]))
                per[s]["bacc"].append(
                    balanced_accuracy_score(y[te][m], yhat[m]))
        results[name] = per

    print(f"\n=== Linear probe: 'which side hosts the HEAVY den?' (label = "
          f"cden_heavy_side, GroupKFold k={args.n_splits} by episode) ===")
    print(f"{'feature set':<14}" + "".join(f"{s:>22}" for s in _STRATA))
    for metric in ("bacc", "auc"):
        print(f"-- {'balanced accuracy' if metric == 'bacc' else 'ROC-AUC'}")
        for name in feature_sets:
            row = f"{name:<14}"
            for s in _STRATA:
                v = results[name][s][metric]
                row += (f"{np.mean(v):>15.3f} +-{np.std(v):.3f}" if v
                        else f"{'n/a':>22}")
            print(row)

    # -- pre-registered gate (plan sec. 6) --
    def _b(name, s):
        v = results[name][s]["bacc"]
        return float(np.mean(v)) if v else float("nan")

    print("\n=== Pre-registered gate (plan sec. 6) ===")
    oracle = _b("ORACLE3", "clean")
    n_clean = counts["clean"]
    sanity_ok = (oracle >= 0.90) and (n_clean >= 500)
    print(f"belief sanity  : bacc(ORACLE3|clean)={oracle:.3f} (>=0.90), "
          f"clean rows={n_clean} (>=500)  -> "
          f"{'PASS' if sanity_ok else 'FAIL'}")

    g = _b("GAUSS5", "clean")
    c = _b("CGF_SPREAD64", "clean")
    gap = c - g
    confirms = (g <= 0.58) and (c >= 0.72) and (gap >= 0.10)
    bug = g > 0.65
    print(f"CONFIRMS       : clean GAUSS5={g:.3f} (<=0.58) and "
          f"CGF_SPREAD64={c:.3f} (>=0.72) and gap={gap:+.3f} (>=0.10)  -> "
          f"{'YES' if confirms else 'no'}")
    bug_msg = ("YES -- the pregate says this cannot be fundamental; fix the "
               "construction and re-probe once" if bug else "no")
    print(f"CONSTRUCTION-BUG: clean GAUSS5={g:.3f} (>0.65)  -> {bug_msg}")
    print(f"secondary      : CGF_INIT64|clean={_b('CGF_INIT64', 'clean'):.3f} "
          f"(expected ~GAUSS5); GAUSS5 leak trend " +
          " -> ".join(f"{_b('GAUSS5', s):.3f}" for s in _STRATA[:-1]) +
          "  (must RISE: the weight leak is genuine Bayesian information)")
    if not sanity_ok:
        verdict = "BELIEF-SANITY-FAIL"
    elif confirms:
        verdict = "CONFIRMS"
    elif bug:
        verdict = "CONSTRUCTION-BUG"
    else:
        verdict = "INCONCLUSIVE (neither CONFIRMS nor bug thresholds met)"
    print(f"\nVERDICT: {verdict}")


# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("pregate",
                        help="dependency-free pre-implementation gate")
    pg.add_argument("--M", type=int, default=1000,
                    help="synthetic beliefs per class")
    pg.add_argument("--n_particles", type=int, default=100)
    pg.add_argument("--n_splits", type=int, default=5)
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--abort_bacc", type=float, default=0.55)
    pg.add_argument("--abort_smd", type=float, default=0.2)
    pg.add_argument("--min_cgf_bacc", type=float, default=0.90)
    pg.add_argument("--leak_dw", type=float, default=0.10)
    pg.set_defaults(func=pregate)

    i = sub.add_parser("invariance",
                       help="executable moment-matching proof (no env needed)")
    i.add_argument("--n_particles", type=int, default=2000)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--mean_bound", type=float, default=None,
                   help="Sampling-noise bound on ||mean|| in normalized "
                        "units. Default 3/sqrt(N).")
    i.add_argument("--cgf_min", type=float, default=0.1)
    i.set_defaults(func=invariance)

    c = sub.add_parser("collect",
                       help="gather belief snapshots from the cdens env")
    c.add_argument("--env_id", type=str, default="pdomains-ant-tag-cdens-v0")
    c.add_argument("--n_episodes", type=int, default=300)
    c.add_argument("--max_steps", type=int, default=300)
    c.add_argument("--num_particles", type=int, default=100)
    c.add_argument("--visitor_frac", type=float, default=0.3,
                   help="Fraction of episodes driven by the den-visitor "
                        "driver instead of the mean-chaser.")
    c.add_argument("--visitor_leg_steps", type=int, default=130)
    c.add_argument("--locomotion_model", type=str,
                   default="models/ant_locomotion_policy.zip")
    c.add_argument("--locomotion_vecnorm", type=str,
                   default="models/locomotion_vecnorm.pkl")
    c.add_argument("--env_seed", type=int, default=3000)
    c.add_argument("--driver_seed", type=int, default=0)
    c.add_argument("--out", type=str, default="data/cdens_probe_dataset.npz")
    c.set_defaults(func=collect)

    q = sub.add_parser("probe", help="run the linear separability probe")
    q.add_argument("--data", type=str, default="data/cdens_probe_dataset.npz")
    q.add_argument("--num_cgf_features", type=int, default=64)
    q.add_argument("--t_init_scale", type=float, default=0.1)
    q.add_argument("--t_clamp", type=float, default=2.0)
    q.add_argument("--n_splits", type=int, default=5)
    q.add_argument("--clean_dw", type=float, default=0.05,
                   help="Clean stratum: |w_hat - w_heavy| must be below this. "
                        "Larger deviations ARE genuine Bayesian information "
                        "about the bit and are reported as leak strata.")
    q.add_argument("--include_ant", action="store_true",
                   help="Concatenate ant_pos/arena_scale to every feature set "
                        "(the Twin-Den probe's behavior). OFF by default here: "
                        "the den positions move with the mirror bit, so the "
                        "driver's own position is a confound, not an encoder "
                        "input. See the comment in probe().")
    q.add_argument("--den_margin", type=float, default=0.3,
                   help="Ant must be farther than spook_radius + this from "
                        "BOTH actual den centers.")
    q.add_argument("--st_checkpoint", type=str, default=None,
                   help="Add an ST64 feature set by running a REAL Set "
                        "Transformer encoder over the beliefs. Either a "
                        "3_train_st.py checkpoint or an SB3 policy .zip; see "
                        "--st_source.")
    q.add_argument("--st_source", choices=["pretrain", "policy"],
                   default="pretrain",
                   help="'pretrain': --st_checkpoint is a 3_train_st.py "
                        "checkpoint. 'policy': it is a 4_train_rl_st.py SB3 "
                        ".zip and the encoder is pulled out of the policy, "
                        "which is how a collapsed encoder gets measured "
                        "instead of inferred.")
    q.set_defaults(func=probe)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "invariance" and args.mean_bound is None:
        args.mean_bound = 3.0 / np.sqrt(args.n_particles)
    args.func(args)
