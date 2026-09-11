"""Record episodes of a trained Ant-Tag policy and render them as belief GIFs.

Variant-generic sibling of render_cden_belief_gif.py (which is tied to the
counterweighted-den env). Each frame shows the cage, the ant with its
visibility and tag rings, the true target (dimmed when the ant cannot see
it), the weighted particle set (marker size = weight), the weighted mean and
a thin, low-alpha pink covariance ellipse; the right panel prints the step,
visibility, ant-target distance, belief-mean error, spread, ESS and the
belief mass within the tag radius of the true target.

Workflow: roll the policy out for --n_episodes, save every trajectory as NPZ
under diagnostics/trajectories/<variant>/, then render --n_tag tagged and
--n_fail failed episodes (longest failures and a mix of short/long tags) to
diagnostics/gifs/<variant>/. Re-render from NPZ with --render_only.

    python3 diagnostics/render_belief_gif.py --variant smart_mid_slow_v15 \
        --model_path <st_agent.zip> --vecnormalize_path <vecnormalize.pkl> \
        --n_episodes 30 --n_tag 3 --n_fail 3 --tag_prefix plain_ft0.1_s0
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
_DIAG = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _ANT_TAG_DIR, _ANT_TAG_DIR / "eval_scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pdomains  # noqa: E402,F401
import variants  # noqa: E402
importlib.import_module("4_train_rl_st")          # registers the ST extractor class for PPO.load
_eval = importlib.import_module("eval_true_reward_cgf")


def _find_pf_wrapper(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "particle_filter"):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("no particle-filter wrapper found")


def _moments(P, W):
    w = np.clip(np.asarray(W, float), 0, None); w /= max(w.sum(), 1e-12)
    mean = (w[:, None] * P).sum(0); c = P - mean
    return mean, np.einsum("n,ni,nj->ij", w, c, c), w


def record(args, variant, cap):
    env_fn = _eval.make_eval_env(num_particles=args.num_particles, obs_mask_indices=[-2, -1], seed=args.seed,
                                 env_id=variant.env_id, particle_filter_class=variant.particle_filter)
    dummy = DummyVecEnv([env_fn])
    vecnorm = VecNormalize.load(args.vecnormalize_path, dummy); vecnorm.training = False; vecnorm.norm_reward = False
    env = vecnorm.venv.envs[0]; raw = env.unwrapped; pf_wrapper = _find_pf_wrapper(env)
    model = PPO.load(args.model_path, device="cpu")
    vis_r, tag_r, half = float(raw.visible_radius), float(raw.tag_radius), float(raw.cage_max_x)
    out_dir = _DIAG / "trajectories" / args.variant; out_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for ep in range(args.n_episodes):
        ep_seed = args.seed * 1000 + ep
        if hasattr(pf_wrapper, "set_particle_filter_seed"):
            pf_wrapper.set_particle_filter_seed(ep_seed + 50_000)
        obs, _ = env.reset(seed=ep_seed)
        rec = {k: [] for k in ("ant", "target", "particles", "weights", "visible")}

        def snap():
            pf = pf_wrapper.particle_filter
            a = np.asarray(raw.data.qpos[:2], float).copy(); t = np.asarray(raw.get_target_pos(), float).copy()
            rec["ant"].append(a); rec["target"].append(t); rec["particles"].append(pf.particles.copy())
            rec["weights"].append(pf.weights.copy()); rec["visible"].append(np.linalg.norm(a - t) < vis_r)
        snap()
        tagged = False
        for step in range(1, cap + 1):
            nobs = vecnorm.normalize_obs({k: np.expand_dims(v, 0) for k, v in obs.items()})
            action, _ = model.predict(nobs, deterministic=True)
            obs, r, term, trunc, info = env.step(action[0]); snap()
            if term or trunc:
                tagged = bool(term and step < cap); break
        data = {k: np.asarray(v) for k, v in rec.items()}
        meta = dict(variant=args.variant, env_id=variant.env_id, model_path=str(Path(args.model_path).resolve()),
                    ep=ep, env_seed=ep_seed, length=int(len(data["ant"]) - 1), tagged=tagged, visible_radius=vis_r,
                    tag_radius=tag_r, half_width=half, sightings=int(data["visible"].sum()))
        path = out_dir / f"{args.tag_prefix}_ep{ep:02d}_{'tag' if tagged else 'fail'}_{meta['length']}steps.npz"
        np.savez_compressed(path, metadata=np.asarray(json.dumps(meta)), **data)
        episodes.append((path, meta))
        print(f"ep {ep:02d}: {'TAG ' if tagged else 'FAIL'} {meta['length']:3d} steps, {meta['sightings']} steps in view")
    vecnorm.close()
    return episodes


def load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item())); data = {k: z[k] for k in z.files if k != "metadata"}
    return data, meta


def ellipse_geom(cov, n_std):
    vals, vecs = np.linalg.eigh(cov); o = np.argsort(vals)[::-1]; vals = np.clip(vals[o], 0, None); vecs = vecs[:, o]
    return 2 * n_std * np.sqrt(vals[0]), 2 * n_std * np.sqrt(vals[1]), np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))


def render(data, meta, out_gif, fps=12, stride=1, n_std=1.0, dpi=80):
    half, vis_r, tag_r = meta["half_width"], meta["visible_radius"], meta["tag_radius"]
    n = len(data["ant"]); idxs = list(range(0, n, max(1, stride)))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    fig = plt.figure(figsize=(10.5, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.6, 0.8)); ax = fig.add_subplot(gs[0, 0]); tx = fig.add_subplot(gs[0, 1]); tx.axis("off")
    ax.set_xlim(-half - 0.4, half + 0.4); ax.set_ylim(-half - 0.4, half + 0.4); ax.set_aspect("equal"); ax.grid(alpha=0.12)
    ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False, edgecolor="black", linewidth=1.5))
    ant_trail = LineCollection([], colors="royalblue", linewidths=1.6, alpha=0.75, zorder=2); ax.add_collection(ant_trail)
    tgt_trail = LineCollection([], colors="firebrick", linewidths=1.2, alpha=0.5, zorder=2); ax.add_collection(tgt_trail)
    parts = ax.scatter([], [], s=[], color="0.25", alpha=0.5, label="PF particles (size = weight)", zorder=3)
    mean_sc = ax.scatter([], [], marker="D", s=80, facecolor="hotpink", edgecolor="black", label="weighted mean", zorder=6)
    ant_sc = ax.scatter([], [], marker="o", s=110, facecolor="royalblue", edgecolor="navy", label="ant", zorder=7)
    tgt_sc = ax.scatter([], [], marker="X", s=140, facecolor="red", edgecolor="darkred", label="true target (dim = unseen)", zorder=7)
    ell = Ellipse((0, 0), 0, 0, fill=False, edgecolor="hotpink", linewidth=1.0, alpha=0.55, label=f"{n_std:g}σ covariance", zorder=5); ax.add_patch(ell)
    vis_ring = Circle((0, 0), vis_r, fill=False, edgecolor="dodgerblue", linestyle=":", linewidth=1.1); ax.add_patch(vis_ring)
    tag_ring = Circle((0, 0), tag_r, fill=False, edgecolor="navy", linestyle="--", linewidth=1.0); ax.add_patch(tag_ring)
    ax.legend(loc="upper left", fontsize=7.5); title = ax.set_title("")
    info = tx.text(0, 1, "", transform=tx.transAxes, va="top", ha="left", family="monospace", fontsize=10, linespacing=1.4)
    A, T = data["ant"], data["target"]
    segA = np.stack([A[:-1], A[1:]], 1) if n > 1 else np.zeros((0, 2, 2)); segT = np.stack([T[:-1], T[1:]], 1) if n > 1 else np.zeros((0, 2, 2))
    blind = 0

    def update(k):
        nonlocal blind
        i = idxs[k]; a, t, P, W = A[i], T[i], data["particles"][i], data["weights"][i]
        mean, cov, w = _moments(P, W); vis = bool(data["visible"][i])
        blind = 0 if vis else blind + (idxs[k] - idxs[k - 1] if k > 0 else 0)
        ant_trail.set_segments(segA[:i]); tgt_trail.set_segments(segT[:i])
        parts.set_offsets(P); parts.set_sizes(8 + 700 * w)
        mean_sc.set_offsets([mean]); ant_sc.set_offsets([a]); tgt_sc.set_offsets([t]); tgt_sc.set_alpha(1.0 if vis else 0.3)
        vis_ring.center = tuple(a); tag_ring.center = tuple(a)
        wdt, hgt, ang = ellipse_geom(cov, n_std); ell.center = tuple(mean); ell.width, ell.height, ell.angle = wdt, hgt, ang
        d_pt = np.linalg.norm(P - t, axis=1); mass_tag = float((w * (d_pt < tag_r)).sum()); ess = 1 / max((w ** 2).sum(), 1e-12)
        spread = float(np.sqrt(np.diag(cov)).mean())
        status = ("TAGGED" if meta["tagged"] else "TIMEOUT") if i == n - 1 else "running"
        title.set_text(f"{meta['variant']}  ep {meta['ep']}  step {i}/{meta['length']}  {status}")
        info.set_text(f"target visible: {vis}\nsteps since sighting: {blind}\n\nant-target dist:  {np.linalg.norm(a - t):5.2f}\n"
                      f"mean-target dist: {np.linalg.norm(mean - t):5.2f}\nnearest particle: {d_pt.min():5.2f}\n\n"
                      f"belief spread:    {spread:5.2f}\nESS (of {len(w)}):     {ess:5.1f}\nmass within {tag_r:g} of\n  true target:    {100 * mass_tag:5.1f}%\n\n"
                      f"policy: {Path(meta['model_path']).parts[-3]}")
        return parts, mean_sc, ant_sc, tgt_sc, ell, vis_ring, tag_ring, ant_trail, tgt_trail, title, info

    anim = animation.FuncAnimation(fig, update, frames=len(idxs), interval=1000 // fps, blit=False)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_gif, writer=animation.PillowWriter(fps=fps), dpi=dpi); plt.close(fig)
    print("wrote", out_gif)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    variants.add_variant_argument(ap, default="smart_mid_slow_v15")
    ap.add_argument("--model_path"); ap.add_argument("--vecnormalize_path")
    ap.add_argument("--num_particles", type=int, default=100); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_episodes", type=int, default=30); ap.add_argument("--n_tag", type=int, default=3); ap.add_argument("--n_fail", type=int, default=3)
    ap.add_argument("--tag_prefix", default="policy")
    ap.add_argument("--render_only", nargs="*", help="NPZ trajectories to render instead of recording")
    ap.add_argument("--fps", type=int, default=12); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ellipse_std", type=float, default=1.0); ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()
    if args.list_variants:
        variants.print_variants(); return
    gif_dir = _DIAG / "gifs" / args.variant
    if args.render_only:
        for p in args.render_only:
            data, meta = load(p); render(data, meta, gif_dir / (Path(p).stem + ".gif"), args.fps, args.stride, args.ellipse_std, args.dpi)
        return
    variant = variants.resolve(args.variant); cap = variants.episode_cap(args.variant)
    eps = record(args, variant, cap)
    tags = sorted([e for e in eps if e[1]["tagged"]], key=lambda e: e[1]["length"])
    fails = sorted([e for e in eps if not e[1]["tagged"]], key=lambda e: -e[1]["sightings"])
    print(f"\n{len(tags)} tagged / {len(fails)} failed of {len(eps)}")
    # tags: shortest, median, longest; fails: the ones with most sightings first (near misses), then blind ones
    pick_t = [tags[j] for j in sorted({0, len(tags) // 2, len(tags) - 1})][: args.n_tag] if tags else []
    pick_f = fails[: max(1, args.n_fail // 2)] + fails[-(args.n_fail - max(1, args.n_fail // 2)):] if len(fails) > args.n_fail else fails
    for path, meta in pick_t + pick_f:
        data, _ = load(path); render(data, meta, gif_dir / (path.stem + ".gif"), args.fps, args.stride, args.ellipse_std, args.dpi)


if __name__ == "__main__":
    main()
