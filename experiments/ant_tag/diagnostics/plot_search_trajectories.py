"""Draw what a trained Ant-Tag policy does while it searches: ant path vs target path.

Rolls out a checkpoint in the belief env (real radius, deterministic), records
ant and target positions every step, and draws one panel per episode: ant path
coloured by time, target path in grey, sightings marked. Also prints per
episode: coverage (fraction of 1x1 cells of the cage the ant entered), wall
fraction of the ant's steps, target path length (is it moving?), and the
closest approach. Failed episodes are drawn first.

    python3 diagnostics/plot_search_trajectories.py --variant smart_mid_slow_v15 \
        --model_path <ckpt zip> --vecnormalize_path <pkl> --n_episodes 30 --out <png>
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_ANT_TAG_DIR), str(_ANT_TAG_DIR / "diagnostics")):
    if p not in sys.path:
        sys.path.insert(0, p)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import variants  # noqa: E402
pbs = importlib.import_module("probe_belief_shift")   # env factory + extractor import


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    variants.add_variant_argument(ap, default="smart_mid_slow_v15")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vecnormalize_path", required=True)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--n_plot", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.list_variants:
        variants.print_variants(); return
    variant = variants.resolve(args.variant)
    cap = variants.episode_cap(args.variant)

    env = DummyVecEnv([pbs._make_belief_env(variant.env_id, variant.particle_filter, 100, args.seed)])
    env = VecNormalize.load(args.vecnormalize_path, env); env.training = False; env.norm_reward = False
    model = PPO.load(args.model_path, env=env, device="cpu")
    env.seed(args.seed)
    raw = env.venv.envs[0].unwrapped
    half = float(raw.arena_limits[1]) if hasattr(raw, "arena_limits") else 4.5
    vis_r = float(raw.visible_radius)

    episodes = []
    for ep in range(args.n_episodes):
        obs = env.reset()
        ant, tgt = [raw.data.qpos[:2].copy()], [np.asarray(raw.get_target_pos()).copy()]
        done, steps = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            steps += 1; done = bool(dones[0])
            if not done:
                ant.append(raw.data.qpos[:2].copy()); tgt.append(np.asarray(raw.get_target_pos()).copy())
        ant, tgt = np.array(ant), np.array(tgt)
        d = np.linalg.norm(ant - tgt, axis=1)
        cells = set(map(tuple, np.floor(ant + half).astype(int)))
        n_cells = int((2 * half) ** 2)
        wall = np.mean(np.max(np.abs(ant), axis=1) > half - 1.0)
        episodes.append(dict(ep=ep, tagged=steps < cap, steps=steps, ant=ant, tgt=tgt, d=d,
                             coverage=len(cells) / n_cells, wall_frac=float(wall),
                             tgt_path=float(np.linalg.norm(np.diff(tgt, axis=0), axis=1).sum()),
                             ant_path=float(np.linalg.norm(np.diff(ant, axis=0), axis=1).sum()),
                             min_d=float(d.min()), sightings=int((d < vis_r).sum())))

    fails = [e for e in episodes if not e["tagged"]]
    print(f"{len(episodes)} episodes, {len(fails)} failures. Per-episode (failures first):")
    print(f"{'ep':>3s} {'out':7s} {'steps':>5s} {'ant path':>8s} {'coverage':>8s} {'wall%':>6s} {'tgt path':>8s} {'min dist':>8s} {'steps in view':>13s}")
    for e in fails + [e for e in episodes if e["tagged"]]:
        print(f"{e['ep']:3d} {'tag' if e['tagged'] else 'FAIL':7s} {e['steps']:5d} {e['ant_path']:8.1f} {e['coverage']:8.2f} "
              f"{100 * e['wall_frac']:6.0f} {e['tgt_path']:8.1f} {e['min_d']:8.2f} {e['sightings']:13d}")
    if fails:
        print(f"\nFAILURES: coverage median {np.median([e['coverage'] for e in fails]):.2f} of the cage's 1x1 cells, "
              f"wall-band steps {100 * np.median([e['wall_frac'] for e in fails]):.0f}%, target path median "
              f"{np.median([e['tgt_path'] for e in fails]):.1f} units (a free 0.3/step random walk would be ~{0.3 * cap:.0f}), "
              f"closest approach median {np.median([e['min_d'] for e in fails]):.2f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order = fails[:args.n_plot] + [e for e in episodes if e["tagged"]][:max(0, args.n_plot - len(fails))]
    n = len(order); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    for ax, e in zip(np.array(axes).flat, order):
        t = np.arange(len(e["ant"]))
        ax.plot(e["tgt"][:, 0], e["tgt"][:, 1], color="gray", lw=1, alpha=0.7, label="target")
        ax.scatter(e["tgt"][0, 0], e["tgt"][0, 1], marker="s", color="gray", s=40)
        ax.scatter(e["tgt"][-1, 0], e["tgt"][-1, 1], marker="*", color="black", s=90, zorder=5, label="target end")
        sc = ax.scatter(e["ant"][:, 0], e["ant"][:, 1], c=t, cmap="viridis", s=6, label="ant (colour = time)")
        ax.scatter(e["ant"][0, 0], e["ant"][0, 1], marker="o", facecolors="none", edgecolors="red", s=80)
        seen = e["d"] < vis_r
        if seen.any():
            ax.scatter(e["ant"][seen, 0], e["ant"][seen, 1], color="red", s=14, zorder=6, label="sighting")
        ax.set_xlim(-half - 0.2, half + 0.2); ax.set_ylim(-half - 0.2, half + 0.2); ax.set_aspect("equal")
        ax.set_title(f"ep {e['ep']} {'TAG' if e['tagged'] else 'FAIL'} {e['steps']} steps | cover {e['coverage']:.2f} "
                     f"wall {100 * e['wall_frac']:.0f}% | min d {e['min_d']:.1f} | tgt moved {e['tgt_path']:.0f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.array(axes).flat[n:]:
        ax.axis("off")
    np.array(axes).flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{args.model_path.split('/')[-4] if '/' in args.model_path else args.model_path}: ant path (viridis = time, red ring = start), "
                 f"target path (grey, black star = end), red dots = target within {vis_r}", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out, dpi=110); print("wrote", args.out)


if __name__ == "__main__":
    main()
