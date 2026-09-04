"""Record and render a Counterweighted-Dens belief trajectory as a GIF.

The arena panel shows the ant, true target, particle belief, weighted mean,
and a covariance ellipse.  The right panel prints the raw and network-scaled
Gaussian moments for every frame.  The trajectory is always saved as NPZ, so
rendering can subsequently be repeated without rerunning MuJoCo.

Outputs are kept separate inside this diagnostics directory:

* GIFs: ``diagnostics/gifs/``
* NPZ recordings: ``diagnostics/trajectories/``

The directory portion of ``--output_gif`` and ``--trajectory_out`` is ignored;
use those arguments to choose filenames only.

Record and render a learned policy::

    python3 diagnostics/render_cden_belief_gif.py \
      --encoder gaussian --model_path <agent.zip> \
      --vecnormalize_path <vecnormalize.pkl> \
      --env_id pdomains-ant-tag-cdens-hard-v0 \
      --env_seed 42001 --pf_seed 52001 \
      --output_gif gaussian_seed42001.gif

Render an existing recording::

    python3 diagnostics/render_cden_belief_gif.py \
      --trajectory_in diagnostics/trajectories/gaussian_seed42001.npz \
      --output_gif gaussian_seed42001_rerender.gif
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Rectangle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
_EVAL_DIR = _ANT_TAG_DIR / "eval_scripts"
_DIAGNOSTICS_DIR = Path(__file__).resolve().parent
_GIF_DIR = _DIAGNOSTICS_DIR / "gifs"
_TRAJECTORY_DIR = _DIAGNOSTICS_DIR / "trajectories"
for path in (_REPO_ROOT, _ANT_TAG_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pdomains  # noqa: E402,F401 - register environments
from set_transformer.rl.particle_filters.ant_tag import (  # noqa: E402
    CounterweightedDenAntTagParticleFilter,
)


def _find_pf_wrapper(env):
    current = env
    while current is not None:
        if hasattr(current, "particle_filter"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find the particle-filter wrapper")


def _weighted_moments(particles, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    weights /= max(float(weights.sum()), 1e-12)
    particles = np.asarray(particles, dtype=np.float64)
    mean = np.sum(weights[:, None] * particles, axis=0)
    centered = particles - mean
    cov = np.einsum("n,ni,nj->ij", weights, centered, centered)
    return mean, cov


def _batch_obs(obs):
    return {key: np.expand_dims(value, axis=0) for key, value in obs.items()}


def _record(args):
    eval_module = importlib.import_module(
        "eval_true_reward_cgf" if args.encoder == "cgf"
        else "eval_true_reward_gaussian"
    )
    env_fn = eval_module.make_eval_env(
        num_particles=args.num_particles,
        obs_mask_indices=[-2, -1],
        seed=args.env_seed,
        env_id=args.env_id,
        particle_filter_class=CounterweightedDenAntTagParticleFilter,
    )

    # VecNormalize is used only as a fixed observation transform here.  We
    # step the underlying Gym env directly so its terminal state remains
    # available for the last GIF frame (DummyVecEnv otherwise auto-resets).
    dummy = DummyVecEnv([env_fn])
    vecnorm = VecNormalize.load(args.vecnormalize_path, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    env = vecnorm.venv.envs[0]
    raw = env.unwrapped
    pf_wrapper = _find_pf_wrapper(env)
    model = PPO.load(args.model_path, device=args.device)

    pf_wrapper.set_particle_filter_seed(args.pf_seed)
    obs, _ = env.reset(seed=args.env_seed)
    metadata = {
        "env_id": args.env_id,
        "encoder": args.encoder,
        "model_path": str(Path(args.model_path).resolve()),
        "vecnormalize_path": str(Path(args.vecnormalize_path).resolve()),
        "env_seed": args.env_seed,
        "pf_seed": args.pf_seed,
        "arena_scale": float(raw.cage_max_x),
        "visible_radius": float(raw.visible_radius),
        "tag_radius": float(raw.tag_radius),
        "den_radius": float(raw.cden_r),
        "spook_radius": float(raw.cden_spook_radius),
        "heavy_pos": np.asarray(raw.cden_heavy_pos).tolist(),
        "light_pos": np.asarray(raw.cden_light_pos).tolist(),
        "candidates": np.asarray(raw.cden_candidates).tolist(),
        "heavy_side": int(raw.cden_heavy_side),
        "occupied_den": "heavy" if raw._occupied_is_heavy else "light",
        "max_steps": args.max_steps,
    }
    lists = {
        "step": [], "ant_pos": [], "target_pos": [], "particles": [],
        "weights": [], "mean": [], "cov": [], "visible": [],
        "spooked": [], "reward": [], "terminated": [], "truncated": [],
    }

    def snapshot(step, reward, terminated=False, truncated=False):
        particles = pf_wrapper.particle_filter.particles.copy()
        weights = pf_wrapper.particle_filter.weights.copy()
        mean, cov = _weighted_moments(particles, weights)
        ant = np.asarray(raw.data.qpos[:2], dtype=np.float64).copy()
        target = np.asarray(raw.get_target_pos(), dtype=np.float64).copy()
        lists["step"].append(step)
        lists["ant_pos"].append(ant)
        lists["target_pos"].append(target)
        lists["particles"].append(particles)
        lists["weights"].append(weights)
        lists["mean"].append(mean)
        lists["cov"].append(cov)
        lists["visible"].append(
            np.linalg.norm(ant - target) < raw.visible_radius)
        lists["spooked"].append(bool(raw.cden_spooked))
        lists["reward"].append(float(reward))
        lists["terminated"].append(bool(terminated))
        lists["truncated"].append(bool(truncated))

    snapshot(0, 0.0)
    final_info = {}
    for step in range(1, args.max_steps + 1):
        normalized_obs = vecnorm.normalize_obs(_batch_obs(obs))
        action, _ = model.predict(normalized_obs, deterministic=True)
        obs, reward, terminated, truncated, final_info = env.step(action[0])
        snapshot(step, reward, terminated, truncated)
        if terminated or truncated:
            break

    data = {key: np.asarray(value) for key, value in lists.items()}
    metadata["length"] = int(data["step"][-1])
    metadata["tagged"] = bool(final_info.get(
        "is_success", data["terminated"][-1]))
    metadata["final_spooked"] = bool(data["spooked"][-1])
    metadata["final_info"] = {
        key: value for key, value in final_info.items()
        if isinstance(value, (str, bool, int, float))
    }
    vecnorm.close()
    return data, metadata


def _save_recording(path, data, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        **data,
    )


def _load_recording(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        data = {key: saved[key] for key in saved.files if key != "metadata"}
    return data, metadata


def _ellipse_geometry(cov, n_std):
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    width, height = 2.0 * n_std * np.sqrt(values)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    return float(width), float(height), float(angle)


def _segments(points):
    if len(points) < 2:
        return np.zeros((0, 2, 2))
    return np.stack([points[:-1], points[1:]], axis=1)


def _render(data, metadata, output_gif, fps, frame_stride,
            ellipse_std, dpi):
    arena = float(metadata["arena_scale"])
    visible_radius = float(metadata["visible_radius"])
    tag_radius = float(metadata["tag_radius"])
    den_radius = float(metadata["den_radius"])
    heavy = np.asarray(metadata["heavy_pos"], dtype=np.float64)
    light = np.asarray(metadata["light_pos"], dtype=np.float64)
    candidates = np.asarray(metadata["candidates"], dtype=np.float64)
    scale = arena

    frame_indices = list(range(0, len(data["step"]), max(frame_stride, 1)))
    if frame_indices[-1] != len(data["step"]) - 1:
        frame_indices.append(len(data["step"]) - 1)

    fig = plt.figure(figsize=(11.5, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 0.9))
    ax = fig.add_subplot(grid[0, 0])
    text_ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(-arena - 0.5, arena + 0.5)
    ax.set_ylim(-arena - 0.5, arena + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.15)
    ax.add_patch(Rectangle((-arena, -arena), 2 * arena, 2 * arena,
                           fill=False, edgecolor="black", linewidth=1.5))

    for candidate in candidates:
        ax.scatter(*candidate, marker="+", s=45, color="0.7", zorder=0)
    ax.add_patch(Circle(heavy, den_radius, facecolor="limegreen",
                        edgecolor="darkgreen", alpha=0.16, linewidth=1.5))
    ax.add_patch(Circle(light, den_radius, facecolor="orange",
                        edgecolor="darkorange", alpha=0.16, linewidth=1.5))
    ax.text(*heavy, "H", ha="center", va="center", color="darkgreen",
            weight="bold", zorder=3)
    ax.text(*light, "L", ha="center", va="center", color="darkorange",
            weight="bold", zorder=3)

    ant_trail = LineCollection([], colors="royalblue", linewidths=1.8,
                               alpha=0.8, zorder=2)
    target_trail = LineCollection([], colors="firebrick", linewidths=1.4,
                                  alpha=0.55, zorder=2)
    ax.add_collection(ant_trail)
    ax.add_collection(target_trail)
    particle_scat = ax.scatter([], [], s=[], color="0.25", alpha=0.48,
                               label="PF particles", zorder=2)
    mean_scat = ax.scatter([], [], marker="D", s=100, facecolor="magenta",
                           edgecolor="black", label="weighted mean", zorder=6)
    ant_scat = ax.scatter([], [], marker="o", s=120, facecolor="royalblue",
                          edgecolor="navy", label="ant", zorder=7)
    target_scat = ax.scatter([], [], marker="X", s=150, facecolor="red",
                             edgecolor="darkred", label="true target", zorder=7)
    ellipse = Ellipse((0, 0), 0, 0, fill=False, edgecolor="magenta",
                      linewidth=2.0, linestyle="--",
                      label=f"{ellipse_std:g}σ covariance", zorder=5)
    ax.add_patch(ellipse)
    vis_ring = Circle((0, 0), visible_radius, fill=False,
                      edgecolor="dodgerblue", linestyle=":", linewidth=1.2)
    tag_ring = Circle((0, 0), tag_radius, fill=False,
                      edgecolor="navy", linestyle="--", linewidth=1.2)
    ax.add_patch(vis_ring)
    ax.add_patch(tag_ring)
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")

    text_ax.axis("off")
    info_text = text_ax.text(
        0.0, 1.0, "", transform=text_ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=10.5, linespacing=1.35)
    text_ax.text(
        0.0, 0.02,
        "Raw moments use arena coordinates.\n"
        "Network moments use x/arena_scale;\n"
        "therefore covariance is divided by scale².",
        transform=text_ax.transAxes, va="bottom", ha="left",
        fontsize=9, color="0.3")

    ant_segments = _segments(data["ant_pos"])
    target_segments = _segments(data["target_pos"])

    def update(animation_index):
        index = frame_indices[animation_index]
        step = int(data["step"][index])
        ant = data["ant_pos"][index]
        target = data["target_pos"][index]
        particles = data["particles"][index]
        weights = np.clip(data["weights"][index], 0.0, None)
        weights /= max(float(weights.sum()), 1e-12)
        mean = data["mean"][index]
        cov = data["cov"][index]
        network_mean = mean / scale
        network_cov = cov / (scale * scale)

        ant_trail.set_segments(ant_segments[:index])
        target_trail.set_segments(target_segments[:index])
        particle_scat.set_offsets(particles)
        particle_scat.set_sizes(10.0 + 900.0 * weights)
        mean_scat.set_offsets([mean])
        ant_scat.set_offsets([ant])
        target_scat.set_offsets([target])
        target_scat.set_alpha(1.0 if data["visible"][index] else 0.32)
        vis_ring.center = tuple(ant)
        tag_ring.center = tuple(ant)
        width, height, angle = _ellipse_geometry(cov, ellipse_std)
        ellipse.center = tuple(mean)
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle

        mode_index = np.linalg.norm(
            particles[:, None, :] - candidates[None, :, :], axis=2
        ).argmin(axis=1)
        mode_mass = np.array([
            weights[mode_index == candidate].sum() for candidate in range(4)
        ])
        status = "TAGGED" if data["terminated"][index] else (
            "TIMEOUT" if data["truncated"][index] else "running")
        title.set_text(
            f"{metadata['encoder'].upper()} belief trajectory — "
            f"step {step}/{metadata['length']} — {status}")
        info_text.set_text(
            f"Environment\n"
            f"  id: {metadata['env_id']}\n"
            f"  env seed: {metadata['env_seed']}\n"
            f"  PF seed:  {metadata['pf_seed']}\n"
            f"  arrangement: heavy side {metadata['heavy_side']}\n"
            f"  occupied: {metadata['occupied_den']}\n"
            f"  visible:  {bool(data['visible'][index])}\n"
            f"  spooked:  {bool(data['spooked'][index])}\n\n"
            f"Raw weighted mean\n"
            f"  [{mean[0]: .6f}, {mean[1]: .6f}]\n\n"
            f"Raw weighted covariance\n"
            f"  [[{cov[0,0]: .6f}, {cov[0,1]: .6f}],\n"
            f"   [{cov[1,0]: .6f}, {cov[1,1]: .6f}]]\n\n"
            f"Network Gaussian features\n"
            f"  mean = [{network_mean[0]: .6f},\n"
            f"          {network_mean[1]: .6f}]\n"
            f"  cov  = [[{network_cov[0,0]: .6f},\n"
            f"           {network_cov[0,1]: .6f}],\n"
            f"          [{network_cov[1,0]: .6f},\n"
            f"           {network_cov[1,1]: .6f}]]\n\n"
            f"Mode mass [-h,+h,-f,+f]\n"
            f"  [{mode_mass[0]:.3f}, {mode_mass[1]:.3f},\n"
            f"   {mode_mass[2]:.3f}, {mode_mass[3]:.3f}]"
        )
        return (ant_trail, target_trail, particle_scat, mean_scat,
                ant_scat, target_scat, ellipse, vis_ring, tag_ring,
                title, info_text)

    gif = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=max(1, 1000 // fps), blit=False)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    gif.save(output_gif, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("trajectory source")
    source.add_argument("--trajectory_in", type=Path)
    source.add_argument("--encoder", choices=("gaussian", "cgf"))
    source.add_argument("--model_path")
    source.add_argument("--vecnormalize_path")
    source.add_argument("--env_id",
                        default="pdomains-ant-tag-cdens-hard-v0")
    source.add_argument("--env_seed", type=int, default=42001)
    source.add_argument("--pf_seed", type=int, default=52001)
    source.add_argument("--num_particles", type=int, default=100)
    source.add_argument("--max_steps", type=int, default=300)
    source.add_argument("--device", default="cpu")

    output = parser.add_argument_group("output")
    output.add_argument("--output_gif", type=Path, required=True)
    output.add_argument("--trajectory_out", type=Path)
    output.add_argument("--fps", type=int, default=12)
    output.add_argument("--frame_stride", type=int, default=1)
    output.add_argument("--ellipse_std", type=float, default=1.0)
    output.add_argument("--dpi", type=int, default=90)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    gif_path = _GIF_DIR / args.output_gif.name
    if args.trajectory_in is not None:
        data, metadata = _load_recording(args.trajectory_in)
        trajectory_path = args.trajectory_in
    else:
        missing = [name for name in ("encoder", "model_path",
                                      "vecnormalize_path")
                   if getattr(args, name) is None]
        if missing:
            parser.error("recording mode requires " + ", ".join(
                "--" + name for name in missing))
        data, metadata = _record(args)
        trajectory_name = (
            args.trajectory_out.name if args.trajectory_out is not None
            else gif_path.with_suffix(".npz").name
        )
        trajectory_path = _TRAJECTORY_DIR / trajectory_name
        _save_recording(trajectory_path, data, metadata)
        print(f"Saved trajectory: {trajectory_path}")

    _render(data, metadata, gif_path, args.fps,
            args.frame_stride, args.ellipse_std, args.dpi)
    print(f"Saved GIF: {gif_path}")
    print(
        f"Episode: length={metadata['length']}, "
        f"tagged={metadata['tagged']}, occupied={metadata['occupied_den']}, "
        f"spooked={metadata['final_spooked']}")
    print(f"Recording source: {trajectory_path}")


if __name__ == "__main__":
    main()
