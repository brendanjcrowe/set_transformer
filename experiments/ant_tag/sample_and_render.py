"""
Sample a few trajectories from a trained AntTag policy and render them.

Each episode is rendered as a 2D arena plot with:
- Ant trajectory (blue), colored by time
- True target trajectory (red), colored by time
- Particle cloud snapshots at a few timesteps (gray dots sized by weight)
- Visibility radius ring around the ant at each snapshot
- Start / end markers, tag indicator

Usage:
    python training/sample_and_render_trajectories.py \
        --model_path sb3_ant_tag_finetune_v2_models/finetune_agent.zip \
        --vecnormalize_path sb3_ant_tag_finetune_v2_models/vecnormalize.pkl \
        --n_episodes 2 \
        --output_dir trajectory_renders/v2_final
"""
import argparse
import importlib

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter
from set_transformer.rl.wrappers.particle_filter import PFDictObservationWrapper

# Sibling module name starts with a digit, so importlib is required.
_train_rl_frozen = importlib.import_module("4_train_rl_frozen")
CurriculumVisibilityWrapper = _train_rl_frozen.CurriculumVisibilityWrapper
_CurriculumRouter = _train_rl_frozen._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_frozen.ant_tag_pf_interaction_mapper


ARENA_MIN, ARENA_MAX = -4.5, 4.5
VISIBILITY_RADIUS = 3.0
TAG_RADIUS = 1.5


def make_eval_env(num_particles, seed):
    """Bare env — no reward shaping, so step rewards are true sparse reward."""
    def _init():
        env = gym.make("pdomains-ant-tag-v0", rendering=False)
        env.reset(seed=seed)
        env = CurriculumVisibilityWrapper(env, initial_visibility_radius=3.0)
        env = PFDictObservationWrapper(
            env=env,
            particle_filter_class=AntTagParticleFilter,
            particle_filter_kwargs={},
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=[-2, -1],
        )
        env = Monitor(env)
        env = _CurriculumRouter(env)
        return env
    return _init


def get_particle_filter(vec_env):
    """Walk the DummyVecEnv's first env to find the PFDictObservationWrapper."""
    e = vec_env.envs[0]
    while e is not None:
        if isinstance(e, PFDictObservationWrapper):
            return e.particle_filter
        e = getattr(e, "env", None)
    return None


def get_unwrapped(vec_env):
    """Get the underlying pdomains AntTag env for true target pos."""
    return vec_env.envs[0].unwrapped


def rollout_one(model, vec_env, max_steps=400):
    """Run one episode, recording full trajectory data.

    DummyVecEnv auto-resets when `done` is returned, so we capture the
    ant/target positions and PF snapshot BEFORE calling step. The final
    frame is the pre-tag state (ant and target within tag_radius).
    """
    obs = vec_env.reset()
    unwrapped = get_unwrapped(vec_env)
    pf = get_particle_filter(vec_env)

    data = {
        "ant_pos": [],
        "target_pos": [],
        "particles": [],
        "weights": [],
        "visible": [],
        "reward": [],
        "step": [],
    }

    def snapshot(t, r):
        ant_pos = unwrapped.data.qpos[:2].copy()
        target_pos = unwrapped.get_target_pos().copy()
        data["ant_pos"].append(ant_pos)
        data["target_pos"].append(target_pos)
        data["particles"].append(pf.particles.copy())
        data["weights"].append(pf.weights.copy())
        data["visible"].append(
            np.linalg.norm(ant_pos - target_pos) < VISIBILITY_RADIUS
        )
        data["reward"].append(r)
        data["step"].append(t)

    snapshot(0, 0.0)

    for t in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        # Capture pre-step state so we can fall back to it on termination
        pre_ant = unwrapped.data.qpos[:2].copy()
        pre_tgt = unwrapped.get_target_pos().copy()
        pre_particles = pf.particles.copy()
        pre_weights = pf.weights.copy()
        obs, r, dones, infos = vec_env.step(action)

        if bool(dones[0]):
            # DummyVecEnv has already reset. The "terminal" ant/target
            # positions are from AFTER the physics step but BEFORE the reset,
            # which we can't recover. Approximate the tag frame by nudging the
            # pre-step state toward the target by one step (env uses 15 frame
            # skips of mj_step) — but simpler: just log the pre-step state with
            # an annotation. Reward reflects the actual tag.
            data["ant_pos"].append(pre_ant)
            data["target_pos"].append(pre_tgt)
            data["particles"].append(pre_particles)
            data["weights"].append(pre_weights)
            data["visible"].append(
                np.linalg.norm(pre_ant - pre_tgt) < VISIBILITY_RADIUS
            )
            data["reward"].append(float(r[0]))
            data["step"].append(t)
            break
        else:
            snapshot(t, float(r[0]))

    data = {k: np.array(v) for k, v in data.items()}
    return data


def render_episode(data, episode_idx, out_path, fps=15):
    """Render one episode as an animated GIF.

    Each frame shows:
      - Ant trail up to time t (blue, darkening with time)
      - Target trail up to time t (red, darkening with time)
      - Current particle cloud (gray dots, sized by weight)
      - Visibility / tag radius rings around the ant
      - Current ant (blue dot) and target (red X, salmon if hidden)
      - Tag star on final frame when tagged
    """
    T = len(data["step"])
    total_r = float(data["reward"].sum())
    tagged = T - 1 < 400

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(ARENA_MIN - 0.5, ARENA_MAX + 0.5)
    ax.set_ylim(ARENA_MIN - 0.5, ARENA_MAX + 0.5)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((ARENA_MIN, ARENA_MIN),
                               ARENA_MAX - ARENA_MIN, ARENA_MAX - ARENA_MIN,
                               fill=False, edgecolor="black", lw=1.5))

    # Persistent artists (updated each frame)
    ant_trail = LineCollection([], cmap="Blues", linewidths=2)
    tgt_trail = LineCollection([], cmap="Reds", linewidths=2)
    ax.add_collection(ant_trail)
    ax.add_collection(tgt_trail)

    vis_ring = Circle((0, 0), VISIBILITY_RADIUS, fill=False,
                      edgecolor="blue", lw=1.2, linestyle="--", zorder=2)
    tag_ring = Circle((0, 0), TAG_RADIUS, fill=False,
                      edgecolor="blue", lw=1.2, linestyle=":", zorder=2)
    ax.add_patch(vis_ring)
    ax.add_patch(tag_ring)

    particle_scat = ax.scatter([], [], s=[], c="gray", alpha=0.45, zorder=1)
    ant_scat = ax.scatter([], [], marker="o", s=120, c="blue", zorder=4)
    tgt_scat = ax.scatter([], [], marker="x", s=160, c="red", linewidths=3, zorder=4)
    tag_star = ax.scatter([], [], marker="*", s=520,
                          facecolor="gold", edgecolor="black", zorder=5)

    title = ax.set_title("")

    # Prepare trail segments once (we'll index up to current t each frame)
    ant = data["ant_pos"]
    tgt = data["target_pos"]
    ant_segs = np.stack([ant[:-1], ant[1:]], axis=1) if len(ant) > 1 else np.zeros((0, 2, 2))
    tgt_segs = np.stack([tgt[:-1], tgt[1:]], axis=1) if len(tgt) > 1 else np.zeros((0, 2, 2))

    def update(frame):
        t = frame
        # Growing trails
        if t > 0:
            ant_trail.set_segments(ant_segs[:t])
            ant_trail.set_array(np.linspace(0.3, 1.0, t))
            tgt_trail.set_segments(tgt_segs[:t])
            tgt_trail.set_array(np.linspace(0.3, 1.0, t))

        # Current ant position + visibility/tag rings
        ant_here = ant[t]
        vis_ring.center = tuple(ant_here)
        tag_ring.center = tuple(ant_here)
        ant_scat.set_offsets([ant_here])

        # Current target (color by visibility flag)
        tgt_here = tgt[t]
        visible = bool(data["visible"][t])
        tgt_scat.set_offsets([tgt_here])
        tgt_scat.set_color("red" if visible else "salmon")

        # Particle cloud (size by weight)
        p = data["particles"][t]
        w = data["weights"][t]
        sizes = 5 + 70 * (w / (w.max() + 1e-12))
        particle_scat.set_offsets(p)
        particle_scat.set_sizes(sizes)

        # Tag star on final frame if tagged
        if tagged and t == T - 1:
            tag_star.set_offsets([ant_here])
        else:
            tag_star.set_offsets(np.empty((0, 2)))

        running_r = float(data["reward"][: t + 1].sum())
        title.set_text(
            f"Episode {episode_idx}  |  t={int(data['step'][t])}/{T-1}  "
            f"|  running reward={running_r:.0f}  "
            f"|  {'target visible' if visible else 'target hidden'}"
            + ("   TAGGED!" if tagged and t == T - 1 else "")
        )
        return (ant_trail, tgt_trail, particle_scat, ant_scat, tgt_scat,
                vis_ring, tag_ring, tag_star, title)

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=False
    )
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--vecnormalize_path", type=str, default=None)
    p.add_argument("--n_episodes", type=int, default=2)
    p.add_argument("--num_particles", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env_fn = make_eval_env(args.num_particles, args.seed)
    vec_env = DummyVecEnv([env_fn])
    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model_path, env=vec_env)
    print(f"Loaded {args.model_path}")

    for ep in range(args.n_episodes):
        data = rollout_one(model, vec_env, max_steps=400)
        T = len(data["step"]) - 1
        tagged = T < 400
        out_path = os.path.join(args.output_dir, f"episode_{ep:02d}.gif")
        render_episode(data, ep, out_path, fps=args.fps)
        print(f"Episode {ep}: len={T}, reward={data['reward'].sum():.0f}, "
              f"{'TAGGED' if tagged else 'timed out'} -> {out_path}")


if __name__ == "__main__":
    main()
