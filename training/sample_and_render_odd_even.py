"""
Sample a few episodes from a trained OddEvenPOMDP policy and render each as a GIF.

Each frame shows:
- Particle distribution as a histogram over the valid means for this episode
- True mean (black dashed vertical)
- Agent's predicted mean for this step (blue solid vertical)
- Per-step accuracy + per-episode running custom (0/-1) reward

Usage:
    python training/sample_and_render_odd_even.py \
        --model_path sb3_odd_even_pretrained_v2_models/best_ppo_odd_even_pretrained/best_model.zip \
        --vecnormalize_path sb3_odd_even_pretrained_v2_models/vecnormalize_odd_even_pretrained.pkl \
        --pretrained_st_model_path set_transformer/particle_reconstruction_results/model.pth \
        --n_episodes 3 \
        --output_dir trajectory_renders/odd_even_v2
"""
import argparse
import os
import sys

_REPO_WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_WORKDIR not in sys.path:
    sys.path.insert(0, _REPO_WORKDIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pdomains.odd_even_pomdp import OddEvenPOMDPConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from train_odd_even_pretrained import make_pretrained_env


def get_unwrapped(vec_env):
    e = vec_env.envs[0]
    while hasattr(e, "env"):
        e = e.env
    return e


def rollout_one(model, vec_env, max_steps=100):
    obs = vec_env.reset()
    pomdp = get_unwrapped(vec_env)
    data = {
        "particles": [pomdp.particles.copy()],
        "true_mean": int(pomdp.mean),
        "hidden_param": str(pomdp.hidden_param),
        "raw_mean": float(pomdp.raw_mean),
        "valid_numbers": pomdp.valid_numbers.copy(),
        "n_dist_size": int(pomdp.n_dist_size),
        "predicted_means": [],
        "correct": [],
    }
    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = vec_env.step(action)
        info = infos[0]
        pred = int(info["predicted_mean"])
        correct = pred == int(info["true_mean"])
        data["predicted_means"].append(pred)
        data["correct"].append(correct)
        data["particles"].append(pomdp.particles.copy())
        if bool(dones[0]):
            break
    return data


def render_episode(data, episode_idx, out_path, fps=8):
    n = data["n_dist_size"]
    T = len(data["predicted_means"])
    valid = data["valid_numbers"]
    truth = data["true_mean"]
    parity = data["hidden_param"]

    fig, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])
    title = ax_p.set_title("")

    bar_container = ax_p.bar(valid, np.zeros(len(valid)), width=0.8,
                             alpha=0.7, color="steelblue", edgecolor="black",
                             label=f"particle histogram ({parity})")
    truth_line = ax_p.axvline(truth, color="black", linestyle="--", lw=2,
                              label=f"true mean = {truth}")
    pred_line = ax_p.axvline(0, color="blue", lw=2,
                             label="predicted mean")
    ax_p.set_xlim(0.3, n + 0.7)
    ax_p.set_xticks(range(1, n + 1))
    ax_p.set_ylim(0, 1.0)
    ax_p.set_xlabel("integer mean")
    ax_p.set_ylabel("particle mass")
    ax_p.grid(True, alpha=0.3, axis="y")
    ax_p.legend(loc="upper right")

    ax_r.set_xlim(0, T)
    ax_r.set_ylim(-T - 1, 1)
    ax_r.set_xlabel("step")
    ax_r.set_ylabel("cumulative custom reward")
    ax_r.axhline(0, color="gray", lw=0.5)
    ax_r.grid(True, alpha=0.3)
    reward_line, = ax_r.plot([], [], color="darkred", lw=1.5)

    cumr = np.cumsum([0.0 if c else -1.0 for c in data["correct"]])

    def update(frame):
        t = frame
        p = data["particles"][t]
        hist, _ = np.histogram(p, bins=np.arange(0.5, n + 1.5, 1))
        mass = hist[valid - 1].astype(float)
        if mass.sum() > 0:
            mass = mass / mass.sum()
        for bar, h in zip(bar_container, mass):
            bar.set_height(h)

        if t < T:
            pred = data["predicted_means"][t]
            pred_line.set_xdata([pred, pred])
            correct = data["correct"][t]
            acc_so_far = sum(data["correct"][: t + 1]) / (t + 1)
            cum_r = cumr[t]
            title.set_text(
                f"Episode {episode_idx}  |  step {t + 1}/{T}  "
                f"|  guess={pred} (true={truth}) {'OK' if correct else 'X'}  "
                f"|  acc={acc_so_far:.2f}  |  reward={cum_r:.0f}"
            )
            reward_line.set_data(np.arange(t + 1), cumr[: t + 1])
        else:
            title.set_text(f"Episode {episode_idx}  |  initial belief  |  truth={truth}")

        return list(bar_container) + [truth_line, pred_line, reward_line, title]

    n_frames = T + 1
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="sb3_odd_even_pretrained_v2_models/best_ppo_odd_even_pretrained/best_model.zip")
    p.add_argument("--vecnormalize_path", type=str,
                   default="sb3_odd_even_pretrained_v2_models/vecnormalize_odd_even_pretrained.pkl")
    p.add_argument("--pretrained_st_model_path", type=str, required=True)
    p.add_argument("--st_processor_device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--n_dist_size", type=int, default=10)
    p.add_argument("--std_dev", type=float, default=2.0)
    p.add_argument("--n_particles", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = OddEvenPOMDPConfig(
        n_dist_size=args.n_dist_size, mean=None, std_dev=args.std_dev,
        seed=None, n_particles=args.n_particles,
        true_particles=True, resample_proportion=0.5,
    )
    env_fn = make_pretrained_env(
        pomdp_config=cfg,
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_processor_device=args.st_processor_device,
        seed=args.seed, monitor_dir=None, max_steps=args.max_steps,
    )
    env = make_vec_env(env_fn, n_envs=1, seed=args.seed, vec_env_cls=DummyVecEnv)
    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model_path, env=env)
    print(f"Loaded {args.model_path}")

    for ep in range(args.n_episodes):
        data = rollout_one(model, env, max_steps=args.max_steps)
        T = len(data["predicted_means"])
        acc = sum(data["correct"]) / T if T else 0.0
        out_path = os.path.join(args.output_dir, f"episode_{ep:02d}.gif")
        render_episode(data, ep, out_path, fps=args.fps)
        print(f"Episode {ep}: parity={data['hidden_param']}, "
              f"raw_mean={data['raw_mean']:.2f}, true_mean={data['true_mean']}, "
              f"acc={acc:.2f} -> {out_path}")


if __name__ == "__main__":
    main()
