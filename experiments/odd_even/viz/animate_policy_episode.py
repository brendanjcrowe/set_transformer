"""Animate the exact posterior over one Odd-Even episode, with a TRAINED
policy's guess marked on the x axis. One panel, no clutter.

What is drawn, per frame (one frame per step, held for --seconds_per_step):

    x axis   the candidate states 1..n, every one labelled
    y axis   probability mass the exact posterior puts on each state
    orange   the x-axis label of the state the policy guesses at THIS step,
             i.e. the action chosen from the belief shown in the frame

The belief shown in frame t is the one the policy had when it chose that
guess: b_0 (after reset's own observation) for step 1, b_1 for step 2, and so
on. The title carries the step number, the guess and whether it hit. The
true state is pinned through the env's `true_state` config field so the same
policy can be watched on a chosen state.

The policy is loaded exactly the way the evaluator loads it
(eval_scripts/eval_true_reward_odd_even.py): the same belief env stack, the
saved VecNormalize statistics with reward normalisation off, and a re-seed
after PPO.load (PITFALLS.md section 2).

Usage:
    python3 viz/animate_policy_episode.py --variant oe50_short \
        --model_path runs/odd_even_cgf_oe50_short/<run>/models/cgf_agent.zip \
        --vecnormalize_path runs/odd_even_cgf_oe50_short/<run>/models/vecnormalize.pkl \
        --true_states 24 25

Writes one .mp4 per true state plus a stacked two-panel .mp4 with both
episodes in lockstep, into --out_dir (default viz/out/).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ODD_EVEN_DIR = _HERE.parent                  # experiments/odd_even
_REPO_ROOT = _HERE.parents[3]                 # repo root (CLAUDE.md convention)
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "set_transformer"), str(_ODD_EVEN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")          # headless: must precede pyplot
import matplotlib.pyplot as plt
from matplotlib import animation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers the pdomains-odd-even-* envs
from set_transformer.rl.domains import odd_even as variants  # noqa: E402 - the registry
from set_transformer.rl.domains.odd_even import make_odd_even_belief_env  # noqa: E402

BAR_COLOUR = "#3b6fb6"
GUESS_COLOUR = "#ff7f0e"


# --------------------------------------------------------------------------
# rollout with the trained policy
# --------------------------------------------------------------------------

def _checkpoint_num_particles(model_path: str) -> int:
    from stable_baselines3.common.save_util import load_from_zip_file
    data, _params, _other = load_from_zip_file(
        model_path, load_data=True, device="cpu", print_system_info=False)
    return int(data["observation_space"]["particles"].shape[0])


def load_policy_and_env(variant: str, model_path: str,
                        vecnormalize_path: str | None, seed: int):
    num_particles = _checkpoint_num_particles(model_path)
    venv = DummyVecEnv([make_odd_even_belief_env(
        num_particles=num_particles, rank=0, seed=seed, variant=variant)])
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        venv = VecNormalize.load(vecnormalize_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        print("WARNING: no VecNormalize stats; if training used them the "
              "policy is seeing a different input distribution.")
    model = PPO.load(model_path, env=venv, device="cpu")
    # PPO.load re-seeds the env with the checkpoint's TRAINING seed.
    venv.seed(seed)
    venv.action_space.seed(seed)
    return model, venv


def rollout_with_policy(model, venv, variant: str, true_state: int, seed: int):
    """One episode with the hidden state pinned, recording belief and guess.

    Returns beliefs [T, n] (belief the policy acted on at each step), guesses
    [T], rewards [T] and the state grid. T is the episode cap.
    """
    cap = variants.episode_cap(variant)
    # VecNormalize -> DummyVecEnv -> Monitor -> ... -> OddEvenPOMDP.
    raw = venv.unwrapped.envs[0].unwrapped
    # `reset()` redraws the state only when config.true_state is None, so
    # pinning the config field here makes the next reset use this state.
    raw.config.true_state = int(true_state)
    try:
        venv.seed(seed)
        obs = venv.reset()
        assert int(raw.true_state) == int(true_state), raw.true_state
        points = raw.belief_points.copy()
        beliefs, guesses, rewards = [], [], []
        belief_now = raw.belief.copy()
        for _ in range(cap):
            action, _ = model.predict(obs, deterministic=True)
            beliefs.append(belief_now)
            obs, reward, dones, infos = venv.step(action)
            info = infos[0]
            guesses.append(int(info["predicted_state"]))
            rewards.append(float(reward[0]))
            belief_now = np.asarray(info["belief"], dtype=float).copy()
            if bool(dones[0]):
                break
    finally:
        raw.config.true_state = None
    return {
        "points": points,
        "beliefs": np.asarray(beliefs),
        "guesses": np.asarray(guesses, dtype=int),
        "rewards": np.asarray(rewards, dtype=float),
        "true_state": int(true_state),
        "n": int(raw.n_dist_size),
        "cap": cap,
    }


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _setup_axes(ax, points, n):
    bars = ax.bar(points, np.zeros_like(points, dtype=float),
                  color=BAR_COLOUR, width=0.8)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(list(points))
    ax.set_xticklabels([str(int(p)) for p in points], fontsize=8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("state")
    ax.set_ylabel("probability mass")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return bars


def _draw_step(ax, bars, roll, i):
    beliefs, guesses, rewards = roll["beliefs"], roll["guesses"], roll["rewards"]
    for bar, h in zip(bars, beliefs[i]):
        bar.set_height(h)
    guess = int(guesses[i])
    for label in ax.get_xticklabels():
        is_guess = int(label.get_text()) == guess
        label.set_color(GUESS_COLOUR if is_guess else "black")
        label.set_fontweight("bold" if is_guess else "normal")
        label.set_fontsize(11 if is_guess else 8)
    hit = "hit" if rewards[i] > 0 else "miss"
    ax.set_title(f"true state {roll['true_state']}    "
                 f"step {i + 1} of {roll['cap']}    "
                 f"policy guess: {guess} ({hit})", fontsize=12)


def _save(fig, draw, n_steps, out_path: Path, seconds_per_step: float,
          fps: int = 2, dpi: int = 120):
    """Hold each step for `seconds_per_step` by repeating its frame."""
    repeat = max(1, int(round(seconds_per_step * fps)))
    frame_ids = np.repeat(np.arange(n_steps), repeat)
    anim = animation.FuncAnimation(fig, draw, frames=frame_ids,
                                   interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p"])   # plays in every common player
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def animate_single(roll, out_path: Path, seconds_per_step: float):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14)
    bars = _setup_axes(ax, roll["points"], roll["n"])

    def draw(i):
        _draw_step(ax, bars, roll, i)
        return bars

    return _save(fig, draw, len(roll["beliefs"]), out_path, seconds_per_step)


def animate_stacked(rolls, out_path: Path, seconds_per_step: float):
    fig, axes = plt.subplots(len(rolls), 1, figsize=(14, 4.6 * len(rolls)))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.07,
                        hspace=0.5)
    bars_list = [_setup_axes(ax, r["points"], r["n"]) for ax, r in zip(axes, rolls)]
    n_steps = min(len(r["beliefs"]) for r in rolls)

    def draw(i):
        for ax, bars, roll in zip(axes, bars_list, rolls):
            _draw_step(ax, bars, roll, i)
        return [b for bars in bars_list for b in bars]

    return _save(fig, draw, n_steps, out_path, seconds_per_step)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    variants.add_variant_argument(ap, default="oe50_short")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vecnormalize_path", default=None,
                    help="vecnormalize.pkl saved next to the checkpoint")
    ap.add_argument("--true_states", type=int, nargs="+", default=[24, 25],
                    help="hidden states to pin, one episode each")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the observation stream and filter")
    ap.add_argument("--seconds_per_step", type=float, default=1.5)
    ap.add_argument("--out_dir", default=str(_HERE / "out"))
    ap.add_argument("--tag", default="cgf")
    ap.add_argument("--no_stacked", action="store_true")
    args = ap.parse_args(argv)
    if args.list_variants:
        variants.print_variants()
        return

    model, venv = load_policy_and_env(
        args.variant, args.model_path, args.vecnormalize_path, args.seed)

    out_dir = Path(args.out_dir)
    rolls = []
    for s in args.true_states:
        roll = rollout_with_policy(model, venv, args.variant, s, args.seed)
        rolls.append(roll)
        out = out_dir / f"{args.variant}_{args.tag}_true{s}_seed{args.seed}.mp4"
        animate_single(roll, out, args.seconds_per_step)
        print(f"video -> {out}")
        hits = int((roll["rewards"] > 0).sum())
        print(f"  true state {s}: guesses {roll['guesses'].tolist()}")
        print(f"  hits {hits} of {len(roll['rewards'])}")
    if len(rolls) > 1 and not args.no_stacked:
        states = "_".join(str(s) for s in args.true_states)
        out = out_dir / f"{args.variant}_{args.tag}_true{states}_seed{args.seed}_stacked.mp4"
        animate_stacked(rolls, out, args.seconds_per_step)
        print(f"video -> {out}")
    venv.close()


if __name__ == "__main__":
    main()
