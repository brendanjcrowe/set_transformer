"""
Evaluate an Odd-Even checkpoint on the RAW -(prediction - s*)^2 reward.

One eval script serves all three arms: 4_train_rl_{cgf,st,gaussian}.py read
the same {"obs", "particles", "weights"} dict observation, so a checkpoint
from any of them runs through this env unchanged.

THERE IS NO "SUCCESS" ON THIS DOMAIN, so the Ant-Tag success-rate metric does
not transfer. What is reported instead, and why each piece is load-bearing:

* **Mean reward per step, SPLIT at the collapse step.** The belief tightens to
  ~2 live candidates by about step 8 and keeps sharpening after that; the
  split separates the informative transient from the settled tail. So the
  pooled mean is largely a measurement of how much uninformed guessing the cap
  dilutes: at n=50 / cap 50 step 1 alone is about 18% of the oracle's pooled
  mean, and before reset() folded in its own observation it was 81%. Worse,
  the pooled mean is not comparable ACROSS CAPS -- the two n=50 variants share
  a protocol and have IDENTICAL transient columns, and their whole pooled
  difference (-0.941 against -0.259) is dilution. **Steady state is the
  headline.**
* **Exact-match rate and mean absolute error**, split the same way. Since
  2026-09-03 the env's reward IS the exact-match indicator, so the reward and
  the exact-match rate now measure the same thing; MAE remains the separate
  signal, saying how wrong the policy is when it misses.
* **The Bayes oracle and the play-the-previous-observation baseline**,
  measured on the SAME episodes. The oracle is info["optimal_prediction"],
  which under the current 0/1 exact-match reward is the posterior MODE.
  (Until 2026-09-03 the reward was -(pred - s*)^2 and that helper returned the
  posterior MEAN, which was Bayes-optimal for THAT rule. The env now derives
  it from the reward in force -- see OddEvenPOMDP.get_optimal_prediction --
  so this script needs no change to follow it, but any number below quoted
  from a squared-error run is not comparable to a current one.) The gap between the oracle's steady state (-0.329) and
  play-the-previous-observation (-9.750) is about 9.4 reward per step: that is
  the value of accumulating evidence, and it is exactly what a belief encoding
  either delivers or loses.

TWO SEEDING RULES, both from PITFALLS.md section 2 and both sharper here than
on Ant-Tag:

1. **Re-seed AFTER PPO.load.** The load restores the TRAINING seed from the
   checkpoint and BaseAlgorithm.set_random_seed re-seeds the env with it, so
   whatever --seed said is overwritten.
2. **Vary the seed per episode.** On this env the hidden state is drawn at
   reset, so THE SEED IS THE EPISODE: reset(seed=42) replays one episode byte
   for byte. A constant seed makes 100 episodes one episode counted 100 times,
   and the tell -- byte-identical results across "different" seeds -- looks
   like a robust policy.

The episode cap comes from the gym registration and num_particles comes off
the checkpoint's observation space. Neither is defaulted (PITFALLS.md
section 5).

Usage:
    python3 eval_scripts/eval_true_reward_odd_even.py --variant oe50 \
        --model_path runs/odd_even_cgf_oe50/<...>/models/cgf_agent.zip \
        --vecnormalize_path runs/odd_even_cgf_oe50/<...>/models/vecnormalize.pkl \
        --n_episodes 400

    # Baselines only, no checkpoint needed -- this reproduces the reference
    # table in domain_mds/oddeven.md. --oracle_only is an accepted alias.
    python3 eval_scripts/eval_true_reward_odd_even.py --variant oe50 \
        --baselines_only --n_episodes 400
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

# Depth matters (PITFALLS.md section 7). parents[3] is the package root from
# experiments/odd_even/eval_scripts/; parents[2] would be correct only at the
# top level of an experiment directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The pipeline scripts live one level up. Only this script's own directory is
# on sys.path by default, so add theirs -- and ONLY theirs. Inserting the
# sibling experiments/ant_tag/ would make `variants` and `4_train_rl_cgf`
# ambiguous flat names (Gap 12).
_ODD_EVEN_DIR = Path(__file__).resolve().parents[1]
if str(_ODD_EVEN_DIR) not in sys.path:
    sys.path.insert(0, str(_ODD_EVEN_DIR))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401,E402

# Siblings are loaded BY PATH through _sibling, not as flat names: `variants`
# names two different files (ant_tag has one too) and sys.modules is
# process-wide, so a plain `import variants` returns whichever was imported
# first anywhere in the process. See experiments/odd_even/_sibling.py.
import _sibling  # noqa: E402
variants = _sibling.load("variants")
make_odd_even_belief_env = _sibling.load(
    "odd_even_belief_env").make_odd_even_belief_env

#: Transient/steady boundary: the measured step at which the n=50 exact
#: posterior has locked on (max(belief) > 0.9). Steps 1..COLLAPSE_STEP are the
#: transient; COLLAPSE_STEP+1..cap are the steady state.
COLLAPSE_STEP = 21


def summarize_episode(rewards, collapse_step: int = COLLAPSE_STEP) -> dict:
    """Split one episode's per-step rewards into transient / steady / pooled.

    `rewards[0]` is step 1. The transient is steps 1..collapse_step and the
    steady state is everything after, so the two are disjoint and their
    lengths reconstruct the episode exactly.

    A steady segment can be EMPTY (an episode shorter than collapse_step), in
    which case its mean is NaN rather than 0.0 -- averaging a missing segment
    as zero would pull a very negative mean toward the oracle and read as
    improvement.
    """
    rewards = np.asarray(rewards, dtype=np.float64).ravel()
    transient = rewards[:collapse_step]
    steady = rewards[collapse_step:]
    return {
        "transient": float(transient.mean()) if transient.size else float("nan"),
        "steady": float(steady.mean()) if steady.size else float("nan"),
        "pooled": float(rewards.mean()) if rewards.size else float("nan"),
        "n_transient": int(transient.size),
        "n_steady": int(steady.size),
        "n_pooled": int(rewards.size),
    }


def _split_metric(values, collapse_step: int = COLLAPSE_STEP) -> dict:
    """Per-episode arrays -> transient / steady / pooled means over all steps.

    MEANS pool across episodes at the STEP level, not by averaging per-episode
    means: episodes here can differ in length, and a per-episode average
    would weight a short episode's steps more heavily.

    SEMs are PER EPISODE: the standard error of the per-episode means over
    the episodes that have any step in the split. Steps within an episode
    are not independent samples -- once the belief locks on, the policy
    repeats the same right or wrong guess, and 82% of oracle episodes have
    all nine steady rewards identical -- so a step-pooled SEM understated
    the uncertainty ~2.5x (0.009 vs 0.022 at 150 episodes; PITFALLS.md
    section 8 item 1). Every number quoted before 2026-09-06 used the
    step-pooled SEM; the means were unaffected.
    """
    transient, steady, pooled = [], [], []
    for row in values:
        row = np.asarray(row, dtype=np.float64).ravel()
        transient.append(row[:collapse_step])
        steady.append(row[collapse_step:])
        pooled.append(row)

    def _mean(chunks):
        flat = np.concatenate(chunks) if chunks else np.array([])
        return float(flat.mean()) if flat.size else float("nan")

    def _sem(chunks):
        per_episode = np.array([c.mean() for c in chunks if c.size], dtype=np.float64)
        return (float(per_episode.std(ddof=1) / np.sqrt(per_episode.size))
                if per_episode.size > 1 else float("nan"))

    return {
        "transient": _mean(transient), "transient_sem": _sem(transient),
        "steady": _mean(steady), "steady_sem": _sem(steady),
        "pooled": _mean(pooled), "pooled_sem": _sem(pooled),
    }


def _episode_metrics(rewards, predictions, true_states,
                     collapse_step: int = COLLAPSE_STEP) -> dict:
    """Reward, exact-match and absolute error, each split the same way."""
    exact, abs_err = [], []
    for preds, truth in zip(predictions, true_states):
        preds = np.asarray(preds, dtype=np.float64)
        exact.append((preds == float(truth)).astype(np.float64))
        abs_err.append(np.abs(preds - float(truth)))
    return {
        "reward": _split_metric(rewards, collapse_step),
        "exact_match": _split_metric(exact, collapse_step),
        "abs_error": _split_metric(abs_err, collapse_step),
    }


def run_reference_policies(variant: str, n_episodes: int, seed: int,
                           collapse_step: int = COLLAPSE_STEP) -> dict:
    """The Bayes oracle (info['optimal_prediction']) and play-the-previous-obs.

    Run on the RAW env, not the belief env: neither policy uses a particle
    filter, and both read what the env already reports in `info`. Deciding
    happens BEFORE each step's observation arrives, because `step()` converts
    the action to a prediction first and only then draws observations -- an
    oracle that peeked at the current step's observation would score about
    -1.098 per step at n=50, which no policy can reach.

    Uses `seed + episode_index`, so these are the same episodes the policy is
    evaluated on when it is given the same --seed.
    """
    resolved = variants.resolve(variant)
    cap = variants.episode_cap(variant)
    env = gym.make(resolved.env_id)
    results = {}
    for name in ("oracle", "prev_obs"):
        rewards, predictions, truths = [], [], []
        for episode in range(n_episodes):
            _obs, info = env.reset(seed=seed + episode)
            previous = int(np.asarray(info["observations"]).ravel()[-1])
            episode_rewards, episode_predictions = [], []
            for _step in range(cap):
                if name == "oracle":
                    prediction = int(info["optimal_prediction"])
                else:
                    prediction = previous
                _obs, reward, terminated, truncated, info = env.step(
                    prediction - 1)
                episode_rewards.append(float(reward))
                episode_predictions.append(float(prediction))
                previous = int(np.asarray(info["observations"]).ravel()[-1])
                if terminated or truncated:
                    break
            rewards.append(episode_rewards)
            predictions.append(episode_predictions)
            truths.append(int(info["true_state"]))
        results[name] = _episode_metrics(rewards, predictions, truths,
                                          collapse_step)
    env.close()
    return results


def _read_run_status(model_path: str) -> dict | None:
    """<model dir>/run_status.json written by train_odd_even (2026-09-06), or
    None for older runs. Also looks one directory up so best_model/*.zip and
    checkpoints/*.zip resolve to their run. Kept local so this script does not
    import the training module (and MuJoCo-free arms stay that way)."""
    import json
    here = Path(model_path).resolve().parent
    for directory in (here, here.parent):
        candidate = directory / "run_status.json"
        if candidate.exists():
            with open(candidate) as handle:
                return json.load(handle)
    return None


def _checkpoint_num_particles(model_path: str) -> int | None:
    """Particle-set size recorded in a saved policy's observation space.

    Read rather than defaulted: the env's particle count must match what the
    policy was trained with or SB3 rejects the observation space, and a wrong
    default silently changes what the encoder reads (PITFALLS.md section 5).
    """
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _params, _other = load_from_zip_file(
            model_path, load_data=True, device="cpu",
            print_system_info=False)
        return int(data["observation_space"]["particles"].shape[0])
    except Exception:  # noqa: BLE001 - best effort, never fatal
        return None


def evaluate_policy(model, env, variant: str, n_episodes: int, seed: int,
                    deterministic: bool = True,
                    collapse_step: int = COLLAPSE_STEP) -> dict:
    """Roll the loaded policy out, re-seeding the env EVERY episode.

    The per-episode reset seed is the whole point (see the module docstring).
    `env.seed(seed + episode)` is applied before each reset, so episode k is
    the same hidden state the reference policies saw for episode k -- which
    makes the three columns a paired comparison rather than three independent
    samples.
    """
    cap = variants.episode_cap(variant)
    rewards, predictions, truths = [], [], []
    for episode in range(n_episodes):
        # THE SEED IS THE EPISODE on this env. Re-seeding per episode is what
        # makes n_episodes into n samples instead of one sample repeated.
        env.seed(seed + episode)
        obs = env.reset()
        episode_rewards, episode_predictions = [], []
        true_state = None
        for _step in range(cap):
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = env.step(action)
            episode_rewards.append(float(reward[0]))
            # The env's own reward is what is being measured, so the
            # prediction is read back from info rather than recomputed from
            # the action -- the env owns the 0-indexed to 1-indexed shift.
            episode_predictions.append(float(infos[0]["predicted_state"]))
            true_state = int(infos[0]["true_state"])
            if bool(dones[0]):
                break
        rewards.append(episode_rewards)
        predictions.append(episode_predictions)
        truths.append(true_state)
    return _episode_metrics(rewards, predictions, truths, collapse_step)


def _print_block(name: str, metrics: dict, collapse_step: int) -> None:
    reward = metrics["reward"]
    exact = metrics["exact_match"]
    error = metrics["abs_error"]
    print(f"\n{name}")
    print(f"  {'':14s} {'steady (HEADLINE)':>20s} {'transient':>14s} "
          f"{'pooled':>14s}")
    print(f"  {'reward/step':14s} {reward['steady']:20.3f} "
          f"{reward['transient']:14.3f} {reward['pooled']:14.3f}")
    print(f"  {'exact match':14s} {exact['steady']:20.3f} "
          f"{exact['transient']:14.3f} {exact['pooled']:14.3f}")
    print(f"  {'abs error':14s} {error['steady']:20.3f} "
          f"{error['transient']:14.3f} {error['pooled']:14.3f}")
    print(f"  (reward SEM: steady {reward['steady_sem']:.3f}, "
          f"transient {reward['transient_sem']:.3f}, "
          f"pooled {reward['pooled_sem']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an Odd-Even checkpoint on the raw "
                    "-(prediction - s*)^2 reward, split transient/steady.")
    variants.add_variant_argument(parser)
    # Not argparse-required: --list_variants and --baselines_only must work
    # without a checkpoint, and argparse enforces required= first.
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--vecnormalize_path", type=str, default=None,
        help="vecnormalize.pkl saved during training. Loaded with "
             "norm_reward=False so the reported reward is RAW.")
    parser.add_argument("--n_episodes", type=int, default=400)
    parser.add_argument(
        "--num_particles", type=int, default=None,
        help="Defaults to the count baked into the checkpoint's observation "
             "space, which is the only value that can work.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--collapse_step", type=int, default=COLLAPSE_STEP,
        help=f"Transient/steady boundary (default {COLLAPSE_STEP}, the "
             "measured n=50 collapse step). Steps 1..this are the transient.")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic",
                        action="store_false")
    parser.add_argument(
        "--baselines_only", "--oracle_only", action="store_true",
        dest="baselines_only",
        help="Report just the Bayes oracle and the prev-obs "
             "reference. No checkpoint needed -- use it to reproduce the "
             "reference table in domain_mds/oddeven.md. --oracle_only is an "
             "alias: the flag reports BOTH references, so neither name is "
             "quite right on its own and both are accepted.")
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return
    if not args.baselines_only and args.model_path is None:
        parser.error("--model_path is required (or pass --baselines_only)")

    resolved = variants.resolve(args.variant)
    # From the registration, never a flag default (PITFALLS.md section 5).
    cap = variants.episode_cap(args.variant)
    print(f"Variant: {args.variant} | env: {resolved.env_id} | "
          f"n={resolved.n_dist_size} | cap={cap} | "
          f"filter: {resolved.particle_filter.__name__}")
    print(f"Episodes: {args.n_episodes}, seeds {args.seed}.."
          f"{args.seed + args.n_episodes - 1} (one per episode: on this env "
          "the hidden state is drawn at reset, so the seed IS the episode)")
    print(f"Transient = steps 1-{args.collapse_step}, "
          f"steady = {args.collapse_step + 1}-{cap}")

    references = run_reference_policies(
        args.variant, args.n_episodes, args.seed, args.collapse_step)
    _print_block("Bayes oracle (info['optimal_prediction'])",
                 references["oracle"], args.collapse_step)
    _print_block("play the previous observation",
                 references["prev_obs"], args.collapse_step)

    if args.baselines_only:
        return

    trained_particles = _checkpoint_num_particles(args.model_path)
    if args.num_particles is None:
        args.num_particles = trained_particles or resolved.n_dist_size
    elif (trained_particles is not None
          and args.num_particles != trained_particles):
        parser.error(
            f"--num_particles {args.num_particles} contradicts the "
            f"checkpoint, which was trained with {trained_particles}. Omit "
            "the flag.")

    env = DummyVecEnv([
        make_odd_even_belief_env(
            num_particles=args.num_particles,
            rank=0,
            seed=args.seed,
            variant=args.variant,
        )
    ])
    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        # The metric is the env's own reward. A normalized reward here would
        # be reported in units of a running standard deviation.
        env.norm_reward = False
        print(f"Loaded VecNormalize from {args.vecnormalize_path}")
    else:
        print("No VecNormalize -- evaluating without obs normalization. If "
              "training used it, this is INVALID: the policy sees a "
              "different input distribution than it was trained on.")

    model = PPO.load(args.model_path, env=env)
    print(f"Loaded model from {args.model_path}")
    run_status = _read_run_status(args.model_path)
    if run_status is None:
        print("  (no run_status.json beside it: a run from before 2026-09-06, "
              "or not written by train_odd_even; check its stdout.log "
              "reached 'Model saved')")
    elif run_status.get("status") != "completed":
        print("  WARNING: run_status.json says this run "
              f"{run_status.get('status', '?').upper()} at step "
              f"{run_status.get('timesteps')} of {run_status.get('total_timesteps')} "
              f"({run_status.get('error')}). The saved agent is a crash "
              "artefact, not a trained policy (PITFALLS.md section 8 item 7).")

    # RE-SEED AFTER THE LOAD. PPO.load restores the TRAINING seed and
    # BaseAlgorithm.set_random_seed re-seeds the env with it, overriding
    # --seed. Every eval of a checkpoint then replays the training episodes.
    # The per-episode reseeding in evaluate_policy is what actually varies
    # the episodes; this line makes --seed take effect at all.
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    print(f"Re-seeded to {args.seed} after PPO.load (the load had restored "
          "the checkpoint's training seed)")

    metrics = evaluate_policy(
        model, env, args.variant, args.n_episodes, args.seed,
        deterministic=args.deterministic, collapse_step=args.collapse_step)
    _print_block(f"policy ({Path(args.model_path).name}, "
                 f"deterministic={args.deterministic})",
                 metrics, args.collapse_step)

    oracle_steady = references["oracle"]["reward"]["steady"]
    prev_steady = references["prev_obs"]["reward"]["steady"]
    policy_steady = metrics["reward"]["steady"]
    span = oracle_steady - prev_steady
    print(f"\nSteady-state placement: oracle {oracle_steady:.3f}, "
          f"policy {policy_steady:.3f}, prev-obs {prev_steady:.3f}")
    if np.isfinite(span) and span != 0:
        # Where the policy sits on the oracle-to-naive span. This is the
        # quantity the encoder comparison is about: 0 means the belief bought
        # nothing, 1 means the encoding delivered the whole value of
        # accumulating evidence.
        print(f"  fraction of the oracle-to-naive span recovered: "
              f"{(policy_steady - prev_steady) / span:.3f}")
    env.close()


if __name__ == "__main__":
    main()
