"""
RL training with a weighted CGF belief encoder on the Odd-Even POMDP.

Step 4 of the pipeline, and the BASELINE arm of the encoder comparison:

    CGF_j = log(sum_i w_i * exp(<t_j, x_i>))

where x_i is a particle (a centred, scaled state value) and w_i is its
particle-filter weight. The t_j directions are learned under PPO.

This module owns the training loop, the CLI and the run bookkeeping.
4_train_rl_st.py and 4_train_rl_gaussian.py import it and swap ONLY the
features extractor, exactly as the three Ant-Tag arms do. All three read the
same {"obs", "particles", "weights"} dict observation from
odd_even_belief_env.make_odd_even_belief_env, so one eval script
(eval_scripts/eval_true_reward_odd_even.py) serves all three.

WEIGHTS ARE THE BELIEF HERE. The hidden state is static, so the particle
support never moves and every particle sits on a state value for the whole
episode. The exact posterior's effective sample size falls to about 1 of 50
within ~20 observations, which means an UNWEIGHTED reading of this particle
set sees the same near-uniform cloud over [1, n] at every step of every
episode and carries no information at all. This is a stronger version of the
Ant-Tag counterweighted-den case (ESS about 11 of 100).

--t_init_mode DEFAULTS TO spread_1d, not to the Ant-Tag default. On Ant-Tag
the spread init is what put the CGF features where the signal already was, so
PPO needed no ~10x growth of ||t_j|| to resolve it; the 0.1-scale linspace
init has to grow into place first. spread_1d is the 1-D analogue: log-spaced
magnitudes in BOTH signs, bounded at 2.0 because in 1-D the elementwise
t_clamp IS the norm bound. It is a separate mode from "spread", whose 8
planar directions and rho_hi=2.8 are intrinsically 2-D.

REWARD NORMALIZATION IS ON. -(pred - s*)^2 reaches -2401 at n=50, and
PITFALLS.md records a value loss of order 10^3 dominating a learned encoder's
parameters when the extractor is shared between the policy and value heads.
Training normalizes the reward; every eval reads the RAW reward.

Usage:
    python3 4_train_rl_cgf.py --variant oe50 --total_timesteps 500000 \
        --run_tag cgf_v1

    python3 4_train_rl_cgf.py --list_variants
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# This directory too, so `import _sibling` works when a test loads this file
# by path rather than running it as a script (where it is sys.path[0]).
# ONLY this directory -- never a sibling experiment directory (Gap 12).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401,E402 - registers the pdomains-odd-even-* envs
# `variants` is loaded BY PATH, not as a flat name: experiments/ant_tag/ has a
# variants.py too, and sys.modules is process-wide, so a plain
# `import variants` returns whichever one was imported first anywhere in the
# process. See _sibling.py -- this was measured happening in the test suite.
import _sibling  # noqa: E402
variants = _sibling.load("variants")
_belief_env = _sibling.load("odd_even_belief_env")
make_odd_even_belief_env = _belief_env.make_odd_even_belief_env
make_vec_env_from_fns = _belief_env.make_vec_env_from_fns
make_vec_normalize = _belief_env.make_vec_normalize

# The encoder pieces come FROM THE PACKAGE. Importing them from an Ant-Tag
# script would pull in MuJoCo and, worse, `4_train_rl_cgf` is now an ambiguous
# flat module name -- both experiment directories hold one (Gap 12).
from set_transformer.rl.feature_extractors.cgf import (  # noqa: E402
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
)


def _default_run_dir(seed: int, run_subdir: str,
                     run_tag: str | None = None) -> str:
    """runs/<run_subdir>/<timestamp>_seed<seed>[_<run_tag>]/.

    Parallel runs with different seeds and variants land in distinct,
    sortable, self-describing folders. run_tag is a free-form human label for
    eyeballing `ls`; it is NOT the source of truth for what a run used --
    that is run_config.json, written alongside with every CLI arg.

    Unlike the Ant-Tag scripts, run_subdir has no default here: it is always
    passed. Those scripts' train_* functions accept a run_subdir and then
    ignore it, so a programmatic call silently writes into the base variant's
    directory. That wart is not reproduced.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{re.sub(r'[^A-Za-z0-9._-]', '_', run_tag)}" if run_tag else ""
    return os.path.join("runs", run_subdir, f"{timestamp}_seed{seed}{suffix}")


def _write_run_config(run_dir: str, **config) -> None:
    """Dump every CLI arg to run_dir/run_config.json.

    The unambiguous record of what a run used. The run_tag in the directory
    name is a hint, not a record.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2, default=str, sort_keys=True)
    print(f"Run config saved to {path}")


def _git_provenance() -> dict:
    """Record WHICH CODE produced this run (PITFALLS.md section 6).

    run_config.json pins every hyperparameter but not the source that read
    them, and two runs with byte-identical configs have already given
    different results in this project because the code changed between them.
    HEAD alone does not close it -- both submodules are routinely dirty -- so
    the SHA-256 of `git diff HEAD` gives an uncommitted tree a stable
    identity. Never raises: a stripped checkout records an error string
    rather than killing a multi-hour run.
    """
    repos = {
        "set_transformer": Path(__file__).resolve().parents[2],
        "pomdp-domains": Path(__file__).resolve().parents[3] / "pomdp-domains",
    }

    def _git(repo: Path, *args: str) -> str:
        return subprocess.run(
            ("git", "-C", str(repo)) + args,
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout

    provenance = {}
    for name, repo in repos.items():
        try:
            head = _git(repo, "rev-parse", "HEAD").strip()
            status = [line for line in
                      _git(repo, "status", "--porcelain").splitlines() if line]
            diff = _git(repo, "diff", "HEAD")
            provenance[name] = {
                "path": str(repo),
                "head": head,
                "dirty": bool(status),
                "diff_sha256": (hashlib.sha256(diff.encode()).hexdigest()
                                if diff else None),
                "status": status,
            }
        except Exception as exc:  # noqa: BLE001 - never abort a run
            provenance[name] = {"path": str(repo),
                                "error": f"{type(exc).__name__}: {exc}"}
    return provenance


class _TeeStream:
    """Duplicates writes to several streams (real stdout + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _tee_stdout_stderr(log_path: str) -> None:
    """Mirror stdout/stderr into log_path as well as the console.

    A background (nohup) run's console output then lands next to that run's
    TensorBoard and model output, instead of depending on the caller to
    redirect into a path that has to be matched back to a run directory
    later.
    """
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)


def build_envs(
    variant: str,
    num_particles: int,
    n_envs: int,
    seed: int,
    monitor_dir: str | None,
    use_vec_normalize: bool,
    particle_filter_class: type | None = None,
):
    """Training and eval vec envs, built from the ONE shared env factory.

    The eval env differs in exactly two ways, both required:

    * `norm_reward=False`, so EvalCallback compares checkpoints on the RAW
      -(pred - s*)^2 reward. A normalized eval reward moves as the running
      statistics move, which makes "best_model" selection a comparison
      against a drifting yardstick.
    * `training=False`, so evaluating does not feed the observation
      statistics that training relies on.

    Its rank is offset past every training worker so it never replays a
    training worker's episode set -- which on this env means never replaying
    a hidden state.
    """
    env_kw = dict(
        variant=variant,
        num_particles=num_particles,
        particle_filter_class=particle_filter_class,
    )
    env_fns = [
        make_odd_even_belief_env(**env_kw, rank=rank, seed=seed,
                                 monitor_dir=monitor_dir)
        for rank in range(n_envs)
    ]
    vec_env = make_vec_env_from_fns(env_fns, n_envs)
    if use_vec_normalize:
        vec_env = make_vec_normalize(vec_env, training=True, norm_reward=True)

    eval_vec_env = DummyVecEnv([
        make_odd_even_belief_env(**env_kw, rank=n_envs + 1, seed=seed)
    ])
    if use_vec_normalize:
        eval_vec_env = make_vec_normalize(eval_vec_env, training=False,
                                          norm_reward=False)
    return vec_env, eval_vec_env


def train_odd_even(
    policy_kwargs: dict,
    encoder: str = "cgf",
    variant: str = "oe50",
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    ppo_n_steps: int = 2048,
    n_epochs: int = 10,
    num_particles: int = 50,
    particle_filter_class: type | None = None,
    device: str = "cpu",
    seed: int = 0,
    run_subdir: str | None = None,
    log_dir: str | None = None,
    model_save_path: str | None = None,
    eval_freq: int = 20_000,
    save_freq: int = 100_000,
    n_eval_episodes: int = 20,
    use_vec_normalize: bool = True,
    lr_anneal: bool = False,
    target_kl: float | None = None,
    progress_bar: bool = False,
    extra_callbacks: list | None = None,
    post_construct=None,
):
    """The shared PPO loop. Every arm calls this with its own policy_kwargs.

    Args:
        policy_kwargs: Must carry `features_extractor_class` and its kwargs.
            This is the ONLY thing that differs between the three arms.
        post_construct: Optional callable(model) run immediately after
            PPO(...) returns. This is where a pretrained encoder is RELOADED:
            SB3's ActorCriticPolicy._build ends with
            self.apply(partial(self.init_weights, ...)), which walks the whole
            policy INCLUDING the features extractor and re-initializes every
            Linear. See PITFALLS.md section 1 -- it cost two 6M-step runs.
        run_subdir: Where under runs/ this goes. Wired through properly, not
            ignored: the Ant-Tag train_* functions accept this parameter and
            then always call their own _default_run_dir(seed), so a
            programmatic call lands in the base variant's directory.
    """
    resolved = variants.resolve(variant)
    if particle_filter_class is None:
        particle_filter_class = resolved.particle_filter
    run_subdir = run_subdir or variants.run_subdir(encoder, variant)

    if log_dir is None or model_save_path is None:
        run_dir = _default_run_dir(seed, run_subdir)
        if log_dir is None:
            log_dir = os.path.join(run_dir, "logs") + "/"
        if model_save_path is None:
            model_save_path = os.path.join(
                run_dir, "models", f"{encoder}_agent.zip")

    print(f"Training PPO on {resolved.env_id} with {encoder} belief features")
    print(f"  variant={variant} n={resolved.n_dist_size} "
          f"cap={variants.episode_cap(variant)} "
          f"filter={particle_filter_class.__name__}")
    print(f"  num_particles={num_particles} device={device} seed={seed}")
    print(f"  log_dir={log_dir}")
    print(f"  model_save_path={model_save_path}")

    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    vec_env, eval_vec_env = build_envs(
        variant=variant,
        num_particles=num_particles,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir,
        use_vec_normalize=use_vec_normalize,
        particle_filter_class=particle_filter_class,
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=(
            (lambda remaining: learning_rate * remaining) if lr_anneal
            else learning_rate),
        n_steps=ppo_n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        target_kl=target_kl,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
        policy_kwargs=policy_kwargs,
        device=device,
    )

    if post_construct is not None:
        post_construct(model)

    callbacks = [
        CheckpointCallback(
            # Divided by n_envs per SB3's convention: a callback counts CALLS,
            # and each call advances n_envs environment steps.
            save_freq=max(save_freq // n_envs, 1),
            save_path=(os.path.join(model_dir, "checkpoints") if model_dir
                       else "checkpoints"),
            name_prefix=f"odd_even_{encoder}",
        ),
        EvalCallback(
            eval_vec_env,
            best_model_save_path=(os.path.join(model_dir, "best_model")
                                  if model_dir else "best_model"),
            log_path=log_dir,
            eval_freq=max(eval_freq // n_envs, 1),
            deterministic=True,
            render=False,
            n_eval_episodes=n_eval_episodes,
        ),
    ]
    callbacks.extend(extra_callbacks or [])

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    progress_bar=progress_bar)
    finally:
        model.save(model_save_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vecnorm_path = (os.path.join(model_dir, "vecnormalize.pkl")
                            if model_dir else "vecnormalize.pkl")
            vec_env.save(vecnorm_path)
            print(f"VecNormalize saved to {vecnorm_path}")
        print(f"Model saved to {model_save_path}")
        vec_env.close()
        eval_vec_env.close()
    return model


def add_common_arguments(parser, encoder: str) -> None:
    """The flags every arm shares. Each arm adds only its encoder's own."""
    variants.add_variant_argument(parser)
    parser.add_argument(
        "--particle_filter", type=str, default=None,
        choices=sorted(variants.PARTICLE_FILTERS),
        help="Override the variant's filter. The bootstrap filter is a "
             "deliberate arm (a lossier belief on the same env).")
    parser.add_argument(
        "--run_subdir", type=str, default=None,
        help=f"Override the derived runs/odd_even_{encoder}_<variant> "
             "directory, for a sweep that needs its own tree.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_tag", type=str, default=None,
        help="Short human label appended to the run directory name, so "
             "parallel runs are distinguishable by eye. Not a substitute "
             "for run_config.json.")
    parser.add_argument(
        "--num_particles", type=int, default=None,
        help="Filter set size. Defaults to the variant's n_dist_size, which "
             "makes the exact-support filter's belief EXACT (one particle "
             "per state) and so isolates encoder loss from filter loss.")
    parser.add_argument(
        "--arena_scale", type=float, default=None,
        help="Particle normalization. Defaults to the state-range half-width "
             "(n-1)/2 from the registry, which maps the states onto about "
             "[-1, 1]. It MUST match the particle_scale recorded in a "
             "pretraining dataset (PITFALLS.md section 4).")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="cpu by default: the encoders here are tiny and the env is "
             "pure Python, so a GPU adds transfer latency without work.")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--no_vec_normalize", action="store_true")
    parser.add_argument(
        "--net_arch", type=str, default=None,
        help="Policy/value MLP sizes, e.g. '256,256'.")
    parser.add_argument(
        "--lr_anneal", action="store_true",
        help="Linearly anneal learning_rate to 0 over training.")
    parser.add_argument(
        "--target_kl", type=float, default=None,
        help="PPO target_kl early-stop threshold per rollout.")
    parser.add_argument("--progress_bar", action="store_true")


def resolve_common(args, encoder: str):
    """Fill in the registry-derived defaults and set up the run directory.

    Shared by all three arms so the run layout, the tee'd log and the
    run_config record cannot drift between them.

    Returns:
        (run_dir, log_dir, model_save_path, run_subdir, resolved_kwargs).
    """
    resolved = variants.resolve(args.variant)
    particle_filter_class = variants.resolve_particle_filter(
        args.variant, args.particle_filter)
    if args.num_particles is None:
        args.num_particles = resolved.n_dist_size
    if args.arena_scale is None:
        args.arena_scale = variants.state_scale(args.variant)
    run_subdir = args.run_subdir or variants.run_subdir(encoder, args.variant)

    run_dir = _default_run_dir(args.seed, run_subdir, run_tag=args.run_tag)
    log_dir = args.log_dir or os.path.join(run_dir, "logs") + "/"
    model_save_path = args.model_save_path or os.path.join(
        run_dir, "models", f"{encoder}_agent.zip")

    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, "stdout.log")
    _tee_stdout_stderr(stdout_log_path)
    print(f"Mirroring stdout/stderr to {stdout_log_path}")

    run_config = vars(args).copy()
    run_config.pop("run_subdir", None)
    run_config.pop("list_variants", None)
    run_config.update(log_dir=log_dir, model_save_path=model_save_path)
    # env_id / filter / cap are derived from --variant, but are written out
    # anyway: run_config.json stays a complete record even if the registry
    # entry is later edited.
    _write_run_config(
        run_dir,
        encoder=encoder,
        env_id=resolved.env_id,
        n_dist_size=resolved.n_dist_size,
        episode_cap=variants.episode_cap(args.variant),
        particle_filter_class=particle_filter_class.__name__,
        run_subdir=run_subdir,
        git=_git_provenance(),
        **run_config,
    )
    return run_dir, log_dir, model_save_path, run_subdir, particle_filter_class


def net_arch_from_args(args):
    return ([int(x) for x in args.net_arch.split(",")]
            if args.net_arch else None)


def main() -> None:
    """Entry point for the CGF arm."""
    encoder = "cgf"
    parser = argparse.ArgumentParser(
        description="RL with weighted CGF particle-belief features on the "
                    "Odd-Even POMDP. --variant picks the env; "
                    "--list_variants shows them.")
    add_common_arguments(parser, encoder)
    parser.add_argument("--num_cgf_features", type=int, default=64)
    parser.add_argument(
        "--t_init_mode", type=str, default="spread_1d",
        choices=["spread_1d", "linspace_all_dims", "linspace_first_dim",
                 "random"],
        help="spread_1d by default: log-spaced magnitudes in both signs, "
             "which starts where the signal already is. The 0.1-scale "
             "linspace inits have to grow ||t|| ~10x into place first. "
             "'spread' is NOT offered -- its 8 planar directions are "
             "intrinsically 2-D and it rejects 1-D particles.")
    parser.add_argument("--t_init_scale", type=float, default=0.1)
    parser.add_argument("--t_clamp", type=float, default=2.0)
    parser.add_argument("--exp_arg_clamp", type=float, default=20.0)
    parser.add_argument(
        "--t_frozen", action="store_true",
        help="Register t_values as a BUFFER, so PPO cannot learn the "
             "projection directions. With --t_init_mode spread_1d this "
             "isolates representational CAPACITY from the optimization "
             "dynamics of t_j growth.")
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    (_run_dir, log_dir, model_save_path, run_subdir,
     particle_filter_class) = resolve_common(args, encoder)

    policy_kwargs = {
        "features_extractor_class": WeightedCGFFeaturesExtractor,
        "features_extractor_kwargs": dict(
            num_cgf_features=args.num_cgf_features,
            arena_scale=args.arena_scale,
            t_init_mode=args.t_init_mode,
            t_init_scale=args.t_init_scale,
            t_clamp=args.t_clamp,
            exp_arg_clamp=args.exp_arg_clamp,
            t_frozen=args.t_frozen,
        ),
    }
    net_arch = net_arch_from_args(args)
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    train_odd_even(
        policy_kwargs=policy_kwargs,
        encoder=encoder,
        variant=args.variant,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        n_epochs=args.n_epochs,
        num_particles=args.num_particles,
        particle_filter_class=particle_filter_class,
        device=args.device,
        seed=args.seed,
        run_subdir=run_subdir,
        log_dir=log_dir,
        model_save_path=model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        n_eval_episodes=args.n_eval_episodes,
        use_vec_normalize=not args.no_vec_normalize,
        lr_anneal=args.lr_anneal,
        target_kl=args.target_kl,
        progress_bar=args.progress_bar,
        # cgf/t_norm_q* in TensorBoard: whether PPO actually grows ||t_j|| is
        # a first-class experimental question, and a FLAT line is the
        # built-in sanity check for --t_frozen.
        extra_callbacks=[TNormLoggingCallback()],
    )


if __name__ == "__main__":
    main()
