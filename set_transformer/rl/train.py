"""The one RL training function and its command line.

Change 4 of the harness centralisation (``refactor_plans.md`` in the parent repo,
2026-09-12). Before it, the PPO loop existed five times -- ``experiments/ant_tag/4_train_rl_
{cgf,st,gaussian,pool}.py`` and ``experiments/odd_even/4_train_rl_cgf.train_odd_even`` (shared
by the three Odd-Even arms) -- and differed only in the encoder, the domain's env factory
and curriculum, and a few extras one copy had grown (resume from a checkpoint, SAC, the
run-status record, VecNormalize snapshots beside checkpoints). :func:`train` is that loop
once, reading a :class:`~set_transformer.rl.domains.base.Domain` record for the problem and
an :class:`~set_transformer.rl.encoders.Encoder` record for the belief encoder. :func:`main`
is the one command line::

    python -m set_transformer.rl.train --domain ant_tag --encoder st --variant smart ...
    python -m set_transformer.rl.train --list_encoders
    python -m set_transformer.rl.train --domain odd_even --encoder cgf --list_variants

The numbered scripts call ``main(argv, domain=..., encoder=..., legacy_layout=True)`` once
they are switched (change 4e), so every recorded command keeps working.

What the loop does, in order, for every domain and encoder: resolve the run directory; build
``n_envs`` training envs through the domain's factory (SubprocVecEnv above one worker) and
one eval env at rank ``n_envs + 1`` (Ant-Tag: real visibility radius, no shaping);
VecNormalize on the ``obs`` key only (reward normalised in training, raw in eval); PPO (or
SAC) with the encoder as the features extractor; the post-construction steps -- reload the
pretrained encoder and verify it (PITFALLS.md section 1), scale its learning rate, arm the
unfreeze step, warm-start the whole policy from a donor zip; the callbacks -- checkpoints with
VecNormalize snapshots, evaluation, the domain's schedules, the encoder's logging, the
domain's per-encoder extras, the finetune loggers; ``learn``; and on ANY exit save the model,
the VecNormalize statistics and ``run_status.json``.

Run directory (plan section 2b): when ``log_dir`` / ``model_save_path`` are not given, the
root-level layout ``<output root>/<domain>/<variant>/rl/<encoder>/<timestamp>_seed<seed>[_<tag>]``
(``run_records.output_root`` resolves the root: ``--output_root`` > ``$RL_BMDP_RUNS`` > the
parent repo's ``runs/`` > this checkout's ``runs/``; never the current directory). Since
change 5.2 the numbered scripts use it too; ``legacy_layout=True`` (the cwd-relative
``runs/<run_subdir>/...`` of the pre-2026-09-12 scripts) remains available to callers.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from set_transformer.rl import domains as _domains
from set_transformer.rl import encoders as _encoders
from set_transformer.rl import run_records
from set_transformer.rl.curriculum import Schedule, ScheduleCallback
from set_transformer.rl.domains.base import Domain
from set_transformer.rl.encoder_finetune import (
    EncoderLRLoggingCallback,
    UnfreezeEncoderCallback,
    scale_encoder_learning_rate,
)
from set_transformer.rl.encoders import Encoder
from set_transformer.rl.pretrained_encoder import policy_extractors, reload_pretrained
from set_transformer.rl.wrappers.obs_history import checkpoint_obs_history_spec, with_obs_history

ALGORITHMS = {"PPO": PPO, "SAC": SAC}


# ---------------------------------------------------------------------------
# The training function
# ---------------------------------------------------------------------------


def train(
    domain: Domain | str,
    variant: str,
    encoder: Encoder | str,
    *,
    # the policy
    features_extractor_kwargs: dict,
    net_arch: Sequence[int] | None = None,
    # the start mode
    pretrained_path: str | None = None,
    frozen: bool = False,
    encoder_lr_scale: float = 1.0,
    unfreeze_at: int | None = None,
    init_policy: str | None = None,
    # the environment
    num_particles: int,
    particle_filter_class: type | None = None,
    env_options: dict | None = None,
    n_envs: int = 4,
    seed: int = 0,
    use_vec_normalize: bool = True,
    # frame stacking (the framestack arm only; 1 = no wrapper, every other arm unchanged)
    obs_history: int = 1,
    obs_history_padding: str = "reset_frame",
    # the schedules (the domain's, with this run's waypoints) and encoder-side options
    schedules: Sequence[Schedule] = (),
    encoder_options: dict | None = None,
    # the algorithm
    algorithm: str = "PPO",
    total_timesteps: int,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    ppo_n_steps: int = 2048,
    n_epochs: int = 10,
    target_kl: float | None = None,
    ent_coef: float = 0.0,
    separate_extractors: bool = False,
    lr_anneal: bool = False,
    device: str = "cpu",
    # the run
    log_dir: str | None = None,
    model_save_path: str | None = None,
    run_tag: str | None = None,
    output_root: str | os.PathLike | None = None,
    eval_freq: int = 20_000,
    save_freq: int = 100_000,
    n_eval_episodes: int = 20,
    progress_bar: bool = False,
    # resuming
    resume_from: str | None = None,
    resume_vecnormalize: str | None = None,
    # hooks for callers that need more (tests; the numbered scripts' wrappers)
    extra_callbacks: Sequence = (),
    post_construct=None,
):
    """Train one agent. Returns the SB3 model. See the module docstring for the steps.

    ``features_extractor_kwargs`` are the extractor's constructor arguments (the command line
    builds them with ``Encoder.extractor_kwargs``); ``env_options`` is what the domain's
    ``resolve_arguments`` returned; ``schedules`` are ``Schedule`` records with this run's
    waypoints (``Domain.schedules(args)``); ``encoder_options`` steer the encoder's callbacks
    (CGF: ``running_norm_update``). ``ent_coef`` is PPO's entropy bonus (0.0 = SB3's default);
    ``separate_extractors`` gives the value network its own features extractor
    (``share_features_extractor=False``, the recorded hunt configuration) instead of sharing
    the actor's; both PPO only (batch 9.1, 2026-09-13). ``obs_history`` is how many consecutive
    base observations the ``"obs"`` key carries (``Encoder.obs_history``; 1 = the current frame
    only and NO wrapper is built, which is every arm but ``framestack``);
    ``obs_history_padding`` is what fills the history slots at reset
    (``Encoder.obs_history_padding``: ``reset_frame`` or ``zeros``; unused at 1).
    ``post_construct(model)`` runs after every built-in post-construction step.
    """
    domain = _domains.get(domain)
    enc = _encoders.get(encoder)
    resolved = domain.resolve(variant)
    env_id = resolved.env_id
    if particle_filter_class is None:
        particle_filter_class = resolved.particle_filter
    env_options = dict(env_options or {})
    encoder_options = dict(encoder_options or {})
    schedules = tuple(schedules)
    algorithm = algorithm.upper()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if (pretrained_path or frozen or unfreeze_at is not None or encoder_lr_scale != 1.0) \
            and not enc.learned:
        raise ValueError(f"encoder {enc.name!r} has no parameters: nothing to pretrain, "
                         "freeze or finetune")
    if frozen and not pretrained_path:
        raise ValueError("frozen=True without a pretrained checkpoint would freeze a RANDOM "
                         "encoder")
    if algorithm != "PPO" and (ent_coef != 0.0 or separate_extractors):
        raise ValueError("--ent_coef and --separate_extractors are PPO options (SAC tunes its "
                         "entropy coefficient itself and never shares the extractor)")
    if init_policy and (pretrained_path or resume_from):
        raise ValueError("--init_policy warm-starts the WHOLE policy from a saved agent; it "
                         "cannot be combined with a pretrained encoder or --resume_from")
    if resume_from:
        # Fork a run from one of its checkpoints (Ant-Tag ST, 2026-09-08): the policy, value
        # head, encoder, Adam moments (when the zip has them) and the step counter come from
        # the zip, the obs-normalization statistics from the matching VecNormalize snapshot,
        # and training continues to `total_timesteps` (the FULL horizon) with every
        # progress-based schedule -- curriculum, reward, evasion, LR anneal -- evaluated at
        # the resumed progress. Not restored: env RNG state, the rollout buffer, PPO's
        # sampling RNG. A fork is not a bit-exact continuation; compare forks with forks.
        if algorithm != "PPO":
            raise ValueError("--resume_from is implemented for PPO only")
        if pretrained_path or frozen:
            raise ValueError("--resume_from restores the encoder from the checkpoint; "
                             "--pretrained_path / --frozen do not apply")
        if use_vec_normalize and not resume_vecnormalize:
            raise ValueError("--resume_from with VecNormalize needs the matching "
                             "--resume_vecnormalize snapshot")

    if log_dir is None or model_save_path is None:
        run_dir = str(run_records.run_dir(domain.name, variant, enc.name, seed, run_tag,
                                          root=output_root))
        if log_dir is None:
            log_dir = os.path.join(run_dir, "logs") + "/"
        if model_save_path is None:
            model_save_path = os.path.join(run_dir, "models", f"{enc.name}_agent.zip")

    print(f"Training {algorithm} on {env_id} ({domain.name}/{variant}) with "
          f"{enc.extractor_class.__name__} features")
    print(f"  encoder kwargs: {features_extractor_kwargs}")
    print(f"  num_particles={num_particles} filter={particle_filter_class.__name__} "
          f"n_envs={n_envs} device={device} seed={seed}")
    if schedules:
        print("  schedules: " + ", ".join(f"{s.name}->{s.target}" for s in schedules))
    print(f"  log_dir={log_dir}")
    print(f"  model_save_path={model_save_path}")

    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    # -- envs: training workers, then the eval env at the next rank -------------------------
    common = dict(num_particles=num_particles, particle_filter_class=particle_filter_class,
                  seed=seed, options=env_options)
    # `with_obs_history` returns the thunk UNCHANGED at obs_history == 1 (every arm but
    # framestack), so no wrapper object exists and the env is the one of every recorded run.
    env_fns = [
        with_obs_history(
            domain.make_env(variant, rank=rank, monitor_dir=monitor_dir, training=True, **common),
            obs_history, obs_history_padding)
        for rank in range(n_envs)
    ]
    vec_env = domain.make_vec_env_from_fns(env_fns, n_envs)
    if use_vec_normalize:
        if resume_from:
            vec_env = VecNormalize.load(resume_vecnormalize, vec_env)
            vec_env.training = True
            print(f"VecNormalize statistics RESUMED from {resume_vecnormalize} "
                  f"(norm_obs_keys={vec_env.norm_obs_keys}, norm_reward={vec_env.norm_reward}, "
                  f"obs count={float(vec_env.obs_rms['obs'].count):.0f})")
        else:
            vec_env = domain.make_vec_normalize(vec_env, training=True, norm_reward=True)
    # Always evaluate at the env's real difficulty and on its true reward (the domain's
    # factory does that for training=False), so the metric does not depend on where the
    # curriculum currently is. EvalCallback syncs the training VecNormalize statistics into
    # this one before every eval.
    eval_vec_env = DummyVecEnv([
        with_obs_history(
            domain.make_env(variant, rank=n_envs + 1, monitor_dir=None, training=False, **common),
            obs_history, obs_history_padding)
    ])
    if use_vec_normalize:
        eval_vec_env = domain.make_vec_normalize(eval_vec_env, training=False, norm_reward=False)

    # -- the model ---------------------------------------------------------------------------
    policy_kwargs = {
        "features_extractor_class": enc.extractor_class,
        "features_extractor_kwargs": dict(features_extractor_kwargs),
    }
    if net_arch is not None:
        policy_kwargs["net_arch"] = list(net_arch)
    if separate_extractors:
        # One extractor for the actor, another for the value network: SB3 builds the second
        # with the same class and kwargs; the pretrained reload / freeze / lr-scale steps below
        # enumerate both (rl/pretrained_encoder.policy_extractors).
        policy_kwargs["share_features_extractor"] = False
    lr_arg = ((lambda progress_remaining: learning_rate * progress_remaining) if lr_anneal
              else learning_rate)
    if resume_from:
        model = PPO.load(
            resume_from, env=vec_env, device=device, force_reset=True,
            # The zip carries the source run's schedule object; rebuild it from THIS run's
            # flags so run_config.json and the optimizer agree.
            custom_objects={"learning_rate": lr_arg, "lr_schedule": lr_arg},
        )
        mismatched = {
            k: (getattr(model, k), v)
            for k, v in dict(n_steps=ppo_n_steps, batch_size=batch_size, n_epochs=n_epochs,
                             target_kl=target_kl, ent_coef=ent_coef, seed=seed).items()
            if getattr(model, k) != v
        }
        # n_stack is NOT a PPO attribute (`getattr(model, "n_stack")` raises on every
        # checkpoint), so it cannot join the dict above. It travels inside the zip's
        # policy_kwargs and is compared separately. Every arm but framestack has k == 1 on
        # both sides: nothing is added and the message is the one of every recorded fork
        # (2026-09-22, change_mds/framestack_arm_2026-09-22.md, plan section 11A). The padding
        # (2026-09-23) travels beside it and is compared the same way; every arm but
        # framestack is "reset_frame" on both sides.
        stored = checkpoint_obs_history_spec(resume_from)
        if stored.n_stack != int(obs_history):
            mismatched["n_stack"] = (stored.n_stack, int(obs_history))
        if stored.padding != str(obs_history_padding):
            mismatched["padding"] = (stored.padding, str(obs_history_padding))
        if mismatched:
            raise ValueError(f"--resume_from checkpoint disagrees with the CLI on "
                             f"{mismatched} (stored, given); pass the source run's values")
        if model.num_timesteps >= total_timesteps:
            raise ValueError(f"checkpoint is at {model.num_timesteps:,} steps, "
                             f"--total_timesteps {total_timesteps:,} must be the FULL horizon "
                             "beyond it")
        # PPO.load restores the SOURCE run's tensorboard_log; point it here.
        model.tensorboard_log = log_dir
        progress = model.num_timesteps / total_timesteps
        print(f"PPO RESUMED from {resume_from}: {model.num_timesteps:,} env steps done "
              f"(progress {progress:.3f}), {model._n_updates} updates; training "
              f"{total_timesteps - model.num_timesteps:,} more steps to {total_timesteps:,}; "
              f"learning rate resumes at {model.lr_schedule(1.0 - progress):.2e}"
              + ("" if model.policy.optimizer.state else
                 " (optimizer moments were not in the zip: Adam restarts)"))
    elif algorithm == "PPO":
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=lr_arg,
            n_steps=ppo_n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            target_kl=target_kl,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=device,
        )
    else:
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=device,
        )

    # -- post-construction: the steps SB3's own construction would undo or cannot do ---------
    finetune_callbacks = []
    if pretrained_path:
        # SB3's _build re-initialises every Linear in the extractor AFTER the constructor
        # loaded them; reload, re-freeze (also for the unfreeze-later mode, which starts
        # frozen) and assert max|delta| == 0 (PITFALLS.md section 1). Also scrubs the
        # absolute path from the saved policy_kwargs.
        reload_pretrained(model, pretrained_path, frozen or unfreeze_at is not None)
    if encoder_lr_scale != 1.0:
        # Finetune-collapse fix 1: the encoder in its own param group at a scaled rate. Must
        # run AFTER the reload and after construction -- it rebuilds the optimizer over the
        # live parameters.
        if algorithm != "PPO":
            raise ValueError("--encoder_lr_scale is implemented for PPO only")
        scale_encoder_learning_rate(model, encoder_lr_scale)
        finetune_callbacks.append(EncoderLRLoggingCallback())
    if unfreeze_at is not None:
        print(f"{enc.extractor_class.__name__}: encoder frozen until step {unfreeze_at:,}, "
              "then finetuned")
        finetune_callbacks.append(UnfreezeEncoderCallback(unfreeze_at))
    if enc.learned:
        n_encoder = sum(p.numel() for p in model.policy.features_extractor.encoder_parameters())
        n_extractors = len(policy_extractors(model))
        print(f"{enc.extractor_class.__name__}: {n_encoder:,} encoder parameters"
              + (f" (x{n_extractors}: separate actor / critic extractors)"
                 if n_extractors > 1 else ""))
    if init_policy:
        # Warm-start the whole policy (extractor + MLP + heads) from a compatible saved agent
        # (Brendan's benchmark option). Strict load: a silent shape mismatch here would look
        # like a normal-but-bad run.
        donor = ALGORITHMS[algorithm].load(init_policy, device=device)
        model.policy.load_state_dict(donor.policy.state_dict())
        print(f"Policy initialised from {init_policy}")
    if post_construct is not None:
        post_construct(model)

    # -- callbacks, in the order every arm had them -------------------------------------------
    checkpoint_cb = CheckpointCallback(
        # Divided by n_envs per SB3's convention: a callback counts CALLS, and each call
        # advances n_envs environment steps.
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(model_dir, "checkpoints") if model_dir else "checkpoints",
        name_prefix=f"{domain.name}_{enc.name}",
        # Snapshot VecNormalize with every checkpoint: without it the checkpoints cannot be
        # evaluated faithfully, and a run killed early is a total loss (PITFALLS.md section 7).
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(model_dir, "best_model") if model_dir else "best_model",
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
    )
    callbacks = [checkpoint_cb, eval_cb]
    if schedules:
        callbacks.append(ScheduleCallback(total_timesteps, schedules, verbose=1))
    callbacks.extend(enc.callbacks(dict(features_extractor_kwargs), encoder_options))
    callbacks.extend(domain.encoder_callbacks(enc.name))
    callbacks.extend(finetune_callbacks)
    callbacks.extend(extra_callbacks)

    # -- learn; save on ANY exit -----------------------------------------------------------
    # The finally: block saves on any exit, so a run that died in its first rollout leaves a
    # directory shaped like a finished one (PITFALLS.md section 8 item 7). run_status.json
    # beside the model records which it was; the eval script warns on anything but
    # "completed".
    completed, error = False, None
    try:
        # SB3 adds num_timesteps to the requested total when the counter is not reset, so a
        # resumed run asks for the REMAINING steps and every progress_remaining-based
        # schedule (LR anneal) continues from where the checkpoint left off; ScheduleCallback
        # divides num_timesteps by the full horizon it was given above.
        model.learn(
            total_timesteps=total_timesteps - (model.num_timesteps if resume_from else 0),
            reset_num_timesteps=not resume_from,
            callback=callbacks,
            progress_bar=progress_bar,
        )
        completed = True
    except BaseException as exc:      # noqa: B902 - recorded, then re-raised
        error = exc
        raise
    finally:
        try:
            model.save(model_save_path)
            if use_vec_normalize and isinstance(vec_env, VecNormalize):
                vecnorm_path = (os.path.join(model_dir, "vecnormalize.pkl") if model_dir
                                else "vecnormalize.pkl")
                vec_env.save(vecnorm_path)
                print(f"VecNormalize saved to {vecnorm_path}")
        finally:
            status = run_records.write_run_status(
                model_save_path, completed=completed, error=error,
                timesteps=int(getattr(model, "num_timesteps", 0)),
                total_timesteps=int(total_timesteps))
            if completed:
                print(f"Model saved to {model_save_path}")
            else:
                print(f"Model saved to {model_save_path} -- but the run "
                      f"FAILED at step {status['timesteps']:,} of "
                      f"{status['total_timesteps']:,} ({status['error']}); "
                      "this is NOT a trained policy. See run_status.json.")
            vec_env.close()
            eval_vec_env.close()
    return model


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser, domain: Domain, encoder: Encoder) -> None:
    """The flags every run has, with this domain's defaults."""
    parser.add_argument(
        "--run_subdir", type=str, default=None,
        help="Override the derived run folder: <output root>/<domain>/<run_subdir>/ (legacy "
             f"layout: runs/<run_subdir>/) instead of the derived "
             f"<variant>/rl/{encoder.name} (legacy: {domain.name}_{encoder.name}[_<variant>]). "
             "For a sweep that needs its own tree.")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=sorted(ALGORITHMS))
    parser.add_argument(
        "--total_timesteps", type=int, default=None,
        help=f"Default: the domain's {domain.default_total_timesteps:,}, or the variant's recorded "
             "horizon where the domain declares one (hunt, msearch). An explicit value is always "
             "honoured (2026-09-19: before this, an explicit value equal to the domain default was "
             "silently swapped for the variant's; debug_plans/ch_fixes.md change C).")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_tag", type=str, default=None,
        help="Short human-readable label appended to the run directory name, so parallel runs "
             "are distinguishable by eye in `ls`. Not a substitute for run_config.json, which "
             "always records the full CLI args.")
    parser.add_argument(
        "--num_particles", type=int, default=None,
        help="Filter set size. Default: the domain's (Ant-Tag 100; Odd-Even the variant's "
             "n_dist_size, which makes the exact-support filter's belief EXACT).")
    parser.add_argument(
        "--arena_scale", type=float, default=None,
        help="Particle normalisation scale for every encoder. Default: the domain's (Ant-Tag: "
             "the live env's arena half-width; Odd-Even: the state-range half-width (n-1)/2). "
             "It MUST match the particle_scale of any pretraining dataset (PITFALLS.md "
             "section 4).")
    parser.add_argument("--device", type=str, default=domain.default_device)
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Default: <run dir>/logs/ (see --output_root for where the run dir goes).")
    parser.add_argument(
        "--model_save_path", type=str, default=None,
        help=f"Default: <run dir>/models/{encoder.name}_agent.zip.")
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Where new run directories go: <root>/<domain>/<variant>/rl/<encoder>/<timestamp>"
             "_seed<seed>[_<tag>]. Default: $RL_BMDP_RUNS, else <parent repo>/runs when this "
             "checkout is a submodule, else <checkout>/runs. Never the current directory.")
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--no_vec_normalize", action="store_true")
    parser.add_argument("--net_arch", type=str, default=None,
                        help="Policy/value MLP sizes, e.g. '256,256'.")
    parser.add_argument("--lr_anneal", action="store_true",
                        help="Linearly anneal learning_rate to 0 over training.")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="PPO target_kl early-stop threshold per rollout (None = disabled).")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="PPO entropy bonus coefficient (default 0.0 = SB3's default). The "
                             "recorded hunt runs used 0.005. PPO only.")
    parser.add_argument("--separate_extractors", action="store_true",
                        help="Give the value network its own features extractor instead of "
                             "sharing the actor's (SB3 share_features_extractor=False; what "
                             "every recorded hunt PPO run did). A pretrained encoder is loaded, "
                             "verified, frozen or lr-scaled in BOTH. Default: shared. PPO only.")
    parser.add_argument("--progress_bar", action="store_true",
                        help="Enable SB3 progress bar. Requires stable-baselines3[extra].")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Fork a run from one of its checkpoint zips (models/checkpoints/<prefix>_<N>_steps"
             f".zip or models/{encoder.name}_agent.zip): policy, encoder, optimizer state and "
             "step counter are restored and training continues to --total_timesteps, which "
             "must be the FULL horizon. Every progress-based schedule is evaluated at the "
             "resumed progress. PPO only; not with --pretrained_path / --frozen; "
             "--encoder_lr_scale may be given. Env RNG and the rollout buffer are NOT "
             "restored: compare forks with forks.")
    parser.add_argument(
        "--resume_vecnormalize", type=str, default=None,
        help="VecNormalize snapshot matching --resume_from. Default: the "
             "<prefix>_vecnormalize_<N>_steps.pkl (or vecnormalize.pkl) beside it.")
    parser.add_argument(
        "--init_policy", type=str, default=None,
        help="Warm-start the WHOLE policy (extractor + MLP + heads) from a saved agent zip of "
             "the same shape (strict load). Not with --pretrained_path or --resume_from.")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Resolve everything and write run_config.json, then stop without building an "
             "env or a model. For checking what a command would train.")


def build_parser(domain: Domain, encoder: Encoder, *, prog: str | None = None,
                 selectors: bool = False) -> argparse.ArgumentParser:
    """The full command line for one (domain, encoder) pair."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"RL on {domain.name} with {encoder.extractor_class.__name__} belief "
                    "features. --variant picks the env; --list_variants shows them.")
    if selectors:
        parser.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES), default=domain.name)
        parser.add_argument("--encoder", choices=sorted(_encoders.ENCODERS), default=encoder.name)
        parser.add_argument("--list_encoders", action="store_true",
                            help="Print the encoder table and exit.")
    domain.add_variant_argument(parser, default=domain.default_variant)
    _add_common_arguments(parser, domain, encoder)
    domain.add_arguments(parser)
    encoder.add_arguments(parser, domain)
    _encoders.add_start_mode_arguments(parser, encoder)
    return parser


def _select(argv: list[str]) -> tuple[str | None, str | None, bool]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES), default=None)
    pre.add_argument("--encoder", choices=sorted(_encoders.ENCODERS), default=None)
    pre.add_argument("--list_encoders", action="store_true")
    known, _ = pre.parse_known_args(argv)
    return known.domain, known.encoder, known.list_encoders


def main(argv: Sequence[str] | None = None, *, domain: Domain | str | None = None,
         encoder: Encoder | str | None = None, legacy_layout: bool = False,
         prog: str | None = None):
    """Parse, resolve, record, train.

    ``domain`` / ``encoder`` given: the numbered scripts' entry (their flags are then a fixed
    pair). Not given: read ``--domain`` / ``--encoder`` from ``argv`` first (the
    ``python -m set_transformer.rl.train`` entry). ``legacy_layout`` keeps today's
    cwd-relative ``runs/<run_subdir>/...`` run directory; the root-level layout is the default.
    Returns the trained model, or None after ``--list_variants`` / ``--list_encoders`` /
    ``--dry_run``.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    selectors = domain is None or encoder is None
    if selectors:
        chosen_domain, chosen_encoder, list_encoders = _select(argv)
        if list_encoders:
            _encoders.print_encoders()
            return None
        if chosen_domain is None or chosen_encoder is None:
            if "-h" in argv or "--help" in argv:
                # A generic help page: the flags depend on the pair, so name the pair first.
                chosen_domain = chosen_domain or sorted(_domains.DOMAIN_NAMES)[0]
                chosen_encoder = chosen_encoder or "cgf"
            else:
                argparse.ArgumentParser(prog=prog).error(
                    "--domain and --encoder are required (e.g. --domain ant_tag --encoder st); "
                    "--list_encoders lists the encoders, --domain <d> --encoder <e> "
                    "--list_variants the variants")
        domain = domain or chosen_domain
        encoder = encoder or chosen_encoder
    domain = _domains.get(domain)
    encoder = _encoders.get(encoder)

    parser = build_parser(domain, encoder, prog=prog, selectors=selectors)
    args = parser.parse_args(argv)
    if args.list_variants:
        domain.print_variants()
        return None
    # Change C (2026-09-19): remember whether the horizon was given, so a domain's per-variant
    # default applies only when it was not (rl/domains/{hunt,msearch}.py::_resolve_arguments).
    args.total_timesteps_given = args.total_timesteps is not None
    if args.total_timesteps is None:
        args.total_timesteps = domain.default_total_timesteps
    _encoders.check_pretrained_path(parser, args, encoder)

    # -- resolution: shared defaults, the domain's flags, the encoder's flags, the start mode --
    resolved = domain.resolve(args.variant)
    env_id = resolved.env_id
    particle_filter_class = domain.particle_filter(args)
    if args.num_particles is None:
        args.num_particles = domain.default_num_particles(args.variant)
    if args.arena_scale is None:
        # Before run_config.json is written, so it records the number that ran (the
        # 2026-09-03 audit found arena_scale recorded as None).
        args.arena_scale = domain.default_arena_scale(args.variant)
    env_options = domain.resolve_arguments(parser, args)
    encoder.resolve_arguments(parser, args, domain)
    start = _encoders.resolve_start_mode(parser, args, encoder, args.algorithm,
                                         resume_from=args.resume_from)
    if args.resume_from:
        if start.pretrained_path or start.frozen:
            parser.error("--resume_from takes the encoder from the checkpoint zip; drop "
                         "--pretrained_path / --frozen")
        if args.algorithm.upper() != "PPO":
            parser.error("--resume_from is implemented for PPO only")
        if not os.path.isfile(args.resume_from):
            parser.error(f"--resume_from {args.resume_from} does not exist")
        if args.resume_vecnormalize is None and not args.no_vec_normalize:
            args.resume_vecnormalize = run_records.resume_vecnormalize_path(args.resume_from)
        if args.resume_vecnormalize and not os.path.isfile(args.resume_vecnormalize):
            parser.error(f"VecNormalize snapshot {args.resume_vecnormalize} does not exist")
    if args.algorithm.upper() != "PPO" and (args.ent_coef != 0.0 or args.separate_extractors):
        parser.error("--ent_coef and --separate_extractors are PPO options (SAC tunes its "
                     "entropy coefficient itself and never shares the extractor)")
    if args.init_policy:
        if start.pretrained_path or args.resume_from:
            parser.error("--init_policy warm-starts the WHOLE policy; not with --pretrained_path "
                         "or --resume_from")
        if not os.path.isfile(args.init_policy):
            parser.error(f"--init_policy {args.init_policy} does not exist")
    net_arch = [int(x) for x in args.net_arch.split(",")] if args.net_arch else None

    # -- the run directory --------------------------------------------------------------------
    run_subdir = args.run_subdir or domain.run_subdir(encoder.name, args.variant)
    if legacy_layout:
        root = None
        run_dir = run_records.default_run_dir(args.seed, run_subdir, run_tag=args.run_tag)
    else:
        root = run_records.output_root(args.output_root)
        if args.run_subdir:
            run_dir = str(root / domain.name / args.run_subdir
                          / run_records.run_leaf(args.seed, args.run_tag))
        else:
            run_dir = str(run_records.run_dir(domain.name, args.variant, encoder.name,
                                              args.seed, args.run_tag, root=root))
    log_dir = args.log_dir or os.path.join(run_dir, "logs") + "/"
    model_save_path = args.model_save_path or os.path.join(
        run_dir, "models", f"{encoder.name}_agent.zip")

    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, "stdout.log")
    run_records.tee_stdout_stderr(stdout_log_path)
    print(f"Mirroring stdout/stderr to {stdout_log_path}")

    # -- the run record: every flag, the derived values, the code --------------------------
    run_config = vars(args).copy()
    # The resolved run_subdir is passed explicitly below; --list_variants, --dry_run and
    # --output_root are not part of the run's identity (the resolved root is recorded).
    for key in ("run_subdir", "list_variants", "dry_run", "output_root", "list_encoders"):
        run_config.pop(key, None)
    run_config.update(
        log_dir=log_dir, model_save_path=model_save_path,
        domain=domain.name, encoder=encoder.name,
        # `run_directory`, not `run_dir`: write_run_config's positional parameter has that name.
        run_directory=os.path.abspath(run_dir), output_root=None if root is None else str(root),
        # The pretraining checkpoint's own config, so "aligned or not", its geometry and its
        # dataset frame are on the RL side too (plan 4d).
        pretrained_config=_encoders.checkpoint_config(start.pretrained_path),
    )
    # env_id / particle_filter_class / run_subdir are derived from --variant, but they are
    # still written out: run_config.json stays a complete record even if the registry entry
    # is later edited.
    run_records.write_run_config(
        run_dir,
        env_id=env_id,
        particle_filter_class=particle_filter_class.__name__,
        run_subdir=run_subdir,
        git=run_records.git_provenance(),
        threads=run_records.thread_settings(),
        **domain.run_config_extras(args),
        **run_config,
    )
    if args.dry_run:
        print("--dry_run: run_config.json written; not training.")
        return None

    # From here the run folder exists with its run_config.json. `train()` writes run_status.json
    # only around `learn()`; an exception before that point (env construction, PPO / feature
    # extractor construction, the pretrained-checkpoint reload and its geometry check) would
    # leave a folder with a config and NO status -- which every reader that keys on "recent
    # write, no status" (the sweep driver's in-progress check) takes for a live run
    # (PITFALLS.md section 13 item 2; batch 10.1, 2026-09-14). Record the failure once, then
    # re-raise. A failure inside `learn()` is already recorded by `train()`'s own finally:
    # block, so the guard below keeps this from writing a second status over it.
    try:
        return _train_from_args(domain, args, encoder, start, net_arch, particle_filter_class,
                                env_options, log_dir, model_save_path)
    except BaseException as exc:      # noqa: B902 - recorded, then re-raised
        status_path = os.path.join(os.path.dirname(model_save_path), run_records.RUN_STATUS_FILENAME)
        if not os.path.exists(status_path):
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            run_records.write_run_status(model_save_path, completed=False, error=exc,
                                         timesteps=0, total_timesteps=int(args.total_timesteps))
            print(f"Run FAILED before training started ({type(exc).__name__}: {exc}); "
                  f"recorded in {status_path}.")
        raise


def _train_from_args(domain, args, encoder, start, net_arch, particle_filter_class, env_options,
                     log_dir, model_save_path):
    """The `train(...)` call of `main`, moved out unchanged so the status guard above wraps it."""
    return train(
        domain, args.variant, encoder,
        features_extractor_kwargs=encoder.extractor_kwargs(args),
        net_arch=net_arch,
        pretrained_path=start.pretrained_path,
        frozen=start.frozen,
        encoder_lr_scale=start.encoder_lr_scale,
        unfreeze_at=start.unfreeze_at,
        init_policy=args.init_policy,
        num_particles=args.num_particles,
        particle_filter_class=particle_filter_class,
        env_options=env_options,
        n_envs=args.n_envs,
        seed=args.seed,
        use_vec_normalize=not args.no_vec_normalize,
        obs_history=encoder.obs_history(args),
        obs_history_padding=encoder.obs_history_padding(args),
        schedules=domain.schedules(args),
        encoder_options=encoder.encoder_options(args),
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
        separate_extractors=args.separate_extractors,
        lr_anneal=args.lr_anneal,
        device=args.device,
        log_dir=log_dir,
        model_save_path=model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        n_eval_episodes=args.n_eval_episodes,
        progress_bar=args.progress_bar,
        resume_from=args.resume_from,
        resume_vecnormalize=args.resume_vecnormalize,
    )


if __name__ == "__main__":
    main()
