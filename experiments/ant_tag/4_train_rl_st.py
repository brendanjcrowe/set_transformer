"""
RL training with a Set Transformer belief encoder for Ant-Tag.

Third arm of the encoder comparison, alongside 4_train_rl_cgf.py (weighted
empirical CGF) and 4_train_rl_gaussian.py (weighted mean + covariance). It
keeps the AntTag particle filter, curriculum, reward shaping, masking and
eval protocol from 4_train_rl_cgf.py untouched and swaps ONLY the belief
encoder: a SetTransformer (ISAB encoder + PMA/SAB decoder head) consumes the
particle set and emits num_encodings x dim_encoder features, which are
concatenated with the base observation exactly like the CGF features are.

Why this file rather than 4_train_rl_frozen.py / 4_train_rl_finetune.py:
those two are an older generation of the pipeline. They hardcode
pdomains-ant-tag-v0 + AntTagParticleFilter, and they lack --run_tag /
run_config.json, --evasion_curriculum, --target_speed_scale, --lr_anneal,
--target_kl and the deterministic per-episode PF seeding that
4_train_rl_cgf.py has gained since. Running them against the CGF numbers
would compare two different experiments. This file reuses the CGF pipeline
verbatim, so the ST arm differs from the CGF arm in the encoder alone.

WEIGHTS. The CGF and Gaussian encoders are both *weighted* — they read
obs_dict["weights"]. The legacy ST extractors ignore the PF weights, which
on the counterweighted-den envs throws away real evidence: the alarm /
silence likelihood in CounterweightedDenAntTagParticleFilter.update()
multiplies weights by ALARM_EPS rather than deleting particles, and weights
stay non-uniform between resamples. So by default the normalized weight is
appended as a third input channel per particle (scaled by num_particles, so
a uniform belief feeds 1.0). Pass --no_st_weight_channel for the legacy
unweighted behavior; the arm is then no longer information-matched to CGF.

Pretrained weights are OPTIONAL and off by default. The CGF baseline learned
its t_values from scratch under PPO, so the matched ST run is likewise
trained end-to-end by PPO. --pretrained_st_model_path (steps 2+3 of the
pipeline: 2_collect_pf_dataset.py then 3_train_st.py) and --st_frozen exist
for the pretrained-encoder variant, mirroring the CGF arm's --t_frozen.

Usage (Counterweighted-Den terminal-phantom env — use the wrapper, which
supplies env_id + PF class):

    python3 4_train_rl_st.py --variant cdens_terminal \
        --total_timesteps 6000000 --seed 0 --device cuda:0 \
        --ppo_n_steps 4096 --n_epochs 10 --target_kl 0.03 --lr_anneal \
        --reward_schedule "0:1:0:0,0.2:1:0:0,0.5:0:0:50,1:0:0:50" \
        --eval_freq 40000 --n_eval_episodes 30 \
        --run_tag terminal_v1_dist0_noent

(--curriculum and --evasion_curriculum come from the variant registry; pass
them explicitly only to override.)
"""

import argparse
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers pdomains-ant-tag-v0
import variants  # noqa: E402 - env/filter/subdir registry
from set_transformer.models import PFSetTransformer
from set_transformer.rl.encoder_finetune import (  # noqa: E402 - shared with experiments/odd_even
    EncoderLRLoggingCallback,
    scale_encoder_learning_rate,
)
from set_transformer.rl.pretrained_encoder import reload_pretrained  # noqa: E402


# Reuse the existing AntTag curriculum/PF/dict-obs utilities. The dict-obs env
# (particles + PF weights) built by 4_train_rl_cgf.py has no CGF-specific
# logic in it, so it is reused as-is for the Set Transformer extractor too.
_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
CurriculumCallback = _train_rl_cgf.CurriculumCallback
CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
PFRewardShapingWrapper = _train_rl_cgf.PFRewardShapingWrapper
_CurriculumRouter = _train_rl_cgf._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs
get_ant_tag_arena_scale = _train_rl_cgf.get_ant_tag_arena_scale
get_env_visible_radius = _train_rl_cgf.get_env_visible_radius
PFDictWithWeightsObservationWrapper = _train_rl_cgf.PFDictWithWeightsObservationWrapper
make_ant_tag_belief_env = _train_rl_cgf.make_ant_tag_cgf_env
_make_vec_normalize = _train_rl_cgf._make_vec_normalize
_make_vec_env_from_fns = _train_rl_cgf._make_vec_env_from_fns
AntTagParticleFilter = _train_rl_cgf.AntTagParticleFilter
_tee_stdout_stderr = _train_rl_cgf._tee_stdout_stderr
_git_provenance = _train_rl_cgf._git_provenance
_write_run_config = _train_rl_cgf._write_run_config
_parse_curriculum = _train_rl_cgf._parse_curriculum
_parse_reward_schedule = _train_rl_cgf._parse_reward_schedule
from set_transformer.rl.run_records import default_run_dir as _shared_default_run_dir  # noqa: E402


def _default_run_dir(seed: int, run_subdir: str = "ant_tag_st",
                      run_tag: str | None = None) -> str:
    """run_records.default_run_dir with this script's historical default subfolder
    (train_ant_tag_st calls it with only a seed when no --log_dir is given)."""
    return _shared_default_run_dir(seed, run_subdir, run_tag)

# The Set Transformer encoder and its logging callback now live in the shared
# package so a second domain can use them without importing an Ant-Tag script.
# They are re-exported here, unchanged and as the SAME objects, because SB3
# pickles a policy's features-extractor CLASS into the saved zip by module
# path: loading a checkpoint saved by this script runs
# getattr(import_module("4_train_rl_st"), "SetTransformerFeaturesExtractor").
from set_transformer.rl.feature_extractors.st import (  # noqa: E402
    STFeatureLoggingCallback,
    SetTransformerFeaturesExtractor,
)


def train_ant_tag_st(
    algorithm: str = "PPO",
    total_timesteps: int = 3_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    ppo_n_steps: int = 2048,
    num_particles: int = 100,
    arena_scale: float | None = None,
    device: str = "cuda:1",
    seed: int = 0,
    log_dir: str | None = None,
    model_save_path: str | None = None,
    eval_freq: int = 20_000,
    save_freq: int = 100_000,
    use_vec_normalize: bool = True,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    curriculum_schedule: list[tuple[float, float]] | None = None,
    reward_schedule: list[tuple[float, ...]] | None = None,
    evasion_schedule: list[tuple[float, float]] | None = None,
    net_arch: list[int] | None = None,
    obs_mask_indices: list[int] | None = None,
    progress_bar: bool = False,
    target_speed_scale: float | None = None,
    env_id: str = "pdomains-ant-tag-v0",
    particle_filter_class: type = AntTagParticleFilter,
    lr_anneal: bool = False,
    target_kl: float | None = None,
    n_epochs: int = 10,
    n_eval_episodes: int = 20,
    num_encodings: int = 8,
    dim_encoder: int = 8,
    num_inds: int = 32,
    dim_hidden: int = 128,
    num_heads: int = 4,
    ln: bool = True,
    weight_channel: bool = True,
    pretrained_st_model_path: str | None = None,
    st_frozen: bool = False,
    st_encoder_lr_scale: float = 1.0,
    resume_from: str | None = None,
    resume_vecnormalize: str | None = None,
):
    if resume_from:
        # Fork a run from one of its 100k-step checkpoints (2026-09-08, the
        # smart_mid_slow_v15 second-half reward ablation): the policy, value
        # head, encoder, Adam moments (when the zip has them; the 0.1x
        # finetune arm's saves collapse the optimizer, encoder_finetune.py)
        # and the step counter come from the zip, the obs-normalization
        # statistics from the matching VecNormalize snapshot, and training
        # continues to `total_timesteps` (the FULL horizon, e.g. 6M) with
        # every progress-based schedule -- curriculum, reward, evasion, LR
        # anneal -- evaluated at the resumed progress. Not restored: env RNG
        # state, the rollout buffer, PPO's sampling RNG. A fork is therefore
        # not a bit-exact continuation; compare forks with forks.
        if algorithm.upper() != "PPO":
            raise ValueError("--resume_from is implemented for PPO only")
        if pretrained_st_model_path or st_frozen:
            raise ValueError("--resume_from restores the encoder from the checkpoint; "
                             "--pretrained_st_model_path / --st_frozen do not apply")
        if use_vec_normalize and not resume_vecnormalize:
            raise ValueError("--resume_from with VecNormalize needs the matching "
                             "--resume_vecnormalize snapshot")
    resolved_arena_scale = (
        arena_scale if arena_scale is not None else get_ant_tag_arena_scale(env_id)
    )
    if log_dir is None or model_save_path is None:
        run_dir = _default_run_dir(seed)
        if log_dir is None:
            log_dir = os.path.join(run_dir, "logs") + "/"
        if model_save_path is None:
            model_save_path = os.path.join(run_dir, "models", "st_agent.zip")

    print(f"Training {algorithm} on AntTag with Set Transformer features")
    print(f"ST features={num_encodings * dim_encoder} "
          f"({num_encodings} encodings x {dim_encoder} dims), "
          f"dim_hidden={dim_hidden}, num_inds={num_inds}, num_heads={num_heads}, "
          f"ln={ln}, weight_channel={weight_channel}, st_frozen={st_frozen}")
    print(f"Arena scale={resolved_arena_scale}")
    print(f"SB3 device={device}")
    print(f"log_dir={log_dir}")
    print(f"model_save_path={model_save_path}")

    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    initial_vis = curriculum_schedule[0][1] if curriculum_schedule else 100.0
    env_kw = dict(
        num_particles=num_particles,
        distance_coeff=distance_coeff,
        entropy_coeff=entropy_coeff,
        tag_bonus_coeff=0.0,
        initial_visibility_radius=initial_vis,
        obs_mask_indices=obs_mask_indices,
        env_id=env_id,
        particle_filter_class=particle_filter_class,
        # In env_kw so the eval envs built from dict(env_kw) below inherit
        # the SAME target speed as training.
        target_speed_scale=target_speed_scale,
    )

    env_fns = [
        make_ant_tag_belief_env(
            **env_kw,
            rank=rank,
            seed=seed,
            monitor_dir=monitor_dir,
        )
        for rank in range(n_envs)
    ]
    vec_env = _make_vec_env_from_fns(env_fns, n_envs)
    if use_vec_normalize:
        if resume_from:
            vec_env = VecNormalize.load(resume_vecnormalize, vec_env)
            vec_env.training = True
            print(f"VecNormalize statistics RESUMED from {resume_vecnormalize} "
                  f"(norm_obs_keys={vec_env.norm_obs_keys}, norm_reward={vec_env.norm_reward}, "
                  f"obs count={float(vec_env.obs_rms['obs'].count):.0f})")
        else:
            vec_env = _make_vec_normalize(vec_env, training=True, norm_reward=True)

    # Eval envs — always evaluate at the env's real POMDP difficulty (its own
    # visible_radius) and on the true sparse reward (no shaping, so the metric
    # doesn't depend on where the training curriculum currently is).
    # (EvalCallback syncs the training VecNormalize stats into this one before
    # every eval, so a resumed run evaluates with the resumed statistics.)
    eval_env_kw = dict(env_kw)
    eval_env_kw["initial_visibility_radius"] = get_env_visible_radius(env_id)
    eval_env_kw["apply_reward_shaping"] = False
    eval_vec_env = DummyVecEnv([
        make_ant_tag_belief_env(
            **eval_env_kw,
            rank=n_envs + 1,
            seed=seed,
        )
    ])
    if use_vec_normalize:
        eval_vec_env = _make_vec_normalize(
            eval_vec_env,
            training=False,
            norm_reward=False,
        )

    policy_kwargs = {
        "features_extractor_class": SetTransformerFeaturesExtractor,
        "features_extractor_kwargs": dict(
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
            arena_scale=resolved_arena_scale,
            weight_channel=weight_channel,
            pretrained_st_model_path=pretrained_st_model_path,
            st_frozen=st_frozen,
        ),
    }
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    lr_arg = (lambda progress_remaining: learning_rate * progress_remaining) if lr_anneal else learning_rate
    if resume_from:
        model = PPO.load(
            resume_from, env=vec_env, device=device, force_reset=True,
            # The zip carries the source run's schedule object; rebuild it from
            # THIS run's flags so run_config.json and the optimizer agree.
            custom_objects={"learning_rate": lr_arg, "lr_schedule": lr_arg},
        )
        mismatched = {
            k: (getattr(model, k), v)
            for k, v in dict(n_steps=ppo_n_steps, batch_size=batch_size, n_epochs=n_epochs,
                             target_kl=target_kl, seed=seed).items()
            if getattr(model, k) != v
        }
        if mismatched:
            raise ValueError(f"--resume_from checkpoint disagrees with the CLI on "
                             f"{mismatched} (stored, given); pass the source run's values")
        if model.num_timesteps >= total_timesteps:
            raise ValueError(f"checkpoint is at {model.num_timesteps:,} steps, "
                             f"--total_timesteps {total_timesteps:,} must be the FULL horizon beyond it")
        # PPO.load restores the SOURCE run's tensorboard_log; point it here.
        model.tensorboard_log = log_dir
        progress = model.num_timesteps / total_timesteps
        print(f"PPO RESUMED from {resume_from}: {model.num_timesteps:,} env steps done "
              f"(progress {progress:.3f}), {model._n_updates} updates; training "
              f"{total_timesteps - model.num_timesteps:,} more steps to {total_timesteps:,}; "
              f"learning rate resumes at {model.lr_schedule(1.0 - progress):.2e}"
              + ("" if model.policy.optimizer.state else
                 " (optimizer moments were not in the zip: Adam restarts)"))
    elif algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=lr_arg,
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
    elif algorithm.upper() == "SAC":
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
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    if pretrained_st_model_path:
        # RELOAD AFTER CONSTRUCTION. The extractor's own __init__ already
        # loaded these weights, but SB3's ActorCriticPolicy._build ends with
        #     self.apply(partial(self.init_weights, gain=...))
        # which walks the WHOLE policy -- features extractor included -- and
        # re-initializes every Linear it finds. So the pretrained encoder was
        # silently overwritten with orthogonal-init noise a moment after being
        # loaded, and --st_frozen then froze that noise. Both pretrained arms
        # trained on a random encoder while the log said "loaded ... FROZEN",
        # because those lines are printed during construction, before the
        # overwrite. Measured max|delta| from the checkpoint was 1.84.
        #
        # Reloading here, after _build, is the fix: nothing re-inits the
        # policy afterwards. The freeze is re-applied for the same reason --
        # requires_grad survives apply(), but re-setting it keeps the two
        # facts in one place.
        # Since change 2 of the harness centralisation (2026-09-12) the shared reload does
        # the same steps for every learned extractor -- reload every extractor on the
        # policy (SAC critics may hold their own), re-freeze, blank the checkpoint path in
        # policy_kwargs so no absolute path is baked into the saved zip -- and ADDS the one
        # this arm never had: assert max|delta| == 0 against the checkpoint (PITFALLS.md
        # section 1), the only positive evidence in the logs that the reload landed.
        reload_pretrained(model, pretrained_st_model_path, st_frozen)

    finetune_callbacks = []
    if st_encoder_lr_scale != 1.0:
        # Finetune-collapse fix 1 (oddeven.md 2026-09-06; on smart_hard the
        # shared-rate finetune arms flipped between 0% and 10% by seed): the
        # encoder gets its own param group at st_encoder_lr_scale x the head
        # rate. Must run AFTER the reload above and after PPO construction --
        # it rebuilds the optimizer over the live parameters.
        if algorithm.upper() != "PPO":
            raise ValueError("--st_encoder_lr_scale is implemented for PPO only")
        scale_encoder_learning_rate(model, st_encoder_lr_scale)
        finetune_callbacks.append(EncoderLRLoggingCallback())

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(model_dir, "checkpoints") if model_dir else "checkpoints",
        name_prefix="ant_tag_st",
        # Also snapshot VecNormalize with every checkpoint: without it the
        # 100k-step checkpoints cannot be evaluated faithfully (the obs
        # normalization is otherwise written once, at the end), and a run
        # killed early is a total loss (PITFALLS.md section 7).
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
    st_log_cb = STFeatureLoggingCallback()
    curriculum_cb = CurriculumCallback(
        total_timesteps=total_timesteps,
        schedule=curriculum_schedule,
        reward_schedule=reward_schedule,
        evasion_schedule=evasion_schedule,
        verbose=1,
    )

    try:
        # SB3 adds num_timesteps to the requested total when the counter is not
        # reset, so a resumed run asks for the REMAINING steps and every
        # progress_remaining-based schedule (LR anneal) continues from where
        # the checkpoint left off; CurriculumCallback divides num_timesteps
        # by the full horizon it was given above.
        model.learn(
            total_timesteps=total_timesteps - (model.num_timesteps if resume_from else 0),
            reset_num_timesteps=not resume_from,
            callback=[checkpoint_cb, eval_cb, curriculum_cb, st_log_cb, *finetune_callbacks],
            progress_bar=progress_bar,
        )
    finally:
        model.save(model_save_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl") if model_dir else "vecnormalize.pkl"
            vec_env.save(vecnorm_path)
            print(f"VecNormalize saved to {vecnorm_path}")
        print(f"Model saved to {model_save_path}")
        vec_env.close()
        eval_vec_env.close()


# The snapshot-beside-checkpoint rule lives in the package since 2026-09-12 (change 4),
# generalised to any <encoder>_agent.zip; imported back under the name the tests read.
from set_transformer.rl.run_records import (  # noqa: E402
    resume_vecnormalize_path as _default_resume_vecnormalize,
)


def main(encoder: str = "st"):
    """Entry point. --variant selects env id, particle filter and run subdir
    together from variants.py, so the three cannot disagree."""
    parser = argparse.ArgumentParser(
        description=(
            "RL with Set Transformer particle-belief features on AntTag. "
            "Use --variant to pick the env; --list_variants to see them."
        )
    )
    variants.add_variant_argument(parser)
    parser.add_argument(
        "--run_subdir", type=str, default=None,
        help="Override the derived runs/ant_tag_st[_<variant>] "
             "directory. For a sweep that needs its own tree "
             "(e.g. ant_tag_cgf_cdens_hard_dist0), which the "
             "derived name cannot express.",
    )
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help=(
            "Optional short human-readable label appended to the run "
            "directory name (e.g. 'terminal_v1_dist0_noent'), so parallel "
            "runs are distinguishable by eye in `ls`. Not a substitute for "
            "run_config.json, which always records the full CLI args."
        ),
    )

    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument(
        "--arena_scale",
        type=float,
        default=None,
        help=(
            "Particle normalization scale for the ST input. Defaults to the "
            "live env's actual arena half-width (cage_max_x), matching the "
            "CGF and Gaussian arms."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:1")

    # -- Set Transformer architecture ------------------------------------
    parser.add_argument(
        "--num_encodings", type=int, default=8,
        help="PMA seed vectors. num_encodings x dim_encoder is the belief "
             "feature width; the default 8 x 8 = 64 matches the CGF arm's "
             "--num_cgf_features 64.",
    )
    parser.add_argument("--dim_encoder", type=int, default=8)
    parser.add_argument("--num_inds", type=int, default=32,
                        help="ISAB inducing points.")
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument(
        "--ln", action="store_true", default=True,
        help="LayerNorm inside the attention blocks. Default on.",
    )
    parser.add_argument("--no_ln", dest="ln", action="store_false")
    parser.add_argument(
        "--st_weight_channel", dest="weight_channel",
        action="store_true", default=True,
        help="Append the normalized PF weight as a third per-particle input "
             "channel, so the ST reads the same weighted belief as the CGF "
             "and Gaussian arms. Default on.",
    )
    parser.add_argument(
        "--no_st_weight_channel", dest="weight_channel", action="store_false",
        help="Legacy unweighted ST input (coordinates only). The arm is then "
             "NOT information-matched to the CGF baseline.",
    )
    parser.add_argument(
        "--pretrained_st_model_path", type=str, default=None,
        help="Optional PFSetTransformer state_dict / trainer checkpoint from "
             "3_train_st.py. Omit to train the encoder from scratch under "
             "PPO, which is what the CGF baseline does.",
    )
    parser.add_argument(
        "--st_encoder_lr_scale", type=float, default=1.0,
        help="Finetune-collapse fix 1: multiply the pretrained encoder's "
             "learning rate by this (heads keep --learning_rate; --lr_anneal "
             "applies to both). 1.0 = off, i.e. the shared-rate finetune. "
             "Requires --pretrained_st_model_path and not --st_frozen. PPO only.")
    parser.add_argument(
        "--st_frozen", action="store_true",
        help="Freeze the ST encoder; only the policy/value MLP learns. The "
             "ST analogue of the CGF arm's --t_frozen. Only meaningful "
             "together with --pretrained_st_model_path.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Defaults to runs/ant_tag_st[_<variant>]/<timestamp>_seed<seed>/logs/",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Defaults to runs/ant_tag_st[_<variant>]/<timestamp>_seed<seed>/models/st_agent.zip",
    )
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--no_vec_normalize", action="store_true")
    parser.add_argument(
        "--net_arch",
        type=str,
        default=None,
        help="Policy/value MLP sizes, e.g. '256,256'.",
    )

    parser.add_argument(
        "--distance_coeff", type=float, default=None,
        help="Constant PF-mean-distance shaping coefficient (default 1.0). "
             "Only honoured with --reward_schedule none: the schedule sets "
             "these coefficients on every step, so combining the two is an "
             "error rather than a silent override.")
    parser.add_argument(
        "--entropy_coeff", type=float, default=None,
        help="Constant PF belief-entropy shaping coefficient (default 0.0). "
             "NOT PPO's entropy bonus. Same rule as --distance_coeff.")
    parser.add_argument(
        "--curriculum",
        type=str,
        default=None,
        help="Visibility curriculum 'frac:radius,...'. Defaults to the "
             "variant's own curriculum, else the base schedule.",
    )
    parser.add_argument(
        "--reward_schedule",
        type=str,
        default=None,
        help="Defaults to the variant's own recipe (variants.py default_reward_schedule), else "
             "'0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50' (PF-entropy 0 throughout). "
             "Shaping schedule 'frac:distance:entropy[:tag_bonus[:spread_gain]],...' (spread_gain pays gain*(spread_{t-1}-spread_t) of the belief's weighted std; 0 when omitted), "
             "interpolated over training progress and applied on every step. "
             "Pass 'none' to run on the constant --distance_coeff / "
             "--entropy_coeff values instead; giving both is an error.",
    )
    parser.add_argument(
        "--evasion_curriculum",
        type=str,
        default=None,
        help=(
            "Optional frac:scale schedule for SmartAntTagEnv.evasion_scale "
            "(0=dumb-target behavior, 1=full smart evasion). E.g. "
            "'0:0.2,0.5:1,1:1' ramps evasion strength up over the first "
            "half of training. Default: constant 1.0 (full strength "
            "throughout, i.e. no curriculum). No-op on envs without an "
            "evasion_scale knob (e.g. the base AntTag env)."
        ),
    )
    parser.add_argument(
        "--mask_target_obs",
        action="store_true",
        default=True,
        help="Zero out obs[-2:] for the agent. Default on.",
    )
    parser.add_argument(
        "--no_mask_target_obs",
        dest="mask_target_obs",
        action="store_false",
    )
    parser.add_argument(
        "--target_speed_scale",
        type=float,
        default=None,
        help=(
            "SmartAntTagEnv only. How much faster the target moves as it gets "
            "cornered: step = target_step * (1 + urgency * scale). Omit to use "
            "the env default (0.0 = constant speed). Applied to both the "
            "training and eval envs."
        ),
    )
    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="Enable SB3 progress bar. Requires stable-baselines3[extra].",
    )
    parser.add_argument(
        "--lr_anneal",
        action="store_true",
        help="Linearly anneal learning_rate to 0 over training (progress_remaining schedule).",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="PPO target_kl early-stop threshold per rollout (None = disabled).",
    )
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=20)

    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Fork a run from one of its checkpoint zips (models/checkpoints/"
             "ant_tag_st_<N>_steps.zip or models/st_agent.zip): policy, encoder, "
             "optimizer state and step counter are restored and training continues "
             "to --total_timesteps, which must be the FULL horizon (e.g. 6000000, "
             "not the remainder). Every progress-based schedule (--curriculum, "
             "--reward_schedule, --evasion_curriculum, --lr_anneal) is evaluated at "
             "the resumed progress, so a schedule that differs from the source "
             "run's only after the checkpoint's progress gives a clean second-half "
             "ablation. PPO only; not with --pretrained_st_model_path / --st_frozen; "
             "--st_encoder_lr_scale may be given (it rebuilds the encoder param group). "
             "Env RNG and the rollout buffer are NOT restored: compare forks with forks.")
    parser.add_argument(
        "--resume_vecnormalize", type=str, default=None,
        help="VecNormalize snapshot matching --resume_from. Default: the "
             "ant_tag_st_vecnormalize_<N>_steps.pkl (or vecnormalize.pkl) beside it.")

    args = parser.parse_args()
    # Before _resolve_reward_shaping: it reads an empty schedule as "use the
    # constant flags", so a None default must be resolved first.
    args.reward_schedule = variants.resolve_schedule(
        args.variant, args.reward_schedule, "default_reward_schedule",
        "0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50")
    _train_rl_cgf._resolve_reward_shaping(parser, args)
    if args.list_variants:
        variants.print_variants()
        return

    variant = variants.resolve(args.variant)
    env_id = variant.env_id
    particle_filter_class = variant.particle_filter
    run_subdir = args.run_subdir or variants.run_subdir(
        encoder, args.variant)
    args.curriculum = variants.resolve_schedule(
        args.variant, args.curriculum, "default_curriculum",
        "0:100,0.3:100,0.7:3,1:3")
    variants.warn_if_not_evading(
        args.variant, args.evasion_curriculum, args.target_speed_scale)
    args.evasion_curriculum = variants.resolve_schedule(
        args.variant, args.evasion_curriculum, "default_evasion_curriculum",
        None)

    net_arch = [int(x) for x in args.net_arch.split(",")] if args.net_arch else None
    obs_mask = [-2, -1] if args.mask_target_obs else None

    if args.resume_from:
        if args.pretrained_st_model_path or args.st_frozen:
            parser.error("--resume_from takes the encoder from the checkpoint zip; "
                         "drop --pretrained_st_model_path / --st_frozen")
        if not os.path.isfile(args.resume_from):
            parser.error(f"--resume_from {args.resume_from} does not exist")
        if args.resume_vecnormalize is None and not args.no_vec_normalize:
            args.resume_vecnormalize = _default_resume_vecnormalize(args.resume_from)
        if args.resume_vecnormalize and not os.path.isfile(args.resume_vecnormalize):
            parser.error(f"VecNormalize snapshot {args.resume_vecnormalize} does not exist")
    if args.st_encoder_lr_scale <= 0:
        parser.error("--st_encoder_lr_scale must be positive")
    if args.st_encoder_lr_scale != 1.0 and not (args.pretrained_st_model_path or args.resume_from):
        parser.error("--st_encoder_lr_scale is a finetune fix for a PRETRAINED "
                     "encoder; pass --pretrained_st_model_path (an end-to-end "
                     "encoder has no pretrained geometry to protect).")
    if args.st_encoder_lr_scale != 1.0 and args.st_frozen:
        parser.error("--st_frozen freezes the encoder for the whole run; "
                     "--st_encoder_lr_scale has nothing to act on. Drop one.")
    if args.st_encoder_lr_scale != 1.0 and args.algorithm.upper() != "PPO":
        parser.error("--st_encoder_lr_scale is implemented for PPO only")
    if args.st_frozen and not args.pretrained_st_model_path:
        raise ValueError(
            "--st_frozen without --pretrained_st_model_path would freeze a "
            "randomly initialized encoder. Pass a checkpoint, or drop "
            "--st_frozen to train the encoder under PPO."
        )

    run_dir = _default_run_dir(args.seed, run_subdir, run_tag=args.run_tag)
    log_dir = args.log_dir or os.path.join(run_dir, "logs") + "/"
    model_save_path = args.model_save_path or os.path.join(run_dir, "models", "st_agent.zip")

    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, "stdout.log")
    _tee_stdout_stderr(stdout_log_path)
    print(f"Mirroring stdout/stderr to {stdout_log_path}")

    run_config = vars(args).copy()
    # The resolved run_subdir is passed explicitly below; drop the raw flag so
    # the two do not collide as duplicate keyword arguments. --list_variants
    # already returned by this point and is not part of the run's identity.
    run_config.pop("run_subdir", None)
    run_config.pop("list_variants", None)
    run_config.update(log_dir=log_dir, model_save_path=model_save_path)
    # env_id / particle_filter_class / run_subdir are derived from --variant,
    # but they are still written out: run_config.json stays a complete record
    # even if the registry entry is later edited.
    _write_run_config(
        run_dir,
        env_id=env_id,
        particle_filter_class=particle_filter_class.__name__,
        run_subdir=run_subdir,
        git=_git_provenance(),
        **run_config,
    )

    train_ant_tag_st(
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        num_particles=args.num_particles,
        arena_scale=args.arena_scale,
        device=args.device,
        seed=args.seed,
        log_dir=log_dir,
        model_save_path=model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_vec_normalize=not args.no_vec_normalize,
        distance_coeff=args.distance_coeff,
        entropy_coeff=args.entropy_coeff,
        curriculum_schedule=_parse_curriculum(args.curriculum),
        reward_schedule=_parse_reward_schedule(args.reward_schedule),
        evasion_schedule=_parse_curriculum(args.evasion_curriculum) if args.evasion_curriculum else None,
        net_arch=net_arch,
        obs_mask_indices=obs_mask,
        progress_bar=args.progress_bar,
        target_speed_scale=args.target_speed_scale,
        env_id=env_id,
        particle_filter_class=particle_filter_class,
        lr_anneal=args.lr_anneal,
        target_kl=args.target_kl,
        n_epochs=args.n_epochs,
        n_eval_episodes=args.n_eval_episodes,
        num_encodings=args.num_encodings,
        dim_encoder=args.dim_encoder,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        ln=args.ln,
        weight_channel=args.weight_channel,
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_frozen=args.st_frozen,
        st_encoder_lr_scale=args.st_encoder_lr_scale,
        resume_from=args.resume_from,
        resume_vecnormalize=args.resume_vecnormalize,
    )


if __name__ == "__main__":
    main()
