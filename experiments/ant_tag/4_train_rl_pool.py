"""
RL training with a pooling / moment belief encoder for Ant-Tag: DeepSet, PointNet or
k-moments (2026-09-11). One trainer, three arms, selected by ``main(encoder=...)`` from the
two-line entry scripts 4_train_rl_deepset.py / 4_train_rl_pointnet.py / 4_train_rl_kmoments.py.

Same contract as the other arms: the AntTag particle filter, curriculum, reward shaping,
PPO settings and eval protocol are 4_train_rl_cgf.py's (imported by name below, like
4_train_rl_st.py and 4_train_rl_gaussian.py do), and ONLY the SB3 features extractor
changes. All three read the {"obs", "particles", "weights"} Dict observation, divide
particles by the arena half-width, keep PF weights in the measure, and hand the policy
[obs, encoder output]. A checkpoint from any of them evaluates through
eval_scripts/eval_true_reward_cgf.py's env.

    deepset   WeightedDeepSetFeaturesExtractor  learned; weighted-mean pool (or --pooling mean)
    pointnet  PointNetFeaturesExtractor         learned; masked-max pool (or --pooling max)
    kmoments  WeightedKMomentsFeaturesExtractor analytic; weighted mean + central moments 2..k

The learned arms take --pretrained_model_path (a DeepSetAE / PointNetAE state_dict or a
Trainer checkpoint), --frozen, and --encoder_lr_scale (PPO only; the ST arm's finetune
fix), with the post-construction reload + verification of PITFALLS.md section 1. Both AEs
are unweighted (D inputs), so load them with --no_weight_channel; the extractor says so.

Run directories: runs/ant_tag_<encoder>[_<variant>]/<timestamp>_seed<seed>[_<run_tag>]/,
model <encoder>_agent.zip, run_config.json with every CLI arg.
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

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import pdomains  # noqa: F401,E402 - registers pdomains-ant-tag-*
import variants  # noqa: E402 - env/filter/subdir registry

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
_parse_curriculum = _train_rl_cgf._parse_curriculum
_parse_reward_schedule = _train_rl_cgf._parse_reward_schedule

from set_transformer.rl.encoder_finetune import (  # noqa: E402
    EncoderLRLoggingCallback,
    scale_encoder_learning_rate,
)
from set_transformer.rl.feature_extractors.pooled import (  # noqa: E402
    PointNetFeaturesExtractor,
    WeightedDeepSetFeaturesExtractor,
    WeightedKMomentsFeaturesExtractor,
    reload_pretrained_pooled,
)

#: encoder name -> (extractor class, learned?)
ENCODERS = {
    "deepset": (WeightedDeepSetFeaturesExtractor, True),
    "pointnet": (PointNetFeaturesExtractor, True),
    "kmoments": (WeightedKMomentsFeaturesExtractor, False),
}


def _default_run_dir(seed: int, run_subdir: str, run_tag: str | None = None) -> str:
    """runs/<run_subdir>/<timestamp>_seed<seed>[_<run_tag>]/ (see 4_train_rl_cgf.py)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{re.sub(r'[^A-Za-z0-9._-]', '_', run_tag)}" if run_tag else ""
    return os.path.join("runs", run_subdir, f"{timestamp}_seed{seed}{suffix}")


def _write_run_config(run_dir: str, **config) -> None:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str, sort_keys=True)
    print(f"Run config saved to {path}")


def build_extractor_kwargs(encoder: str, arena_scale: float, *, num_encodings: int,
                           dim_encoder: int, dim_hidden: int, weight_channel: bool,
                           pooling: str | None, pretrained_model_path: str | None,
                           frozen: bool, k_moments: int) -> dict:
    """The features_extractor_kwargs for one arm; what run_config.json should agree with."""
    if encoder == "kmoments":
        return dict(k=k_moments, arena_scale=arena_scale)
    return dict(num_encodings=num_encodings, dim_encoder=dim_encoder, dim_hidden=dim_hidden,
                arena_scale=arena_scale, weight_channel=weight_channel, pooling=pooling,
                pretrained_model_path=pretrained_model_path, frozen=frozen)


def train_ant_tag_pool(
    encoder: str,
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
    # encoder
    num_encodings: int = 8,
    dim_encoder: int = 8,
    dim_hidden: int = 128,
    weight_channel: bool = True,
    pooling: str | None = None,
    pretrained_model_path: str | None = None,
    frozen: bool = False,
    encoder_lr_scale: float = 1.0,
    k_moments: int = 4,
):
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {sorted(ENCODERS)}, got {encoder!r}")
    extractor_cls, learned = ENCODERS[encoder]
    resolved_arena_scale = (
        arena_scale if arena_scale is not None else get_ant_tag_arena_scale(env_id)
    )
    if log_dir is None or model_save_path is None:
        run_dir = _default_run_dir(seed, f"ant_tag_{encoder}")
        log_dir = log_dir or os.path.join(run_dir, "logs") + "/"
        model_save_path = model_save_path or os.path.join(run_dir, "models", f"{encoder}_agent.zip")
    print(f"Training {algorithm} on AntTag with {extractor_cls.__name__} features")
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
        target_speed_scale=target_speed_scale,
    )
    env_fns = [
        make_ant_tag_belief_env(**env_kw, rank=rank, seed=seed, monitor_dir=monitor_dir)
        for rank in range(n_envs)
    ]
    vec_env = _make_vec_env_from_fns(env_fns, n_envs)
    if use_vec_normalize:
        vec_env = _make_vec_normalize(vec_env, training=True, norm_reward=True)
    eval_env_kw = dict(env_kw)
    eval_env_kw["initial_visibility_radius"] = get_env_visible_radius(env_id)
    eval_env_kw["apply_reward_shaping"] = False
    eval_vec_env = DummyVecEnv([
        make_ant_tag_belief_env(**eval_env_kw, rank=n_envs + 1, seed=seed)
    ])
    if use_vec_normalize:
        eval_vec_env = _make_vec_normalize(eval_vec_env, training=False, norm_reward=False)

    policy_kwargs = {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": build_extractor_kwargs(
            encoder, resolved_arena_scale, num_encodings=num_encodings,
            dim_encoder=dim_encoder, dim_hidden=dim_hidden, weight_channel=weight_channel,
            pooling=pooling, pretrained_model_path=pretrained_model_path, frozen=frozen,
            k_moments=k_moments),
    }
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch
    lr_arg = (lambda progress_remaining: learning_rate * progress_remaining) if lr_anneal else learning_rate
    if algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy", vec_env, learning_rate=lr_arg, n_steps=ppo_n_steps,
            batch_size=batch_size, n_epochs=n_epochs, target_kl=target_kl, verbose=1,
            tensorboard_log=log_dir, seed=seed, policy_kwargs=policy_kwargs, device=device,
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MultiInputPolicy", vec_env, learning_rate=learning_rate, batch_size=batch_size,
            verbose=1, tensorboard_log=log_dir, seed=seed, policy_kwargs=policy_kwargs,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    extra_callbacks = []
    if learned and pretrained_model_path:
        reload_pretrained_pooled(model, pretrained_model_path, frozen)
    if learned and encoder_lr_scale != 1.0:
        if algorithm.upper() != "PPO":
            raise ValueError("--encoder_lr_scale is implemented for PPO only")
        scale_encoder_learning_rate(model, encoder_lr_scale)
        extra_callbacks.append(EncoderLRLoggingCallback())
    if learned:
        print(f"{extractor_cls.__name__}: {model.policy.features_extractor.encoder_parameter_count():,} "
              "encoder parameters")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(model_dir, "checkpoints") if model_dir else "checkpoints",
        name_prefix=f"ant_tag_{encoder}",
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
    curriculum_cb = CurriculumCallback(
        total_timesteps=total_timesteps,
        schedule=curriculum_schedule,
        reward_schedule=reward_schedule,
        evasion_schedule=evasion_schedule,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, curriculum_cb, *extra_callbacks],
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


def main(encoder: str = "deepset"):
    """Entry point shared by the three pooling arms. --variant selects env id,
    particle filter and run subdir together from variants.py."""
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {sorted(ENCODERS)}, got {encoder!r}")
    extractor_cls, learned = ENCODERS[encoder]
    parser = argparse.ArgumentParser(
        description=(f"RL with {extractor_cls.__name__} particle-belief features on AntTag. "
                     "Use --variant to pick the env; --list_variants to see them."))
    variants.add_variant_argument(parser)
    parser.add_argument("--run_subdir", type=str, default=None,
                        help=f"Override the derived runs/ant_tag_{encoder}[_<variant>] directory.")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Short human-readable label appended to the run directory name.")
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--arena_scale", type=float, default=None,
                        help="Particle normalisation scale. Defaults to the live env's arena "
                             "half-width (cage_max_x).")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--no_vec_normalize", action="store_true")
    parser.add_argument("--net_arch", type=str, default=None, help="Policy/value MLP sizes, e.g. '256,256'.")
    parser.add_argument("--distance_coeff", type=float, default=None,
                        help="Constant PF-mean-distance shaping coefficient; only with --reward_schedule none.")
    parser.add_argument("--entropy_coeff", type=float, default=None,
                        help="Constant PF belief-entropy shaping coefficient (NOT PPO's entropy bonus); "
                             "only with --reward_schedule none.")
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Visibility curriculum 'frac:radius,...'; defaults to the variant's.")
    parser.add_argument("--reward_schedule", type=str, default=None,
                        help="'frac:distance:entropy[:tag_bonus],...'; defaults to the variant's recipe; "
                             "'none' for the constant coefficients.")
    parser.add_argument("--evasion_curriculum", type=str, default=None,
                        help="'frac:scale,...' for SmartAntTagEnv.evasion_scale; defaults to the variant's.")
    parser.add_argument("--mask_target_obs", action="store_true", default=True)
    parser.add_argument("--no_mask_target_obs", dest="mask_target_obs", action="store_false")
    parser.add_argument("--target_speed_scale", type=float, default=None)
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--lr_anneal", action="store_true")
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    enc = parser.add_argument_group(f"{encoder} encoder")
    if learned:
        enc.add_argument("--num_encodings", type=int, default=8)
        enc.add_argument("--dim_encoder", type=int, default=8,
                         help="Code = num_encodings x dim_encoder features (the ST arm's default 8 x 8 = 64).")
        enc.add_argument("--dim_hidden", type=int, default=128)
        enc.add_argument("--weight_channel", dest="weight_channel", action="store_true", default=True,
                         help="Feed the PF mass as an extra input channel (default on).")
        enc.add_argument("--no_weight_channel", dest="weight_channel", action="store_false",
                         help="Coordinates only; required to load an unweighted DeepSetAE / PointNetAE checkpoint.")
        enc.add_argument("--pooling", type=str, default=None, choices=extractor_cls.POOLINGS,
                         help=f"Pool operator; default {extractor_cls.POOLINGS[0]!r}.")
        enc.add_argument("--pretrained_model_path", type=str, default=None,
                         help="DeepSetAE / PointNetAE state_dict or Trainer checkpoint; encoder keys loaded strict.")
        enc.add_argument("--frozen", action="store_true", help="Freeze the pretrained encoder (heads only learn).")
        enc.add_argument("--encoder_lr_scale", type=float, default=1.0,
                         help="Encoder param group at this x the head LR (finetune fix; PPO only).")
    else:
        enc.add_argument("--k", dest="k_moments", type=int, default=4,
                         help="Moment orders 1..k per coordinate (k=2: mean + variance).")
    args = parser.parse_args()
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
    run_subdir = args.run_subdir or variants.run_subdir(encoder, args.variant)
    args.curriculum = variants.resolve_schedule(
        args.variant, args.curriculum, "default_curriculum", "0:100,0.3:100,0.7:3,1:3")
    variants.warn_if_not_evading(args.variant, args.evasion_curriculum, args.target_speed_scale)
    args.evasion_curriculum = variants.resolve_schedule(
        args.variant, args.evasion_curriculum, "default_evasion_curriculum", None)
    net_arch = [int(x) for x in args.net_arch.split(",")] if args.net_arch else None
    obs_mask = [-2, -1] if args.mask_target_obs else None
    if learned:
        if args.frozen and not args.pretrained_model_path:
            parser.error("--frozen without --pretrained_model_path would freeze a random encoder")
        if args.encoder_lr_scale <= 0:
            parser.error("--encoder_lr_scale must be positive")
        if args.encoder_lr_scale != 1.0 and not args.pretrained_model_path:
            parser.error("--encoder_lr_scale is a finetune fix for a PRETRAINED encoder")
        if args.encoder_lr_scale != 1.0 and args.frozen:
            parser.error("--frozen and --encoder_lr_scale contradict")
        if args.encoder_lr_scale != 1.0 and args.algorithm.upper() != "PPO":
            parser.error("--encoder_lr_scale is implemented for PPO only")
        if args.pretrained_model_path and not os.path.isfile(args.pretrained_model_path):
            parser.error(f"--pretrained_model_path {args.pretrained_model_path} does not exist")

    run_dir = _default_run_dir(args.seed, run_subdir, run_tag=args.run_tag)
    log_dir = args.log_dir or os.path.join(run_dir, "logs") + "/"
    model_save_path = args.model_save_path or os.path.join(run_dir, "models", f"{encoder}_agent.zip")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, "stdout.log")
    _tee_stdout_stderr(stdout_log_path)
    print(f"Mirroring stdout/stderr to {stdout_log_path}")
    run_config = vars(args).copy()
    run_config.pop("run_subdir", None)
    run_config.pop("list_variants", None)
    run_config.update(log_dir=log_dir, model_save_path=model_save_path, encoder=encoder)
    _write_run_config(run_dir, env_id=env_id, particle_filter_class=particle_filter_class.__name__,
                      run_subdir=run_subdir, git=_git_provenance(), **run_config)
    encoder_kwargs = (dict(num_encodings=args.num_encodings, dim_encoder=args.dim_encoder,
                           dim_hidden=args.dim_hidden, weight_channel=args.weight_channel,
                           pooling=args.pooling, pretrained_model_path=args.pretrained_model_path,
                           frozen=args.frozen, encoder_lr_scale=args.encoder_lr_scale)
                      if learned else dict(k_moments=args.k_moments))
    train_ant_tag_pool(
        encoder,
        algorithm=args.algorithm, total_timesteps=args.total_timesteps, n_envs=args.n_envs,
        learning_rate=args.learning_rate, batch_size=args.batch_size, ppo_n_steps=args.ppo_n_steps,
        num_particles=args.num_particles, arena_scale=args.arena_scale, device=args.device,
        seed=args.seed, log_dir=log_dir, model_save_path=model_save_path, eval_freq=args.eval_freq,
        save_freq=args.save_freq, use_vec_normalize=not args.no_vec_normalize,
        distance_coeff=args.distance_coeff, entropy_coeff=args.entropy_coeff,
        curriculum_schedule=_parse_curriculum(args.curriculum),
        reward_schedule=_parse_reward_schedule(args.reward_schedule),
        evasion_schedule=_parse_curriculum(args.evasion_curriculum) if args.evasion_curriculum else None,
        net_arch=net_arch, obs_mask_indices=obs_mask, progress_bar=args.progress_bar,
        target_speed_scale=args.target_speed_scale, env_id=env_id,
        particle_filter_class=particle_filter_class, lr_anneal=args.lr_anneal,
        target_kl=args.target_kl, n_epochs=args.n_epochs, n_eval_episodes=args.n_eval_episodes,
        **encoder_kwargs,
    )


if __name__ == "__main__":
    main()
