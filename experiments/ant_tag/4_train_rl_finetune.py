"""
RL training with a fine-tunable (pretrained) Set Transformer for Ant-Tag.

Same pipeline as train_ant_tag_pretrained.py except the ST is inside the
policy as a custom SB3 BaseFeaturesExtractor — so PPO backprops through it
and its weights update during RL training.

Usage:
    python training/train_ant_tag_finetune.py \
        --pretrained_st_model_path <path_to_checkpoint> \
        --algorithm PPO --total_timesteps 3000000 --n_envs 4
"""

import argparse
import importlib

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401 — registers pdomains-ant-tag-v0
from set_transformer.models import PFSetTransformer
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter
from set_transformer.rl.wrappers.particle_filter import PFDictObservationWrapper

# Reuse shared components from the frozen-ST training script. The sibling
# module name starts with a digit, so the normal ``from X import Y`` syntax
# can't be used; importlib accepts the name as a string.
_train_rl_frozen = importlib.import_module("4_train_rl_frozen")
CurriculumCallback = _train_rl_frozen.CurriculumCallback
CurriculumVisibilityWrapper = _train_rl_frozen.CurriculumVisibilityWrapper
PFRewardShapingWrapper = _train_rl_frozen.PFRewardShapingWrapper
_CurriculumRouter = _train_rl_frozen._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_frozen.ant_tag_pf_interaction_mapper


# ---------------------------------------------------------------------------
# Custom features extractor: pretrained ST inside the policy, trainable.
# ---------------------------------------------------------------------------

class FineTunableSTFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that runs a PFSetTransformer encoder over the particle
    set and concatenates its flattened output with the base observation.

    Gradients flow through the encoder, so the pretrained weights are
    fine-tuned during RL training.

    Expected observation space: gym.spaces.Dict({
        "obs": Box(shape=(obs_dim,)),
        "particles": Box(shape=(num_particles, dim_particles)),
    })
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pretrained_st_model_path: str | None = None,
        num_particles: int = 100,
        dim_particles: int = 2,
        num_encodings: int = 8,
        dim_encoder: int = 2,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = True,
    ):
        obs_dim = observation_space["obs"].shape[0]
        st_output_dim = num_encodings * dim_encoder
        super().__init__(observation_space, features_dim=obs_dim + st_output_dim)

        self.pf_st = PFSetTransformer(
            num_particles=num_particles,
            dim_particles=dim_particles,
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )

        if pretrained_st_model_path:
            loaded = torch.load(
                pretrained_st_model_path, map_location="cpu", weights_only=False
            )
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                state_dict = loaded["model_state_dict"]
            elif isinstance(loaded, dict):
                state_dict = loaded
            else:
                raise ValueError(
                    f"Expected state_dict or trainer checkpoint, got {type(loaded)}"
                )
            self.pf_st.load_state_dict(state_dict)
            print(f"FineTunableSTFeaturesExtractor: loaded {pretrained_st_model_path}")
        else:
            print("FineTunableSTFeaturesExtractor: training from scratch (no pretrained weights)")

        self.encoder = self.pf_st.set_transformer
        self._obs_dim = obs_dim
        self._st_output_dim = st_output_dim

    def forward(self, obs_dict: dict) -> torch.Tensor:
        base_obs = obs_dict["obs"]          # [B, obs_dim]
        particles = obs_dict["particles"]   # [B, N, dim_particles]
        enc = self.encoder(particles)       # [B, num_encodings, dim_encoder]
        flat = enc.reshape(enc.size(0), -1) # [B, num_encodings * dim_encoder]
        return torch.cat([base_obs, flat], dim=-1)


# ---------------------------------------------------------------------------
# Environment factory (uses PFDictObservationWrapper instead of PlusFeatures)
# ---------------------------------------------------------------------------

def make_ant_tag_finetune_env(
    num_particles: int,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    tag_bonus_coeff: float = 0.0,
    initial_visibility_radius: float = 100.0,
    obs_mask_indices: list[int] | None = None,
):
    """Return a callable that creates a wrapped AntTag env with Dict obs."""

    def _init():
        env = gym.make("pdomains-ant-tag-v0", rendering=False)
        env.reset(seed=seed + rank)

        env = CurriculumVisibilityWrapper(env, initial_visibility_radius=initial_visibility_radius)

        env = PFDictObservationWrapper(
            env=env,
            particle_filter_class=AntTagParticleFilter,
            particle_filter_kwargs={},
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=obs_mask_indices,
        )

        env = PFRewardShapingWrapper(
            env, distance_coeff=distance_coeff, entropy_coeff=entropy_coeff,
            tag_bonus_coeff=tag_bonus_coeff,
        )

        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        else:
            env = Monitor(env)

        env = _CurriculumRouter(env)
        return env

    return _init


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_ant_tag_finetune(
    pretrained_st_model_path: str | None,
    algorithm: str = "PPO",
    total_timesteps: int = 3_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    ppo_n_steps: int = 2048,
    num_particles: int = 100,
    dim_particles: int = 2,
    num_encodings: int = 8,
    dim_encoder: int = 2,
    num_inds: int = 32,
    dim_hidden: int = 128,
    num_heads: int = 4,
    ln: bool = True,
    seed: int = 0,
    log_dir: str = "./sb3_ant_tag_finetune_logs/",
    model_save_path: str = "./sb3_ant_tag_finetune_models/finetune_agent.zip",
    eval_freq: int = 20_000,
    save_freq: int = 100_000,
    use_vec_normalize: bool = True,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    curriculum_schedule: list[tuple[float, float]] | None = None,
    reward_schedule: list[tuple[float, ...]] | None = None,
    net_arch: list[int] | None = None,
    obs_mask_indices: list[int] | None = None,
):
    print(f"Training {algorithm} on AntTag with fine-tunable ST: {pretrained_st_model_path}")
    print(f"Initial reward: distance_coeff={distance_coeff}, entropy_coeff={entropy_coeff}")
    if net_arch:
        print(f"Policy net_arch: {net_arch}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    initial_vis = 100.0
    if curriculum_schedule:
        initial_vis = curriculum_schedule[0][1]

    env_kw = dict(
        num_particles=num_particles,
        distance_coeff=distance_coeff,
        entropy_coeff=entropy_coeff,
        tag_bonus_coeff=0.0,
        initial_visibility_radius=initial_vis,
        obs_mask_indices=obs_mask_indices,
    )

    env_fn = make_ant_tag_finetune_env(**env_kw, seed=seed, monitor_dir=monitor_dir)
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)
    if use_vec_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_env_kw = dict(env_kw)
    eval_env_kw["initial_visibility_radius"] = 3.0
    eval_env_fn = make_ant_tag_finetune_env(
        **eval_env_kw, rank=n_envs + 1, seed=seed
    )
    eval_vec_env = make_vec_env(eval_env_fn, n_envs=1)
    if use_vec_normalize:
        eval_vec_env = VecNormalize(
            eval_vec_env, training=False, norm_obs=True, norm_reward=False
        )

    policy_kwargs = {
        "features_extractor_class": FineTunableSTFeaturesExtractor,
        "features_extractor_kwargs": dict(
            pretrained_st_model_path=pretrained_st_model_path,
            num_particles=num_particles,
            dim_particles=dim_particles,
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        ),
    }
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    if algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy", vec_env,
            learning_rate=learning_rate,
            n_steps=ppo_n_steps,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MultiInputPolicy", vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    print(f"Policy architecture:\n{model.policy}")
    n_st_params = sum(p.numel() for p in model.policy.features_extractor.parameters() if p.requires_grad)
    print(f"Trainable ST feature-extractor params: {n_st_params:,}")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(os.path.dirname(model_save_path), "checkpoints"),
        name_prefix="ant_tag_finetune",
    )
    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(
            os.path.dirname(model_save_path), "best_model"
        ),
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    curriculum_cb = CurriculumCallback(
        total_timesteps=total_timesteps,
        schedule=curriculum_schedule,
        reward_schedule=reward_schedule,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, curriculum_cb],
            progress_bar=True,
        )
    finally:
        model.save(model_save_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vec_env.save(
                os.path.join(os.path.dirname(model_save_path), "vecnormalize.pkl")
            )
        print(f"Model saved to {model_save_path}")
        vec_env.close()
        eval_vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RL with fine-tunable pretrained Set Transformer on Ant-Tag"
    )

    parser.add_argument(
        "--pretrained_st_model_path", type=str, default=None,
        help="Optional pretrained ST checkpoint. If omitted, ST is trained from random init.",
    )

    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--dim_particles", type=int, default=2)
    parser.add_argument("--num_encodings", type=int, default=8)
    parser.add_argument("--dim_encoder", type=int, default=2)
    parser.add_argument("--num_inds", type=int, default=32)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--no_ln", action="store_true", help="Disable layer norm in ST.")

    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log_dir", type=str, default="./sb3_ant_tag_finetune_logs/")
    parser.add_argument(
        "--model_save_path", type=str,
        default="./sb3_ant_tag_finetune_models/finetune_agent.zip",
    )
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--no_vec_normalize", action="store_true")
    parser.add_argument(
        "--net_arch", type=str, default=None,
        help="Policy/value MLP sizes, e.g. '256,256'. Default: SB3 default (64,64).",
    )

    parser.add_argument("--distance_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.0)

    parser.add_argument(
        "--curriculum", type=str, default=None,
        help="frac:radius pairs, e.g. '0:100,0.3:100,0.7:3,1:3'.",
    )
    parser.add_argument(
        "--reward_schedule", type=str, default=None,
        help="frac:dist:ent:tag_bonus quads, e.g. '0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50'.",
    )
    parser.add_argument(
        "--mask_target_obs", action="store_true", default=True,
        help="Zero out obs[-2:] (true target) for the agent (PF still sees it). Default on.",
    )
    parser.add_argument(
        "--no_mask_target_obs", dest="mask_target_obs", action="store_false",
        help="Disable target masking.",
    )

    args = parser.parse_args()

    curriculum_str = args.curriculum or "0:100,0.3:100,0.7:3,1:3"
    curriculum_schedule = []
    try:
        for pair in curriculum_str.split(","):
            frac, rad = pair.strip().split(":")
            curriculum_schedule.append((float(frac), float(rad)))
    except Exception as e:
        parser.error(f"Invalid --curriculum format: {e}")

    reward_str = args.reward_schedule or "0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50"
    reward_schedule = []
    try:
        for entry in reward_str.split(","):
            parts = [float(p) for p in entry.strip().split(":")]
            if len(parts) == 3:
                parts.append(0.0)
            reward_schedule.append(tuple(parts))
    except Exception as e:
        parser.error(f"Invalid --reward_schedule format: {e}")

    net_arch = None
    if args.net_arch:
        net_arch = [int(x) for x in args.net_arch.split(",")]

    obs_mask = [-2, -1] if args.mask_target_obs else None

    train_ant_tag_finetune(
        pretrained_st_model_path=args.pretrained_st_model_path,
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        num_particles=args.num_particles,
        dim_particles=args.dim_particles,
        num_encodings=args.num_encodings,
        dim_encoder=args.dim_encoder,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        ln=not args.no_ln,
        seed=args.seed,
        log_dir=args.log_dir,
        model_save_path=args.model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_vec_normalize=not args.no_vec_normalize,
        distance_coeff=args.distance_coeff,
        entropy_coeff=args.entropy_coeff,
        curriculum_schedule=curriculum_schedule,
        reward_schedule=reward_schedule,
        net_arch=net_arch,
        obs_mask_indices=obs_mask,
    )
