import argparse
import importlib
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from features_extractors.set_transformer_pretrained_processor import (
    PretrainedSetTransformerProcessor,
)
from particle_filters.base_pf import BaseParticleFilter  # For type hinting

# Refactored components
from wrappers.particle_filter_wrappers import PFPlusFeaturesObservationWrapper

# Placeholder for dynamic import of PF and mapper (similar to train_e2e.py)
# from particle_filters.ant_tag_pf import AntTagParticleFilter
# from environments.ant_tag_utils import ant_tag_pf_interaction_mapper

def get_env_module_and_pf(env_id_str: str): # Duplicated for now, ideally in a shared util
    if "ant_tag" in env_id_str.lower():
        pf_module_name = "particle_filters.ant_tag_pf"
        pf_class_name = "AntTagParticleFilter"
        mapper_module_name = "environments.ant_tag_utils"
        mapper_func_name = "ant_tag_pf_interaction_mapper"
    else:
        raise ValueError(f"Unsupported env_id for dynamic PF/mapper loading: {env_id_str}.")
    try:
        pf_module = importlib.import_module(pf_module_name)
        particle_filter_class = getattr(pf_module, pf_class_name)
        mapper_module = importlib.import_module(mapper_module_name)
        pf_interaction_mapper = getattr(mapper_module, mapper_func_name)
        return particle_filter_class, pf_interaction_mapper
    except ImportError as e:
        print(f"Error importing modules for {env_id_str}: {e}")
        raise

def make_pretrained_env(
    env_id: str,
    particle_filter_class: type[BaseParticleFilter],
    particle_filter_kwargs: dict,
    pf_interaction_mapper: callable,
    pretrained_st_model_path: str,
    st_processor_device: str,
    num_particles: int,
    rank: int,
    seed: int = 0,
    monitor_dir: str | None = None
):
    """Utility function for multiprocessed env creation for training with pretrained ST."""
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)

        st_processor = PretrainedSetTransformerProcessor(
            model_path=pretrained_st_model_path, 
            device=st_processor_device
        )
        
        env = PFPlusFeaturesObservationWrapper(
            env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs=particle_filter_kwargs,
            pretrained_st_processor=st_processor,
            num_particles=num_particles,
            pf_interaction_mapper=pf_interaction_mapper
        )
        if monitor_dir:
            log_file = os.path.join(monitor_dir, str(rank))
            env = Monitor(env, log_file)
        else:
            env = Monitor(env)
        return env
    return _init

def train_with_pretrained(
    env_id: str,
    pretrained_st_model_path: str,
    st_processor_device: str = "auto",
    # Particle Filter params
    num_particles: int = 100,
    particle_filter_config: dict = {},
    # SB3 Agent params
    algorithm: str = "PPO",
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64, # For PPO, this is n_steps * n_envs. For SAC, training batch_size.
    ppo_n_steps: int = 2048,
    policy_type: str = "MlpPolicy", # Standard policy as ST is part of wrapper
    seed: int = 0,
    log_dir: str = "./sb3_pretrained_logs/",
    model_save_path: str = "./sb3_pretrained_models/pretrained_agent",
    eval_freq: int = 20000,
    save_freq: int = 100000,
    use_vec_normalize: bool = True
):
    """
    Main training loop for RL with a Pretrained Set Transformer as a feature processor.
    """
    print(f"Starting training for environment: {env_id} using pretrained ST: {pretrained_st_model_path}")

    try:
        particle_filter_class, pf_interaction_mapper = get_env_module_and_pf(env_id)
    except (ValueError, ImportError) as e:
        print(f"Could not load PF components for {env_id}: {e}")
        return

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    env_fn = make_pretrained_env(
        env_id=env_id,
        particle_filter_class=particle_filter_class,
        particle_filter_kwargs=particle_filter_config,
        pf_interaction_mapper=pf_interaction_mapper,
        pretrained_st_model_path=pretrained_st_model_path,
        st_processor_device=st_processor_device,
        num_particles=num_particles,
        monitor_dir=monitor_dir
    )
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)

    if use_vec_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Standard policy, as feature extraction is handled by the wrapper
    policy_kwargs = {} 

    if algorithm.upper() == "PPO":
        model = PPO(policy_type, vec_env, policy_kwargs=policy_kwargs, 
                    learning_rate=learning_rate, n_steps=ppo_n_steps, 
                    batch_size=batch_size, 
                    verbose=1, tensorboard_log=log_dir, seed=seed)
    elif algorithm.upper() == "SAC":
        model = SAC(policy_type, vec_env, policy_kwargs=policy_kwargs, 
                    learning_rate=learning_rate, batch_size=batch_size, 
                    verbose=1, tensorboard_log=log_dir, seed=seed, buffer_size=int(1e6))
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose PPO or SAC.")

    print(f"Policy architecture: {model.policy}")

    checkpoint_callback = CheckpointCallback(save_freq=save_freq // n_envs, 
                                           save_path=os.path.join(os.path.dirname(model_save_path), f"{algorithm.lower()}_{env_id.replace('/', '_')}_pretrained_checkpoints"), 
                                           name_prefix="pretrained_model")
    
    eval_env_fn = make_pretrained_env(
        env_id=env_id,
        particle_filter_class=particle_filter_class,
        particle_filter_kwargs=particle_filter_config,
        pf_interaction_mapper=pf_interaction_mapper,
        pretrained_st_model_path=pretrained_st_model_path,
        st_processor_device=st_processor_device,
        num_particles=num_particles,
        monitor_dir=None,
        rank=n_envs + 1, 
        seed=seed
    )
    eval_vec_env = make_vec_env(eval_env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
    if use_vec_normalize:
        eval_vec_env = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(eval_vec_env, best_model_save_path=os.path.join(os.path.dirname(model_save_path),f"best_{algorithm.lower()}_{env_id.replace('/', '_')}_pretrained"),
                               log_path=log_dir, eval_freq=max(eval_freq // n_envs, 1),
                               deterministic=True, render=False, n_eval_episodes=5)

    print(f"Starting training with {algorithm} using pretrained ST processor...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    finally:
        model.save(model_save_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vec_env.save(os.path.join(os.path.dirname(model_save_path), f"vecnormalize_pretrained_{env_id.replace('/', '_')}.pkl"))
        print(f"Training finished. Model saved to {model_save_path}")
        vec_env.close()
        eval_vec_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Training with Pretrained Set Transformer Feature Processor")
    parser.add_argument("--env_id", type=str, required=True, help="Gym environment ID")
    parser.add_argument("--pretrained_st_model_path", type=str, required=True, help="Path to the pretrained Set Transformer .pth model file")
    parser.add_argument("--st_processor_device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for ST processor")
    
    # PF args (similar to train_e2e)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--pf_initial_spread_std", type=float, default=5.0)
    parser.add_argument("--pf_arena_min", type=float, default=-10.0)
    parser.add_argument("--pf_arena_max", type=float, default=10.0)
    parser.add_argument("--pf_obs_noise_std", type=float, default=0.1)
    parser.add_argument("--pf_process_noise_std", type=float, default=0.2)
    parser.add_argument("--pf_avoidance_factor", type=float, default=0.8)
    parser.add_argument("--pf_max_avoidance_dist", type=float, default=7.0)
    parser.add_argument("--pf_visibility_radius", type=float, default=3.0)

    # SB3 Agent args
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--policy_type", type=str, default="MlpPolicy", help="SB3 policy type (e.g., MlpPolicy, CnnPolicy)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="./sb3_pretrained_logs/")
    parser.add_argument("--model_save_path", type=str, default="./sb3_pretrained_models/pretrained_agent.zip")
    parser.add_argument("--eval_freq", type=int, default=20000)
    parser.add_argument("--save_freq", type=int, default=100000)
    parser.add_argument("--no_vec_normalize", action="store_true")

    args = parser.parse_args()

    pf_config = {
        "initial_spread_std": args.pf_initial_spread_std,
        "arena_limits": (args.pf_arena_min, args.pf_arena_max),
        "obs_noise_std": args.pf_obs_noise_std,
        "process_noise_std": args.pf_process_noise_std,
        "avoidance_factor": args.pf_avoidance_factor,
        "max_avoidance_dist": args.pf_max_avoidance_dist,
        "visibility_radius": args.pf_visibility_radius
    }

    train_with_pretrained(
        env_id=args.env_id,
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_processor_device=args.st_processor_device,
        num_particles=args.num_particles,
        particle_filter_config=pf_config,
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        policy_type=args.policy_type,
        seed=args.seed,
        log_dir=args.log_dir,
        model_save_path=args.model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_vec_normalize=not args.no_vec_normalize
    ) 