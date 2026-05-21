import argparse
import importlib
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from set_transformer.rl.feature_extractors.e2e import (
    CustomSetTransformerExtractor,
)
from set_transformer.rl.feature_extractors.pretrained import (
    PretrainedSetTransformerProcessor,
)
from set_transformer.rl.particle_filters.base import BaseParticleFilter
from set_transformer.rl.wrappers.particle_filter import (
    PFDictObservationWrapper,
    PFPlusFeaturesObservationWrapper,
)


# Placeholder for dynamic import of PF and mapper
def get_env_module_and_pf(env_id_str: str): # Duplicated, ideally in a shared util
    if "ant_tag" in env_id_str.lower():
        pf_module_name = "set_transformer.rl.particle_filters.ant_tag"
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

def make_eval_env(env_id, num_particles, particle_filter_class, particle_filter_config, pf_interaction_mapper,
                    training_mode, # 'e2e' or 'pretrained'
                    # For e2e specific:
                    st_params=None,
                    # For pretrained specific:
                    pretrained_st_model_path=None,
                    st_processor_device=None,
                    seed=0):
    """Helper function to create a single evaluation environment."""
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)

        if training_mode == 'e2e':
            env = PFDictObservationWrapper(
                env,
                particle_filter_class=particle_filter_class,
                particle_filter_kwargs=particle_filter_config,
                num_particles=num_particles,
                pf_interaction_mapper=pf_interaction_mapper
            )
        elif training_mode == 'pretrained':
            st_processor = PretrainedSetTransformerProcessor(
                model_path=pretrained_st_model_path, 
                device=st_processor_device
            )
            env = PFPlusFeaturesObservationWrapper(
                env,
                particle_filter_class=particle_filter_class,
                particle_filter_kwargs=particle_filter_config,
                pretrained_st_processor=st_processor,
                num_particles=num_particles,
                pf_interaction_mapper=pf_interaction_mapper
            )
        else:
            raise ValueError(f"Unknown training_mode: {training_mode}")
        return env
    return _init

def evaluate_agent(
    env_id: str,
    model_path: str,
    training_mode: str, # 'e2e' or 'pretrained'
    num_particles: int,
    particle_filter_config: dict,
    # For e2e specific (needed to reconstruct policy for CustomSetTransformerExtractor)
    st_output_dim: int = 64, st_hidden_dim: int = 128, st_num_heads: int = 4, 
    st_num_inds: int = 32, st_num_outputs: int = 1, st_ln: bool = False,
    obs_mlp_hidden_dims: list[int] = [64, 64], features_dim: int = 128,
    # For pretrained specific
    pretrained_st_model_path: str | None = None, # Required if training_mode == 'pretrained'
    st_processor_device: str = "auto",
    # Eval params
    n_eval_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
    vec_normalize_path: str | None = None,
    seed: int = 0
):
    set_random_seed(seed)

    particle_filter_class, pf_interaction_mapper = get_env_module_and_pf(env_id)

    # Create evaluation environment
    eval_env_fn = make_eval_env(
        env_id,
        num_particles,
        particle_filter_class,
        particle_filter_config,
        pf_interaction_mapper,
        training_mode,
        st_params= {
            "st_output_dim": st_output_dim, "st_hidden_dim": st_hidden_dim,
            "st_num_heads": st_num_heads, "st_num_inds": st_num_inds,
            "st_num_outputs": st_num_outputs, "st_ln": st_ln,
            "obs_mlp_hidden_dims": obs_mlp_hidden_dims, "features_dim": features_dim
        } if training_mode == 'e2e' else None,
        pretrained_st_model_path=pretrained_st_model_path if training_mode == 'pretrained' else None,
        st_processor_device=st_processor_device if training_mode == 'pretrained' else None,
        seed=seed
    )
    # Use DummyVecEnv for evaluation as it's simpler and typically used for single env eval
    eval_env = DummyVecEnv([eval_env_fn]) 

    if vec_normalize_path:
        if os.path.exists(vec_normalize_path):
            print(f"Loading VecNormalize stats from {vec_normalize_path}")
            eval_env = VecNormalize.load(vec_normalize_path, eval_env)
            eval_env.training = False # Important: do not update running stats during eval
            eval_env.norm_reward = False # Typically, don't normalize rewards for evaluation
        else:
            print(f"Warning: VecNormalize path {vec_normalize_path} not found. Evaluating without normalization.")

    # Load the trained agent
    # Determine algorithm from model_path or add as arg if necessary
    # This is a simplification; SB3 model loading might need more info or specific class.
    if "ppo" in model_path.lower():
        model = PPO.load(model_path, env=eval_env) # env is sometimes needed for policy reconstruction
    elif "sac" in model_path.lower():
        model = SAC.load(model_path, env=eval_env)
    else:
        # Fallback or raise error if algo cannot be inferred
        print(f"Could not infer algorithm from model_path: {model_path}. Trying PPO by default.")
        try:
            model = PPO.load(model_path, env=eval_env)
        except Exception as e1:
            print(f"Failed to load as PPO ({e1}), trying SAC...")
            try:
                model = SAC.load(model_path, env=eval_env)
            except Exception as e2:
                raise ValueError(f"Could not load model. Ensure model_path is correct and algo type matches. Errors: PPO-{e1}, SAC-{e2}")
    
    print(f"Loaded model from {model_path}")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        render=render,
        deterministic=deterministic,
        return_episode_rewards=False, # Set to True if you want list of all episode rewards
        warn=True
    )

    print(f"Evaluation Results ({n_eval_episodes} episodes):")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a trained RL agent with Particle Filter processing")
    parser.add_argument("--env_id", type=str, required=True, help="Gym environment ID")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained agent .zip file")
    parser.add_argument("--training_mode", type=str, required=True, choices=['e2e', 'pretrained'], help="Mode used for training the agent")
    
    # PF args
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--pf_initial_spread_std", type=float, default=5.0)
    parser.add_argument("--pf_arena_min", type=float, default=-10.0)
    parser.add_argument("--pf_arena_max", type=float, default=10.0)
    # ... (add other PF config args as in training scripts)

    # E2E specific ST params (needed if training_mode='e2e' for policy reconstruction)
    parser.add_argument("--st_output_dim", type=int, default=64)
    # ... (add other st_* and obs_mlp_hidden_dims, features_dim args as in train_e2e.py)

    # Pretrained specific
    parser.add_argument("--pretrained_st_model_path", type=str, help="Path to pretrained ST model (required if training_mode='pretrained')")
    parser.add_argument("--st_processor_device", type=str, default="auto")

    # Eval params
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_false", dest="deterministic_eval", help="Use stochastic actions for evaluation")
    parser.add_argument("--vec_normalize_path", type=str, help="Path to VecNormalize stats .pkl file")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.training_mode == 'pretrained' and not args.pretrained_st_model_path:
        parser.error("--pretrained_st_model_path is required when training_mode is 'pretrained'")

    # Reconstruct pf_config (simplified, match with training scripts)
    pf_config = {
        "initial_spread_std": args.pf_initial_spread_std,
        "arena_limits": (args.pf_arena_min, args.pf_arena_max),
        # ... add other pf params from parser
    }
    # Reconstruct st_params for e2e if needed (simplified)
    st_params_e2e = {
        "st_output_dim": args.st_output_dim,
        # ... add other st params from parser
    }

    evaluate_agent(
        env_id=args.env_id,
        model_path=args.model_path,
        training_mode=args.training_mode,
        num_particles=args.num_particles,
        particle_filter_config=pf_config,
        st_output_dim=args.st_output_dim if args.training_mode == 'e2e' else 64, # provide defaults or ensure parsed
        # ... pass all other relevant st_params for e2e
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_processor_device=args.st_processor_device,
        n_eval_episodes=args.n_eval_episodes,
        render=args.render,
        deterministic=args.deterministic_eval,
        vec_normalize_path=args.vec_normalize_path,
        seed=args.seed
    ) 