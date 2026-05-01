import argparse
import importlib
import os

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from features_extractors.set_transformer_e2e_extractor import (
    CustomSetTransformerExtractor,
)
from particle_filters.ant_tag_pf import (
    AntTagParticleFilter,  # Assuming we have this for AntTag
)

# BaseParticleFilter is needed for type hinting if we pass the class directly
from particle_filters.base_pf import BaseParticleFilter

# Refactored components
from wrappers.particle_filter_wrappers import PFDictObservationWrapper


def get_env_and_pf_components(env_module_name: str):
    """
    Dynamically imports the environment class and its PF interaction mapper
    from the specified environment module.
    It also selects the appropriate Particle Filter class from this project.
    """
    # Derive environment class name and mapper function name from module name
    # e.g., env_module_name 'ant_tag_env' -> EnvClass 'AntTagEnv', mapper_func 'ant_tag_env_pf_interaction_mapper'
    
    # Ensure casing is handled for module name if user provides it differently
    base_name_parts = env_module_name.split('.')[-1].split('_') # e.g., "ant_tag_env" -> ["ant", "tag", "env"]
    
    # Heuristic for common Gym env naming (e.g., AntTagEnv, CarFlagEnv)
    # If "env" is the last part, remove it for class name generation, otherwise use all parts.
    if base_name_parts[-1].lower() == "env":
        class_name_parts = base_name_parts[:-1]
    else:
        class_name_parts = base_name_parts
    
    env_class_name = "".join(part.capitalize() for part in class_name_parts) + "Env" # AntTagEnv
    
    mapper_func_name = f"{env_module_name.replace('.', '_')}_pf_interaction_mapper" # ant_tag_env_pf_interaction_mapper

    # Select Particle Filter class based on the environment module name
    # This mapping should be extended for new environments.
    if "ant_tag" in env_module_name.lower():
        # Specific PF for AntTag, located within this project
        particle_filter_class = AntTagParticleFilter
        gym_env_id = "AntTag-v0" # Assumed registered name for gym.make()
    # elif "car_flag" in env_module_name.lower():
    #     # particle_filter_class = CarFlagParticleFilter # To be created in particle_filters/
    #     gym_env_id = "CarFlag-v0"
    # elif "ant_heaven_hell" in env_module_name.lower():
    #     # particle_filter_class = AntHeavenHellParticleFilter # To be created in particle_filters/
    #     gym_env_id = "AntHeavenHell-v0"
    # elif "two_boxes" in env_module_name.lower():
    #     # particle_filter_class = TwoBoxesParticleFilter # To be created in particle_filters/
    #     gym_env_id = "TwoBoxes-v0"
    # elif "ur5_reacher" in env_module_name.lower():
    #     # particle_filter_class = UR5ReacherParticleFilter # To be created in particle_filters/
    #     gym_env_id = "UR5Reacher-v0"
    else:
        raise ValueError(f"Unsupported env_module_name for PF selection: {env_module_name}. Please update get_env_and_pf_components.")

    try:
        # Import the environment module (user is responsible for it being in PYTHONPATH)
        env_module = importlib.import_module(env_module_name)
        
        # Get the environment class from the imported module
        # env_class = getattr(env_module, env_class_name) # We'll use gym.make with gym_env_id instead

        # Get the PF interaction mapper function from the imported module
        pf_interaction_mapper = getattr(env_module, mapper_func_name)
        
        # The particle_filter_class is sourced from this project's particle_filters
        return gym_env_id, particle_filter_class, pf_interaction_mapper
        
    except ImportError as e:
        print(f"Error importing environment module '{env_module_name}': {e}")
        print(f"Please ensure '{env_module_name}.py' is in your PYTHONPATH, contains a class '{env_class_name}', and a function '{mapper_func_name}'.")
        raise
    except AttributeError as e:
        print(f"Error accessing components in '{env_module_name}': {e}")
        print(f"Please ensure class '{env_class_name}' and function '{mapper_func_name}' are defined in '{env_module_name}.py'.")
        raise


def make_e2e_env(gym_env_id: str, # Now gym_env_id
                   particle_filter_class: type[BaseParticleFilter],
                   particle_filter_kwargs: dict,
                   pf_interaction_mapper: callable,
                   num_particles: int,
                   rank: int, # for make_vec_env
                   seed: int = 0,
                   monitor_dir: str | None = None,
                   env_kwargs: dict | None = None): # Added env_kwargs
    """Utility function for multiprocessed env creation for end-to-end training."""
    def _init():
        # Base environment
        # The user's environment module (e.g., ant_tag_env.py) should register itself with Gym
        # if it wants to be callable by a simple ID like "AntTag-v0".
        # Alternatively, gym.make can sometimes resolve "module:ClassName"
        env = gym.make(gym_env_id, **(env_kwargs or {}))
        env.reset(seed=seed + rank)
        
        # Wrap with Particle Filter Dict Observation Wrapper
        env = PFDictObservationWrapper(
            env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs=particle_filter_kwargs,
            num_particles=num_particles,
            pf_interaction_mapper=pf_interaction_mapper
        )
        if monitor_dir:
            # Ensure monitor_dir exists for each process if SubprocVecEnv is used.
            # os.makedirs(monitor_dir, exist_ok=True) # Handled in main train function for global monitor dir
            log_file = os.path.join(monitor_dir, str(rank))
            env = Monitor(env, log_file)
        else:
            env = Monitor(env) # Basic monitoring even if no specific dir
        return env
    return _init


def train_e2e(
    env_module_name: str, # Changed from env_id to env_module_name
    # Particle Filter params
    num_particles: int,
    particle_filter_config: dict, # Kwargs for the specific PF class
    # CustomSetTransformerExtractor params
    st_output_dim: int = 64,
    st_hidden_dim: int = 128,
    st_num_heads: int = 4,
    st_num_inds: int = 32,
    st_num_outputs: int = 1, # Number of outputs from the ST head (e.g., 1 for a value prediction)
    st_ln: bool = False,
    obs_mlp_hidden_dims: list[int] = [64, 64], # MLP for the original observation part
    features_dim: int = 128, # Final features_dim for SB3 policy
    # SB3 Agent params
    algorithm: str = "PPO",
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64, 
    ppo_n_steps: int = 2048,
    seed: int = 0,
    log_dir: str = "./sb3_e2e_logs/",
    model_save_path: str = "./sb3_e2e_models/e2e_agent",
    eval_freq: int = 20000,
    save_freq: int = 100000,
    use_vec_normalize: bool = True,
    env_kwargs: dict | None = None # Added env_kwargs
):
    """
    Main training loop for end-to-end RL with Set Transformer Feature Extractor.
    """
    print(f"Starting end-to-end training for environment module: {env_module_name}")
    
    try:
        gym_env_id, particle_filter_class, pf_interaction_mapper = get_env_and_pf_components(env_module_name)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Could not load components for {env_module_name}: {e}")
        print("Please ensure your environment module is in PYTHONPATH, defines the required class and mapper function,")
        print("and the environment is registered with Gym or accessible via 'module:ClassName'.")
        return

    # Prepare paths
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_save_path)
    os.makedirs(model_dir, exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    # Create vectorized environments
    env_fn_args = {
        'gym_env_id': gym_env_id,
        'particle_filter_class': particle_filter_class,
        'particle_filter_kwargs': particle_filter_config,
        'pf_interaction_mapper': pf_interaction_mapper,
        'num_particles': num_particles,
        'monitor_dir': monitor_dir,
        'env_kwargs': env_kwargs or {}
    }
    vec_env = make_vec_env(make_e2e_env, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_fn_args)
    
    # Note: make_vec_env passes rank and seed to make_e2e_env's _init via its wrapper.
    # We pass other make_e2e_env args via env_kwargs in make_vec_env.

    if use_vec_normalize:
        print("Applying VecNormalize to the training environment.")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Policy kwargs for the agent, including the custom feature extractor
    policy_kwargs = {
        "features_extractor_class": CustomSetTransformerExtractor,
        "features_extractor_kwargs": {
            "st_output_dim": st_output_dim,
            "st_hidden_dim": st_hidden_dim,
            "st_num_heads": st_num_heads,
            "st_num_inds": st_num_inds,
            "st_num_outputs": st_num_outputs, # This should match the number of heads for PMA if used directly
            "st_ln": st_ln,
            "obs_mlp_hidden_dims": obs_mlp_hidden_dims,
            "features_dim": features_dim
        }
    }

    # Create the RL agent
    if algorithm.upper() == "PPO":
        model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, 
                    learning_rate=learning_rate, n_steps=ppo_n_steps // n_envs, # n_steps per env
                    batch_size=batch_size, # mini-batch size for PPO updates
                    n_epochs=10, # Default PPO epochs
                    verbose=1, tensorboard_log=log_dir, seed=seed)
    elif algorithm.upper() == "SAC":
        # SAC typically doesn't use n_steps in the same way as PPO for rollout collection.
        # batch_size is the training mini-batch size.
        model = SAC("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, 
                    learning_rate=learning_rate, batch_size=batch_size,
                    buffer_size=int(1e6), # Default SAC buffer size
                    learning_starts=10000, # Default SAC learning starts
                    verbose=1, tensorboard_log=log_dir, seed=seed)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose PPO or SAC.")

    print(f"Policy architecture: {model.policy}")
    
    # Checkpoint save path needs to be robust to env_module_name containing dots
    safe_env_name = env_module_name.replace('.', '_')
    checkpoint_save_dir = os.path.join(model_dir, f"{algorithm.lower()}_{safe_env_name}_checkpoints")

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=max(save_freq // n_envs, 1), 
                                           save_path=checkpoint_save_dir, 
                                           name_prefix="e2e_model")
    
    # Create a separate eval env (not normalized initially for PPO evaluation)
    eval_env_fn_args = {
        'gym_env_id': gym_env_id,
        'particle_filter_class': particle_filter_class,
        'particle_filter_kwargs': particle_filter_config,
        'pf_interaction_mapper': pf_interaction_mapper,
        'num_particles': num_particles,
        'monitor_dir': None, # No separate monitoring for the raw eval env here
        'env_kwargs': env_kwargs or {}
        # rank and seed are handled by make_vec_env
    }

    # Eval env should not be a SubprocVecEnv if only 1 env for faster debugging, but for consistency with training:
    eval_vec_env = make_vec_env(make_e2e_env, n_envs=1, seed=seed + n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=eval_env_fn_args)

    if use_vec_normalize:
        print("Applying VecNormalize to the evaluation environment (in inference mode).")
        # Load running averages from training VecNormalize, but do not update them during eval
        # The VecNormalize wrapper for eval_vec_env needs to be loaded from the training one if we want to use its stats.
        # For now, we'll normalize it independently, which is standard for eval unless loading stats.
        # If vec_env was VecNormalize, eval_vec_env is now also wrapped by VecNormalize by make_vec_env's wrapper logic if not careful
        # It's better to handle VecNormalize for eval separately AFTER model.learn if we want to load stats.
        # For EvalCallback, it's often simpler to pass a VecNormalize instance that is synced with the training one.
        
        # Create a new VecNormalize for eval, then load stats if available
        eval_vec_env_normalized = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False)
        # We will sync it with the training vec_env before evaluation callback uses it.

    eval_callback_env = eval_vec_env_normalized if use_vec_normalize else eval_vec_env

    eval_callback = EvalCallback(eval_callback_env, 
                               best_model_save_path=os.path.join(model_dir, f"best_{algorithm.lower()}_{safe_env_name}"),
                               log_path=log_dir, eval_freq=max(eval_freq // n_envs, 1),
                               deterministic=True, render=False, n_eval_episodes=5,
                               # For VecNormalize, EvalCallback needs to sync the running stats
                               callback_on_new_best=None, # Can add a custom callback here
                               # callback_after_eval=None # Can add custom callback after each eval
                               )
    if use_vec_normalize:
        # Custom handler to sync VecNormalize for evaluation
        class SyncVecNormalizeCallback(EvalCallback):
            def _on_step(self) -> bool:
                if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                    if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                        # Sync obs mean and std
                        self.eval_env.obs_rms = self.training_env.obs_rms
                        self.eval_env.epsilon = self.training_env.epsilon
                    # For SB3 > 2.0, VecNormalize stores running_mean and running_var
                    # For older SB3 or custom VecNormalize, it might be obs_rms.
                    # Let's assume SB3 VecNormalize structure:
                    if hasattr(self.training_env, 'normalize_obs') and hasattr(self.eval_env, 'normalize_obs'):
                         # For SB3 VecNormalize, sync obs_rms
                        if hasattr(self.training_env, 'obs_rms') and self.training_env.obs_rms is not None:
                            self.eval_env.obs_rms = self.training_env.obs_rms
                        # For older versions or if reward normalization is also used in eval_env
                        if hasattr(self.training_env, 'ret_rms') and self.training_env.ret_rms is not None and self.eval_env.norm_reward:
                             self.eval_env.ret_rms = self.training_env.ret_rms

                return super()._on_step()
        
        eval_callback = SyncVecNormalizeCallback(eval_callback_env, 
                                   best_model_save_path=os.path.join(model_dir, f"best_{algorithm.lower()}_{safe_env_name}"),
                                   log_path=log_dir, eval_freq=max(eval_freq // n_envs, 1),
                                   deterministic=True, render=False, n_eval_episodes=5)


    # Train the agent
    print(f"Starting training with {algorithm}...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    finally:
        # Save the final model and VecNormalize stats
        final_model_path = os.path.join(model_dir, f"{algorithm.lower()}_{safe_env_name}_final.zip")
        model.save(final_model_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vec_normalize_path = os.path.join(model_dir, f"vecnormalize_{safe_env_name}.pkl")
            vec_env.save(vec_normalize_path)
            print(f"VecNormalize stats saved to {vec_normalize_path}")
        print(f"Training finished. Final model saved to {final_model_path}")
        
        vec_env.close() 
        eval_vec_env.close() # eval_vec_env_normalized shares the same underlying envs
        # No need to close eval_vec_env_normalized separately if it's just a wrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End RL Training with Set Transformer Feature Extractor")
    # Changed --env_id to --env_module_name
    parser.add_argument("--env_module_name", type=str, required=True, 
                        help="Python module name of the environment (e.g., 'ant_tag_env' or 'my_custom_envs.ant_tag_env'). "
                             "This module should be in PYTHONPATH and contain the Env class and pf_interaction_mapper function. "
                             "The environment should also be registered with Gym for gym.make() to work (e.g. as 'AntTag-v0').")
    
    # PF args - these are for AntTagParticleFilter primarily
    parser.add_argument("--num_particles", type=int, default=100, help="Number of particles for the filter")
    parser.add_argument("--pf_initial_spread_std", type=float, default=5.0, help="PF: Initial spread of particles (std dev)")
    parser.add_argument("--pf_arena_min", type=float, default=-10.0, help="PF: Arena min boundary for particle state")
    parser.add_argument("--pf_arena_max", type=float, default=10.0, help="PF: Arena max boundary for particle state")
    parser.add_argument("--pf_obs_noise_std", type=float, default=0.1, help="PF: Std dev of observation noise model")
    parser.add_argument("--pf_process_noise_std", type=float, default=0.2, help="PF: Std dev of process noise model")
    parser.add_argument("--pf_avoidance_factor", type=float, default=0.8, help="PF: Target avoidance factor for particle movement")
    
    # ST Extractor args
    parser.add_argument("--st_output_dim", type=int, default=64, help="Output dimension of the Set Transformer's PMA/final processing for features.")
    parser.add_argument("--st_hidden_dim", type=int, default=128, help="Hidden dimension within Set Transformer blocks (MAB, ISAB).")
    parser.add_argument("--st_num_heads", type=int, default=4, help="Number of attention heads in MABs.")
    parser.add_argument("--st_num_inds", type=int, default=32, help="Number of inducing points in ISABs.")
    parser.add_argument("--st_num_outputs", type=int, default=1, help="Number of output vectors from PMA (if PMA is the final layer of ST feature part). Usually 1 for SB3 features.")
    parser.add_argument("--st_ln", action='store_true', help="Use LayerNorm in Set Transformer blocks.")
    parser.add_argument("--obs_mlp_hidden_dims", type=int, nargs='*', default=[64, 64], help="Hidden layer dimensions for the MLP processing the original (non-particle) observation. Empty list means no MLP.")
    parser.add_argument("--features_dim", type=int, default=128, help="Final combined features dimension for the SB3 policy.")

    # SB3 Agent args
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"], help="RL algorithm to use.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timesteps for training.")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (PPO: minibatch size for updates; SAC: training minibatch size).") # PPO default often 64, SAC 256
    parser.add_argument("--ppo_n_steps", type=int, default=2048, help="PPO: Number of steps to run for each environment per update (i.e. rollout buffer size = ppo_n_steps * n_envs).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--log_dir", type=str, default="./sb3_e2e_logs/", help="Directory for TensorBoard logs.")
    parser.add_argument("--model_save_path_prefix", type=str, default="./sb3_e2e_models/e2e_agent", help="Prefix for saving models (env_id and algo will be appended).")
    parser.add_argument("--eval_freq", type=int, default=20000, help="Evaluate the agent every N steps (per environment).")
    parser.add_argument("--save_freq", type=int, default=100000, help="Save a checkpoint every N steps (per environment).")
    parser.add_argument("--no_vec_normalize", action='store_true', help="Do not use VecNormalize.")
    
    # Env Kwargs (passed as JSON string)
    parser.add_argument("--env_kwargs_json", type=str, default='{}', help="JSON string of keyword arguments for environment initialization. E.g., \'{\"rendering\": true}\'")


    args = parser.parse_args()

    # Construct particle_filter_config from args
    # This is specific to AntTagParticleFilter currently.
    # A more general solution would involve a config file or more dynamic arg parsing.
    pf_config = {
        "initial_spread_std": args.pf_initial_spread_std,
        "arena_limits": [args.pf_arena_min, args.pf_arena_max],
        "obs_noise_std": args.pf_obs_noise_std,
        "process_noise_std": args.pf_process_noise_std,
        "avoidance_factor": args.pf_avoidance_factor,
        # "num_particles": args.num_particles # num_particles is passed separately to the wrapper
    }
    
    import json
    try:
        env_kwargs = json.loads(args.env_kwargs_json)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse --env_kwargs_json \'{args.env_kwargs_json}\'. Using empty dict. Ensure it is valid JSON.")
        env_kwargs = {}


    # Adjust model_save_path to include algorithm and env_module_name for clarity
    safe_env_name = args.env_module_name.replace('.', '_')
    model_save_file = f"{args.model_save_path_prefix}_{args.algorithm.lower()}_{safe_env_name}"


    train_e2e(
        env_module_name=args.env_module_name,
        num_particles=args.num_particles,
        particle_filter_config=pf_config,
        st_output_dim=args.st_output_dim,
        st_hidden_dim=args.st_hidden_dim,
        st_num_heads=args.st_num_heads,
        st_num_inds=args.st_num_inds,
        st_num_outputs=args.st_num_outputs,
        st_ln=args.st_ln,
        obs_mlp_hidden_dims=args.obs_mlp_hidden_dims,
        features_dim=args.features_dim,
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        seed=args.seed,
        log_dir=args.log_dir,
        model_save_path=model_save_file, # Use the constructed path
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_vec_normalize=not args.no_vec_normalize,
        env_kwargs=env_kwargs
    ) 