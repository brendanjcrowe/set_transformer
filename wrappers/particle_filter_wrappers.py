import gymnasium as gym
import numpy as np
import torch  # For PFPlusFeaturesObservationWrapper if converting to torch tensor for ST

# Assuming PretrainedSetTransformerProcessor is available for the second wrapper
from features_extractors.set_transformer_pretrained_processor import (
    PretrainedSetTransformerProcessor,
)

# Assuming BaseParticleFilter is in particle_filters.base_pf
from particle_filters.base_pf import BaseParticleFilter


class PFDictObservationWrapper(gym.Wrapper):
    """
    Wraps an environment to include particle filter beliefs in a Dict observation space.
    The observation space will be a gym.spaces.Dict with:
    - "obs": The original environment observation.
    - "particles": The particle set [num_particles, particle_dim] from the PF.

    Designed for end-to-end training where the Set Transformer is part of the policy.
    """
    def __init__(self,
                 env: gym.Env,
                 particle_filter_class: type[BaseParticleFilter],
                 particle_filter_kwargs: dict,
                 num_particles: int,
                 # This function will be responsible for extracting necessary info from
                 # base_env_obs for pf.update() and pf.predict()
                 # It should return a dict like {"predict": {kwargs for pf.predict}, "update": {kwargs for pf.update}}
                 pf_interaction_mapper: callable = None,
                 obs_mask_indices: list[int] | None = None,
                ):
        super().__init__(env)
        self.particle_filter_class = particle_filter_class
        self.particle_filter_kwargs = particle_filter_kwargs
        self.num_particles = num_particles
        self.particle_filter: BaseParticleFilter | None = None
        self.obs_mask_indices = obs_mask_indices
        
        # This mapper is crucial and env-specific. It defines how to get args for pf methods from env data.
        # Example for AntTag: 
        # def ant_tag_pf_mapper(base_env_obs, base_env_info, base_env_action=None, unwrapped_env=None):
        #     ant_pos = base_env_obs[:2]
        #     predict_args = {"ant_current_pos_from_obs": ant_pos} # action is passed directly by wrapper
        #     observed_target_pos = np.array([np.nan, np.nan]) # Default if not visible
        #     if unwrapped_env: # hasattr(unwrapped_env, 'get_target_pos'): 
        #            # This part is tricky as direct access to true opponent for visibility check might not always be clean
        #            # For simplicity, AntTagPF's update might need to take the full obs and deduce visibility itself.
        #            true_opponent_pos = unwrapped_env.get_target_pos() 
        #            if np.linalg.norm(ant_pos - true_opponent_pos) <= unwrapped_env.visibility_radius: 
        #                observed_target_pos = base_env_obs[unwrapped_env.observation_space.shape[0]-2:] # Assuming target is last 2
        #     update_args = {"observed_target_pos": observed_target_pos, "ant_current_pos_from_obs": ant_pos}
        #     return {"predict_args": predict_args, "update_args": update_args}
        self.pf_interaction_mapper = pf_interaction_mapper
        if self.pf_interaction_mapper is None:
            print("Warning: pf_interaction_mapper is not provided. PF predict/update calls will only receive action/obs_from_env directly.")

        # Determine particle_dim by instantiating a temporary PF
        # This is a bit of a hack. A better way would be if PF class had a static method or property.
        _temp_obs, _ = self.env.reset() # Need an initial obs for PF instantiation
        _temp_pf_kwargs = particle_filter_kwargs.copy()
        _temp_pf_kwargs['initial_env_obs'] = _temp_obs
        _temp_pf = particle_filter_class(num_particles=self.num_particles, **_temp_pf_kwargs)
        self.particle_dim = _temp_pf.particle_dim
        del _temp_pf, _temp_obs, _ # Clean up
        self.env.reset() # Reset again to ensure clean state for actual first reset

        self.observation_space = gym.spaces.Dict({
            "obs": self.env.observation_space,
            "particles": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_particles, self.particle_dim), 
                dtype=np.float32
            )
        })
        print(f"PFDictObservationWrapper: Original obs space: {self.env.observation_space}")
        print(f"PFDictObservationWrapper: New Dict obs space defined with particle_dim: {self.particle_dim}")

    def reset(self, **kwargs) -> tuple[dict, dict]:
        base_env_obs, info = self.env.reset(**kwargs)
        base_env_obs_float = base_env_obs.astype(np.float32)
        
        pf_init_kwargs = self.particle_filter_kwargs.copy()
        pf_init_kwargs['initial_env_obs'] = base_env_obs_float # Pass initial obs for PF internal init
        self.particle_filter = self.particle_filter_class(
            num_particles=self.num_particles, 
            **pf_init_kwargs
        )
        return self._get_dict_obs(base_env_obs_float), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        base_env_obs, reward, terminated, truncated, info = self.env.step(action)
        base_env_obs_float = base_env_obs.astype(np.float32)

        # Prepare arguments for PF predict and update using the mapper
        predict_call_kwargs = {}
        update_call_kwargs = {}
        if self.pf_interaction_mapper:
            mapped_args = self.pf_interaction_mapper(
                base_env_obs=base_env_obs_float, 
                base_env_info=info, 
                base_env_action=action, 
                unwrapped_env=self.env.unwrapped
            )
            predict_call_kwargs = mapped_args.get("predict_args", {})
            update_call_kwargs = mapped_args.get("update_args", {})

        self.particle_filter.predict(action, **predict_call_kwargs)

        # When a mapper is provided, update_call_kwargs contains all needed args
        # (e.g. observed_target_pos, ant_current_pos_from_obs for AntTag).
        # Passing base_env_obs_float as a positional would conflict with those kwargs.
        if self.pf_interaction_mapper:
            self.particle_filter.update(**update_call_kwargs)
        else:
            self.particle_filter.update(base_env_obs_float)

        return self._get_dict_obs(base_env_obs_float), reward, terminated, truncated, info

    def _get_dict_obs(self, base_env_obs: np.ndarray) -> dict:
        particles_state = self.particle_filter.particles.astype(np.float32)
        # If particles also include weights as the last dim, make sure ST network expects that.
        # The BaseParticleFilter defines particle_dim, which should be used by the ST network.
        agent_obs = base_env_obs
        if self.obs_mask_indices is not None:
            agent_obs = base_env_obs.copy()
            agent_obs[self.obs_mask_indices] = 0.0
        return {"obs": agent_obs, "particles": particles_state}


class PFPlusFeaturesObservationWrapper(gym.Wrapper):
    """
    Wraps an environment to concatenate features extracted from a particle filter
    (by a pretrained Set Transformer) to the original environment observation.

    Designed for RL with a pretrained Set Transformer as a fixed feature processor.

    If obs_mask_indices is provided, those indices are zeroed out in the
    observation passed to the agent (but NOT in the obs used by the PF mapper).
    This prevents the agent from seeing privileged info (e.g. true target
    position) that the PF legitimately needs to update its belief.
    """
    def __init__(self,
                 env: gym.Env,
                 particle_filter_class: type[BaseParticleFilter],
                 particle_filter_kwargs: dict,
                 pretrained_st_processor: PretrainedSetTransformerProcessor,
                 num_particles: int,
                 pf_interaction_mapper: callable = None,
                 obs_mask_indices: list[int] | None = None,
                ):
        super().__init__(env)
        self.particle_filter_class = particle_filter_class
        self.particle_filter_kwargs = particle_filter_kwargs
        self.st_processor = pretrained_st_processor
        self.num_particles = num_particles
        self.particle_filter: BaseParticleFilter | None = None
        self.pf_interaction_mapper = pf_interaction_mapper
        self.obs_mask_indices = obs_mask_indices
        if self.pf_interaction_mapper is None:
            print("Warning: pf_interaction_mapper is not provided. PF predict/update calls will only receive action/obs_from_env directly.")

        orig_obs_shape = self.env.observation_space.shape
        if not orig_obs_shape or len(orig_obs_shape) == 0:
            raise ValueError("Original observation space shape is not defined or empty.")
        
        self.st_feature_dim = self.st_processor.st_output_dim 

        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(orig_obs_shape[0] + self.st_feature_dim,), 
            dtype=np.float32
        )
        print(f"PFPlusFeaturesObservationWrapper: Original obs_dim: {orig_obs_shape[0]}")
        print(f"PFPlusFeaturesObservationWrapper: ST feature_dim: {self.st_feature_dim}")
        print(f"PFPlusFeaturesObservationWrapper: New obs_dim: {orig_obs_shape[0] + self.st_feature_dim}")


    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        base_env_obs, info = self.env.reset(**kwargs)
        base_env_obs_float = base_env_obs.astype(np.float32)
        
        pf_init_kwargs = self.particle_filter_kwargs.copy()
        pf_init_kwargs['initial_env_obs'] = base_env_obs_float
        self.particle_filter = self.particle_filter_class(
            num_particles=self.num_particles, 
            **pf_init_kwargs
        )
        return self._get_concatenated_obs(base_env_obs_float), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        base_env_obs, reward, terminated, truncated, info = self.env.step(action)
        base_env_obs_float = base_env_obs.astype(np.float32)

        predict_call_kwargs = {}
        update_call_kwargs = {}
        if self.pf_interaction_mapper:
            mapped_args = self.pf_interaction_mapper(
                base_env_obs=base_env_obs_float, 
                base_env_info=info, 
                base_env_action=action, 
                unwrapped_env=self.env.unwrapped
            )
            predict_call_kwargs = mapped_args.get("predict_args", {})
            update_call_kwargs = mapped_args.get("update_args", {})

        self.particle_filter.predict(action, **predict_call_kwargs)

        if self.pf_interaction_mapper:
            self.particle_filter.update(**update_call_kwargs)
        else:
            self.particle_filter.update(base_env_obs_float)

        return self._get_concatenated_obs(base_env_obs_float), reward, terminated, truncated, info

    def _get_concatenated_obs(self, base_env_obs: np.ndarray) -> np.ndarray:
        particles_state = self.particle_filter.particles.astype(np.float32)
        # The PretrainedSetTransformerProcessor expects particle_dim to match its model.dim_input
        # It might need particles and weights separately or combined, depending on its implementation.
        
        # We need to ensure the particle_set tensor matches the expected input of st_processor.
        # The st_processor.dim_particle_input should guide this.
        # For now, assuming st_processor.process_particles_numpy handles this (e.g. by taking particles and weights)
        # or that particle_filter.particles already has the correct format (e.g. [N, particle_dim_for_ST]).
        
        # This is a simplified call; complex PFs might store weights separately.
        # The AntTagPF stores particles as [N,2] and weights as [N]. 
        # The PretrainedSetTransformerProcessor needs to handle this, e.g. by its process_particles_numpy method.
        if self.st_processor.dim_particle_input == self.particle_filter.particle_dim + 1 and self.particle_filter.weights is not None:
            # Assumes ST model wants weights concatenated
            st_features = self.st_processor.process_particles_numpy(particles_state, self.particle_filter.weights)
        elif self.st_processor.dim_particle_input == self.particle_filter.particle_dim:
            # Assumes ST model wants only particle coordinates
            st_features = self.st_processor.process_particles_numpy(particles_state, None)
        else:
            # Fallback: create a torch tensor and call process_particles directly.
            # This requires particles_state to be in [N, ST_input_dim] format already.
            # If particle_filter.particle_dim is just coords, and ST wants coords+weights, this will fail.
            # This highlights the need for careful configuration of PF output and ST input expectation.
            print(f"Warning: Mismatch or ambiguity between PF output dim ({self.particle_filter.particle_dim}) and ST input dim ({self.st_processor.dim_particle_input}). Attempting direct processing.")
            particle_set_torch = torch.tensor(particles_state, dtype=torch.float32).unsqueeze(0) # Add batch dim
            st_features = self.st_processor.process_particles(particle_set_torch).squeeze(0) # Remove batch dim

        agent_obs = base_env_obs.copy()
        if self.obs_mask_indices is not None:
            agent_obs[self.obs_mask_indices] = 0.0
        return np.concatenate([agent_obs, st_features.astype(np.float32)])