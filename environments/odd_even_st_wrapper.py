"""
Wrapper for OddEvenPOMDP that processes particles with pretrained Set Transformer.
"""

import gymnasium as gym
import numpy as np
import torch

from features_extractors.set_transformer_pretrained_processor import (
    PretrainedSetTransformerProcessor,
)


class OddEvenSTWrapper(gym.Wrapper):
    """
    Wrapper for OddEvenPOMDP that processes particles with a pretrained Set Transformer.
    
    The environment already provides particles as observations, so we just need to:
    1. Process particles through the Set Transformer to get features
    2. Use those features as the observation (or concatenate if needed)
    """
    
    def __init__(self, env: gym.Env, pretrained_st_processor: PretrainedSetTransformerProcessor):
        """
        Args:
            env: The OddEvenPOMDP environment
            pretrained_st_processor: The pretrained Set Transformer processor
        """
        super().__init__(env)
        self.st_processor = pretrained_st_processor
        
        # Get the feature dimension from the processor
        self.st_feature_dim = self.st_processor.st_output_dim
        
        # The new observation space is just the features from the Set Transformer
        # (or we could concatenate with something else if needed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.st_feature_dim,),
            dtype=np.float32
        )
        
        print(f"OddEvenSTWrapper: Original obs shape: {env.observation_space.shape}")
        print(f"OddEvenSTWrapper: ST feature dim: {self.st_feature_dim}")
        print(f"OddEvenSTWrapper: New obs shape: {self.observation_space.shape}")
    
    def reset(self, **kwargs):
        """Reset the environment and process initial particles."""
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_particles(obs)
        return processed_obs, info
    
    def step(self, action):
        """Step the environment and process particles."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_particles(obs)
        return processed_obs, reward, terminated, truncated, info
    
    def _process_particles(self, particles: np.ndarray) -> np.ndarray:
        """
        Process particles through the pretrained Set Transformer.
        
        Args:
            particles: Particle observation from the environment, shape [n_particles,]
        
        Returns:
            features: Processed features, shape [st_feature_dim,]
        """
        # Reshape particles to [n_particles, 1] since the ST expects [N, particle_dim]
        # The particles are 1D values, so we need to add a dimension
        particles_reshaped = particles.reshape(-1, 1).astype(np.float32)
        
        # Process through Set Transformer
        # The processor expects particles in the format it was trained on
        # Check if we need to add weights or if particles alone are sufficient
        if self.st_processor.dim_particle_input == 1:
            # Model expects just particle values
            st_features = self.st_processor.process_particles_numpy(particles_reshaped, None)
        elif self.st_processor.dim_particle_input == 2:
            # Model might expect [value, weight] - but we don't have weights from the env
            # So we'll just use particles and let the processor handle it
            # Or create uniform weights
            weights = np.ones(len(particles)) / len(particles)
            st_features = self.st_processor.process_particles_numpy(particles_reshaped, weights)
        else:
            # Fallback: try direct processing
            particle_set_torch = torch.tensor(particles_reshaped, dtype=torch.float32).unsqueeze(0)
            st_features = self.st_processor.process_particles(particle_set_torch).squeeze(0)
        
        return st_features.astype(np.float32)

