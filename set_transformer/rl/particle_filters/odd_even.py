"""
Particle filter for OddEvenPOMDP.

This particle filter tracks the belief over the mean value of the distribution.
Particles represent possible mean values in the range [1, n].
"""

import numpy as np
from filterpy.monte_carlo import systematic_resample

from .base_pf import BaseParticleFilter


class OddEvenParticleFilter(BaseParticleFilter):
    """
    Particle filter for the Odd-Even POMDP.
    Tracks the belief over the mean value of the distribution.
    """
    
    def __init__(self, num_particles: int, initial_env_obs: np.ndarray,
                 n_dist_size: int = 10,
                 process_noise_std: float = 0.5,
                 obs_noise_std: float = 1.0,
                 **kwargs):
        """
        Args:
            num_particles: Number of particles to use.
            initial_env_obs: Initial observation (particles from POMDP).
            n_dist_size: Maximum number in range [1, n].
            process_noise_std: Standard deviation of process noise (how much mean can change).
            obs_noise_std: Standard deviation of observation noise for likelihood computation.
        """
        self.n_dist_size = n_dist_size
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std
        
        # Particle dimension is 1 (just the mean value)
        self._particle_dim = 1
        
        # Call super().__init__ which will call _initialize_particles
        # Note: base class expects env_obs_space as second arg, but we use initial_env_obs
        super().__init__(num_particles, initial_env_obs, **kwargs)
    
    def _initialize_particles(self, initial_env_obs: np.ndarray, **kwargs) -> None:
        """Initialize particle states and weights.
        Particles are initialized uniformly in range [1, n_dist_size].
        """
        # Initialize particles uniformly in the valid range
        self._particles = np.random.uniform(1.0, float(self.n_dist_size), (self.num_particles, 1))
        self._weights = np.ones(self.num_particles) / self.num_particles
    
    def predict(self, action: np.ndarray, **kwargs) -> None:
        """Predict the next state of particles.
        The mean doesn't change much, so we add small process noise.
        
        Args:
            action: The predicted mean (not used for prediction, but could be)
        """
        # Add small process noise (the true mean is fixed, but we don't know that)
        self._particles += np.random.normal(0, self.process_noise_std, self._particles.shape)
        
        # Clip particles to valid range
        self._particles = np.clip(self._particles, 1.0, float(self.n_dist_size))
    
    def update(self, obs_from_env: np.ndarray, **kwargs) -> None:
        """Update particle weights based on the observation.
        
        Args:
            obs_from_env: Observation from the environment (particles from POMDP).
                          We use the mean of these particles as the observation.
        """
        # Use the mean of the observed particles as the observation
        observed_value = np.mean(obs_from_env)
        
        # Compute likelihood for each particle
        # Likelihood: probability of observing this value given the particle's mean
        # We use a Gaussian likelihood
        distances_sq = (self._particles.squeeze() - observed_value) ** 2
        self._weights *= np.exp(-distances_sq / (2 * self.obs_noise_std ** 2))
        
        # Avoid numerical issues
        self._weights += 1.e-300
        self._weights /= np.sum(self._weights)
        
        # Resample if effective sample size is too small
        n_eff = 1.0 / np.sum(self._weights ** 2)
        if n_eff < self.num_particles / 2.0:
            self._resample_particles()
    
    def _resample_particles(self) -> None:
        """Resample particles based on their weights using systematic resampling."""
        indices = systematic_resample(self._weights)
        self._particles = self._particles[indices]
        self._weights = np.ones(self.num_particles) / self.num_particles
    
    @property
    def particles(self) -> np.ndarray:
        """Return particles, shape [num_particles, 1]."""
        return self._particles
    
    @property
    def weights(self) -> np.ndarray:
        """Return particle weights, shape [num_particles]."""
        return self._weights
    
    @property
    def particle_dim(self) -> int:
        """Return particle dimension (1 for mean value)."""
        return self._particle_dim
    
    def estimate_mean(self) -> float:
        """Estimate the mean as the weighted average of particles."""
        return float(np.average(self._particles, weights=self._weights))

