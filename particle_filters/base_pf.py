from abc import ABC, abstractmethod

import numpy as np


class BaseParticleFilter(ABC):
    """
    Abstract base class for particle filters.
    Defines the interface expected by the environment wrappers.
    """
    def __init__(self, num_particles: int, env_obs_space: np.ndarray, **kwargs):
        """
        Args:
            num_particles (int): Number of particles to maintain.
            env_obs_space (np.ndarray): The observation from the base environment used for initialization.
            **kwargs: Additional arguments for specific PF implementations.
        """
        self.num_particles = num_particles
        self._initialize_particles(env_obs_space, **kwargs)

    @abstractmethod
    def _initialize_particles(self, env_obs: np.ndarray, **kwargs) -> None:
        """Initialize particle states and weights based on an initial environment observation."""
        pass

    @abstractmethod
    def predict(self, action: np.ndarray, **kwargs) -> None:
        """Predict the next state of particles based on the action taken."""
        pass

    @abstractmethod
    def update(self, obs_from_env: np.ndarray, **kwargs) -> None:
        """Update particle weights based on the new observation from the environment."""
        pass

    @property
    @abstractmethod
    def particles(self) -> np.ndarray:
        """Return the current particle states, shape [num_particles, particle_dim]."""
        pass

    @property
    @abstractmethod
    def weights(self) -> np.ndarray:
        """Return the current particle weights, shape [num_particles]."""
        pass

    @property
    @abstractmethod
    def particle_dim(self) -> int:
        """Return the dimension of a single particle in the set (e.g., 2 for (x,y), 3 for (x,y,w))."""
        pass 