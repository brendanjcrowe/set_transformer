import numpy as np
from filterpy.monte_carlo import systematic_resample

from .base_pf import BaseParticleFilter  # Use relative import


class AntTagParticleFilter(BaseParticleFilter):
    """
    Particle filter implementation for the Ant-Tag environment.
    Tracks the position of the opponent.
    """
    
    def __init__(self, num_particles: int, initial_env_obs: np.ndarray, 
                 initial_spread_std: float = 5.0, 
                 arena_limits: tuple[float, float] = (-10.0, 10.0),
                 obs_noise_std: float = 0.1,
                 process_noise_std: float = 0.2,
                 avoidance_factor: float = 0.8,
                 max_avoidance_dist: float = 7.0,
                 visibility_radius: float = 3.0,
                 **kwargs # To accommodate other BaseParticleFilter args
                 ):
        """
        Args:
            num_particles: Number of particles to use.
            initial_env_obs: Initial observation from the AntTag environment.
                               Assumed to contain ant's initial x,y at obs[0], obs[1].
            initial_spread_std: Standard deviation for initial particle distribution around a central point.
            arena_limits: Tuple (min_coord, max_coord) for the square arena.
            obs_noise_std: Standard deviation of the observation noise.
            process_noise_std: Standard deviation of the process noise (opponent movement).
            avoidance_factor: How strongly the opponent avoids the ant.
            max_avoidance_dist: Maximum distance at which avoidance behavior is significant.
            visibility_radius: Radius within which the target is considered visible.
        """
        self.arena_min, self.arena_max = arena_limits
        self.obs_noise_std = obs_noise_std
        self.process_noise_std = process_noise_std
        self.avoidance_factor = avoidance_factor
        self.max_avoidance_dist = max_avoidance_dist
        self.visibility_radius = visibility_radius
        
        # The actual particle state is [x, y] for the opponent
        self._particle_dim = 2 

        # Call super().__init__ which will call _initialize_particles
        super().__init__(num_particles, initial_env_obs, initial_spread_std=initial_spread_std, **kwargs)

    def _initialize_particles(self, initial_env_obs: np.ndarray, initial_spread_std: float, **kwargs) -> None:
        """Initialize particle states and weights.
        Particles are spread uniformly within arena_limits.
        """
        self._particles = np.random.uniform(self.arena_min, self.arena_max, (self.num_particles, self.particle_dim))
        self._weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, action: np.ndarray, ant_current_pos_from_obs: np.ndarray, **kwargs) -> None:
        """Predict the next state of particles (opponent movement).
        Opponent has a simple policy: move away from ant if close, else random walk.
        Args:
            action: The ant's action (not directly used for opponent model here, but could be).
            ant_current_pos_from_obs: Current [x,y] position of the ant from environment observation.
        """
        # Add random process noise (random walk for opponent)
        self._particles += np.random.normal(0, self.process_noise_std, self._particles.shape)
        
        # Opponent avoidance behavior
        for i, p_opponent in enumerate(self._particles):
            dist_to_ant = np.linalg.norm(p_opponent - ant_current_pos_from_obs)
            if dist_to_ant < self.max_avoidance_dist and dist_to_ant > 0: # Avoid division by zero
                # Move away from ant
                avoid_direction = (p_opponent - ant_current_pos_from_obs) / dist_to_ant
                avoid_step = self.avoidance_factor * (self.max_avoidance_dist - dist_to_ant) / self.max_avoidance_dist
                self._particles[i] += avoid_direction * avoid_step * self.process_noise_std # Scale by process_std to keep movement magnitude reasonable

        # Clip particles to stay within arena boundaries
        self._particles = np.clip(self._particles, self.arena_min, self.arena_max)

    def update(self, observed_target_pos: np.ndarray, ant_current_pos_from_obs: np.ndarray, **kwargs) -> None:
        """Update particle weights based on the observed target position (if any) and ant's position.
        Args:
            observed_target_pos: Observed [x,y] of the target. Can be [np.nan, np.nan] if not visible.
            ant_current_pos_from_obs: Current [x,y] position of the ant.
        """
        opponent_is_visible = not np.isnan(observed_target_pos[0])

        if opponent_is_visible:
            # Target is visible, update weights based on distance to observation
            # This is a likelihood function: N(observed_target_pos | particle_pos, obs_std^2)
            distances_sq = np.sum((self._particles - observed_target_pos)**2, axis=1)
            self._weights *= (1. / (np.sqrt(2 * np.pi) * self.obs_noise_std)) * np.exp(-distances_sq / (2 * self.obs_noise_std**2))
        else:
            # Target is not visible. We can't directly update based on target observation.
            # However, we know the target is *not* within the ant's visibility_radius.
            # Penalize particles that are within the ant's visibility radius.
            for i, p_opponent in enumerate(self._particles):
                dist_to_ant = np.linalg.norm(p_opponent - ant_current_pos_from_obs)
                if dist_to_ant < self.visibility_radius:
                    self._weights[i] *= 0.1 # Penalize particles inside the visible but unobserved zone
        
        self._weights += 1.e-300 # Avoid round-off to zero
        self._weights /= np.sum(self._weights) # Normalize
        
        # Resample if weights become too skewed (N_eff < N/2)
        if 1. / np.sum(self._weights**2) < self.num_particles / 2.:
            self._resample_particles()

    def _resample_particles(self) -> None:
        """Resample particles based on their weights using systematic resampling."""
        indices = systematic_resample(self._weights)
        self._particles = self._particles[indices]
        self._weights = np.ones(self.num_particles) / self.num_particles

    @property
    def particles(self) -> np.ndarray:
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    @property
    def particle_dim(self) -> int:
        return self._particle_dim

    def estimate_opponent_pos(self) -> np.ndarray:
        """Estimate the opponent's position as the mean of the particles."""
        return np.average(self.particles, weights=self.weights, axis=0) 