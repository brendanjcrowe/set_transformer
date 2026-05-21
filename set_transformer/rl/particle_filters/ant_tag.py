import numpy as np
from filterpy.monte_carlo import systematic_resample

from .base import BaseParticleFilter  # Use relative import


class AntTagParticleFilter(BaseParticleFilter):
    """
    Particle filter implementation for the Ant-Tag environment.
    Tracks the position of the opponent.
    """
    
    def __init__(self, num_particles: int, initial_env_obs: np.ndarray,
                 initial_spread_std: float = 5.0,
                 arena_limits: tuple[float, float] = (-4.5, 4.5),
                 obs_noise_std: float = 0.1,
                 target_step: float = 0.5,
                 visibility_radius: float = 3.0,
                 **kwargs # To accommodate other BaseParticleFilter args
                 ):
        """
        Args:
            num_particles: Number of particles to use.
            initial_env_obs: Initial observation from the AntTag environment.
            initial_spread_std: Standard deviation for initial particle distribution.
            arena_limits: Tuple (min_coord, max_coord) for the square arena.
            obs_noise_std: Standard deviation of the observation noise.
            target_step: Step size of target movement (matches env's target_step).
            visibility_radius: Radius within which the target is considered visible.
        """
        self.arena_min, self.arena_max = arena_limits
        self.obs_noise_std = obs_noise_std
        self.target_step = target_step
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
        """Predict the next state of particles using the true target motion model.

        The real target picks uniformly from 4 actions each step:
          - perpendicular left  (25%)
          - perpendicular right (25%)
          - directly away       (25%)
          - stay still          (25%)
        Each move has magnitude target_step (0.5). If a move would exit
        the arena, the target stays put instead.
        """
        n = self.num_particles
        ant = ant_current_pos_from_obs

        # Unit vector from each particle toward ant
        diff = ant - self._particles  # [N, 2]
        dists = np.linalg.norm(diff, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-8)  # avoid div by zero
        target2ant = diff / dists  # [N, 2]

        # Build the 4 direction options for each particle
        perp_left = np.column_stack([target2ant[:, 1], -target2ant[:, 0]])
        perp_right = np.column_stack([-target2ant[:, 1], target2ant[:, 0]])
        away = -target2ant
        stay = np.zeros_like(target2ant)

        # Randomly choose one of the 4 options per particle (uniform 25% each)
        choices = np.random.randint(0, 4, size=n)

        directions = np.where(
            (choices == 0)[:, None], perp_left,
            np.where(
                (choices == 1)[:, None], perp_right,
                np.where(
                    (choices == 2)[:, None], away,
                    stay
                )
            )
        )

        new_positions = self._particles + directions * self.target_step

        # If move would go out of bounds, stay put (matches env behavior)
        out_of_bounds = (
            (np.abs(new_positions[:, 0]) > -self.arena_min) |
            (np.abs(new_positions[:, 1]) > -self.arena_min)
        )
        new_positions[out_of_bounds] = self._particles[out_of_bounds]

        self._particles = new_positions

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
        """Partial resampling: only replace low-weight ('dead') particles.

        Particles with weight below 1/(2N) are replaced by noisy copies of
        high-weight particles.  Surviving particles keep their weights, so the
        weight distribution stays non-uniform and Shannon entropy remains an
        informative measure of belief uncertainty.
        """
        threshold = 1.0 / (2 * self.num_particles)
        dead = self._weights < threshold
        alive = ~dead

        n_dead = dead.sum()
        if n_dead == 0:
            return
        if alive.sum() == 0:
            # Degenerate case: all particles are dead — fall back to full resample
            indices = systematic_resample(self._weights)
            self._particles = self._particles[indices]
            self._weights = np.ones(self.num_particles) / self.num_particles
            return

        # Draw donors from alive particles proportional to their weights
        alive_weights = self._weights[alive]
        alive_probs = alive_weights / alive_weights.sum()
        donor_idx = np.random.choice(
            np.where(alive)[0], size=n_dead, p=alive_probs
        )

        # Replace dead particles with noisy copies of donors
        noise = np.random.normal(0, self.target_step * 0.5, (n_dead, self._particle_dim))
        self._particles[dead] = self._particles[donor_idx] + noise
        self._particles = np.clip(self._particles, self.arena_min, self.arena_max)

        # Give resampled particles a small weight
        self._weights[dead] = threshold
        self._weights /= self._weights.sum()

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