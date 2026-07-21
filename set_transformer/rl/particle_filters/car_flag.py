"""Particle filter for the Car-Flag POMDP (``pdomains-car-flag-v0``).

Car-Flag hides a single binary latent: ``heaven_position`` in ``{+1, -1}`` (hell is the
opposite flag), drawn 50/50 at reset and **static** thereafter. Position and velocity are
fully observed; the only clue to the latent is the observation's third component,
``direction``, which is ``0`` everywhere except inside the priest region
``[priest - delta, priest + delta]``, where it equals the true heaven side (+1 = right,
-1 = left).

So each particle is a one-dimensional hypothesis for ``heaven_position`` (``+1`` or ``-1``):

* prior            — half the particles at ``+1``, half at ``-1`` (uniform belief);
* ``predict``      — no-op (the latent never moves), the classic static-parameter PF;
* ``update``       — when ``direction != 0`` the priest reveals the truth, so particles
                     whose sign disagrees are driven to ~zero weight and resampled away,
                     collapsing the belief to a delta; when ``direction == 0`` the
                     observation is uninformative and the belief is unchanged.

No mapper or privileged access is needed — ``direction`` is part of the ordinary
observation — so this PF is driven exactly like :class:`OddEvenParticleFilter`, with the
wrapper calling ``update(base_env_obs)`` positionally.
"""

import numpy as np
from filterpy.monte_carlo import systematic_resample

from .base import BaseParticleFilter

#: Index of the priest's direction reading within the Car-Flag observation
#: ``[position, velocity, direction]``.
DIRECTION_OBS_INDEX = 2


class CarFlagParticleFilter(BaseParticleFilter):
    """Belief over the binary, static ``heaven_position`` latent of Car-Flag."""

    def __init__(self, num_particles: int, initial_env_obs: np.ndarray,
                 mismatch_likelihood: float = 1e-6,
                 **kwargs):
        """
        Args:
            num_particles: Number of particles to use.
            initial_env_obs: Initial observation (unused for init; the prior is fixed).
            mismatch_likelihood: Likelihood assigned to a particle whose sign disagrees
                with the priest's revealed direction. The priest is a noiseless oracle,
                so this is ~0; a tiny positive value keeps weights numerically stable.
        """
        self.mismatch_likelihood = mismatch_likelihood
        self._particle_dim = 1  # a single scalar: the hypothesised heaven side (+1/-1)
        super().__init__(num_particles, initial_env_obs, **kwargs)

    def _initialize_particles(self, initial_env_obs: np.ndarray, **kwargs) -> None:
        """Uniform prior over the binary latent: half at +1, half at -1.

        An exact half/half split is a lower-variance representation of the 50/50 prior
        than iid Bernoulli draws, and keeps the initial belief mean at exactly 0.
        """
        half = self.num_particles // 2
        signs = np.ones(self.num_particles)
        signs[:half] = -1.0
        np.random.shuffle(signs)  # order carries no meaning; avoid a fixed layout
        self._particles = signs.reshape(self.num_particles, 1)
        self._weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, action: np.ndarray, **kwargs) -> None:
        """No-op: ``heaven_position`` is fixed for the whole episode."""
        return

    def update(self, obs_from_env: np.ndarray, **kwargs) -> None:
        """Reweight on the priest's reading; collapse the belief once it fires.

        Args:
            obs_from_env: The full Car-Flag observation ``[position, velocity, direction]``.
        """
        direction = float(np.asarray(obs_from_env).reshape(-1)[DIRECTION_OBS_INDEX])
        if direction == 0.0:
            return  # outside the priest region: no information about the latent

        # Priest fired: particles agreeing with the revealed side keep full likelihood,
        # disagreeing ones are driven to ~0.
        agrees = np.sign(self._particles.squeeze(-1)) == np.sign(direction)
        likelihood = np.where(agrees, 1.0, self.mismatch_likelihood)
        self._weights *= likelihood

        self._weights += 1.0e-300  # guard against all-zero weights
        self._weights /= np.sum(self._weights)

        # Always resample on a reveal. Downstream feature extractors read the particle
        # *positions* unweighted (the Dict wrapper does not expose weights), so the belief
        # has to collapse physically, not just in the weight vector — otherwise the
        # encoders would still see the uninformative 50/50 set. (Note an exact 50/50 prior
        # gives n_eff == N/2 after a one-sided reveal, so a ``< N/2`` guard would not fire.)
        self._resample_particles()

    def _resample_particles(self) -> None:
        """Systematic resampling, then reset to uniform weights."""
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
        """Return particle dimension (1: the scalar heaven side)."""
        return self._particle_dim

    def heaven_right_prob(self) -> float:
        """Belief that heaven is on the right (weighted fraction of +1 particles)."""
        return float(np.average(self._particles.squeeze(-1) > 0, weights=self._weights))
