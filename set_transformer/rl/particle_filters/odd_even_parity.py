"""Parity-aware particle filter for OddEvenPOMDP.

The generative model is: a hidden integer ``true_state`` in ``[1, n]`` is fixed for the
episode, and every observation is an integer drawn from a Gaussian centred on
``true_state`` **restricted to integers of true_state's own parity**. Parity is therefore
not a nuisance -- it is observed exactly, from the first sample, and it rules out half the
state space outright.

:class:`~set_transformer.rl.particle_filters.odd_even.OddEvenParticleFilter` ignores all
of that: it puts particles anywhere in the continuous interval ``[1, n]``, diffuses them
with process noise, and scores them against the sample mean with a plain Gaussian. The
resulting belief is a smeared blob spanning both parities -- it assigns mass to states the
observations have already excluded.

This filter instead represents the exact posterior: particles live on the integer grid,
the likelihood is the environment's own observation model, and wrong-parity states are
zeroed. The belief that comes out is a **comb** -- mass at 3 and 7 with a hole at 5 -- and
that shape is the point. A Gaussian summary of a comb has its mean at an *impossible*
state (the mean of {3, 5} is 4, which the wrong parity forbids), so mean+covariance is
provably insufficient here in a way it is not for a unimodal belief. That is exactly the
structure a set encoder should be able to exploit and the analytic baselines cannot.

Two properties worth stating because they differ from the older filter:

* ``predict`` is a **no-op**. ``true_state`` is constant within an episode, so process
  noise does not model anything real -- it just diffuses the comb back into a blob.
* Resampling is **physical** (particle positions move, not just weights). Downstream
  feature extractors read the particle set unweighted, so a belief that collapses only in
  the weight vector would still look uniform to them. Same lesson as the Car-Flag filter.
"""

from __future__ import annotations

import numpy as np
from filterpy.monte_carlo import systematic_resample

from .base import BaseParticleFilter


class ParityAwareOddEvenParticleFilter(BaseParticleFilter):
    """Exact-model particle filter over the integer states of OddEvenPOMDP."""

    def __init__(
        self,
        num_particles: int,
        initial_env_obs: np.ndarray,
        n_dist_size: int = 10,
        std_dev: float = 2.0,
        resample_threshold: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            num_particles: Number of particles.
            initial_env_obs: First observation (one or more integer samples).
            n_dist_size: States are the integers ``[1, n_dist_size]``.
            std_dev: Observation-model spread; must match the environment's ``std_dev``,
                since this filter reproduces that model exactly rather than approximating
                it.
            resample_threshold: Resample when ``n_eff`` falls below this fraction of
                ``num_particles``.
        """
        self.n_dist_size = int(n_dist_size)
        self.std_dev = float(std_dev)
        self.resample_threshold = float(resample_threshold)
        self._particle_dim = 1

        self._grid = np.arange(1, self.n_dist_size + 1, dtype=np.float64)
        self._grid_is_odd = (self._grid.astype(int) % 2 == 1)
        # Per-candidate observation model, normalized over that candidate's own parity --
        # the environment's `_compute_observation_probability`, precomputed as a table:
        # _obs_lut[c_idx, v_idx] = P(observe grid[v_idx] | true_state = grid[c_idx]).
        self._obs_lut = self._build_observation_table()

        super().__init__(num_particles, initial_env_obs, **kwargs)

    def _build_observation_table(self) -> np.ndarray:
        n = len(self._grid)
        lut = np.zeros((n, n))
        for ci, c in enumerate(self._grid):
            same = self._grid_is_odd == self._grid_is_odd[ci]
            dens = np.exp(-0.5 * ((self._grid[same] - c) / self.std_dev) ** 2)
            lut[ci, same] = dens / dens.sum()
        return lut

    def _initialize_particles(self, initial_env_obs: np.ndarray, **kwargs) -> None:
        """Uniform prior over every integer state, both parities.

        Spread deterministically across the grid rather than drawn iid: an exact uniform
        is a lower-variance representation of the same prior, and it guarantees every
        state is represented even when ``num_particles`` is small.

        The order is then shuffled, which is load-bearing. ``np.resize`` lays the states
        out periodically (1, 2, ..., n, 1, 2, ...), and systematic resampling walks the
        weight cumsum with fixed-stride pointers -- against a periodic layout those
        pointers alias, and the resampled belief comes out visibly quantized (0.3/0.4/0.2
        /0.1 in place of the correct 0.321/0.393/0.227/0.053). Systematic resampling
        assumes no correlation between particle index and weight; shuffling restores that.
        """
        self._particles = self._shuffled_grid(self._grid).reshape(self.num_particles, 1)
        self._weights = np.ones(self.num_particles) / self.num_particles

    def _shuffled_grid(self, states: np.ndarray) -> np.ndarray:
        """``num_particles`` particles spread evenly over ``states``, in random order."""
        reps = np.resize(np.asarray(states, dtype=np.float64), self.num_particles)
        np.random.default_rng().shuffle(reps)
        return reps

    def predict(self, action: np.ndarray, **kwargs) -> None:
        """No-op: ``true_state`` is fixed for the whole episode.

        Adding process noise here (as the non-parity-aware filter does) would model a
        drift that does not exist, and would blur the parity comb the belief is supposed
        to hold.
        """
        return

    def _log_likelihood(self, samples: np.ndarray) -> np.ndarray:
        """Log P(samples | c) for every grid state ``c``; ``-inf`` where impossible."""
        idx = np.rint(samples).astype(int) - 1
        idx = idx[(idx >= 0) & (idx < len(self._grid))]
        if idx.size == 0:
            return np.zeros(len(self._grid))
        with np.errstate(divide="ignore"):
            return np.log(self._obs_lut[:, idx]).sum(axis=1)

    def update(self, obs_from_env: np.ndarray, **kwargs) -> None:
        """Exact Bayesian update against the environment's observation model.

        Args:
            obs_from_env: One or more integer observation samples. Every sample shares
                ``true_state``'s parity, so the first one already rules out half the grid.
        """
        samples = np.atleast_1d(np.asarray(obs_from_env, dtype=np.float64)).ravel()
        log_lik = self._log_likelihood(samples)

        # Score each particle by its own state, in log space: with many samples per step
        # the raw likelihoods underflow float64 long before the belief is actually
        # degenerate.
        state_idx = np.clip(np.rint(self._particles.ravel()).astype(int) - 1,
                            0, len(self._grid) - 1)
        log_w = np.log(self._weights + 1e-300) + log_lik[state_idx]

        if not np.isfinite(log_w).any():
            # Every surviving particle is impossible under this observation -- the belief
            # had already collapsed onto the wrong parity (possible after an unlucky
            # resample). Reinitialize on the states the observation actually permits
            # rather than raising: a filter that dies mid-episode would take the training
            # run with it.
            self._reinitialize_from_observation(samples)
            return

        log_w -= log_w.max()
        weights = np.exp(log_w)
        self._weights = weights / weights.sum()

        n_eff = 1.0 / np.sum(self._weights ** 2)
        if n_eff < self.resample_threshold * self.num_particles:
            self._resample_particles()

    def _reinitialize_from_observation(self, samples: np.ndarray) -> None:
        """Uniform over the states consistent with the observed parity."""
        log_lik = self._log_likelihood(samples)
        allowed = self._grid[np.isfinite(log_lik)]
        if allowed.size == 0:
            allowed = self._grid
        self._particles = self._shuffled_grid(allowed).reshape(self.num_particles, 1)
        self._weights = np.ones(self.num_particles) / self.num_particles

    def _resample_particles(self) -> None:
        """Systematic resampling. Moves particle *positions*, not just weights, because
        the feature extractors read the particle set unweighted."""
        indices = systematic_resample(self._weights)
        self._particles = self._particles[indices]
        self._weights = np.ones(self.num_particles) / self.num_particles

    @property
    def particles(self) -> np.ndarray:
        """Particles, shape ``[num_particles, 1]``, valued on the integer grid."""
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def particle_dim(self) -> int:
        return self._particle_dim

    def belief_over_states(self) -> np.ndarray:
        """Weighted belief mass per grid state — the comb, as a length-n vector."""
        state_idx = np.clip(np.rint(self._particles.ravel()).astype(int) - 1,
                            0, len(self._grid) - 1)
        return np.bincount(state_idx, weights=self._weights, minlength=len(self._grid))

    def estimate_mean(self) -> float:
        return float(np.average(self._particles, weights=self._weights))
