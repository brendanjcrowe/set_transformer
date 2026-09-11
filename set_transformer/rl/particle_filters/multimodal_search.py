"""Particle filter for :class:`MultimodalSearchEnv` — search by negative information.

The belief is over the static target's position. It starts as the environment's prior: an
equal share of particles drawn from each of the K random Gaussian modes, giving a
genuinely multimodal cloud whose mean is pinned at the origin by construction.

What drives it is **negative information**, and because the target is static the exact
posterior has a simple closed form: it is the **prior restricted to the region not yet
observed**. Nothing else moves it. So the belief here is left untouched from step to step
until the agent's visibility disc actually passes over particles; those particles are then
either explained (the target is there, and the episode ends) or refuted — their
probability goes to zero and they are redrawn.

Refuted particles are redrawn *from the prior, rejecting the swept region*, rather than
copied from surviving neighbours. Copying would be a second approximation on top of an
exact posterior: duplicates accumulate until the cloud degenerates into a handful of
atoms, and the usual remedy — roughening the copies with noise — quietly diffuses a
belief that should not diffuse at all, since the target never moves. Sampling the
restricted prior directly has neither problem and is exactly right.

Sweeping a mode therefore erases it from the belief and the remaining mass sits on the
modes still unvisited. That is what makes "visit each mode in turn" the optimal policy,
and it is information a mean cannot carry: after eliminating a mode the mean simply slides
to the centroid of what is left, which is empty space between modes.

``predict`` is a **no-op**: the target is static, so process noise would model a drift
that does not exist and would blur the modes the belief exists to represent.
"""

from __future__ import annotations

import numpy as np

from set_transformer.rl.envs.multimodal_search import (
    BASE_OBS_DIM,
    segment_distance,
    unpack_modes,
)

from .base import BaseParticleFilter


class MultimodalSearchParticleFilter(BaseParticleFilter):
    """Belief over a static hidden target, updated by refuting swept regions."""

    def __init__(
        self,
        num_particles: int,
        initial_env_obs: np.ndarray,
        k_max: int = 10,
        arena_half: float = 14.0,
        visibility_radius: float = 1.5,
        **kwargs,
    ):
        self.k_max = int(k_max)
        self.arena_half = float(arena_half)
        self.visibility_radius = float(visibility_radius)
        self._particle_dim = 2
        self._rng = np.random.default_rng()
        super().__init__(num_particles, initial_env_obs, **kwargs)

    def _initialize_particles(self, initial_env_obs: np.ndarray, **kwargs) -> None:
        """The environment's prior: an equal share of particles per mode.

        Equal shares (rather than multinomial draws) match the environment's uniform
        choice of hiding mode exactly and remove sampling noise from the prior, so every
        mode is represented even at small ``num_particles``.
        """
        means, covs = unpack_modes(np.asarray(initial_env_obs), self.k_max)
        self._prior_means, self._prior_covs = means, covs
        # Centres of every disc observed so far. The posterior is the prior minus these,
        # so this list *is* the filter's memory -- there is nothing else to carry.
        self._swept_centres: list[np.ndarray] = []
        # Prior mass not yet ruled out. The posterior is the prior restricted to the
        # unobserved region, so this single number IS the belief's remaining uncertainty:
        # the information gathered so far is -log(surviving_mass) nats, and it is exact
        # rather than an estimate off the particle cloud.
        self.surviving_mass = 1.0
        if len(means) == 0:                       # no prior available: fall back to flat
            self._particles = self._rng.uniform(
                -self.arena_half, self.arena_half, size=(self.num_particles, 2))
        else:
            counts = np.full(len(means), self.num_particles // len(means))
            counts[: self.num_particles - counts.sum()] += 1
            self._particles = np.concatenate([
                self._rng.multivariate_normal(mu, cov, size=n)
                for mu, cov, n in zip(means, covs, counts) if n > 0
            ])
            np.clip(self._particles, -self.arena_half, self.arena_half,
                    out=self._particles)
        # Shuffle so no downstream consumer can pick up mode identity from particle
        # order: the prior above is laid out mode by mode.
        self._rng.shuffle(self._particles)
        self._weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, action: np.ndarray, **kwargs) -> None:
        """No-op: the target does not move."""
        return

    def update(self, obs_from_env: np.ndarray, **kwargs) -> None:
        """Refute the swept disc, or collapse onto a sighting.

        Args:
            obs_from_env: the environment observation — agent xy, target offset (zeros
                when not visible), and the visibility flag.
        """
        obs = np.asarray(obs_from_env, dtype=np.float64).ravel()
        agent_pos, prev_pos = obs[0:2], obs[2:4]
        visible = obs[6] > 0.5

        if visible:
            # Perfect detection: the belief is now a delta on the observed position.
            self._particles = np.tile(agent_pos + obs[4:6], (self.num_particles, 1))
            self._weights = np.ones(self.num_particles) / self.num_particles
            return

        # Refute the whole swath swept between the previous and current positions. With a
        # step comparable to the visibility radius, refuting only an endpoint disc would
        # leave unobserved gaps that the belief would wrongly keep believing in.
        swept = (segment_distance(self._particles, prev_pos, agent_pos)
                 <= self.visibility_radius)
        if not swept.any():
            return                                  # nothing refuted; belief unchanged

        self._remember_sweep(agent_pos, prev_pos)
        n_refuted = int(swept.sum())
        # Each particle carries 1/N of the surviving mass, so refuting n of them scales
        # the remaining prior mass by (1 - n/N).
        self.surviving_mass *= 1.0 - n_refuted / self.num_particles
        replacements = self._sample_unswept(n_refuted)
        if replacements is None:
            # Every hypothesis has been refuted and the unswept arena is empty too, so
            # restart flat rather than dying: an exhausted belief would otherwise emit
            # NaNs into the policy for the rest of the episode.
            self._reinitialize_flat()
            return
        self._particles[swept] = replacements
        # Weights stay uniform throughout: the belief is carried entirely by particle
        # POSITIONS. That is load-bearing -- the feature extractors read the particle set
        # unweighted, so a belief that collapsed only in the weight vector would be
        # invisible to them. It matters here more than usual: sweeping one of K modes
        # refutes just 1/K of the mass, so n_eff would stay well above any sane
        # resampling threshold and a conventional filter would never move a particle.
        self._weights = np.ones(self.num_particles) / self.num_particles

    def _remember_sweep(self, agent_pos: np.ndarray, prev_pos=None) -> None:
        """Record an observed disc, skipping ones that add nothing.

        The agent moves a fraction of the visibility radius per step, so consecutive
        discs overlap almost entirely; keeping every one would grow the rejection test
        without shrinking the accepted region.
        """
        # Store the swath as a chain of discs dense enough to cover it.
        pts = [np.asarray(agent_pos, dtype=np.float64)]
        if prev_pos is not None:
            seg = np.asarray(agent_pos, dtype=np.float64) - np.asarray(prev_pos, dtype=np.float64)
            n = int(np.linalg.norm(seg) / (0.5 * self.visibility_radius))
            pts = [np.asarray(prev_pos, dtype=np.float64) + seg * f
                   for f in np.linspace(0.0, 1.0, max(n, 1) + 1)]
        for pt in pts:
            if self._swept_centres:
                d = np.linalg.norm(np.asarray(self._swept_centres) - pt, axis=1)
                if d.min() < 0.25 * self.visibility_radius:
                    continue
            self._swept_centres.append(pt.copy())

    def _is_swept(self, pts: np.ndarray) -> np.ndarray:
        if not self._swept_centres:
            return np.zeros(len(pts), dtype=bool)
        centres = np.asarray(self._swept_centres)
        d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=-1)
        return (d <= self.visibility_radius).any(axis=1)

    def _sample_unswept(self, n: int, attempts: int = 24):
        """``n`` draws from the prior, rejecting anything already observed.

        Falls back to the unswept arena when the prior is exhausted (the target must be
        somewhere, and if every mode has been ruled out it is outside all of them).
        """
        if n == 0:
            return np.empty((0, 2))
        kept = []
        need = n
        for source_is_prior in (True, False):
            for _ in range(attempts):
                cand = (self._draw_prior(need * 4) if source_is_prior
                        else self._rng.uniform(-self.arena_half, self.arena_half,
                                               size=(need * 4, 2)))
                if cand is None:
                    break
                cand = cand[~self._is_swept(cand)]
                if len(cand):
                    kept.append(cand)
                    need = n - sum(len(c) for c in kept)
                    if need <= 0:
                        return np.concatenate(kept)[:n]
        return np.concatenate(kept)[:n] if kept and sum(map(len, kept)) >= n else None

    def _draw_prior(self, n: int):
        """Draw from the mixture prior: uniform over modes, then that mode's Gaussian."""
        if len(self._prior_means) == 0:
            return None
        which = self._rng.integers(len(self._prior_means), size=n)
        out = np.empty((n, 2))
        for k in range(len(self._prior_means)):
            m = which == k
            if m.any():
                out[m] = self._rng.multivariate_normal(
                    self._prior_means[k], self._prior_covs[k], size=int(m.sum()))
        return np.clip(out, -self.arena_half, self.arena_half)

    def _reinitialize_flat(self) -> None:
        self._particles = self._rng.uniform(
            -self.arena_half, self.arena_half, size=(self.num_particles, 2))
        self._weights = np.ones(self.num_particles) / self.num_particles

    @property
    def information_gain(self) -> float:
        """Nats of information gathered so far: ``-log(surviving prior mass)``.

        Zero at the start and monotonically increasing as regions are ruled out. Used as
        a shaping potential — it rewards sweeping *belief mass* rather than distance, so
        it is only exploitable by an agent that can tell where the mass is.
        """
        return -float(np.log(max(self.surviving_mass, 1e-12)))

    @property
    def particles(self) -> np.ndarray:
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def particle_dim(self) -> int:
        return self._particle_dim
