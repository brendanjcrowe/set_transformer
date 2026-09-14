"""Pass-through "filter" for envs that emit the belief cloud themselves (the hunt tasks).

**This class performs no inference.** On Cluster-Hunt, least-mass and most-var
(``pdomains.hunt``) the particle cloud IS the belief: the env redraws 100 particles from its
live clusters at every step and hands them out as part of the observation. There is nothing
to predict or update. The shared harness still expects a particle-filter object behind
:class:`~set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`
(that is how every other domain's belief reaches the encoders, and what the collector, the
pretraining round-trip check and the evaluation script build), so this class fills that slot
by COPYING the env's cloud after every step (plan section 9 of the parent repo's
``refactor_plans.md``, decision B, 2026-09-13).

Two conventions, both fixed here so every encoder sees the same input:

* **Agent-relative, raw arena units.** The emitted particles are ``env.particles - env.pos``
  in the env's own [0, 20]^2 coordinates (decision 9c-2). Every recorded hunt arm subtracted
  the agent position inside its features extractor (``src/hunt_tasks/encoders/extractors.py::
  _Base.forward``, "agent-centric"); the harness extractors do not, so the subtraction moves
  here. The extractors then divide by ``arena_scale`` = 10 (``pdomains.hunt.SCALE``), which
  gives exactly the tensor the recorded arms computed.
* **Uniform weights ``1/N``.** The env draws its cloud by sampling, so every particle carries
  equal mass; the weight channel of the weighted encoders is then constant, as it was in the
  recorded (unweighted) arms.

The filter reads the env directly (``env=`` in the filter kwargs, the unwrapped env), so no
``pf_interaction_mapper`` is needed: the wrapper's positional ``update(obs)`` call is enough.
``num_particles`` must equal the env's cloud size, or construction raises: a smaller filter
would silently drop particles, a larger one would have nothing to fill the rest with.
"""

from __future__ import annotations

import numpy as np

from .base import BaseParticleFilter


class EnvEmittedBeliefFilter(BaseParticleFilter):
    """Copies the env's own particle cloud, agent-relative, with uniform weights. No inference."""

    def __init__(self, num_particles: int, initial_env_obs=None, *, env,
                 particles_attr: str = "particles", position_attr: str = "pos",
                 rng_seed=None, **kwargs):
        """
        Args:
            num_particles: Must equal the size of the env's cloud (``len(env.<particles_attr>)``).
            initial_env_obs: Ignored (the wrapper passes it; the cloud is read off the env).
            env: The UNWRAPPED env holding the cloud and the agent position as attributes.
            particles_attr / position_attr: Attribute names on ``env`` ([N, 2] and [2]).
            rng_seed: Ignored (the wrapper passes one per episode; nothing here is random).
        """
        if kwargs:
            raise TypeError(f"EnvEmittedBeliefFilter takes no extra kwargs, got {sorted(kwargs)}")
        self._env = env
        self._particles_attr = particles_attr
        self._position_attr = position_attr
        self._particles = None
        self._weights = None
        super().__init__(num_particles, initial_env_obs)

    # -- BaseParticleFilter interface ------------------------------------------------------------

    def _initialize_particles(self, env_obs, **kwargs) -> None:
        self._copy_from_env()

    def predict(self, action, **kwargs) -> None:
        """No motion model: the env redraws the cloud itself."""

    def update(self, obs_from_env=None, **kwargs) -> None:
        """Copy the cloud the env holds AFTER the step just taken."""
        self._copy_from_env()

    @property
    def particles(self) -> np.ndarray:
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def particle_dim(self) -> int:
        return 2

    # -- the copy -------------------------------------------------------------------------------

    def _copy_from_env(self) -> None:
        cloud = np.asarray(getattr(self._env, self._particles_attr), dtype=np.float64)
        pos = np.asarray(getattr(self._env, self._position_attr), dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 2 or pos.shape != (2,):
            raise ValueError(
                f"{type(self._env).__name__}.{self._particles_attr} has shape {cloud.shape} and "
                f".{self._position_attr} has shape {pos.shape}; expected [N, 2] and [2]")
        if cloud.shape[0] != self.num_particles:
            raise ValueError(
                f"EnvEmittedBeliefFilter was built for num_particles={self.num_particles} but "
                f"{type(self._env).__name__} emits {cloud.shape[0]} particles. The filter copies "
                "the env's cloud, so the two must agree; pass --num_particles "
                f"{cloud.shape[0]} (or leave it to the domain default).")
        # Agent-relative, raw arena units (decision 9c-2). The extractors divide by
        # arena_scale, so the encoder sees (particles - pos) / SCALE, exactly what the recorded
        # arms' agent-centric extractors computed from the env's scaled observation.
        self._particles = (cloud - pos[None, :]).astype(np.float32)
        self._weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float32)
