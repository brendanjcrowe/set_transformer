"""Particle filters for the ``pdomains`` Odd-Even POMDP.

The hidden state is one INTEGER in ``[1, n_dist_size]``, drawn at reset and
**static** for the whole episode. Two facts about that state drive every
choice in this file:

* **The state is discrete and its parity defines the observation support.**
  Observations only ever share the state's parity, so a candidate whose
  parity differs from an observation is refuted outright. A continuous
  particle has no parity and cannot express that, which is why particles here
  are integer state values, never floats.
* **The state never moves, so ``predict()`` is a no-op.** Process noise on a
  static state erases accumulated information every step and holds the belief
  at the noise floor. The information only ever accumulates, so the whole
  belief lives in the WEIGHTS: at ``n_dist_size = 50`` the exact posterior's
  effective sample size falls to about 1 of 50 within ~20 observations. An
  unweighted reading of these particles sees a near-uniform cloud over the
  full state range and learns almost nothing.

Both filters consume ``initial_env_obs``, so the belief they start from is
``b0 = P(s | o0)`` -- the same posterior the env's ``reset()`` returns. That
is not a detail: the reset observation is the only evidence available to the
first action, and at ``n_dist_size = 50`` over a 50-step cap the first step
alone is 81% of the optimal policy's pooled mean reward.

.. warning::

   ``_likelihood_table`` below MIRRORS
   ``pdomains.odd_even_pomdp.OddEvenPOMDP._compute_observation_probability``:
   the Gaussian is evaluated pointwise on the CANDIDATE state's own parity
   set and renormalized over just that set, so a candidate of the wrong
   parity gets likelihood exactly 0. **The two must be changed together.** If
   the env's observation model and this table drift apart, belief propagation
   diverges from the env silently -- no error is raised, the weights are just
   wrong. ``test_exact_support_filter_matches_the_env_posterior`` in
   ``tests/test_odd_even_pomdp_contract.py`` is the guard: it asserts the
   filter's weights equal the env's own posterior to 1e-9.

Two filters live here, and they answer different questions:

``OddEvenExactSupportParticleFilter``
    Particles are the state values themselves, fixed for the episode; the
    weights are the exact posterior. With ``num_particles == n_dist_size`` the
    belief is exact, so an encoder comparison built on it measures encoder
    loss with no filter loss mixed in. This is the primary arm.

``OddEvenBootstrapParticleFilter``
    The same, plus resampling on low effective sample size. That resampling is
    the realistic degradation, and on a static state it is irreversible: with
    no process noise to re-diversify, a state that loses its last particle can
    never be recovered.
"""

import numpy as np

from .base import BaseParticleFilter

#: Matches ``OddEvenPOMDPConfig.sigma_divisor``. See ``_default_std_dev``.
DEFAULT_SIGMA_DIVISOR = float(np.sqrt(10))


class _OddEvenParticleFilterBase(BaseParticleFilter):
    """Shared state representation, likelihood and weight arithmetic.

    Subclasses differ only in how the support is initialized and in what
    happens after a weight update (resampling, or nothing).
    """

    def __init__(self, num_particles: int, initial_env_obs: np.ndarray,
                 n_dist_size: int = 10,
                 std_dev: float | None = None,
                 sigma_divisor: float = DEFAULT_SIGMA_DIVISOR,
                 rng_seed: int | None = None,
                 **kwargs):
        """
        Args:
            num_particles: Number of particles to maintain. Must be at least
                ``n_dist_size``: below that, some state has no particle, and
                on a static state a state with no particle can never be
                recovered.
            initial_env_obs: The env's reset observation, folded in as
                evidence: the filter's starting belief is
                ``b0 = P(s | initial_env_obs)``, not the uniform prior. The
                env's ``reset()`` consumes the same observation, so the two
                beliefs stay level -- change one side and the other must
                change with it. Pass ``None`` for a filter that starts from
                the bare uniform prior.
            n_dist_size: Upper end of the state range ``[1, n_dist_size]``.
            std_dev: Observation-Gaussian width. Defaults to the env's own
                ``sqrt(n_dist_size)/sigma_divisor + 1``.
            sigma_divisor: Divisor in that default.
            rng_seed: Seed for a filter-private generator. The global
                ``np.random`` is never touched, so a per-episode seed makes a
                run reproducible -- ``PFDictWithWeightsObservationWrapper``
                derives one per episode and passes it here.
        """
        self.n_dist_size = int(n_dist_size)
        self.sigma_divisor = float(sigma_divisor)
        self.std_dev = (self._default_std_dev() if std_dev is None
                        else float(std_dev))
        self.rng_seed = None if rng_seed is None else int(rng_seed)
        # A private Generator, always. The old filter swallowed rng_seed in
        # **kwargs and drew from the process-global np.random, so two filters
        # built with the same seed disagreed and no run was reproducible.
        self._rng = np.random.default_rng(self.rng_seed)

        self._particle_dim = 1
        self._states = np.arange(1, self.n_dist_size + 1)
        self._likelihood_table = self._build_likelihood_table()

        # Set by _initialize_particles in the subclass.
        self._state_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._particles: np.ndarray = np.empty((0, 1), dtype=np.float64)
        self._weights: np.ndarray = np.empty(0, dtype=np.float64)

        super().__init__(num_particles, initial_env_obs, **kwargs)

    # -- model ------------------------------------------------------------

    def _default_std_dev(self) -> float:
        """The env's own default width. Mirrors ``OddEvenPOMDPConfig``."""
        return float(np.sqrt(self.n_dist_size) / self.sigma_divisor + 1.0)

    def _build_likelihood_table(self) -> np.ndarray:
        """``table[o - 1, s - 1] = P(observation o | state s)``.

        MIRRORS ``OddEvenPOMDP._compute_observation_probability``, evaluated
        once for every (observation, candidate) pair because both live on the
        same ``n_dist_size``-point grid. Precomputing turns the update into a
        table lookup and, more importantly, means there is exactly ONE place
        in this file where the observation model is written down.

        Column ``s`` sums to 1 over ``s``'s own parity set and is exactly 0
        off it, which is the whole of the env's model.
        """
        states = self._states
        gaussian = np.exp(
            -0.5 * ((states[:, None] - states[None, :]) / self.std_dev) ** 2)
        same_parity = (states[:, None] % 2) == (states[None, :] % 2)
        table = np.where(same_parity, gaussian, 0.0)
        # Per-candidate normalization over that candidate's own parity set.
        return table / table.sum(axis=0, keepdims=True)

    # -- interface --------------------------------------------------------

    def predict(self, action: np.ndarray = None, **kwargs) -> None:
        """No-op: the hidden state never changes.

        Deliberately not "small process noise for safety". Adding N(0, s) here
        would destroy the accumulated posterior every step and stop the belief
        from ever sharpening past that noise floor -- and the action is a
        prediction, so it does not move the state either.
        """
        return None

    def update(self, obs_from_env: np.ndarray, **kwargs) -> None:
        """Multiply in the env's likelihood for every emitted observation.

        Args:
            obs_from_env: The env's observation for this step, an array of
                ``obs_per_step`` integer values (shape ``(k,)``, or a scalar).
                Each one is independent evidence given the state, so all of
                them are folded in -- averaging them, as the old filter did,
                throws away k - 1 observations and mis-states the variance of
                the one it keeps.
        """
        self._fold_in(np.atleast_1d(np.asarray(obs_from_env)).ravel())
        self._after_update()

    def _fold_in_initial_observation(self,
                                     initial_env_obs: np.ndarray) -> None:
        """Turn the uniform prior into ``b0 = P(s | initial_env_obs)``.

        Called at the end of every ``_initialize_particles``. It multiplies in
        the likelihood exactly as ``update()`` does, but WITHOUT the
        post-update hook, so the bootstrap filter does not resample before it
        has propagated anything: resampling here would throw away the
        stratified full-state coverage that the initialization exists to
        guarantee, in the one place where no observation has yet refuted
        anything.

        ``None`` or an empty array leaves the uniform prior in place, for a
        caller that has no initial observation.
        """
        if initial_env_obs is None:
            return
        observations = np.atleast_1d(np.asarray(initial_env_obs)).ravel()
        if observations.size == 0:
            return
        self._fold_in(observations)

    def _fold_in(self, observations: np.ndarray) -> None:
        """Multiply in the env's likelihood for each observation, in order."""
        for observation in observations:
            value = int(round(float(observation)))
            if not 1 <= value <= self.n_dist_size:
                raise ValueError(
                    f"observation {observation} is outside the state range "
                    f"[1, {self.n_dist_size}]; this filter and its env "
                    "disagree about the observation model")
            self._weights = self._weights * self._likelihood_table[
                value - 1, self._state_idx]
            self._normalize_weights(value)

    def _normalize_weights(self, observation: int) -> None:
        """Renormalize by the TRUE sum, after every single observation.

        Three deliberate choices here:

        * **Divide by the true sum, never by ``sum + eps``.** With weights
          near 1e-30 an epsilon dominates the denominator and the total mass
          comes out near 1e-17; that bug once turned a Sinkhorn loss of 0.157
          into 2831.
        * **Renormalize per observation, not per step.** The largest weight
          then stays of order 1/N, so the weight vector as a whole cannot
          underflow to zero mass however long the episode runs.
        * **Let the smallest weights flush to exactly 0.** This belief goes
          near one-hot -- effective sample size about 1 of 50 -- so refuted
          states underflow. Zero is the right value for them: it is what the
          env's own float64 Bayes update produces, so the two stay
          comparable, and a zero weight is still a valid measure (finite,
          non-negative, positive total mass). Clamping to a floor instead
          would resurrect states the observations have already refuted.
        """
        total = float(self._weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            # Every particle is impossible under this observation. Raise
            # rather than resetting to uniform, exactly as the env's
            # update_belief() does: a silent recovery hides either an
            # env/filter mismatch or a corrupted observation behind a
            # plausible-looking belief.
            raise ValueError(
                f"observation {observation} has total probability {total} "
                "under every particle. Either this filter and its env "
                "disagree about the observation model, or the support died "
                "(resampling on a static state is irreversible).")
        self._weights = self._weights / total

    def _after_update(self) -> None:
        """Hook: the bootstrap filter resamples here, the exact one does not."""
        return None

    @property
    def particles(self) -> np.ndarray:
        """Integer state values, shape ``[num_particles, 1]``, float64."""
        return self._particles

    @property
    def weights(self) -> np.ndarray:
        """Normalized posterior mass, shape ``[num_particles]``, float64."""
        return self._weights

    @property
    def particle_dim(self) -> int:
        return self._particle_dim

    # -- readouts ---------------------------------------------------------

    def state_belief(self) -> np.ndarray:
        """Mass POOLED per state, shape ``[n_dist_size]``.

        Duplicate particles of one state split that state's mass, so the
        per-state belief is the pooled sum, not any single weight. This is
        the vector to compare against ``env.belief`` and to use as a probe
        label.
        """
        pooled = np.zeros(self.n_dist_size, dtype=np.float64)
        np.add.at(pooled, self._state_idx, self._weights)
        return pooled

    def effective_sample_size(self) -> float:
        """1 / sum(w^2). Reaches about 1.0 once the belief locks on."""
        return float(1.0 / np.sum(self._weights ** 2))

    def map_state(self) -> int:
        """Most probable state -- the greedy-argmax oracle's prediction."""
        return int(np.argmax(self.state_belief()) + 1)

    def estimate_mean(self) -> float:
        """Posterior mean state. Kept under its historical name."""
        return float(np.sum(self._states * self.state_belief()))

    # -- shared initialization helpers -------------------------------------

    def _require_full_coverage(self) -> None:
        if self.num_particles < self.n_dist_size:
            raise ValueError(
                f"num_particles={self.num_particles} is below "
                f"n_dist_size={self.n_dist_size}, so some state gets no "
                "particle. The state is static, so there is no process noise "
                "to re-diversify and a state with no particle can never be "
                "recovered -- it would be refuted before the first "
                "observation. Use num_particles >= n_dist_size.")

    def _set_support(self, states: np.ndarray) -> None:
        """Install a support and the exactly-uniform prior over it.

        The prior over a state drawn uniformly at reset is uniform, so a
        state's mass must be 1/n whatever number of particles happen to sit
        on it. Splitting equally among duplicates keeps the POOLED prior
        exactly uniform; leaving the per-particle weights uniform instead
        would tilt the prior toward whichever states got duplicated.
        """
        states = np.asarray(states, dtype=np.int64).ravel()
        self._state_idx = states - 1
        self._particles = states.astype(np.float64).reshape(-1, 1)
        copies = np.bincount(self._state_idx, minlength=self.n_dist_size)
        self._weights = 1.0 / (self.n_dist_size * copies[self._state_idx])


class OddEvenExactSupportParticleFilter(_OddEvenParticleFilterBase):
    """Fixed support on every state; the weights ARE the exact posterior.

    With ``num_particles == n_dist_size`` there is one particle per state and
    the weight vector equals ``env.belief`` to float64 precision, from the
    reset observation onward. Nothing is
    approximated and nothing is resampled, so this arm isolates encoder loss
    from filter loss -- which is what makes "does the belief encoder matter?"
    answerable on this domain.

    It is also the strongest possible case for weighted sets: the support is
    the same constant every episode and at every step, so the mass is the
    entire signal.
    """

    def _initialize_particles(self, initial_env_obs: np.ndarray,
                              **kwargs) -> None:
        """One particle per state, plus evenly tiled duplicates if asked."""
        self._require_full_coverage()
        # Tile states as evenly as num_particles allows. Uneven counts are
        # harmless because _set_support splits each state's mass among its
        # own copies.
        base, remainder = divmod(self.num_particles, self.n_dist_size)
        copies = np.full(self.n_dist_size, base, dtype=np.int64)
        copies[:remainder] += 1
        self._set_support(np.repeat(self._states, copies))
        self._fold_in_initial_observation(initial_env_obs)


class OddEvenBootstrapParticleFilter(_OddEvenParticleFilterBase):
    """Sampled support, true likelihood, resampling on low ESS.

    The realistic filter, and the one whose failure mode is worth measuring
    rather than hiding. Two Odd-Even-specific points:

    * **Initialization is stratified, not uniform.** Every state gets at
      least one particle, and only the surplus is drawn at random. A uniform
      draw misses the true state in 36.4% of episodes at N=50 and 13.3% at
      N=100 (n=50) -- 100 uniform draws over 50 states cover only about 43 of
      them -- and a missed state is refuted before the first observation, with
      no process noise to ever bring it back.
    * **Resampling is irreversible here.** It is what separates this filter
      from the exact-support one: each resample kills the states it does not
      duplicate, and on a static state they never come back. That is the cost
      this arm exists to measure.
    """

    def __init__(self, num_particles: int, initial_env_obs: np.ndarray,
                 ess_resample_fraction: float = 0.5,
                 **kwargs):
        """
        Args:
            ess_resample_fraction: Resample once the effective sample size
                falls below this fraction of ``num_particles``. Set it to 0 to
                never resample, which recovers the exact-support behaviour on
                a sampled support.
        """
        self.ess_resample_fraction = float(ess_resample_fraction)
        self.resample_count = 0
        super().__init__(num_particles, initial_env_obs, **kwargs)

    def _initialize_particles(self, initial_env_obs: np.ndarray,
                              **kwargs) -> None:
        """Full stratified coverage first, then the surplus from the prior."""
        self._require_full_coverage()
        surplus = self.num_particles - self.n_dist_size
        states = self._states
        if surplus > 0:
            # The prior is uniform, so the surplus is drawn uniformly. This
            # is the only random draw at init, and it comes from the
            # filter-private generator, so rng_seed reaches it.
            states = np.concatenate(
                [states,
                 self._rng.integers(1, self.n_dist_size + 1, size=surplus)])
        self._set_support(states)
        self._fold_in_initial_observation(initial_env_obs)

    def _after_update(self) -> None:
        if self.ess_resample_fraction <= 0.0:
            return
        if (self.effective_sample_size()
                < self.ess_resample_fraction * self.num_particles):
            self._resample()

    def _resample(self) -> None:
        """Systematic resampling, from the filter-private generator.

        Not ``filterpy.monte_carlo.systematic_resample``: that draws its
        single offset from the process-global ``np.random``, which would put
        an unseeded draw back into an otherwise reproducible filter.
        """
        n = self.num_particles
        positions = (self._rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(self._weights)
        cumulative[-1] = 1.0            # guard the float64 tail of the sum
        indices = np.searchsorted(cumulative, positions)
        self._state_idx = self._state_idx[indices]
        self._particles = self._particles[indices]
        # Resampling moves the mass into the support, so the weights go flat.
        self._weights = np.full(n, 1.0 / n, dtype=np.float64)
        self.resample_count += 1


#: Deprecated name, kept so ``from ... import OddEvenParticleFilter`` still
#: works. It now resolves to the exact-support filter -- the primary arm --
#: because the class that used to carry this name modelled a continuously
#: drifting state with its own unrelated Gaussian likelihood and could not
#: represent this domain's belief at all. Prefer the explicit names.
OddEvenParticleFilter = OddEvenExactSupportParticleFilter
