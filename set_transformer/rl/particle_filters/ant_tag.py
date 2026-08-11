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
                 min_initial_distance: float = 5.0,
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
            min_initial_distance: Minimum initial ant-target distance from env reset.
        """
        self.arena_min, self.arena_max = arena_limits
        self.obs_noise_std = obs_noise_std
        self.target_step = target_step
        self.visibility_radius = visibility_radius
        self.min_initial_distance = min_initial_distance
        
        # The actual particle state is [x, y] for the opponent
        self._particle_dim = 2 

        # Call super().__init__ which will call _initialize_particles
        super().__init__(num_particles, initial_env_obs, initial_spread_std=initial_spread_std, **kwargs)

    def _initialize_particles(self, initial_env_obs: np.ndarray, initial_spread_std: float, **kwargs) -> None:
        """Initialize particle states and weights.
        Particles match the AntTag reset prior: uniform in the arena, conditioned
        on starting far enough from the ant.
        """
        ant_pos = initial_env_obs[:2]
        particles = np.empty((self.num_particles, self.particle_dim))
        filled = 0

        while filled < self.num_particles:
            batch_size = max(2 * (self.num_particles - filled), self.num_particles)
            candidates = np.random.uniform(
                self.arena_min,
                self.arena_max,
                (batch_size, self.particle_dim),
            )
            distances = np.linalg.norm(candidates - ant_pos, axis=1)
            valid = candidates[distances > self.min_initial_distance]
            n_take = min(len(valid), self.num_particles - filled)
            if n_take > 0:
                particles[filled:filled + n_take] = valid[:n_take]
                filled += n_take

        self._particles = particles
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
            (new_positions[:, 0] < self.arena_min) |
            (new_positions[:, 0] > self.arena_max) |
            (new_positions[:, 1] < self.arena_min) |
            (new_positions[:, 1] > self.arena_max)
        )
        new_positions[out_of_bounds] = self._particles[out_of_bounds]

        self._particles = new_positions

    def update(self, observed_target_pos: np.ndarray, ant_current_pos_from_obs: np.ndarray,
               visibility_radius: float | None = None, **kwargs) -> None:
        """Update particle weights based on the observed target position (if any) and ant's position.
        Args:
            observed_target_pos: Observed [x,y] of the target. Can be [np.nan, np.nan] if not visible.
            ant_current_pos_from_obs: Current [x,y] position of the ant.
            visibility_radius: The radius that actually decided reveal/no-reveal
                this step. Overrides self.visibility_radius when provided, so a
                curriculum-annealed radius (see CurriculumVisibilityWrapper) is
                honored instead of the value fixed at construction time.
        """
        opponent_is_visible = not np.isnan(observed_target_pos[0])
        radius = self.visibility_radius if visibility_radius is None else visibility_radius

        if opponent_is_visible:
            # Target is visible, update weights based on distance to observation
            # This is a likelihood function: N(observed_target_pos | particle_pos, obs_std^2)
            distances_sq = np.sum((self._particles - observed_target_pos)**2, axis=1)
            self._weights *= (1. / (np.sqrt(2 * np.pi) * self.obs_noise_std)) * np.exp(-distances_sq / (2 * self.obs_noise_std**2))
        else:
            # Target is not visible. We can't directly update based on target observation.
            # However, we know the target is *not* within the ant's visibility radius.
            # Penalize particles that are within the ant's visibility radius.
            for i, p_opponent in enumerate(self._particles):
                dist_to_ant = np.linalg.norm(p_opponent - ant_current_pos_from_obs)
                if dist_to_ant < radius:
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


class SmartAntTagParticleFilter(AntTagParticleFilter):
    """
    Particle filter matching `pdomains.ant_tag.SmartAntTagEnv`'s target motion
    model: as the ant closes in, the target flees directly away more often
    (and, if the env's target_speed_scale knob is on, moves faster), and it
    slides along the arena wall instead of freezing against it. Everything
    else (init, update, resampling) is inherited unchanged from
    AntTagParticleFilter.
    """

    def __init__(self, num_particles: int, initial_env_obs: np.ndarray,
                 tag_radius: float = 1.5, **kwargs):
        self.tag_radius = tag_radius
        super().__init__(num_particles, initial_env_obs, **kwargs)

    def predict(self, action: np.ndarray, ant_current_pos_from_obs: np.ndarray,
                evasion_scale: float = 1.0, target_speed_scale: float = 0.0,
                **kwargs) -> None:
        """Predict the next state of particles using the smart target motion model.

        Per particle, urgency scales from 0 (ant at/beyond visibility_radius,
        flat 25/25/25/25 as in the base filter) to 1 (ant within tag_radius,
        flee-heavy), matching SmartAntTagEnv._move_target.
        evasion_scale mirrors the env's curriculum knob of the same name, so
        belief propagation tracks whatever evasion strength is currently
        active in the real env instead of assuming full strength.

        target_speed_scale likewise mirrors SmartAntTagEnv.target_speed_scale
        and MUST match it, or the belief propagates the target at a different
        speed than the env actually moves it. It defaults to 0.0 (constant
        target_step) to match the env's own default; the caller
        (ant_tag_pf_interaction_mapper) reads the live value off the env each
        step rather than relying on this default.
        """
        ant = ant_current_pos_from_obs

        diff = ant - self._particles  # [N, 2]
        dists = np.linalg.norm(diff, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-8)
        target2ant = diff / dists

        perp_left = np.column_stack([target2ant[:, 1], -target2ant[:, 0]])
        perp_right = np.column_stack([-target2ant[:, 1], target2ant[:, 0]])
        away = -target2ant
        stay = np.zeros_like(target2ant)

        urgency = np.clip(
            (self.visibility_radius - dists[:, 0]) / (self.visibility_radius - self.tag_radius),
            0.0, 1.0,
        ) * evasion_scale
        p_flee = 0.25 + urgency * 0.45
        p_stay = 0.25 - urgency * 0.20
        p_side = (1.0 - p_flee - p_stay) / 2.0
        cum_probs = np.stack([p_side, 2 * p_side, 2 * p_side + p_flee, np.ones_like(p_flee)], axis=1)

        u = np.random.rand(self.num_particles, 1)
        choices = (u > cum_probs).sum(axis=1)

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

        step_size = (self.target_step * (1.0 + urgency * target_speed_scale))[:, None]
        new_positions = self._particles + directions * step_size

        # Slide along the wall instead of freezing (matches SmartAntTagEnv)
        new_positions[:, 0] = np.clip(new_positions[:, 0], self.arena_min, self.arena_max)
        new_positions[:, 1] = np.clip(new_positions[:, 1], self.arena_min, self.arena_max)

        self._particles = new_positions


class GhostAntTagParticleFilter(SmartAntTagParticleFilter):
    """PF matching pdomains.ant_tag.GhostAntTagEnv: SmartAntTag dynamics
    (predict inherited unchanged) plus a clutter-robust mixture update for
    the ghost-ping sensor.

    When no visual detection but a ping z arrived:
        w_i *= [ beta * N2(z; x_i, sigma^2 I) + (1-beta)/A ] * neg_info(x_i)
    where N2 is the PROPERLY NORMALIZED 2D Gaussian density (the 1/(2*pi*s^2)
    prefactor matters here: in a mixture the constants do NOT cancel),
    A is the arena area, and neg_info is the existing 0.1 penalty for
    particles inside the visibility radius (not-seeing is still evidence).

    Ghost-birth: if the belief has (nearly) no support near a fresh ping --
    a particle-depletion artifact, since the true posterior holds ~beta mass
    there -- the lowest-weight particles are reborn around z with collective
    weight w_birth (= beta, the Bayes-consistent posterior share under a
    locally uniform depleted prior).
    """

    BIRTH_TRIGGER_WEIGHT = 0.05
    N_BIRTH = 15
    W_BIRTH = 0.35

    def update(self, observed_target_pos: np.ndarray,
               ant_current_pos_from_obs: np.ndarray,
               visibility_radius: float | None = None,
               ghost_ping: np.ndarray | None = None,
               ping_beta: float = 0.35,
               ping_sigma: float = 0.8,
               **kwargs) -> None:
        visible = not np.isnan(observed_target_pos[0])
        if visible or ghost_ping is None:
            # Visual detection, or nothing at all: base behavior unchanged.
            super().update(observed_target_pos, ant_current_pos_from_obs,
                           visibility_radius=visibility_radius, **kwargs)
            return

        radius = self.visibility_radius if visibility_radius is None else visibility_radius
        arena_area = (self.arena_max - self.arena_min) ** 2

        ghost_ping = np.asarray(ghost_ping, dtype=float).reshape(-1)

        d2 = np.sum((self._particles - ghost_ping) ** 2, axis=1)
        gauss = np.exp(-d2 / (2.0 * ping_sigma ** 2)) / (2.0 * np.pi * ping_sigma ** 2)
        likelihood = ping_beta * gauss + (1.0 - ping_beta) / arena_area

        dist_to_ant = np.linalg.norm(
            self._particles - ant_current_pos_from_obs, axis=1)
        neg_info = np.where(dist_to_ant < radius, 0.1, 1.0)

        self._weights *= likelihood * neg_info
        self._weights += 1.e-300
        self._weights /= np.sum(self._weights)

        # Ghost-birth on depletion near the ping.
        near = np.linalg.norm(self._particles - ghost_ping, axis=1) < 2.0 * ping_sigma
        if float(self._weights[near].sum()) < self.BIRTH_TRIGGER_WEIGHT:
            n_birth = self.N_BIRTH
            idx = np.argsort(self._weights)[:n_birth]
            self._particles[idx] = np.clip(
                ghost_ping + np.random.normal(0.0, ping_sigma, (n_birth, self._particle_dim)),
                self.arena_min, self.arena_max,
            )
            keep = np.ones(self.num_particles, dtype=bool)
            keep[idx] = False
            keep_sum = float(self._weights[keep].sum())
            if keep_sum > 0.0:
                self._weights[keep] *= (1.0 - self.W_BIRTH) / keep_sum
            self._weights[idx] = self.W_BIRTH / n_birth
            self._weights /= np.sum(self._weights)

        # Same N_eff resample trigger as the base class.
        if 1. / np.sum(self._weights ** 2) < self.num_particles / 2.:
            self._resample_particles()


class TwinDenAntTagParticleFilter(SmartAntTagParticleFilter):
    """PF matching pdomains.ant_tag.TwinDenAntTagEnv: nearest-den commitment,
    deterministic transit, leashed smart-flee jitter in-den. update() is
    inherited unchanged (visual likelihood + negative info); only predict()
    changes. Den geometry and the per-episode tight/loose assignment arrive
    per-step via predict kwargs (forwarded from live env attributes by
    ant_tag_pf_interaction_mapper) — the assignment is a motion-model
    parameter the filter is entitled to know, exactly like evasion_scale.
    The hidden episode latent (which den the target actually committed to)
    is represented only by the particle population itself.
    """

    def predict(self, action: np.ndarray, ant_current_pos_from_obs: np.ndarray,
                evasion_scale: float = 1.0, target_speed_scale: float = 0.0,
                den_tight: int | None = None,
                den_positions: np.ndarray | None = None,
                den_radius_tight: float | None = None,
                den_radius_loose: float | None = None,
                **kwargs) -> None:
        if den_tight is None or den_positions is None \
                or den_radius_tight is None or den_radius_loose is None:
            raise ValueError(
                "TwinDenAntTagParticleFilter.predict requires den_tight, "
                "den_positions, den_radius_tight, den_radius_loose (is the "
                "env TwinDenAntTagEnv and ant_tag_pf_interaction_mapper "
                "up to date?)"
            )
        n = self.num_particles
        X = self._particles
        dens = np.asarray(den_positions, dtype=np.float64)  # [2, 2]

        d_all = np.linalg.norm(X[:, None, :] - dens[None, :, :], axis=2)  # [N,2]
        den_idx = np.argmin(d_all, axis=1)                                # [N]
        dist_to_den = d_all[np.arange(n), den_idx]                        # [N]
        den_pos = dens[den_idx]                                           # [N,2]
        radius = np.where(den_idx == int(den_tight),
                          float(den_radius_tight), float(den_radius_loose))
        in_den = dist_to_den <= radius

        # --- Transit branch: straight to the den center, landing exactly.
        step = np.minimum(self.target_step, dist_to_den)
        dir_to_den = (den_pos - X) / np.maximum(dist_to_den, 1e-8)[:, None]
        transit_pos = X + dir_to_den * step[:, None]

        # --- In-den branch: parent's smart urgency-flee step, leashed.
        ant = ant_current_pos_from_obs
        diff = ant - X
        dists = np.linalg.norm(diff, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-8)
        target2ant = diff / dists
        perp_left = np.column_stack([target2ant[:, 1], -target2ant[:, 0]])
        perp_right = -perp_left
        away = -target2ant
        stay = np.zeros_like(target2ant)
        urgency = np.clip(
            (self.visibility_radius - dists[:, 0])
            / (self.visibility_radius - self.tag_radius), 0.0, 1.0
        ) * evasion_scale
        p_flee = 0.25 + urgency * 0.45
        p_stay = 0.25 - urgency * 0.20
        p_side = (1.0 - p_flee - p_stay) / 2.0
        cum_probs = np.stack(
            [p_side, 2 * p_side, 2 * p_side + p_flee, np.ones_like(p_flee)],
            axis=1)
        u = np.random.rand(n, 1)
        choices = (u > cum_probs).sum(axis=1)
        directions = np.where(
            (choices == 0)[:, None], perp_left,
            np.where((choices == 1)[:, None], perp_right,
                     np.where((choices == 2)[:, None], away, stay)))
        candidate = X + directions * self.target_step
        off = candidate - den_pos
        off_norm = np.linalg.norm(off, axis=1)
        scale = np.where(off_norm > radius,
                         radius / np.maximum(off_norm, 1e-8), 1.0)
        leashed_pos = den_pos + off * scale[:, None]

        new_positions = np.where(in_den[:, None], leashed_pos, transit_pos)
        new_positions[:, 0] = np.clip(new_positions[:, 0], self.arena_min, self.arena_max)
        new_positions[:, 1] = np.clip(new_positions[:, 1], self.arena_min, self.arena_max)
        self._particles = new_positions


class CounterweightedDenAntTagParticleFilter(SmartAntTagParticleFilter):
    """PF matching pdomains.ant_tag.CounterweightedDenAntTagEnv.

    Differences from TwinDenAntTagParticleFilter:

    * There is NO transit branch -- the target spawns settled in its den, so
      every particle is always leashed to its nearest den disc (the leash
      projection also snaps resampling-noise strays back onto the disc).
    * Both dens share a single radius `cden_r`; the signal carrier is the
      weight x distance counterweight (heavy-near occupied w.p.
      w = f/(h+f), light-far otherwise), not a radius asymmetry.
    * Two-stage init. `_initialize_particles` builds the s-MARGINALIZED
      four-candidate prior (the correct Bayesian belief before the mirror bit
      is known), because the wrapper rebuilds the PF from build-time-static
      kwargs and cannot deliver the per-episode geometry. The first
      `predict` call -- the first moment the mapper hands over live
      `cden_heavy_pos` / `cden_light_pos` -- conditions on the bit by
      resampling into the two ACTUAL discs.
    * `update` applies an exact deterministic alarm/silence likelihood before
      the inherited visual + negative-info update.

    Den geometry, the occupancy prior and the spook state are motion- and
    observation-model parameters the filter is entitled to know (the
    evasion_scale convention); the hidden episode latent (WHICH den is
    actually occupied) is represented only by the particle population.
    """

    #: Soft-hard zero for deterministic-channel evidence, matching the
    #: codebase's preference for multiplicative penalties over exact zeros.
    ALARM_EPS = 1e-6

    def __init__(self, num_particles: int, initial_env_obs: np.ndarray,
                 cden_h: float = 2.4, cden_f: float = 6.75,
                 cden_r: float = 0.4, **kwargs):
        # Set BEFORE super().__init__, which calls _initialize_particles.
        self.cden_h = float(cden_h)
        self.cden_f = float(cden_f)
        self.cden_r = float(cden_r)
        self.cden_w_heavy = self.cden_f / (self.cden_h + self.cden_f)
        self._diag = np.array([1.0, 1.0]) / np.sqrt(2.0)
        # Same order as the env's cden_candidates: [-h, +h, -f, +f].
        self.cden_candidates = np.stack([s * d * self._diag
            for d in (self.cden_h, self.cden_f) for s in (-1.0, 1.0)])
        self._needs_prior_init = True
        self._alarm_processed = False
        super().__init__(num_particles, initial_env_obs, **kwargs)

    # -- helpers ----------------------------------------------------------

    def _sample_in_disc(self, centers: np.ndarray) -> np.ndarray:
        """One uniform-in-disc(radius cden_r) draw per row of `centers`."""
        n = centers.shape[0]
        rr = self.cden_r * np.sqrt(np.random.rand(n))
        th = np.random.uniform(0.0, 2.0 * np.pi, n)
        return centers + rr[:, None] * np.stack([np.cos(th), np.sin(th)],
                                                axis=1)

    # -- two-stage init ---------------------------------------------------

    def _initialize_particles(self, initial_env_obs: np.ndarray,
                              initial_spread_std: float, **kwargs) -> None:
        """s-marginalized prior: the four STATIC candidate discs, heavy
        candidates carrying w/2 each and light candidates (1-w)/2 each.
        Marginalizing over the (uniform) mirror bit is exactly this."""
        w = self.cden_w_heavy
        probs = np.array([w / 2.0, w / 2.0, (1 - w) / 2.0, (1 - w) / 2.0])
        pick = np.random.choice(4, size=self.num_particles, p=probs)
        self._particles = self._sample_in_disc(self.cden_candidates[pick])
        self._weights = np.ones(self.num_particles) / self.num_particles
        self._needs_prior_init = True
        self._alarm_processed = False

    def _condition_on_mirror_bit(self, heavy_pos, light_pos, w_heavy):
        """First-predict stage 2: collapse the four-candidate marginal prior
        onto the two ACTUAL discs, heavy with probability w_heavy."""
        take_heavy = np.random.rand(self.num_particles) < float(w_heavy)
        centers = np.where(take_heavy[:, None], heavy_pos[None, :],
                           light_pos[None, :])
        self._particles = self._sample_in_disc(centers)
        self._weights = np.ones(self.num_particles) / self.num_particles
        self._needs_prior_init = False

    def _nearest_den(self, dens: np.ndarray):
        """dens: [2, 2] -> (idx [N], dist [N], den_pos [N, 2])."""
        d_all = np.linalg.norm(self._particles[:, None, :] - dens[None, :, :],
                               axis=2)
        idx = np.argmin(d_all, axis=1)
        return idx, d_all[np.arange(self.num_particles), idx], dens[idx]

    # -- motion model -----------------------------------------------------

    def predict(self, action: np.ndarray, ant_current_pos_from_obs: np.ndarray,
                evasion_scale: float = 1.0, target_speed_scale: float = 0.0,
                cden_heavy_pos: np.ndarray | None = None,
                cden_light_pos: np.ndarray | None = None,
                cden_w_heavy: float | None = None,
                cden_r: float | None = None,
                cden_spooked: bool = False,
                **kwargs) -> None:
        if (cden_heavy_pos is None or cden_light_pos is None
                or cden_w_heavy is None or cden_r is None):
            raise ValueError(
                "CounterweightedDenAntTagParticleFilter.predict requires "
                "cden_heavy_pos, cden_light_pos, cden_w_heavy, cden_r (is the "
                "env CounterweightedDenAntTagEnv and "
                "ant_tag_pf_interaction_mapper up to date?)"
            )
        heavy = np.asarray(cden_heavy_pos, dtype=np.float64).reshape(2)
        light = np.asarray(cden_light_pos, dtype=np.float64).reshape(2)

        if self._needs_prior_init:
            # Loud consistency check: the constructor duplicates the env's
            # geometry, so a mismatch means a stale mapper or the wrong env.
            live_h = float(np.linalg.norm(heavy))
            live_f = float(np.linalg.norm(light))
            if not (np.isclose(live_h, self.cden_h, atol=1e-6)
                    and np.isclose(live_f, self.cden_f, atol=1e-6)
                    and np.isclose(float(cden_r), self.cden_r, atol=1e-9)
                    and np.isclose(float(cden_w_heavy), self.cden_w_heavy,
                                   atol=1e-9)):
                raise ValueError(
                    "cdens PF geometry disagrees with the live env: PF has "
                    f"h={self.cden_h}, f={self.cden_f}, r={self.cden_r}, "
                    f"w={self.cden_w_heavy:.6f}; env delivered h={live_h:.6f}, "
                    f"f={live_f:.6f}, r={float(cden_r)}, "
                    f"w={float(cden_w_heavy):.6f}."
                )
            # First predict = conditioning on the mirror bit. No motion step:
            # the t=0 marginal prior is consumed by the reset observation only
            # (one step out of 300, identically across all encoder arms).
            self._condition_on_mirror_bit(heavy, light, cden_w_heavy)
            return

        if cden_spooked:
            # Alarm raised: the leash dissolved env-side on the same step.
            SmartAntTagParticleFilter.predict(
                self, action, ant_current_pos_from_obs,
                evasion_scale=evasion_scale,
                target_speed_scale=target_speed_scale)
            return

        n = self.num_particles
        X = self._particles
        dens = np.stack([heavy, light])
        _, _, den_pos = self._nearest_den(dens)
        radius = float(cden_r)

        # --- in-den: parent's smart urgency-flee step, leashed to the disc.
        # KEEP TEXTUALLY PARALLEL with
        # CounterweightedDenAntTagEnv._move_target's in-den branch. ---
        ant = ant_current_pos_from_obs
        diff = ant - X
        dists = np.linalg.norm(diff, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-8)
        target2ant = diff / dists
        perp_left = np.column_stack([target2ant[:, 1], -target2ant[:, 0]])
        perp_right = -perp_left
        away = -target2ant
        toward = target2ant
        stay = np.zeros_like(target2ant)
        urgency = np.clip(
            (self.visibility_radius - dists[:, 0])
            / (self.visibility_radius - self.tag_radius), 0.0, 1.0
        ) * evasion_scale
        # ISOTROPIC-AT-ZERO-URGENCY: exact mirror of
        # CounterweightedDenAntTagEnv._move_target's five-way split. The
        # inherited {perp_left, perp_right, away, STAY} set has no "toward",
        # so flat 25/25/25/25 drifts 0.125 u/step away from the ant even at
        # urgency 0; that bearing-locked drift leaks the mirror bit into the
        # pooled mean. Reassigning the urgency-0 stay mass to `toward` makes
        # the no-threat wander isotropic (mean 0, covariance 0.5*step^2*I)
        # while leaving the urgency-1 cornered split bit-identical.
        p_flee = 0.25 + urgency * 0.45
        p_toward = 0.25 * (1.0 - urgency)
        p_stay = 0.05 * urgency
        p_side = (1.0 - p_flee - p_toward - p_stay) / 2.0
        cum_probs = np.stack(
            [p_side, 2 * p_side, 2 * p_side + p_flee,
             2 * p_side + p_flee + p_toward, np.ones_like(p_flee)],
            axis=1)
        u = np.random.rand(n, 1)
        choices = (u > cum_probs).sum(axis=1)
        directions = np.where(
            (choices == 0)[:, None], perp_left,
            np.where((choices == 1)[:, None], perp_right,
                     np.where((choices == 2)[:, None], away,
                              np.where((choices == 3)[:, None], toward,
                                       stay))))
        candidate = X + directions * self.target_step
        off = candidate - den_pos
        off_norm = np.linalg.norm(off, axis=1)
        scale = np.where(off_norm > radius,
                         radius / np.maximum(off_norm, 1e-8), 1.0)
        new_positions = den_pos + off * scale[:, None]

        new_positions[:, 0] = np.clip(new_positions[:, 0], self.arena_min, self.arena_max)
        new_positions[:, 1] = np.clip(new_positions[:, 1], self.arena_min, self.arena_max)
        self._particles = new_positions

    # -- observation model ------------------------------------------------

    def update(self, observed_target_pos: np.ndarray,
               ant_current_pos_from_obs: np.ndarray,
               visibility_radius: float | None = None,
               cden_heavy_pos: np.ndarray | None = None,
               cden_light_pos: np.ndarray | None = None,
               cden_r: float | None = None,
               cden_spook_enabled: bool = False,
               cden_spooked: bool = False,
               cden_spook_pos: np.ndarray | None = None,
               cden_spook_radius: float | None = None,
               **kwargs) -> None:
        """Exact deterministic alarm/silence likelihood, then the inherited
        visual + negative-info update.

        The alarm is a deterministic observation channel: with the ant inside
        den k's trigger zone, the alarm fires IFF den k is empty. Hence

          silence  : P(no alarm | target in den k) = 1,
                     P(no alarm | target elsewhere) = 0
                     -> every particle whose nearest den != k gets ALARM_EPS.
                     (This is the alarm-channel analogue of the existing
                     negative-visual-information penalty.)
          alarm    : the den nearest cden_spook_pos is certainly EMPTY
                     -> its particles get ALARM_EPS, applied exactly once.
          after    : no further den-based evidence; the target is free.
        """
        if (cden_spook_enabled and cden_heavy_pos is not None
                and cden_light_pos is not None):
            dens = np.stack([
                np.asarray(cden_heavy_pos, dtype=np.float64).reshape(2),
                np.asarray(cden_light_pos, dtype=np.float64).reshape(2)])
            near_idx, _, _ = self._nearest_den(dens)

            if cden_spooked:
                if not self._alarm_processed and cden_spook_pos is not None:
                    spook = np.asarray(cden_spook_pos,
                                       dtype=np.float64).reshape(2)
                    j = int(np.argmin(np.linalg.norm(dens - spook, axis=1)))
                    self._weights = np.where(near_idx == j,
                                             self._weights * self.ALARM_EPS,
                                             self._weights)
                    self._alarm_processed = True
            elif cden_spook_radius is not None:
                ant = np.asarray(ant_current_pos_from_obs,
                                 dtype=np.float64).reshape(2)
                d_ant = np.linalg.norm(dens - ant, axis=1)
                for k in range(2):
                    if d_ant[k] < float(cden_spook_radius):
                        self._weights = np.where(
                            near_idx != k,
                            self._weights * self.ALARM_EPS, self._weights)

        super().update(observed_target_pos, ant_current_pos_from_obs,
                       visibility_radius=visibility_radius, **kwargs)
