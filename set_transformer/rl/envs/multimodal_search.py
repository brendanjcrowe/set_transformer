"""Multimodal Search — a POMDP built so a Gaussian belief summary provably cannot win.

A static target hides in one of ``K`` Gaussian modes. The agent sees perfectly inside a
small radius and nothing outside it, so the task is a search: visit modes in turn until
the target turns up. The optimal policy needs the **locations and shapes of the individual
modes**; a mean-and-covariance summary of the belief cannot express either.

Two construction choices make that airtight rather than merely plausible.

* **The belief mean is pinned at the origin.** Mode centres are drawn at random and then
  translated so their centroid is exactly zero, so the belief mean is the same constant
  every episode and carries *zero* information about the target. This is the same trick as
  the collaborator's Counterweighted-Den Ant-Tag variant, applied to positions.
* **The geometry stays random.** Pinning the centroid by placing modes on a fixed ring
  would also zero the mean, but then an agent could learn to sweep that ring and ignore
  the belief entirely. Here the centres are unconstrained apart from the centroid and a
  minimum separation, so mode radii range over roughly 0.3-13.5 and there is no fixed
  path to memorise.

Each mode is a random full Gaussian (random rotation applied to random anisotropic
scales), matching ``set_transformer.data.mixture_of_gaussians``. That is what forces the
belief to convey mode *shape* as well as position: a diffuse mode costs many steps to
sweep and a tight one costs one visit, so an optimal searcher weighs footprint against
distance. A single global covariance reports the spread of the whole configuration and
says nothing about any individual mode.

**The mode footprint constraint is load-bearing.** The environment discriminates only
while the modes occupy a small fraction of the arena; if they cover most of it, sweeping
the modes costs about as much as sweeping everything and the whole point evaporates.
Measured over 3,000 sampled episodes at K ~ U[2,10]: with ``mode_scale_hi = 1.0`` an
informed tour takes 77 steps against 307 for a full lawnmower (a 4.0x gap), but at
``mode_scale_hi = 2.0`` that collapses to 1.4x. :meth:`MultimodalSearchConfig.__post_init__`
enforces a bound; do not raise it without re-measuring the gap.

The horizon sits deliberately *between* the two: 250 steps is ample for a mode tour and
not enough for a lawnmower, so an agent that cannot localise the modes runs out of time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

#: Fixed per-step observation entries: agent xy, previous agent xy, target offset xy,
#: visibility flag. The previous position is carried so the filter can refute the whole
#: swath swept between the two, not just a disc at the endpoint.
BASE_OBS_DIM = 7
#: Per-mode entries packed into the observation: validity, mean xy, cov (xx, xy, yy).
MODE_OBS_DIM = 6


@dataclass
class MultimodalSearchConfig:
    """Environment parameters. Defaults are the measured 4.0x operating point."""

    arena_half: float = 14.0
    k_min: int = 2
    k_max: int = 10
    #: Per-axis standard deviations of each mode, drawn uniformly from this range.
    mode_scale_lo: float = 0.3
    mode_scale_hi: float = 1.0
    min_separation: float = 5.0
    visibility_radius: float = 1.5
    #: Distance covered per action. Chosen large relative to the visibility radius on
    #: purpose. What matters for whether the task is searchable is the PATH BUDGET
    #: (``speed * max_steps``), which must exclude a full-arena lawnmower (~307 units)
    #: while admitting a mode tour (~38); at a fixed budget, step size and decisions per
    #: episode are inversely locked. Bigger steps buy exploration: a random walk's net
    #: displacement goes as sqrt(budget * step), so raising the speed from 0.5 to 3.0
    #: takes random-policy success from 0.08 to 0.15 and net displacement from 4.9 to 9.9
    #: units in a 28-wide arena -- while an informed tour is unaffected (0.95 either way).
    #: This is exactly equivalent to action repeat, without the extra concept.
    speed: float = 3.0
    #: Horizon. Deliberately short: the path budget (speed * max_steps = 126 units)
    #: admits an informed mode tour (~38 units) but excludes a full-arena lawnmower
    #: (~307), so a belief-blind policy cannot simply sweep everything.
    #:
    #: A longer horizon was tried, to make the headline metric *time to find* rather than
    #: success rate. It did not work: tripling the budget does not make a partially
    #: trained policy cover the arena, because a random walk's coverage grows only as
    #: sqrt(t). Success stayed at 0.13-0.26 instead of saturating, so episode length just
    #: measured how often the agent timed out, and the larger timeout penalty diluted the
    #: learning signal -- st_scratch reached the same success at horizon 150 in 1M steps
    #: as at horizon 42 in 600k. Revisit only once a policy reliably solves the task.
    max_steps: int = 42
    step_penalty: float = -1.0
    #: Paid once on finding the target. One more than ``max_steps`` so a positive return
    #: means "found" and a negative one means "did not", with no ambiguous middle even
    #: when the target is found on the final step.
    find_bonus: float = 43.0
    #: Reject a configuration whose modes cover more than this fraction of the arena --
    #: past it, sweeping the modes is no cheaper than sweeping everything.
    max_footprint_fraction: float = 0.35
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 1 <= self.k_min <= self.k_max:
            raise ValueError(f"need 1 <= k_min <= k_max, got {self.k_min}, {self.k_max}")
        if not 0 < self.mode_scale_lo <= self.mode_scale_hi:
            raise ValueError("need 0 < mode_scale_lo <= mode_scale_hi")
        # Worst case: k_max modes each at the largest scale. A mode's ~2-sigma footprint
        # is pi * (2*s)^2; the arena is (2 * arena_half)^2.
        footprint = self.k_max * np.pi * (2.0 * self.mode_scale_hi) ** 2
        fraction = footprint / (2.0 * self.arena_half) ** 2
        if fraction > self.max_footprint_fraction:
            raise ValueError(
                f"modes would cover {fraction:.0%} of the arena (limit "
                f"{self.max_footprint_fraction:.0%}). At that density, visiting every "
                f"mode costs about as much as sweeping the whole arena and the "
                f"environment stops discriminating belief encoders -- which is its only "
                f"purpose. Lower k_max or mode_scale_hi, or enlarge arena_half."
            )

    @property
    def obs_dim(self) -> int:
        return BASE_OBS_DIM + MODE_OBS_DIM * self.k_max

    @property
    def mode_obs_indices(self) -> list[int]:
        """Observation indices holding the prior's mode parameters.

        These are read by the particle filter to build the belief and **masked from the
        agent** (``EnvSpec.obs_mask_indices``): the agent is meant to learn the mode
        structure through its belief encoder, not to be handed it as a vector.
        """
        return list(range(BASE_OBS_DIM, self.obs_dim))


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment ``a -> b``.

    A moving sensor sweeps a swath, not a disc: with a step comparable to the visibility
    radius, testing only the endpoint would let the agent jump over the target and would
    leave unobserved gaps between consecutive positions.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(points - a, axis=-1)
    t = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * ab), axis=-1)


def pack_observation(agent_pos, prev_pos, target_offset, visible, means, covs,
                     k_max) -> np.ndarray:
    """Lay out one observation. Shared with the particle filter so the two cannot drift."""
    obs = np.zeros(BASE_OBS_DIM + MODE_OBS_DIM * k_max, dtype=np.float32)
    obs[0:2] = agent_pos
    obs[2:4] = prev_pos
    obs[4:6] = target_offset
    obs[6] = float(visible)
    for k, (mu, cov) in enumerate(zip(means, covs)):
        base = BASE_OBS_DIM + MODE_OBS_DIM * k
        obs[base] = 1.0                                   # validity
        obs[base + 1:base + 3] = mu
        obs[base + 3] = cov[0, 0]
        obs[base + 4] = cov[0, 1]
        obs[base + 5] = cov[1, 1]
    return obs


def unpack_modes(obs: np.ndarray, k_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recover ``(means, covs)`` for the valid modes packed into ``obs``."""
    means, covs = [], []
    for k in range(k_max):
        base = BASE_OBS_DIM + MODE_OBS_DIM * k
        if obs[base] < 0.5:
            continue
        means.append(obs[base + 1:base + 3].astype(np.float64))
        cxx, cxy, cyy = (float(obs[base + 3]), float(obs[base + 4]), float(obs[base + 5]))
        covs.append(np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64))
    return np.asarray(means).reshape(-1, 2), np.asarray(covs).reshape(-1, 2, 2)


class MultimodalSearchEnv(gym.Env):
    """Search for a static target hidden in one of K random Gaussian modes."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[MultimodalSearchConfig] = None):
        super().__init__()
        self.config = config or MultimodalSearchConfig()
        c = self.config
        self.rng = np.random.default_rng(c.seed)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (c.obs_dim,), dtype=np.float32)
        self.mode_means = np.zeros((0, 2))
        self.mode_covs = np.zeros((0, 2, 2))
        self.target_pos = np.zeros(2)
        self.agent_pos = np.zeros(2)
        self.prev_pos = np.zeros(2)
        self.step_count = 0

    # --- episode construction -------------------------------------------------
    def _sample_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """K modes whose centres have centroid exactly zero and pairwise separation.

        Rejection sampling: draw, translate to zero-centroid, keep if every centre is
        still inside the arena and no two are closer than ``min_separation``. Measured at
        ~233 attempts for the hardest case (K=10, arena 14, separation 5), i.e. under
        2 ms, which is negligible against a 250-step episode.
        """
        c = self.config
        k = int(self.rng.integers(c.k_min, c.k_max + 1))
        for _ in range(20_000):
            mu = self.rng.uniform(-c.arena_half, c.arena_half, size=(k, 2))
            mu = mu - mu.mean(axis=0)          # pin the belief mean at the origin
            if np.abs(mu).max() > c.arena_half:
                continue
            if k > 1:
                d = np.linalg.norm(mu[:, None] - mu[None, :], axis=-1) + np.eye(k) * 1e9
                if d.min() < c.min_separation:
                    continue
            covs = np.empty((k, 2, 2))
            for j in range(k):
                scales = self.rng.uniform(c.mode_scale_lo, c.mode_scale_hi, size=2)
                rot = _rotation(self.rng.uniform(0.0, np.pi))
                covs[j] = rot @ np.diag(scales ** 2) @ rot.T
            return mu, covs
        raise RuntimeError(
            f"could not place {k} modes with separation {c.min_separation} in an arena of "
            f"half-width {c.arena_half}; loosen min_separation or enlarge the arena.")

    def _observation(self) -> np.ndarray:
        delta = self.target_pos - self.agent_pos
        swept = float(segment_distance(self.target_pos[None, :], self.prev_pos,
                                       self.agent_pos)[0])
        visible = float(swept <= self.config.visibility_radius)
        return pack_observation(
            self.agent_pos, self.prev_pos, delta if visible else np.zeros(2), visible,
            self.mode_means, self.mode_covs, self.config.k_max)

    # --- gym API --------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.mode_means, self.mode_covs = self._sample_modes()
        # The target is equally likely to be in any mode, then drawn from that mode's own
        # Gaussian -- so the belief's mode weights are uniform and the optimal policy is
        # a pure cost-ordered tour, depending only on where the modes are and how big.
        self.target_mode = int(self.rng.integers(len(self.mode_means)))
        self.target_pos = self.rng.multivariate_normal(
            self.mode_means[self.target_mode], self.mode_covs[self.target_mode])
        self.target_pos = np.clip(self.target_pos, -c.arena_half, c.arena_half)
        self.agent_pos = self.rng.uniform(-c.arena_half, c.arena_half, size=2)
        self.prev_pos = self.agent_pos.copy()
        self.step_count = 0
        return self._observation(), self._info(found=False)

    def step(self, action: np.ndarray):
        c = self.config
        act = np.clip(np.asarray(action, dtype=np.float64).ravel()[:2], -1.0, 1.0)
        norm = np.linalg.norm(act)
        if norm > 1.0:                     # a direction, not a diagonal speed bonus
            act = act / norm
        self.prev_pos = self.agent_pos.copy()
        self.agent_pos = np.clip(self.agent_pos + c.speed * act,
                                 -c.arena_half, c.arena_half)
        self.step_count += 1

        # Found if the target came within the visibility radius of the PATH travelled,
        # not merely of the endpoint -- otherwise a step longer than the radius could
        # pass straight over it.
        found = bool(segment_distance(self.target_pos[None, :], self.prev_pos,
                                      self.agent_pos)[0] <= c.visibility_radius)
        reward = c.step_penalty + (c.find_bonus if found else 0.0)
        truncated = (not found) and self.step_count >= c.max_steps
        return self._observation(), reward, found, truncated, self._info(found)

    def _info(self, found: bool) -> dict:
        return {
            "found": found,
            "target_pos": self.target_pos.copy(),
            "agent_pos": self.agent_pos.copy(),
            "mode_means": self.mode_means.copy(),
            "mode_covs": self.mode_covs.copy(),
            "target_mode": getattr(self, "target_mode", -1),
            "num_modes": len(self.mode_means),
        }
