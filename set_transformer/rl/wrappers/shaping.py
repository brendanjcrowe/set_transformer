"""Potential-based reward shaping (PBRS) for the belief-encoder benchmark.

Shaping is **env-level and method-agnostic**: the same wrapper with the same potential is
applied to every method for a given environment, so the encoder under test is the only
thing that differs. Training uses the shaped reward; **evaluation always uses the true
reward** (build eval envs without this wrapper — the true per-step reward is also exposed
in ``info["true_reward"]``).

PBRS (Ng, Harada & Russell 1999): ``F(s, s') = gamma * Phi(s') - Phi(s)`` added to the
env reward provably preserves the optimal policy. At *terminated* states we use the
standard episodic convention ``Phi(s_T) = 0`` so the discounted shaping sum telescopes to
``-Phi(s_0)`` and no termination bias is introduced; truncation (time limit) keeps
``Phi(s')`` since the state is not absorbing.

Two potential families are provided:

- **True-state potentials** (e.g. Ant-Tag): ``potential_fn`` reads privileged state from
  ``env.unwrapped``. Safe when the latent is a position the agent must physically reach
  anyway; NOT used on hidden-side envs (Car-Flag / Ant-Heaven-Hell / Two-Boxes), where it
  would leak the latent through the reward.
- **PF-belief potentials** (:func:`pf_belief_expected_distance_potential`): ``Phi`` is
  computed from the *shared* particle filter, leaking nothing beyond what the belief
  already encodes. Fallback for hidden-side envs if they can't be learned sparse.

Wrapper order: ``base env -> PFDictObservationWrapper -> PotentialBasedShapingWrapper``
so belief potentials can reach the particle filter (found by walking the wrapper chain).
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np


def find_particle_filter(env: gym.Env):
    """Walk the wrapper chain looking for a wrapper holding a ``particle_filter``.

    Returns the innermost wrapper *attribute* (the live ``BaseParticleFilter``), or
    raises if none is found. Note ``PFDictObservationWrapper`` re-creates its filter on
    every ``reset()``, so callers must re-fetch per step rather than caching.
    """
    current = env
    while current is not None:
        pf = getattr(current, "particle_filter", None)
        if pf is not None:
            return pf
        current = getattr(current, "env", None)
    raise AttributeError(
        "No wrapper with a live 'particle_filter' found below the shaping wrapper. "
        "Belief potentials require PFDictObservationWrapper inside "
        "PotentialBasedShapingWrapper."
    )


def pf_belief_expected_distance_potential(
    get_agent_pos: Callable[[gym.Env], np.ndarray],
    scale: float = 1.0,
) -> Callable[[gym.Env], float]:
    """Potential ``Phi(b) = -scale * E_{s~belief}||agent - s||`` from the shared PF.

    Args:
        get_agent_pos: maps the *wrapped env below the shaping wrapper* to the agent's
            position (e.g. ``lambda env: env.unwrapped.data.qpos[:2]`` for Ant envs).
        scale: multiplier on the expected distance.
    """

    def _potential(env: gym.Env) -> float:
        pf = find_particle_filter(env)
        agent = np.asarray(get_agent_pos(env), dtype=np.float64)
        particles = np.asarray(pf.particles, dtype=np.float64)  # [N, d]
        dists = np.linalg.norm(particles - agent[None, :], axis=1)  # [N]
        weights = pf.weights
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w_sum = w.sum()
            expected = float((dists * w).sum() / w_sum) if w_sum > 0 else float(dists.mean())
        else:
            expected = float(dists.mean())
        return -scale * expected

    return _potential


def pf_belief_information_potential(scale: float = 30.0) -> Callable[[gym.Env], float]:
    """Potential ``Phi(b) = scale * (nats of information gathered so far)``.

    Rewards *reducing belief uncertainty* rather than approaching a point. That matters on
    a multimodal search task, where the belief mean sits between the modes: a
    distance-to-belief potential would pull the agent into empty space, while this one
    pays only for sweeping regions that actually hold probability mass. It is therefore
    still discriminating — collecting it requires knowing *where* the mass is, which is
    exactly what a mean-and-covariance summary cannot say.

    Being potential-based, it leaves the optimal policy unchanged (the wrapper zeroes
    ``Phi`` at true termination, so the discounted shaping telescopes to zero over a
    successful episode); it only redistributes reward in time so the learner is not
    waiting on one sparse event.

    ``scale`` sets the size of that redistribution. Keep ``scale * max_expected_nats``
    comfortably below the task's terminal reward, or the negative shaping paid back at
    termination would cancel the incentive to finish. With K <= 10 modes the realistic
    ceiling is ``log(10) ~ 2.3`` nats, so the default 30 tops out near 70 against a
    250-point find bonus.

    Requires a particle filter exposing ``information_gain``.
    """

    def _potential(env: gym.Env) -> float:
        pf = find_particle_filter(env)
        gain = getattr(pf, "information_gain", None)
        if gain is None:
            raise AttributeError(
                f"{type(pf).__name__} exposes no `information_gain`; this potential needs "
                "a filter that tracks how much prior mass it has ruled out.")
        return scale * float(gain)

    return _potential


class PotentialBasedShapingWrapper(gym.Wrapper):
    """Adds ``gamma * Phi(s') - Phi(s)`` to the reward each step.

    Args:
        env: env to wrap (with ``PFDictObservationWrapper`` already applied if the
            potential is belief-based).
        potential_fn: ``Phi`` — called with ``self.env`` after each transition; reads
            whatever it needs (``env.unwrapped`` state or the particle filter).
        gamma: MUST match the RL algorithm's discount factor, otherwise the
            policy-invariance guarantee is lost.

    Every step's ``info`` gains ``true_reward``, ``potential`` and ``shaping_reward``
    (so shaped == ``true_reward + shaping_reward``) for diagnostics and logging.
    """

    def __init__(
        self,
        env: gym.Env,
        potential_fn: Callable[[gym.Env], float],
        gamma: float = 0.99,
        zero_at_termination: bool = True,
    ):
        """
        Args:
            zero_at_termination: keep ``Phi(terminal) = 0``, the convention that makes
                the discounted shaping telescope to zero and leaves the optimal policy
                exactly unchanged. Set False to use the potential as an outright **proxy
                reward** instead: the agent then keeps what it accrued, so with
                ``gamma=1`` an episode's shaping totals ``Phi_T - Phi_0``. That trades the
                policy-invariance guarantee for a signal that survives a sparse task --
                worth it only when the unshaped reward is too rare to learn from, and only
                when evaluation stays on the true reward (which the trainer enforces by
                building eval envs unshaped).
        """
        super().__init__(env)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.zero_at_termination = zero_at_termination
        self._last_potential: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_potential = float(self.potential_fn(self.env))
        info["potential"] = self._last_potential
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Episodic PBRS convention: Phi(absorbing state) = 0 on true termination.
        new_potential = float(self.potential_fn(self.env))
        if terminated and self.zero_at_termination:
            new_potential = 0.0
        shaping = self.gamma * new_potential - self._last_potential
        self._last_potential = new_potential

        info["true_reward"] = reward
        info["potential"] = new_potential
        info["shaping_reward"] = shaping
        return obs, reward + shaping, terminated, truncated, info
