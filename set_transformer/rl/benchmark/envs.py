"""Per-environment base-env factories and potential functions for the benchmark.

Each ``make_*_base_env`` returns the raw POMDP (with its Gymnasium ``TimeLimit`` /
truncation handling), *before* the shared PF and shaping wrappers that the trainer adds.
Potential functions implement the environment's shaping ``Phi`` (true-state where the
latent is a position the agent must reach; belief-based fallbacks live in
:mod:`set_transformer.rl.wrappers.shaping`).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


# --- Ant-Tag -------------------------------------------------------------------

def make_ant_tag_base_env(seed: int = 0, rendering: bool = False) -> gym.Env:
    """Registered Ant-Tag POMDP (native target visibility radius 3.0, TimeLimit 400).

    No curriculum wrapper: the base env already zeros ``obs[-2:]`` when the target is out
    of visual range, giving a genuine POMDP.
    """
    import pdomains  # noqa: F401  (registers pdomains-* envs on import)

    env = gym.make("pdomains-ant-tag-v0", rendering=rendering)
    env.reset(seed=seed)
    return env


def ant_tag_true_state_potential(env: gym.Env) -> float:
    """Phi(s) = -||ant - target||  (true-state; the target is a position to reach)."""
    u = env.unwrapped
    ant = np.asarray(u.data.qpos[:2], dtype=np.float64)
    target = np.asarray(u.get_target_pos(), dtype=np.float64)
    return -float(np.linalg.norm(ant - target))


def ant_tag_agent_pos(env: gym.Env) -> np.ndarray:
    """Agent xy for belief-based potentials (unused by default; fallback helper)."""
    return np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float64)


# --- Car-Flag ------------------------------------------------------------------

# Car-Flag's stock reward (-1/step, heaven 0 or +1, hell -5) makes information worthless:
# the priest detour costs ~27 steps while knowing the answer is worth only (heaven-hell)/2
# = 3, so the *optimal* policy ignores the latent and gambles on the nearest flag. Measured
# directly: a hand-coded gambler scores -38.1 (45% heaven) vs -61.5 for a policy that uses
# the priest, and PPO duly converged to the gambler (-37.7, success 0.55 = chance). Since
# potential-based shaping preserves the optimal policy by construction, no shaping term can
# repair this — the task reward itself has to change.
#
# These constants restore the standard Car-Flag trade-off, giving the ordering
# heaven (+1) > hell (-1) > timeout (-1.6): committing beats stalling, correct beats wrong,
# and buying information is worth +0.83 over gambling.
CAR_FLAG_STEP_PENALTY = -0.01
CAR_FLAG_HEAVEN_REWARD = 1.0
CAR_FLAG_HELL_REWARD = -1.0


class CarFlagRewardWrapper(gym.Wrapper):
    """Replace Car-Flag's reward so that gathering information is worth its cost.

    Observations, dynamics, and termination are untouched — only the scalar reward is
    recomputed. It sits below every method in the wrapper stack, so all methods see the
    identical task and cross-method fairness is unaffected. Reading the true heaven side
    here is not an information leak: it determines the reward, which the agent only
    receives at termination (exactly as in the stock env), never the observation.
    """

    def __init__(self, env: gym.Env,
                 step_penalty: float = CAR_FLAG_STEP_PENALTY,
                 heaven_reward: float = CAR_FLAG_HEAVEN_REWARD,
                 hell_reward: float = CAR_FLAG_HELL_REWARD):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.heaven_reward = heaven_reward
        self.hell_reward = hell_reward

    def step(self, action):
        obs, _stock_reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            # The env terminates exactly when |position| >= 1.0, i.e. at one of the flags.
            position = float(np.asarray(obs).reshape(-1)[0])
            heaven = float(self.env.unwrapped.heaven_position)
            reached_heaven = np.sign(position) == np.sign(heaven)
            reward = self.heaven_reward if reached_heaven else self.hell_reward
            info["reached_heaven"] = bool(reached_heaven)
        else:
            reward = self.step_penalty
        return obs, reward, terminated, truncated, info


def make_car_flag_base_env(seed: int = 0, rendering: bool = False) -> gym.Env:
    """Car-Flag POMDP with the corrected reward (obs ``[position, velocity, direction]``).

    ``direction`` is 0 except in the priest region, where it reveals the (static, hidden)
    heaven side. ``gym.make`` applies the ``TimeLimit`` (160) from the registration;
    :class:`CarFlagRewardWrapper` then makes information worth buying (see above).
    """
    import pdomains  # noqa: F401  (registers pdomains-* envs on import)

    env = gym.make("pdomains-car-flag-v0", rendering=rendering)
    env = CarFlagRewardWrapper(env)
    env.reset(seed=seed)
    return env


def car_flag_success(episode_return: float, episode_length: float) -> bool:
    """True iff the car reached heaven (not hell, not a timeout).

    Reconstructs the terminal reward from the episode statistics: every non-terminal step
    pays exactly ``CAR_FLAG_STEP_PENALTY``, so
    ``terminal = return - step_penalty * (length - 1)``. That leaves +1 for heaven, -1 for
    hell, and ``step_penalty`` (-0.01) for a timeout, so a midpoint threshold separates
    success from both failure modes.
    """
    terminal_reward = episode_return - CAR_FLAG_STEP_PENALTY * (episode_length - 1)
    return terminal_reward >= 0.5 * CAR_FLAG_HEAVEN_REWARD


def car_flag_belief_potential(env: gym.Env) -> float:
    """Belief-based (leak-free) shaping potential for the fallback path.

    Pulls the car toward the flag the *belief* thinks is heaven, weighted by confidence:
    ``Phi = -E_{h~belief}|position - h|`` with ``h`` the hypothesised heaven coordinate
    (+1/-1) carried by each particle. Uses only the shared particle filter — nothing the
    agent doesn't already receive — so it introduces no true-state leak. Kept off by
    default (Car-Flag runs sparse first); wire it into the EnvSpec only if PPO can't learn.
    """
    from set_transformer.rl.wrappers.shaping import find_particle_filter

    pf = find_particle_filter(env)
    position = float(np.asarray(env.unwrapped.state).reshape(-1)[0])
    heaven_hypotheses = pf.particles.reshape(-1)  # +1 / -1 per particle
    return -float(np.average(np.abs(position - heaven_hypotheses), weights=pf.weights))


# --- Odd-Even ------------------------------------------------------------------

class OddEvenPOMDPGymAdapter(gym.Wrapper):
    """Adds fixed-horizon truncation to ``OddEvenPOMDP``.

    Lifted from ``experiments/odd_even/train_rl_pretrained.py`` so the benchmark trainer
    does not depend on that script.
    """

    def __init__(self, pomdp: gym.Env, max_steps: int = 100):
        super().__init__(pomdp)
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


def make_odd_even_base_env(
    seed: int = 0,
    n_dist_size: int = 10,
    std_dev: float = 2.0,
    max_steps: int = 100,
) -> gym.Env:
    """OddEvenPOMDP (raw-particle observations, dense ``-squared_error`` reward)."""
    from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig

    cfg = OddEvenPOMDPConfig(n_dist_size=n_dist_size, std_dev=std_dev, seed=seed)
    env = OddEvenPOMDPGymAdapter(OddEvenPOMDP(config=cfg), max_steps=max_steps)
    env.reset(seed=seed)
    return env
