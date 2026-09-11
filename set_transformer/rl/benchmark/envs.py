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

#: Ant-Tag tag bonus. The stock reward is -1/step with 0 on the tag, so tagging is worth
#: only the steps it saves and the terminal event is not distinguished at all. This makes
#: reaching the target explicitly valuable.
ANT_TAG_TAG_BONUS = 100.0
#: Step penalty paired with the bonus. Smaller than the stock -1 so a long successful
#: episode still beats a short unsuccessful one: 400 steps cost 40, well under the bonus.
ANT_TAG_STEP_PENALTY = -0.1


class AntTagRewardWrapper(gym.Wrapper):
    """Pay an explicit bonus for tagging, with a smaller per-step cost.

    Stock Ant-Tag pays -1/step and 0 on the tag, so a tag is worth only the steps it
    saves and carries no distinct signal. Measured over a 2-seed x 8-method x 2M-step
    pilot, PPO never tagged once: every run sat at exactly -400.0. Potential-based shaping
    cannot repair that -- shaping provably preserves the optimal policy, so it can guide a
    learner that is already exploring the right region but cannot make an unexplored
    terminal event worth finding.

    That the task is reachable at all is established separately: the pretrained locomotion
    policy tags 4/20 episodes and moves the ant 4.9 units, where random actions tag 0/20
    and move 2.2. The gap is motor learning and exploration, not feasibility.

    Only the scalar reward changes; observations, dynamics and termination are untouched,
    and the wrapper sits below every method so all methods see one task.
    """

    def __init__(self, env: gym.Env, tag_bonus: float = ANT_TAG_TAG_BONUS,
                 step_penalty: float = ANT_TAG_STEP_PENALTY):
        super().__init__(env)
        self.tag_bonus = tag_bonus
        self.step_penalty = step_penalty

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.step_penalty + (self.tag_bonus if terminated else 0.0)
        info["tagged"] = bool(terminated)
        return obs, reward, terminated, truncated, info


def ant_tag_success(episode_return: float, episode_length: int) -> bool:
    """Success = the target was tagged before the horizon.

    Recovered from the return rather than the length so it stays correct under either
    reward: with the bonus, only a tagging episode can finish above zero.
    """
    return episode_return > 0.0


def make_ant_tag_base_env(seed: int = 0, rendering: bool = False,
                          tag_bonus_reward: bool = False) -> gym.Env:
    """Registered Ant-Tag POMDP (native target visibility radius 3.0, TimeLimit 400).

    No curriculum wrapper: the base env already zeros ``obs[-2:]`` when the target is out
    of visual range, giving a genuine POMDP.

    ``tag_bonus_reward`` applies :class:`AntTagRewardWrapper`; see it for why the stock
    reward could not be learned from.
    """
    import pdomains  # noqa: F401  (registers pdomains-* envs on import)

    env = gym.make("pdomains-ant-tag-v0", rendering=rendering)
    if tag_bonus_reward:
        env = AntTagRewardWrapper(env)
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


class OddEvenParityRewardWrapper(gym.Wrapper):
    """Gate Odd-Even's squared-error reward on getting the parity right.

    The stock reward is ``-(predicted - true_state)^2``. Under it the optimal action is
    ``argmin_a E[(a - s)^2]`` = round(E[s]) -- a function of the belief **mean alone**.
    Measured over 24,000 real belief states, the optimal action differs from round(mean)
    in 0.0% of them, so mean+covariance is a sufficient statistic and no belief encoder
    can beat the Gaussian baseline however well it represents the posterior. That is the
    same defect Car-Flag had, and potential-based shaping cannot repair it: shaping
    preserves the optimal policy, so the *task reward* has to change.

    Here a prediction of the wrong parity earns the worst reward the task can give,
    ``-(n - 1)^2``, instead of being scored on distance. Parity is knowable exactly (every
    observation shares ``true_state``'s parity), so this asks the agent for something the
    belief genuinely contains -- and it is precisely what a mean cannot express: the mean
    of a comb sits *between* its teeth, on a state of the opposite parity. Under the gated
    reward the optimal action differs from round(mean) 39.2% of the time, and the expected
    reward gap is ~31 per step (-1.70 optimal vs -32.65 for round(mean)).

    Only the scalar reward is recomputed; observations, dynamics and termination are
    untouched, and the wrapper sits below every method so all methods see one task.
    Reading ``true_state`` here is not a leak -- it determines the reward, never the
    observation, exactly as the stock reward already does.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n = int(env.unwrapped.n_dist_size)
        self.n_dist_size = n
        self.worst_reward = -float((n - 1) ** 2)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        predicted_state = int(action) + 1          # actions are 0-indexed predictions
        true_state = int(self.env.unwrapped.true_state)
        if predicted_state % 2 == true_state % 2:
            reward = -float((predicted_state - true_state) ** 2)
        else:
            reward = self.worst_reward
        info["predicted_state"] = predicted_state
        info["true_state"] = true_state
        info["parity_correct"] = predicted_state % 2 == true_state % 2
        return obs, reward, terminated, truncated, info


#: Worst per-step reward at the registry's n_dist_size=10, i.e. -(n-1)^2.
ODD_EVEN_WORST_REWARD = -81.0


#: Success threshold on mean per-step reward. Parity dominates the reward (-81 for a
#: wrong-parity guess against 0 to -16 for a right-parity one), so the per-step mean is
#: essentially -81 x (fraction of steps on the wrong parity). -2.0 therefore means the
#: agent spends under ~2.5% of steps on an impossible state. Calibrated against the probe
#: policies so the metric separates the two strategies rather than saturating: tracking
#: the posterior mode scores -0.19/step (passes by 10x) while rounding the posterior mean
#: -- the Gaussian's own rule -- scores -4.24/step (fails by 2x). A looser gate passes
#: both, because the mean-tracker does recover the parity late in an episode; the whole
#: difference lives in the early, genuinely multimodal steps.
ODD_EVEN_SUCCESS_PER_STEP = -2.0


def odd_even_success(episode_return: float, episode_length: int) -> bool:
    """Success = the agent tracked the belief closely enough to stay on the true parity."""
    if episode_length <= 0:
        return False
    return (episode_return / episode_length) > ODD_EVEN_SUCCESS_PER_STEP


def make_odd_even_base_env(
    seed: int = 0,
    n_dist_size: int = 10,
    std_dev: float = 2.0,
    max_steps: int = 100,
    n_obs_samples: int = 1,
    parity_gated_reward: bool = True,
) -> gym.Env:
    """OddEvenPOMDP with raw observation samples and a parity-gated dense reward.

    ``parity_gated_reward`` applies :class:`OddEvenParityRewardWrapper` -- required for
    this env to discriminate belief encoders at all (see that class). Set it False only to
    reproduce the stock ``-squared_error`` task.

    ``n_obs_samples`` is how many iid observation draws the agent gets per step, and it
    decides whether this environment is a belief benchmark at all. The env's default of
    100 makes the exact posterior collapse to a **point mass after a single step** -- with
    that much evidence the latent is effectively observed, every method sees the same
    delta, and no belief encoder can differentiate. At 1 sample/step the posterior stays
    genuinely multimodal over the parity comb for ~10 steps (mean support 3.9 states at
    t=1, 1.8 at t=10), which is the regime where the belief representation matters.
    """
    from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig

    cfg = OddEvenPOMDPConfig(n_dist_size=n_dist_size, std_dev=std_dev, seed=seed,
                             n_particles=n_obs_samples)
    env = OddEvenPOMDPGymAdapter(OddEvenPOMDP(config=cfg), max_steps=max_steps)
    if parity_gated_reward:
        env = OddEvenParityRewardWrapper(env)
    env.reset(seed=seed)
    return env


# --- Multimodal Search ---------------------------------------------------------

def make_msearch_base_env(seed: int = 0, **config_kwargs) -> gym.Env:
    """Multimodal Search: a static target hidden in one of K random Gaussian modes.

    Built so a Gaussian belief summary provably cannot compete — the belief mean is pinned
    at the origin every episode, so it carries zero information about the target, while
    the mode geometry stays random so there is no fixed sweep to memorise instead. See
    :mod:`set_transformer.rl.envs.multimodal_search`.
    """
    from set_transformer.rl.envs.multimodal_search import (
        MultimodalSearchConfig,
        MultimodalSearchEnv,
    )

    cfg = MultimodalSearchConfig(seed=seed, **config_kwargs)
    env = MultimodalSearchEnv(cfg)
    env.reset(seed=seed)
    return env


def msearch_success(episode_return: float, episode_length: int) -> bool:
    """Success = the target was found.

    The find bonus equals the horizon and each step costs 1, so a found episode returns
    ``find_bonus - steps > 0`` and an unfound one returns ``-max_steps < 0``. There is no
    ambiguous middle.
    """
    return episode_return > 0.0
