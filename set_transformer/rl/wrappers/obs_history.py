"""Frame stacking on the ``obs`` key of a PF Dict observation (the ``framestack`` arm).

Added 2026-09-22 with the ``framestack`` encoder
(``change_mds/framestack_arm_2026-09-22.md``; design in
``change_mds/framestack_arm_plan_2026-09-22.md``). The arm's policy input is the last ``k``
base observations concatenated and nothing else, so it answers "does a belief encoder beat
simply remembering the last k frames?".

The history CANNOT be built inside the features extractor: SB3 calls ``forward()`` on shuffled
minibatches drawn from the rollout buffer, so there is no per-env temporal order to accumulate
over, and the buffer stores whatever the observation space declares. It has to be in the
observation before SB3 sees it, which is what :class:`ObsHistoryDictWrapper` does.

Three pieces, used by the four doors that build an env (plan section 6): the wrapper itself,
:func:`with_obs_history` for the three doors that own a thunk (training workers, the
in-training eval env, the standalone eval env) and :func:`checkpoint_obs_history`, which reads
``k`` back off a saved zip so eval and collect never need a flag.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class ObsHistoryDictWrapper(gym.ObservationWrapper):
    """Stack the last ``n_stack`` base observations into the ``"obs"`` key of a PF Dict
    observation; ``particles`` and ``weights`` pass through untouched.

    The frames are concatenated NEWEST LAST, so ``obs[-D:]`` is always the current frame and
    an extractor that ignores the history reads the same slice it always did.

    Placement: this wrapper is the OUTERMOST one, above
    :class:`~set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`.
    Below the PF wrapper it would corrupt belief propagation (the filter's interaction mapper
    indexes the base observation positionally -- ``base_env_obs[:2]``, ``base_env_obs[-2:]`` on
    Ant-Tag -- and ``obs_mask_indices`` would mask only the newest frame, leaking ``k-1``
    unmasked copies of the true target position). Plan section 4 has the argument. It also
    keeps the PF wrapper's two dimension-probing resets below this wrapper, where they never
    disturb the deque.

    On Ant-Tag this sits above ``_CurriculumRouter``, which is documented as that domain's
    outermost wrapper. Both curriculum paths still reach the setter: ``curriculum.apply_to_env``
    walks ``.env`` itself, and ``VecEnv.env_method`` goes through
    ``gym.Wrapper.get_wrapper_attr``, which walks the chain too (plan section 11F). A caller
    that instead reaches for the setter as a plain attribute is served by ``__getattr__``
    below.

    ``n_stack == 1`` is the exact identity -- the space and every emitted array are unchanged --
    but the harness does not build the object at all in that case (:func:`with_obs_history`),
    which is what makes every other arm bit-identical.
    """

    def __init__(self, env: gym.Env, n_stack: int):
        super().__init__(env)
        self.n_stack = int(n_stack)
        base = env.observation_space["obs"]
        self._frames: deque = deque(maxlen=self.n_stack)
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            "obs": gym.spaces.Box(
                low=np.tile(base.low, self.n_stack),
                high=np.tile(base.high, self.n_stack),
                dtype=base.dtype,
            ),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # A new episode starts with no history: pad with the reset frame, so the observation
        # is always n_stack * D wide and never carries the previous episode's frames.
        self._frames.clear()
        for _ in range(self.n_stack - 1):
            self._frames.append(np.asarray(obs["obs"]))
        return self.observation(obs), info

    def observation(self, obs: dict) -> dict:
        self._frames.append(np.asarray(obs["obs"]))
        stacked = np.concatenate(list(self._frames)).astype(
            self.observation_space["obs"].dtype, copy=False)
        return {**obs, "obs": stacked}

    def __getattr__(self, name: str):
        """Forward an unknown attribute down the wrapper chain.

        gymnasium 1.2.3's ``Wrapper`` has NO ``__getattr__``, so a caller that holds the
        outermost env and calls a domain setter on it directly breaks as soon as one more
        wrapper sits on top. Two collection callbacks do exactly that:
        ``hunt._collect_begin_episode`` calls ``env.set_n_active(...)`` and
        ``ant_tag._collect_begin_episode`` calls ``env.set_curriculum_radius(...)``. This
        forwards the same way ``Wrapper.get_wrapper_attr`` and ``VecEnv.env_method`` already
        reach those setters, so door 4 works on every domain without a domain edit.

        Private names and ``env`` itself are NOT forwarded: ``__getattr__`` runs during
        unpickling and copying, before ``self.env`` exists, and forwarding ``__setstate__`` /
        ``__deepcopy__`` -- or looking ``env`` up through ``self.env`` -- would recurse
        (measured 2026-09-22: without the ``env`` guard, ``hasattr(w, "x")`` on a wrapper whose
        ``env`` is not yet set raised ``RecursionError``).
        """
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


def with_obs_history(thunk, n_stack: int):
    """Wrap an env thunk so the env it builds stacks the last ``n_stack`` base observations.

    ``n_stack <= 1`` returns the thunk UNCHANGED -- the same object, not an equivalent one --
    so no wrapper exists and nothing about an existing arm's env, observation space or pickled
    thunk moves. The returned callable is a module-level class instance (not a closure) so
    cloudpickle round-trips it for ``SubprocVecEnv``.
    """
    if int(n_stack) <= 1:
        return thunk
    return _ObsHistoryThunk(thunk, int(n_stack))


class _ObsHistoryThunk:
    """``with_obs_history``'s picklable callable: build the env, then stack above it."""

    def __init__(self, thunk, n_stack: int):
        self.thunk = thunk
        self.n_stack = int(n_stack)

    def __call__(self):
        return ObsHistoryDictWrapper(self.thunk(), self.n_stack)


def checkpoint_obs_history(model_path: str | None) -> int:
    """How many frames the saved policy's env stacked, read off the zip; 1 when it cannot be
    read (every recorded checkpoint, which declares no ``n_stack``).

    BEST EFFORT, the same shape as
    :func:`~set_transformer.rl.eval_true_reward.checkpoint_num_particles`: the read unpickles
    ``policy_kwargs``, so it needs the features-extractor class importable, and a zip whose
    class pickles under a flat module name (the recorded Ant-Tag zips, until
    ``_make_extractor_classes_importable`` has run) raises. Falling back to 1 is FAIL-SAFE: for
    an existing arm 1 is the correct answer, and for a framestack checkpoint the eval env is
    then built one frame wide and SB3's ``check_for_correct_spaces`` refuses the load loudly
    rather than evaluating a policy against the wrong observation (plan section 11D).
    """
    if not model_path:
        return 1
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _params, _other = load_from_zip_file(
            model_path, load_data=True, device="cpu", print_system_info=False)
        return int(data["policy_kwargs"]["features_extractor_kwargs"]["n_stack"])
    except Exception:  # noqa: BLE001 - a best-effort default, never fatal
        return 1
