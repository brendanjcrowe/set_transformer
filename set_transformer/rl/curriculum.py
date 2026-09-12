"""Training-progress schedules for the RL harness (changes 1d and 3, 2026-09-12).

A schedule is a list of waypoints ``(fraction of training progress, value, ...)``, linearly
interpolated between waypoints and clamped outside them. A domain declares the quantities it
anneals as :class:`Schedule` descriptions (Ant-Tag: visibility radius, four reward-shaping
coefficients, evasion scale; Odd-Even: none); :class:`ScheduleCallback` interpolates every
declared schedule at the current progress on every step and pushes the values into the env
wrappers by setter name -- directly with one worker, through :class:`ScheduleRouter` and
``SubprocVecEnv.env_method`` with several. Nothing here knows about a particular domain: a
new problem writes a wrapper with a ``set_<something>`` method, declares a ``Schedule``
whose ``target`` is that name, and gets the callback, the router and the status line for free.

History: the two parsers came verbatim from ``experiments/ant_tag/4_train_rl_cgf.py``
(change 1d); :func:`interpolate` and the callback's behaviour came verbatim from the Ant-Tag
``CurriculumCallback`` (change 3), which is now a thin adapter over :class:`ScheduleCallback`
in ``rl/domains/ant_tag.py`` and goes away with the numbered scripts (change 5).

A schedule STRING is ``frac:value[:value...][,frac:value...]``; see the two parsers below.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass, field

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


def parse_curriculum(curriculum: str) -> list[tuple[float, float]]:
    """``"0:100,0.4:1.5,1:1.5"`` -> ``[(0.0, 100.0), (0.4, 1.5), (1.0, 1.5)]``.

    One value per waypoint: the visibility-radius curriculum and the evasion-scale
    schedule both use this shape.
    """
    schedule = []
    for pair in curriculum.split(","):
        frac, radius = pair.strip().split(":")
        schedule.append((float(frac), float(radius)))
    return schedule


def parse_reward_schedule(reward_schedule: str) -> list[tuple[float, ...]]:
    """``"frac:distance:entropy[:tag_bonus[:spread_gain]],..."`` -> 5-tuples.

    Missing trailing fields are 0, so every pre-2026-09-08 three- or four-field schedule
    keeps its meaning; fewer than three or more than five fields raise.
    """
    schedule = []
    for entry in reward_schedule.split(","):
        parts = [float(part) for part in entry.strip().split(":")]
        # frac:distance:entropy[:tag_bonus[:spread_gain]] -- missing trailing
        # fields are 0, so every pre-2026-09-08 schedule keeps its meaning.
        if len(parts) in (3, 4):
            parts.extend([0.0] * (5 - len(parts)))
        if len(parts) != 5:
            raise ValueError("Each reward schedule entry must have 3 to 5 values: "
                             "frac:distance:entropy[:tag_bonus[:spread_gain]]")
        schedule.append(tuple(parts))
    return schedule


# ---------------------------------------------------------------------------
# Interpolation, schedules, and pushing values into the env wrappers (change 3)
# ---------------------------------------------------------------------------


def interpolate(schedule, progress: float):
    """Linearly interpolate a schedule. Returns all values after the fraction.

    ``schedule`` is a list of waypoints sorted by fraction; clamped to the first waypoint's
    values before it and the last waypoint's after it. Moved verbatim from the Ant-Tag
    ``CurriculumCallback._interpolate_schedule``.
    """
    if progress <= schedule[0][0]:
        return schedule[0][1:]
    if progress >= schedule[-1][0]:
        return schedule[-1][1:]
    for i in range(len(schedule) - 1):
        frac_lo = schedule[i][0]
        frac_hi = schedule[i + 1][0]
        if frac_lo <= progress <= frac_hi:
            t = (progress - frac_lo) / (frac_hi - frac_lo) if frac_hi > frac_lo else 1.0
            vals_lo = schedule[i][1:]
            vals_hi = schedule[i + 1][1:]
            return tuple(lo + t * (hi - lo) for lo, hi in zip(vals_lo, vals_hi))
    return schedule[-1][1:]


@dataclass(frozen=True)
class Schedule:
    """One annealed quantity: what it is called, where its values go, and its waypoints.

    ``target`` is the name of the setter method on the env wrapper that applies the values
    (``set_curriculum_radius`` on Ant-Tag's visibility wrapper, ...); the callback calls
    ``<target>(*values)`` on the first wrapper, from the outside in, that has it. ``n_values``
    is how many values a waypoint carries after the fraction; a waypoint with fewer is padded
    with zeros (the historical rule for the reward schedule, whose ``tag_bonus`` and
    ``spread_gain`` fields were added later), and extra values are ignored. ``labels`` are
    ``(label, format spec)`` per value for the callback's status line.
    """

    name: str
    target: str
    waypoints: tuple[tuple[float, ...], ...]
    n_values: int = 1
    labels: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self):
        waypoints = tuple(tuple(w) for w in self.waypoints)
        if not waypoints:
            raise ValueError(f"schedule {self.name!r} has no waypoints")
        object.__setattr__(self, "waypoints", tuple(sorted(waypoints, key=lambda w: w[0])))
        if not self.labels:
            labels = ((self.name, ".3g"),) if self.n_values == 1 else tuple(
                (f"{self.name}[{i}]", ".3g") for i in range(self.n_values))
            object.__setattr__(self, "labels", labels)
        if len(self.labels) != self.n_values:
            raise ValueError(f"schedule {self.name!r}: {len(self.labels)} labels for "
                             f"{self.n_values} values")

    def values_at(self, progress: float) -> tuple[float, ...]:
        """The ``n_values`` values in force at ``progress`` (padded / truncated as above)."""
        values = tuple(interpolate(self.waypoints, progress))
        return values[:self.n_values] + (0.0,) * (self.n_values - len(values))


def apply_to_env(env, target: str, values) -> bool:
    """Call ``<target>(*values)`` on the first object, from ``env`` inwards, that has it.

    Walks the wrapper stack through ``.env``; returns False (and does nothing) when no wrapper
    has the setter, which is what the old per-wrapper helpers did for an env without that
    wrapper (an eval env without reward shaping, say).
    """
    e = env
    while e is not None:
        method = getattr(e, target, None)
        if callable(method):
            method(*values)
            return True
        e = getattr(e, "env", None)
    return False


class ScheduleRouter(gym.Wrapper):
    """Outermost wrapper that exposes each declared setter, so ``SubprocVecEnv.env_method``
    (which can only call methods of the outermost object in a worker) reaches the inner
    wrapper that has it. ``ScheduleRouter(env, targets=("set_curriculum_radius", ...))``
    gives the instance a method of each name that forwards down the stack.
    """

    def __init__(self, env: gym.Env, targets: Sequence[str]):
        super().__init__(env)
        self.targets = tuple(targets)
        for target in self.targets:
            setattr(self, target, functools.partial(self.apply, target))

    def apply(self, target: str, *values) -> bool:
        return apply_to_env(self.env, target, values)


class ScheduleCallback(BaseCallback):
    """SB3 callback: on every step, interpolate every declared schedule at the current
    training progress and push the values into all training envs.

    ``_on_training_start`` applies the schedules at the CURRENT progress before the first env
    step: for a fresh run that repeats the construction-time values; for a run resumed from a
    mid-run checkpoint the envs were just built with progress-0 values, and without this the
    first ``n_envs`` steps would run under the wrong schedule. With one worker
    (``DummyVecEnv``) the setters are called directly on each env's wrapper stack; with
    several (``SubprocVecEnv``) through ``env_method`` on the router. A status line is
    printed at training start and about every 10,000 steps when ``verbose > 0``.
    """

    def __init__(self, total_timesteps: int, schedules: Sequence[Schedule], verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.schedules = tuple(schedules)

    def values_at(self, progress: float) -> dict[str, tuple[float, ...]]:
        return {schedule.name: schedule.values_at(progress) for schedule in self.schedules}

    def _on_training_start(self) -> None:
        self._apply(self.num_timesteps / self.total_timesteps, announce=True)

    def _on_step(self) -> bool:
        self._apply(self.num_timesteps / self.total_timesteps)
        return True

    def _apply(self, progress: float, announce: bool = False) -> None:
        values = [(schedule, schedule.values_at(progress)) for schedule in self.schedules]

        # Update all training envs (works through VecNormalize -> SubprocVecEnv)
        vec_env = self.training_env
        while hasattr(vec_env, "venv"):
            vec_env = vec_env.venv
        if hasattr(vec_env, "envs"):
            # DummyVecEnv -- direct access
            for env in vec_env.envs:
                for schedule, vals in values:
                    apply_to_env(env, schedule.target, vals)
        elif hasattr(vec_env, "env_method"):
            # SubprocVecEnv -- call into subprocesses
            for schedule, vals in values:
                vec_env.env_method(schedule.target, *vals)

        if self.verbose > 0 and (announce or self.num_timesteps % 10000 < (self.training_env.num_envs if self.training_env else 1)):
            print(self.status_line(progress, values, announce))

    def status_line(self, progress: float, values=None, announce: bool = False) -> str:
        if values is None:
            values = [(schedule, schedule.values_at(progress)) for schedule in self.schedules]
        parts = [f"{label}={value:{fmt}}"
                 for schedule, vals in values
                 for (label, fmt), value in zip(schedule.labels, vals)]
        return (f"[Curriculum] step={self.num_timesteps}, progress={progress:.2f}, "
                + ", ".join(parts)
                + (" (applied at training start)" if announce else ""))
