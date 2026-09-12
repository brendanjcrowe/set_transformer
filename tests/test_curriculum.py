"""rl/curriculum.py, change 3 of the harness centralisation (2026-09-12): the generic schedule
machinery -- interpolate, Schedule, apply_to_env, ScheduleRouter, ScheduleCallback -- and the
Ant-Tag adapter over it (CurriculumCallback, _CurriculumRouter, ANT_TAG_SCHEDULES).

The adapter was checked against the pre-change class at 3,927 progress points (env_method
calls and status lines identical) when the change was made; these tests pin the behaviour with
hand-computed values so it stays that way. The Ant-Tag scripts' use of the adapter is pinned
by tests/test_harness_behaviour_inventory.py, test_ant_tag_spread_gain_shaping.py and
test_ant_tag_st_resume.py, all unchanged.
"""

import contextlib
import io
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from set_transformer.rl import curriculum as cu
from set_transformer.rl.curriculum import Schedule, ScheduleCallback, ScheduleRouter, apply_to_env, interpolate


# ---------------------------------------------------------------------------
# interpolate / Schedule
# ---------------------------------------------------------------------------

def test_interpolate_is_linear_between_waypoints_and_clamped_outside():
    schedule = [(0.0, 100.0), (0.2, 100.0), (0.5, 3.0), (1.0, 3.0)]
    assert interpolate(schedule, -1.0) == (100.0,) and interpolate(schedule, 0.1) == (100.0,)
    assert interpolate(schedule, 0.35) == pytest.approx((51.5,))
    assert interpolate(schedule, 0.5) == (3.0,) and interpolate(schedule, 7.0) == (3.0,)
    # every field of a multi-value waypoint, independently
    multi = [(0.0, 1.0, 2.0, 0.0, 0.0), (1.0, 0.0, 2.0, 50.0, 10.0)]
    assert interpolate(multi, 0.5) == pytest.approx((0.5, 2.0, 25.0, 5.0))
    # two waypoints at the same fraction: the later one wins from that fraction on
    step = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 1.0)]
    assert interpolate(step, 0.49) == pytest.approx((0.0,)) and interpolate(step, 0.51) == (1.0,)


def test_schedule_sorts_waypoints_pads_short_ones_and_checks_labels():
    s = Schedule("reward", target="set_reward_coeffs", n_values=4,
                 waypoints=((1.0, 0.0, 2.0), (0.0, 1.0, 2.0)))            # unsorted, 2 of 4 values
    assert s.waypoints == ((0.0, 1.0, 2.0), (1.0, 0.0, 2.0))
    assert s.values_at(0.5) == pytest.approx((0.5, 2.0, 0.0, 0.0))        # padded with zeros
    assert s.labels == (("reward[0]", ".3g"), ("reward[1]", ".3g"), ("reward[2]", ".3g"), ("reward[3]", ".3g"))
    wide = Schedule("x", target="set_x", n_values=1, waypoints=((0.0, 1.0, 9.0), (1.0, 3.0, 9.0)))
    assert wide.values_at(0.5) == (2.0,)                                   # extra values ignored
    assert Schedule("v", target="set_v", waypoints=((0.0, 1.0),)).labels == (("v", ".3g"),)
    with pytest.raises(ValueError, match="no waypoints"):
        Schedule("empty", target="set_e", waypoints=())
    with pytest.raises(ValueError, match="labels"):
        Schedule("v", target="set_v", n_values=2, waypoints=((0.0, 1.0, 2.0),), labels=(("a", ".2f"),))


# ---------------------------------------------------------------------------
# A small wrapper stack of a made-up domain: nothing here is Ant-Tag
# ---------------------------------------------------------------------------

class _Base(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 0.0, False, False, {}


class _NoiseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.noise = None
        self.calls = []

    def set_noise(self, sigma):
        self.noise = sigma
        self.calls.append(("noise", sigma))


class _BonusWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.bonus = None

    def set_bonus(self, a, b, c=None):
        self.bonus = (a, b, c)


class _Plain(gym.Wrapper):
    """A wrapper with no setters at all (Monitor stands here in the real stacks)."""


def _stack():
    noise = _NoiseWrapper(_Base())
    bonus = _BonusWrapper(noise)
    return _Plain(bonus), noise, bonus


def test_apply_to_env_reaches_the_first_wrapper_with_the_setter_or_does_nothing():
    outer, noise, bonus = _stack()
    assert apply_to_env(outer, "set_noise", (0.3,)) is True and noise.noise == 0.3
    assert apply_to_env(outer, "set_bonus", (1.0, 2.0)) is True and bonus.bonus == (1.0, 2.0, None)
    assert apply_to_env(outer, "set_bonus", (1.0, 2.0, 3.0)) is True and bonus.bonus == (1.0, 2.0, 3.0)
    assert apply_to_env(outer, "set_missing", (1.0,)) is False              # silently nothing, as before


def test_schedule_router_exposes_each_target_and_forwards_down_the_stack():
    outer, noise, bonus = _stack()
    router = ScheduleRouter(outer, targets=("set_noise", "set_bonus"))
    assert router.targets == ("set_noise", "set_bonus")
    router.set_noise(0.7)                                                   # what env_method calls
    router.set_bonus(4.0, 5.0)
    assert noise.noise == 0.7 and bonus.bonus == (4.0, 5.0, None)
    assert router.apply("set_bonus", 1.0, 1.0, 1.0) is True and bonus.bonus == (1.0, 1.0, 1.0)
    assert router.apply("set_missing") is False
    assert not hasattr(router, "set_missing")


def _schedules():
    return (Schedule("noise", target="set_noise", waypoints=((0.0, 1.0), (0.5, 0.0), (1.0, 0.0)),
                     labels=(("sigma", ".2f"),)),
            Schedule("bonus", target="set_bonus", n_values=3, waypoints=((0.0, 0.0, 0.0), (1.0, 10.0, 20.0)),
                     labels=(("a", ".1f"), ("b", ".1f"), ("c", ".1f"))))


def test_schedule_callback_pushes_every_schedule_directly_with_one_worker():
    outer, noise, bonus = _stack()
    venv = SimpleNamespace(envs=[outer], num_envs=1)                        # DummyVecEnv shape
    cb = ScheduleCallback(total_timesteps=100, schedules=_schedules())
    cb.model = SimpleNamespace(get_env=lambda: venv)
    cb.num_timesteps = 25
    cb._on_step()
    assert noise.noise == pytest.approx(0.5) and bonus.bonus == pytest.approx((2.5, 5.0, 0.0))
    assert cb.values_at(0.25) == {"noise": (pytest.approx(0.5),), "bonus": pytest.approx((2.5, 5.0, 0.0))}


def test_schedule_callback_uses_env_method_with_several_workers_and_applies_at_training_start():
    calls = []

    class _Venv:                                                             # SubprocVecEnv shape
        num_envs = 4

        def env_method(self, name, *args):
            calls.append((name, args))

    venv = _Venv()
    cb = ScheduleCallback(total_timesteps=100, schedules=_schedules(), verbose=1)
    cb.model = SimpleNamespace(get_env=lambda: venv)
    cb.num_timesteps = 50                                                    # a resumed run
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cb._on_training_start()
    assert calls == [("set_noise", (0.0,)), ("set_bonus", (5.0, 10.0, 0.0))]
    assert buf.getvalue().strip() == ("[Curriculum] step=50, progress=0.50, sigma=0.00, a=5.0, b=10.0, c=0.0"
                                      " (applied at training start)")
    # The 10k-step print: at step 10000 with 4 workers the condition holds, at 10005 it does not.
    for step, printed in ((10000, True), (10005, False)):
        calls.clear(); buf = io.StringIO(); cb.num_timesteps = step
        with contextlib.redirect_stdout(buf):
            cb._on_step()
        assert bool(buf.getvalue()) is printed and len(calls) == 2


def test_schedule_callback_sees_through_vecnormalize():
    outer, noise, _bonus = _stack()
    inner = SimpleNamespace(envs=[outer], num_envs=1)
    wrapped = SimpleNamespace(venv=inner, num_envs=1)                        # VecNormalize shape
    cb = ScheduleCallback(total_timesteps=10, schedules=_schedules()[:1])
    cb.model = SimpleNamespace(get_env=lambda: wrapped)
    cb.num_timesteps = 0
    cb._on_step()
    assert noise.noise == 1.0


# ---------------------------------------------------------------------------
# The Ant-Tag adapter over the generic pieces
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ant_tag():
    pytest.importorskip("pdomains")
    from set_transformer.rl.domains import ant_tag as module
    return module


def test_ant_tag_declares_three_schedules_with_the_historical_script_defaults(ant_tag):
    by_name = {s.name: s for s in ant_tag.ANT_TAG_SCHEDULES}
    assert list(by_name) == ["visibility", "reward", "evasion"]
    assert by_name["visibility"].target == "set_curriculum_radius" and by_name["visibility"].n_values == 1
    assert by_name["visibility"].waypoints == ((0.0, 100.0), (0.3, 100.0), (0.7, 3.0), (1.0, 3.0))
    assert by_name["reward"].target == "set_reward_coeffs" and by_name["reward"].n_values == 4
    assert by_name["reward"].waypoints == ((0.0, 1.0, 0.0, 0.0), (0.3, 1.0, 0.0, 0.0),
                                           (0.7, 0.0, 0.0, 50.0), (1.0, 0.0, 0.0, 50.0))
    assert by_name["evasion"].target == "set_evasion_scale" and by_name["evasion"].waypoints == ((0.0, 1.0), (1.0, 1.0))
    # the adapter's defaults are exactly these
    cb = ant_tag.CurriculumCallback(total_timesteps=100)
    assert cb.schedule == list(by_name["visibility"].waypoints)
    assert cb.reward_schedule == list(by_name["reward"].waypoints)
    assert cb.evasion_schedule == list(by_name["evasion"].waypoints)
    assert [s.name for s in cb.schedules] == ["visibility", "reward", "evasion"]


def test_ant_tag_adapter_equals_the_generic_callback_on_the_same_schedules(ant_tag):
    from dataclasses import replace
    vis = [(0.0, 100.0), (0.2, 100.0), (0.4, 1.5), (1.0, 1.5)]
    rew = [(0.0, 1.0, 2.0, 0.0), (0.2, 1.0, 2.0, 0.0), (0.5, 0.15, 2.0, 50.0), (1.0, 0.15, 2.0, 50.0)]
    eva = [(0.0, 0.0), (0.2, 0.0), (0.5, 1.0), (1.0, 1.0)]
    adapter = ant_tag.CurriculumCallback(total_timesteps=100, schedule=vis, reward_schedule=rew, evasion_schedule=eva)
    by_name = {s.name: s for s in ant_tag.ANT_TAG_SCHEDULES}
    generic = ScheduleCallback(total_timesteps=100, schedules=(
        replace(by_name["visibility"], waypoints=tuple(vis)), replace(by_name["reward"], waypoints=tuple(rew)),
        replace(by_name["evasion"], waypoints=tuple(eva))))
    for progress in np.linspace(-0.1, 1.1, 61):
        assert adapter.values_at(progress) == generic.values_at(progress)
    # progress 0.3: visibility halfway down its 0.2 -> 0.4 ramp, reward and evasion a third
    # of the way along theirs (no exact .xx5 ties, so the formatted line is unambiguous)
    assert adapter.values_at(0.3) == {"visibility": (pytest.approx(50.75),),
                                      "reward": pytest.approx((1 - 0.85 / 3, 2.0, 50 / 3, 0.0)),
                                      "evasion": (pytest.approx(1 / 3),)}
    assert adapter._interpolate_schedule(adapter.reward_schedule, 0.3) == pytest.approx((1 - 0.85 / 3, 2.0, 50 / 3))
    adapter.num_timesteps = 30
    assert adapter.status_line(0.3) == ("[Curriculum] step=30, progress=0.30, vis_radius=50.75, dist_coeff=0.717, "
                                        "ent_coeff=2.000, tag_bonus=16.7, gain_coeff=0.00, evasion_scale=0.33")


def test_ant_tag_router_is_a_schedule_router_over_the_three_setters(ant_tag):
    router = ant_tag._CurriculumRouter(_Plain(_Base()))
    assert isinstance(router, ScheduleRouter)
    assert router.targets == ("set_curriculum_radius", "set_reward_coeffs", "set_evasion_scale")
    for target in router.targets:
        assert callable(getattr(router, target))
    assert router.set_curriculum_radius(1.0) is False                      # nothing below has it: no-op
