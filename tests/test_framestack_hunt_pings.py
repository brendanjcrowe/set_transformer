"""Hunt ``--policy_obs pings``: anonymous beacon pings as the framestack arm's input (2026-09-24).

Pins ``change_mds/framestack_hunt_pings_2026-09-24.md``:

* ``policy_obs="agent"`` (the default) is the recorded env: the same wrapper chain and the same
  arrays step for step as a factory call without the option, and a default ``run_config.json``
  gains no key;
* under ``pings`` the env itself is untouched: layouts, clouds, rewards and infos are the same as
  under ``agent`` on the same seeds and actions, and the global numpy stream is not drawn from;
* the frame: 2 + 3 * n_clusters = 17 wide, the agent part is today's obs, every frame is
  re-derived from the documented draw order (``(clip(fix, 0, 20) - pos) / 10``), empty slots are
  zero, every frame is in the Box;
* the ping model per env: on cluster_hunt the pings are the live clusters and a collected one
  never pings again; on least_mass a cluster pings at ``counts / max(counts)``; on most_var five
  clusters ping at every step at k = 5;
* the slot order is random; the same reset seed gives the same pings, another seed other pings;
* the refusals: another encoder, ``--stack_padding reset_frame`` and VecNormalize at train,
  ``pings`` at the collect door, a pings agent under ``--behaviour policy``;
* a short real framestack run trains through the pings env (two subprocess workers), a
  standalone eval with ``--policy_obs pings`` rebuilds it off the zip, writes its JSON and
  repeats itself, and the same eval without the flag is refused with the flag named.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains", reason="envs are registered by pdomains")
pytest.importorskip("pdomains.hunt", reason="needs the pomdp-domains hunt envs")

import gymnasium as gym  # noqa: E402

from set_transformer.rl import collect as collect_mod  # noqa: E402
from set_transformer.rl import eval_true_reward as eval_mod  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import hunt  # noqa: E402
from set_transformer.rl.wrappers.obs_history import (  # noqa: E402
    ObsHistoryDictWrapper,
    checkpoint_obs_history_spec,
    checkpoint_obs_width,
)

VARIANTS = ("cluster_hunt", "least_mass", "most_var")
N = 100          # the envs' own cloud size; the pass-through filter must match it
WIDTH = 17       # 2 + 3 * n_clusters at the registered five clusters


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """PITFALLS.md 13.36: the file sets the variable itself."""
    monkeypatch.setenv("WANDB_MODE", "offline")


def _build(variant, seed=0, rank=0, **options):
    """The EVAL env (every cluster active, the registered hit radius), with ``options``."""
    return hunt.HUNT.make_env(variant, num_particles=N,
                              particle_filter_class=hunt.resolve(variant).particle_filter,
                              seed=seed, rank=rank, monitor_dir=None, training=False,
                              options=options)()


def _chain(env):
    names = []
    while env is not None:
        names.append(type(env).__name__)
        env = getattr(env, "env", None)
    return names


def _pursuit(u):
    """Head for the nearest live cluster (cluster_hunt) or the nearest cluster: episodes end."""
    centers = u.centers[u.alive] if hasattr(u, "alive") else u.centers[: u.k]
    if len(centers) == 0:
        return np.zeros(2, np.float32)
    d = centers[np.argmin(np.linalg.norm(centers - u.pos, axis=-1))] - u.pos
    return np.clip(d, -1.0, 1.0).astype(np.float32)


def _state(u):
    """The hidden state the ping model reads, copied."""
    out = dict(pos=u.pos.copy(), centers=u.centers.copy(), sigmas=u.sigmas.copy(),
               particles=u.particles.copy())
    if hasattr(u, "alive"):
        out["alive"] = u.alive.copy()
    else:
        out["counts"] = u.counts.copy()
    return out


def _roll(env, seed, steps=60, act=_pursuit):
    """Reset on ``seed``, step with ``act(unwrapped)``; every (obs, reward, done, info, state)."""
    obs, info = env.reset(seed=seed)
    out = [(obs, 0.0, False, info, _state(env.unwrapped))]
    for _ in range(steps):
        obs, reward, term, trunc, info = env.step(act(env.unwrapped))
        out.append((obs, reward, term or trunc, info, _state(env.unwrapped)))
        if term or trunc:
            break
    return out


def _same_info(a, b) -> bool:
    """Equal step infos; Monitor's ``episode`` record is compared without its wall-clock time."""
    def strip(info):
        episode = {k: v for k, v in info.get("episode", {}).items() if k != "t"}
        return {**{k: v for k, v in info.items() if k != "episode"}, "episode": episode}
    return strip(a) == strip(b)


def _slots(frame):
    return np.asarray(frame["obs"] if isinstance(frame, dict) else frame)[2:].reshape(-1, 3)


class _NoGpsError:
    """A ping generator with the GPS error set to zero, so every ping is a true centre and can
    be attributed exactly (the Bernoulli draws and the slot permutation are the real ones)."""

    def __init__(self, seed):
        self._rng = np.random.default_rng([int(seed), hunt.PING_SEED_TAG])

    def random(self, *shape):
        return self._rng.random(*shape)

    def permutation(self, n):
        return self._rng.permutation(n)

    def standard_normal(self, shape):
        return np.zeros(shape)


class _HugeGpsError(_NoGpsError):
    """A GPS error of +-50 on every axis: every fix leaves the arena, so the clip decides it."""

    def standard_normal(self, shape):
        return 50.0 * np.where(self._rng.random(shape) < 0.5, -1.0, 1.0)


def _attribute(frame, state, tol=1e-4):
    """The cluster index of every ping under ``_NoGpsError`` (the exact centre, clipped)."""
    slots = _slots(frame)
    fixes = state["pos"] + hunt.ARENA_SCALE * slots[slots[:, 0] > 0.5, 1:].astype(np.float64)
    centres = np.clip(state["centers"], 0.0, 2 * hunt.ARENA_SCALE)
    d = np.linalg.norm(fixes[:, None] - centres[None], axis=-1)
    assert np.all(d.min(axis=1) < tol), "a ping is not at a true centre"
    return np.sort(d.argmin(axis=1))


# --------------------------------------------------------------------------
# 1. agent is the recorded env
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_agent_is_byte_identical_to_the_factory_without_the_option(variant):
    plain = hunt.make_hunt_belief_env(num_particles=N, seed=3, variant=variant)()
    agent = hunt.make_hunt_belief_env(num_particles=N, seed=3, variant=variant, policy_obs="agent")()
    try:
        assert _chain(plain) == _chain(agent)
        assert "HuntAgentObsWrapper" in _chain(agent) and "HuntPingObsWrapper" not in _chain(agent)
        assert plain.observation_space == agent.observation_space
        assert agent.observation_space["obs"] == gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        for seed in (5, 6):
            for a, b in zip(_roll(plain, seed), _roll(agent, seed), strict=True):
                assert set(a[0]) == set(b[0]) == {"obs", "particles", "weights"}
                for key in a[0]:
                    assert a[0][key].dtype == b[0][key].dtype, key
                    assert a[0][key].tobytes() == b[0][key].tobytes(), key
                assert a[1] == b[1] and a[2] == b[2] and _same_info(a[3], b[3])
    finally:
        plain.close()
        agent.close()


def test_an_unknown_policy_obs_and_an_oracle_env_are_refused():
    with pytest.raises(ValueError, match="unknown policy_obs"):
        hunt.make_hunt_belief_env(num_particles=N, variant="least_mass", policy_obs="beacons")
    import pdomains  # noqa: F401,PLC0415
    raw = gym.make("pdomains-least-mass-v0", include_oracle=True)
    try:
        with pytest.raises(ValueError, match="include_oracle"):
            hunt.HuntPingObsWrapper(raw)
    finally:
        raw.close()


# --------------------------------------------------------------------------
# 2. pings leave the env alone
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_pings_leave_the_layouts_the_clouds_and_the_episodes_alone(variant):
    agent, pings = _build(variant), _build(variant, policy_obs="pings")
    try:
        assert _chain(pings) == [n if n != "HuntAgentObsWrapper" else "HuntPingObsWrapper"
                                 for n in _chain(agent)]
        for seed in (0, 7, 8):
            for a, p in zip(_roll(agent, seed), _roll(pings, seed), strict=True):
                for key, value in a[4].items():            # the hidden state, cloud included
                    assert np.array_equal(value, p[4][key]), key
                for key in ("particles", "weights"):
                    assert a[0][key].tobytes() == p[0][key].tobytes(), key
                assert a[1] == p[1] and a[2] == p[2] and _same_info(a[3], p[3])
    finally:
        agent.close()
        pings.close()


def test_pings_draw_nothing_from_the_global_numpy_stream():
    np.random.seed(1234)
    before = np.random.get_state()[1].copy()
    env = _build("least_mass", policy_obs="pings")
    try:
        _roll(env, 2)
        assert np.array_equal(np.random.get_state()[1], before)
    finally:
        env.close()


# --------------------------------------------------------------------------
# 3. the frame
# --------------------------------------------------------------------------

def _replay(states, seed, model, n_clusters=5):
    """Every frame's ping part from the documented draw order, independently of the wrapper."""
    rng = np.random.default_rng([seed, hunt.PING_SEED_TAG])
    frames = []
    for s in states:
        if model == "every_live":
            beacons = np.flatnonzero(s["alive"])
        else:
            counts = s["counts"].astype(np.float64)
            beacons = np.flatnonzero(rng.random(len(counts)) < counts / counts.max())
        fixes = s["centers"][beacons] + s["sigmas"][beacons, None] * rng.standard_normal(
            (len(beacons), 2))
        fixes = np.clip(fixes, 0.0, 20.0)
        slots = rng.permutation(n_clusters)[: len(beacons)]
        frame = np.zeros((n_clusters, 3))
        frame[slots, 0] = 1.0
        frame[slots, 1:] = (fixes - s["pos"]) / 10.0
        frames.append(frame.ravel().astype(np.float32))
    return frames


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_frame_is_the_agent_obs_then_the_pings_of_the_documented_model(variant):
    agent, pings = _build(variant), _build(variant, policy_obs="pings")
    try:
        space = pings.observation_space["obs"]
        assert space == gym.spaces.Box(-2.0, 2.0, (WIDTH,), np.float32)
        assert hunt.ping_frame_width(pings.unwrapped.cfg.n_clusters) == WIDTH
        model = "every_live" if variant == "cluster_hunt" else "by_mass"
        n_empty = 0
        for seed in (1, 4, 9):
            rolled_a, rolled_p = _roll(agent, seed), _roll(pings, seed)
            replayed = _replay([r[4] for r in rolled_p], seed, model)
            for a, p, expected in zip(rolled_a, rolled_p, replayed, strict=True):
                frame = p[0]["obs"]
                assert frame.dtype == np.float32 and frame.shape == (WIDTH,)
                assert space.contains(frame)
                assert frame[:2].tobytes() == a[0]["obs"].tobytes()     # today's obs, byte for byte
                assert frame[2:].tobytes() == expected.tobytes()
                slots = _slots(frame)
                assert set(np.unique(slots[:, 0])) <= {0.0, 1.0}
                empty = slots[:, 0] == 0.0
                assert np.all(slots[empty] == 0.0)
                n_empty += int(empty.sum())
        if variant == "least_mass":
            assert n_empty > 0, "least_mass never left a slot empty: every cluster pinged"
    finally:
        agent.close()
        pings.close()


# --------------------------------------------------------------------------
# 4-6. the ping model per env (exact, with the GPS error set to zero)
# --------------------------------------------------------------------------

def test_cluster_hunt_pings_are_the_live_clusters_and_a_collected_one_goes_silent(monkeypatch):
    monkeypatch.setattr(hunt, "_ping_generator", _NoGpsError)
    env = _build("cluster_hunt", policy_obs="pings")
    try:
        collections = 0
        for seed in range(6):
            rolled = _roll(env, seed)
            silent = set()
            for obs, _r, _done, info, state in rolled:
                assert list(_attribute(obs, state)) == list(np.flatnonzero(state["alive"]))
                if info.get("collected"):
                    collections += 1
                    # collected in THIS step: it is dead in the post-step state and did not ping
                    silent |= set(np.flatnonzero(~state["alive"]))
                assert not silent & set(_attribute(obs, state))
        assert collections >= 5, f"only {collections} collections: the pursuit did not reach clusters"
    finally:
        env.close()


def test_least_mass_clusters_ping_at_counts_over_the_largest_count():
    """Nearest-true-centre attribution of the real (noisy) pings; the bias of the attribution is
    under 3 % on these seeds, the tolerance is 10 %."""
    env = _build("least_mass", policy_obs="pings")
    observed = {"lightest": 0, "rate_below_1": 0, "rate_1": 0}
    expected = dict.fromkeys(observed, 0.0)
    try:
        for seed in range(1000, 1060):
            u = env.unwrapped
            for obs, *_rest, state in _roll(env, seed, act=lambda u: np.zeros(2, np.float32)):
                slots = _slots(obs)
                fixes = state["pos"] + hunt.ARENA_SCALE * slots[slots[:, 0] > 0.5, 1:]
                near = np.linalg.norm(fixes[:, None] - state["centers"][None], axis=-1).argmin(1)
                counts = np.bincount(near, minlength=len(state["counts"]))
                rates = state["counts"] / state["counts"].max()
                for j, rate in enumerate(rates):
                    key = ("lightest" if j == np.argmin(state["counts"]) else
                           "rate_1" if rate == 1.0 else "rate_below_1")
                    observed[key] += int(counts[j])
                    expected[key] += float(rate)
            assert u.k == 5
    finally:
        env.close()
    for key in observed:
        assert observed[key] == pytest.approx(expected[key], rel=0.10), (key, observed, expected)
    assert expected["lightest"] < 0.6 * expected["rate_1"], "the lightest must ping least often"


def test_most_var_every_cluster_pings_once_at_every_step(monkeypatch):
    monkeypatch.setattr(hunt, "_ping_generator", _NoGpsError)
    env = _build("most_var", policy_obs="pings")
    try:
        for seed in range(4):
            for obs, *_rest, state in _roll(env, seed, act=lambda u: np.zeros(2, np.float32)):
                assert np.all(state["counts"] == 20)
                assert _slots(obs)[:, 0].sum() == 5
                assert list(_attribute(obs, state)) == [0, 1, 2, 3, 4]
    finally:
        env.close()


# --------------------------------------------------------------------------
# 7-8. slot order and seeding
# --------------------------------------------------------------------------

def test_the_slot_order_is_drawn_fresh_at_every_step(monkeypatch):
    monkeypatch.setattr(hunt, "_ping_generator", _NoGpsError)
    env = _build("most_var", policy_obs="pings")
    try:
        in_slot_0 = set()
        for obs, *_rest, state in _roll(env, 3, act=lambda u: np.zeros(2, np.float32)):
            fix = state["pos"] + hunt.ARENA_SCALE * _slots(obs)[0, 1:].astype(np.float64)
            in_slot_0.add(int(np.linalg.norm(state["centers"] - fix, axis=-1).argmin()))
        assert in_slot_0 == {0, 1, 2, 3, 4}
    finally:
        env.close()
    env = _build("least_mass", policy_obs="pings")
    try:
        # an empty slot BEFORE an occupied one: the pings are not packed into the first slots
        bits = [_slots(obs)[:, 0] for obs, *_ in _roll(env, 5, act=lambda u: np.zeros(2, np.float32))]
        assert any(np.any(np.diff(b) > 0) for b in bits)
    finally:
        env.close()


def test_the_same_reset_seed_gives_the_same_pings_and_another_seed_other_pings():
    a = _build("least_mass", seed=0, policy_obs="pings")
    b = _build("least_mass", seed=9, rank=3, policy_obs="pings")      # another constructor seed
    try:
        for x, y in zip(_roll(a, 21), _roll(b, 21), strict=True):
            assert x[0]["obs"].tobytes() == y[0]["obs"].tobytes()
        again = _roll(a, 21)
        assert again[0][0]["obs"].tobytes() == _roll(b, 21)[0][0]["obs"].tobytes()
        other = _roll(a, 22)
        assert not np.array_equal(again[0][0]["obs"][2:], other[0][0]["obs"][2:])
    finally:
        a.close()
        b.close()
    # Without a reset seed the constructor's seed + rank decides: the same pair, the same pings.
    c, d = (_build("most_var", seed=4, rank=1, policy_obs="pings") for _ in range(2))
    try:
        for x, y in zip(_roll(c, None), _roll(d, None), strict=True):
            assert x[0]["obs"].tobytes() == y[0]["obs"].tobytes()
    finally:
        c.close()
        d.close()


def test_the_constructor_seeds_the_ping_generator_with_seed_plus_rank(monkeypatch):
    seeds, real = [], hunt._ping_generator
    monkeypatch.setattr(hunt, "_ping_generator", lambda seed: (seeds.append(seed), real(seed))[1])
    _build("most_var", seed=4, rank=1, policy_obs="pings").close()
    assert seeds[0] == 5


def test_a_fix_outside_the_arena_is_clipped_to_the_wall_before_it_is_made_relative(monkeypatch):
    monkeypatch.setattr(hunt, "_ping_generator", _HugeGpsError)
    env = _build("most_var", policy_obs="pings")
    try:
        for obs, *_rest, state in _roll(env, 2, steps=10, act=lambda u: np.zeros(2, np.float32)):
            assert env.observation_space["obs"].contains(obs["obs"])
            slots = _slots(obs)
            assert slots[:, 0].sum() == 5
            fixes = state["pos"] + hunt.ARENA_SCALE * slots[:, 1:].astype(np.float64)
            on_wall = np.isclose(fixes, 0.0, atol=1e-4) | np.isclose(fixes, 20.0, atol=1e-4)
            assert on_wall.all()
    finally:
        env.close()


# --------------------------------------------------------------------------
# 9. the doors: the options, the record, the refusals
# --------------------------------------------------------------------------

@pytest.fixture
def _quiet_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    return tmp_path


def _dry(tmp_path, *extra, encoder="framestack", tag="t", variant="cluster_hunt"):
    root = tmp_path / tag
    train_mod.main(["--variant", variant, "--output_root", str(root), "--dry_run",
                    "--run_tag", tag, *extra], domain="hunt", encoder=encoder)
    [config] = list(root.rglob("run_config.json"))
    return json.loads(config.read_text())


_PATHS = ("log_dir", "model_save_path", "run_directory", "output_root", "run_tag")


def test_a_default_hunt_record_gains_no_key_and_explicit_agent_is_the_default(_quiet_runs):
    default = _dry(_quiet_runs, encoder="gaussian", tag="a")
    explicit = _dry(_quiet_runs, "--policy_obs", "agent", encoder="gaussian", tag="b")
    assert "policy_obs" not in default and "ping_frame" not in default
    assert {k: v for k, v in default.items() if k not in _PATHS} == \
           {k: v for k, v in explicit.items() if k not in _PATHS}
    framestack = _dry(_quiet_runs, "--n_stack", "4", tag="c")
    assert "policy_obs" not in framestack and framestack["stack_padding"] == "reset_frame"


def test_pings_travel_in_the_options_and_the_record_only_under_pings(_quiet_runs):
    parser = train_mod.build_parser(hunt.HUNT, train_mod._encoders.get("framestack"))
    args = parser.parse_args(["--variant", "most_var"])
    assert hunt._resolve_arguments(parser, args) == {}
    args = parser.parse_args(["--variant", "most_var", "--policy_obs", "pings",
                              "--stack_padding", "zeros", "--no_vec_normalize"])
    assert hunt._resolve_arguments(parser, args) == {"policy_obs": "pings"}
    config = _dry(_quiet_runs, "--n_stack", "20", "--policy_obs", "pings", "--stack_padding",
                  "zeros", "--no_vec_normalize", variant="least_mass")
    assert (config["policy_obs"], config["n_stack"], config["stack_padding"],
            config["no_vec_normalize"]) == ("pings", 20, "zeros", True)
    frame = config["ping_frame"]
    assert (frame["width"], frame["n_clusters"], frame["ping_model"]) == (WIDTH, 5, "by_mass")
    assert hunt.ping_frame_record("cluster_hunt")["ping_model"] == "every_live"
    eval_args = type("A", (), {})()
    eval_args.variant = "most_var"
    assert hunt._eval_options(eval_args) == {}
    eval_args.policy_obs = "agent"
    assert hunt._eval_options(eval_args) == {}
    eval_args.policy_obs = "pings"
    assert hunt._eval_options(eval_args) == {"policy_obs": "pings"}
    assert "match --policy_obs pings" in hunt._eval_frame_width_hint(eval_args, 17, 17)
    assert "match --policy_obs agent" in hunt._eval_frame_width_hint(eval_args, 2, 17)
    # a width that is neither hunt frame (a 31-wide Ant-Tag frame): no flag is suggested
    foreign = hunt._eval_frame_width_hint(eval_args, 31, 17)
    assert "match neither" in foreign and "match --policy_obs" not in foreign


def test_every_train_refusal_names_its_fix(_quiet_runs, capsys):
    ok = ("--policy_obs", "pings", "--no_vec_normalize")    # --stack_padding is framestack's
    for encoder in ("gaussian", "st"):
        with pytest.raises(SystemExit):
            _dry(_quiet_runs, *ok, encoder=encoder, tag=f"e_{encoder}")
        assert "Use --encoder framestack" in capsys.readouterr().err
    with pytest.raises(SystemExit):     # the `python -m set_transformer.rl.train` entry
        train_mod.main(["--domain", "hunt", "--encoder", "cgf", "--variant", "cluster_hunt",
                        "--output_root", str(_quiet_runs / "sel"), "--dry_run", *ok])
    assert "--encoder cgf reads the belief" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _dry(_quiet_runs, "--policy_obs", "pings", "--no_vec_normalize", tag="pad")
    assert "needs --stack_padding zeros (got reset_frame)" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _dry(_quiet_runs, "--policy_obs", "pings", "--stack_padding", "zeros", tag="vn")
    assert "needs --no_vec_normalize" in capsys.readouterr().err
    assert not list(_quiet_runs.rglob("run_config.json"))


def test_pings_are_refused_at_the_collect_door(_quiet_runs, capsys):
    with pytest.raises(SystemExit):
        collect_mod.main(["--domain", "hunt", "--variant", "least_mass", "--policy_obs", "pings",
                          "--num_episodes", "1", "--output_root", str(_quiet_runs / "root")])
    assert "--policy_obs pings cannot collect a hunt dataset" in capsys.readouterr().err
    out = collect_mod.main(["--domain", "hunt", "--variant", "least_mass", "--policy_obs", "agent",
                            "--num_episodes", "1", "--output_root", str(_quiet_runs / "root")])
    with np.load(out, allow_pickle=True) as z:
        assert "policy_obs" not in json.loads(str(z["metadata"]))["args"]


def test_the_history_wrapper_stacks_the_ping_frames_behind_zero_frames():
    env = ObsHistoryDictWrapper(_build("most_var", policy_obs="pings"), 3, padding="zeros")
    try:
        assert env.observation_space["obs"] == gym.spaces.Box(-2.0, 2.0, (3 * WIDTH,), np.float32)
        obs, _ = env.reset(seed=0)
        assert np.all(obs["obs"][: 2 * WIDTH] == 0.0) and obs["obs"][-WIDTH:][2::3].sum() == 5
        assert env.observation_space["obs"].contains(obs["obs"])
    finally:
        env.close()


# --------------------------------------------------------------------------
# 10. End to end: train through the pings env, eval with and without --policy_obs pings
# --------------------------------------------------------------------------

def test_train_then_eval_with_pings_and_the_refusals_without(_quiet_runs, capsys):
    tmp_path = _quiet_runs
    model_path = tmp_path / "models" / "framestack_agent.zip"
    model = train_mod.main(
        # The fixed hunt recipe's flags, shrunk to 2,048 steps on two subprocess workers.
        ["--variant", "cluster_hunt", "--n_stack", "3", "--stack_padding", "zeros",
         "--policy_obs", "pings", "--no_vec_normalize", "--total_timesteps", "2048",
         "--n_envs", "2", "--ppo_n_steps", "512", "--batch_size", "256", "--n_epochs", "1",
         "--ent_coef", "0.005", "--separate_extractors", "--net_arch", "32,32", "--device", "cpu",
         "--eval_freq", "1024", "--n_eval_episodes", "1", "--save_freq", "100000",
         "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="hunt", encoder="framestack")
    capsys.readouterr()
    assert model.num_timesteps >= 2048
    assert model.observation_space["obs"] == gym.spaces.Box(-2.0, 2.0, (3 * WIDTH,), np.float32)
    assert model.policy.features_extractor.features_dim == 3 * WIDTH
    assert checkpoint_obs_history_spec(str(model_path)) == (3, "zeros")
    assert checkpoint_obs_width(str(model_path)) == 3 * WIDTH
    assert not (model_path.parent / "vecnormalize.pkl").exists()
    [config] = list((tmp_path / "root").rglob("run_config.json"))
    assert json.loads(config.read_text())["policy_obs"] == "pings"

    common = ["--variant", "cluster_hunt", "--model_path", str(model_path), "--n_episodes", "3",
              "--seed", "42", "--output_root", str(tmp_path / "root")]
    summaries = []
    for run in (1, 2):
        summary = tmp_path / f"eval_{run}.json"
        episodes = eval_mod.main(common + ["--policy_obs", "pings", "--summary_path", str(summary)],
                                 domain="hunt")
        out = capsys.readouterr().out
        assert len(episodes) == 3
        assert f"obs width {3 * WIDTH} = 3 frames x {WIDTH}" in out
        summaries.append(json.loads(summary.read_text()))
    assert summaries[0]["report"]["n_episodes"] == 3 and summaries[0]["domain"] == "hunt"
    assert summaries[0]["report"]["episodes"] == summaries[1]["report"]["episodes"]

    with pytest.raises(SystemExit):
        eval_mod.main(common + ["--no_summary"], domain="hunt")
    err = capsys.readouterr().err
    assert f"{3 * WIDTH} wide (3 frame(s) of {WIDTH})" in err
    assert "match --policy_obs pings" in err

    with pytest.raises(SystemExit):
        collect_mod.main(["--domain", "hunt", "--variant", "cluster_hunt", "--num_episodes", "1",
                          "--behaviour", "policy", "--policy_path", str(model_path),
                          "--output_root", str(tmp_path / "root")])
    assert "was trained with --policy_obs pings (3 frame(s) of 17)" in capsys.readouterr().err
