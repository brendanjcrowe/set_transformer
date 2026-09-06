"""Regression gate for the Ant-Tag pieces that are about to be moved.

Four env-independent pieces are scheduled to move out of
`experiments/ant_tag/` into the shared package `set_transformer/rl/`, so a
second domain (Odd-Even) can reuse them:

    PFDictWithWeightsObservationWrapper   4_train_rl_cgf.py
    WeightedCGFFeaturesExtractor          4_train_rl_cgf.py
    SetTransformerFeaturesExtractor       4_train_rl_st.py
    WeightedGaussianFeaturesExtractor     4_train_rl_gaussian.py

The Ant-Tag arms must not change behaviour when that happens. This file is
the executable form of that requirement. It locks six things:

1.  **The import-by-name surface.** Every symbol the Ant-Tag scripts take off
    each other must stay reachable under its ORIGINAL module and name. A move
    that forgets one re-export breaks a sibling script at import time.
2.  **Saved checkpoints.** SB3 pickles the extractor CLASS into the zip, so a
    load is `getattr(import_module('4_train_rl_cgf'), 'WeightedCGF...')`.
    Move the class without a re-export and every checkpoint on disk dies.
    This test loads real ones.
3.  **Golden numeric forward outputs** for all three extractors, against
    hard-coded literals.
4.  **A seeded dict-obs rollout** through the real belief env.
5.  **The variant registry**: env id, filter class name and episode cap.
6.  **A short real training run** for each of the three arms.

The related invariant from domain_mds/PITFALLS.md section 1 -- that a
pretrained ST encoder survives PPO construction, because
`ActorCriticPolicy._build` re-initializes the whole policy including the
features extractor -- lives in `tests/test_st_pretrained_load.py`. It is not
duplicated here.

Other pitfalls guarded here:
  section 5  the episode cap must come from the gym registration
  section 7  digit-leading sibling modules, and sys.path depth
"""

import copy
import importlib
import json
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANT_TAG_DIR = _ST_ROOT / "experiments" / "ant_tag"
_RUNS_DIR = _ANT_TAG_DIR / "runs"
# Only the package roots go on sys.path permanently. The experiment directory
# must not: see the ant_tag fixture.
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
# AntTag uses the modern `mujoco` binding, not the legacy mujoco_py.
pytest.importorskip("mujoco", reason="AntTag needs MuJoCo")
pytest.importorskip("pdomains", reason="AntTag envs are registered by pdomains")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402


# --------------------------------------------------------------------------
# Module loading
# --------------------------------------------------------------------------

@contextmanager
def _sys_path(*directories):
    """Temporarily prepend directories, then restore sys.path exactly.

    Same helper as tests/test_odd_even_pomdp_contract.py. Leaving an
    experiment directory on the global sys.path is how a flat module name
    silently resolves to the wrong file, and experiments/odd_even/ is about to
    hold same-named numbered scripts (4_train_rl_cgf.py and friends).
    """
    saved = list(sys.path)
    try:
        for directory in directories:
            sys.path.insert(0, str(directory))
        yield
    finally:
        sys.path[:] = saved


#: The names the Ant-Tag scripts import each other by. They start with a
#: digit, so `import` cannot spell them and importlib is required
#: (PITFALLS.md section 7).
_MODULE_NAMES = ("variants", "4_train_rl_frozen", "4_train_rl_cgf",
                 "4_train_rl_st", "4_train_rl_gaussian")


@pytest.fixture(scope="module")
def ant_tag():
    """The Ant-Tag pipeline modules, imported under their FLAT names.

    The flat name matters and cannot be replaced by a load-by-path here: a
    saved SB3 policy records its features extractor as
    `<class '4_train_rl_cgf.WeightedCGFFeaturesExtractor'>`, so unpickling
    does `getattr(sys.modules['4_train_rl_cgf'], ...)`. Loading the file
    under a unique key would not satisfy that.

    experiments/ant_tag/ is therefore on sys.path only for the duration of
    the imports, and every module this fixture newly created is dropped at
    teardown, so no later test module can resolve `4_train_rl_cgf` to the
    wrong directory.
    """
    preexisting = set(sys.modules)
    with _sys_path(_ANT_TAG_DIR):
        modules = {name: importlib.import_module(name)
                   for name in _MODULE_NAMES}
    try:
        yield modules
    finally:
        for name in set(sys.modules) - preexisting:
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None) or ""
            if str(_ST_ROOT / "experiments") in file or name in _MODULE_NAMES:
                sys.modules.pop(name, None)


# --------------------------------------------------------------------------
# 1. Import-by-name surface
# --------------------------------------------------------------------------

#: What each Ant-Tag module must expose, keyed by module name. Every entry is
#: read off another file's module-top import block (or off an eval /
#: diagnostics script), so a missing one is an ImportError in the pipeline.
_REQUIRED_SURFACE = {
    # 4_train_rl_cgf pulls these six off the older frozen script.
    "4_train_rl_frozen": (
        "CurriculumCallback",
        "CurriculumVisibilityWrapper",
        "PFRewardShapingWrapper",
        "_CurriculumRouter",
        "ant_tag_pf_interaction_mapper",
        "get_ant_tag_pf_kwargs",
    ),
    # 4_train_rl_st and 4_train_rl_gaussian pull this whole block off
    # 4_train_rl_cgf; eval_scripts/ and diagnostics/ pull a subset.
    "4_train_rl_cgf": (
        "AntTagParticleFilter",
        "CurriculumCallback",
        "CurriculumVisibilityWrapper",
        "PFDictWithWeightsObservationWrapper",
        "PFRewardShapingWrapper",
        "TNormLoggingCallback",
        "WeightedCGFFeaturesExtractor",
        "_CurriculumRouter",
        "_default_run_dir",
        "_git_provenance",
        "_make_vec_env_from_fns",
        "_make_vec_normalize",
        "_parse_curriculum",
        "_parse_reward_schedule",
        "_tee_stdout_stderr",
        "_write_run_config",
        "ant_tag_pf_interaction_mapper",
        "get_ant_tag_arena_scale",
        "get_ant_tag_pf_kwargs",
        "get_env_visible_radius",
        "main",
        "make_ant_tag_cgf_env",
        "train_ant_tag_cgf",
    ),
    "4_train_rl_st": (
        "PFDictWithWeightsObservationWrapper",
        "STFeatureLoggingCallback",
        "SetTransformerFeaturesExtractor",
        "main",
        "make_ant_tag_belief_env",
        "train_ant_tag_st",
    ),
    "4_train_rl_gaussian": (
        "CurriculumVisibilityWrapper",
        "PFDictWithWeightsObservationWrapper",
        "WeightedGaussianFeaturesExtractor",
        "_CurriculumRouter",
        "ant_tag_pf_interaction_mapper",
        "get_ant_tag_pf_kwargs",
        "main",
        "make_ant_tag_belief_env",
        "train_ant_tag_gaussian",
    ),
    "variants": (
        "EVADING",
        "VARIANTS",
        "Variant",
        "add_variant_argument",
        "episode_cap",
        "print_variants",
        "resolve",
        "resolve_schedule",
        "run_subdir",
        "warn_if_not_evading",
        "warn_if_override_contradicts",
    ),
}


@pytest.mark.parametrize("module_name", sorted(_REQUIRED_SURFACE))
def test_import_by_name_surface_is_intact(ant_tag, module_name):
    """Every cross-script symbol stays reachable under its original name.

    The Ant-Tag scripts are wired to each other by name, not by package:
    4_train_rl_st.py and 4_train_rl_gaussian.py each open with a block of
    `X = _train_rl_cgf.X` assignments. Moving a class out of 4_train_rl_cgf.py
    without leaving a re-export behind breaks those two files, plus every
    eval and diagnostics script, at import time. This test is the checklist.
    """
    module = ant_tag[module_name]
    missing = [name for name in _REQUIRED_SURFACE[module_name]
               if not hasattr(module, name)]
    assert not missing, (
        f"{module_name} no longer exposes {missing}. Another Ant-Tag script "
        f"imports each of these by name; re-export them from {module_name} "
        "even after moving the definition."
    )


def test_moved_classes_are_the_same_object_everywhere(ant_tag):
    """The re-exports must be the SAME class, not a second copy.

    4_train_rl_st.py and 4_train_rl_gaussian.py both re-export
    PFDictWithWeightsObservationWrapper. If a move left a shim class behind
    instead of an alias, `isinstance` checks and pickled class identity would
    diverge between arms while every import still succeeded.
    """
    cgf = ant_tag["4_train_rl_cgf"]
    for module_name in ("4_train_rl_st", "4_train_rl_gaussian"):
        module = ant_tag[module_name]
        assert (module.PFDictWithWeightsObservationWrapper
                is cgf.PFDictWithWeightsObservationWrapper)
        assert module.make_ant_tag_belief_env is cgf.make_ant_tag_cgf_env
        assert module.get_ant_tag_pf_kwargs is cgf.get_ant_tag_pf_kwargs
        assert module._CurriculumRouter is cgf._CurriculumRouter


# --------------------------------------------------------------------------
# 2. Saved-checkpoint compatibility
# --------------------------------------------------------------------------

#: (label, zip path relative to experiments/ant_tag/, module that defines the
#: extractor class). One per encoder arm. These are real training outputs.
_CHECKPOINTS = [
    (
        "cgf",
        "runs/ant_tag_cgf_smart/20260731_190521_seed0_stab_v1_entropy_flat_newtarget"
        "/models/cgf_agent.zip",
        "4_train_rl_cgf",
        "WeightedCGFFeaturesExtractor",
    ),
    (
        "gaussian",
        "runs/ant_tag_gaussian_cdens/20260801_094312_seed0_smoke_gauss"
        "/models/gaussian_agent.zip",
        "4_train_rl_gaussian",
        "WeightedGaussianFeaturesExtractor",
    ),
    (
        "st",
        "runs/ant_tag_st_cdens_terminal/"
        "20260902_183959_seed1_terminal_v1_dist0_noent_frozen_fixed"
        "/models/st_agent.zip",
        "4_train_rl_st",
        "SetTransformerFeaturesExtractor",
    ),
]


def _recorded_extractor_class(zip_path):
    """The extractor class string SB3 wrote into the checkpoint's `data`."""
    with zipfile.ZipFile(zip_path) as archive:
        data = json.loads(archive.read("data").decode())
    return data["policy_kwargs"]["features_extractor_class"]


@pytest.mark.parametrize("label,relative_path,module_name,class_name",
                         _CHECKPOINTS,
                         ids=[entry[0] for entry in _CHECKPOINTS])
def test_saved_checkpoint_still_loads(ant_tag, label, relative_path,
                                      module_name, class_name):
    """A checkpoint on disk must still load. This is the costliest invariant.

    SB3 does not store the extractor's source; it stores the CLASS, so the
    zip carries a literal string such as
    `<class '4_train_rl_cgf.WeightedCGFFeaturesExtractor'>` and unpickling is
    `getattr(import_module('4_train_rl_cgf'), 'WeightedCGFFeaturesExtractor')`.
    Move the class to set_transformer/rl/ without re-exporting it from its
    original module and those checkpoints become unloadable, with no way to
    regenerate them short of re-running multi-million-step training. Counted
    on 2026-09-03, experiments/ant_tag/runs/ holds 2788 zips, and 2074 of
    them -- 1257 keyed to 4_train_rl_cgf and 817 to 4_train_rl_gaussian --
    resolve their extractor through exactly that module lookup. The
    module-keyed set is the large majority of the corpus, not a marginal
    case; see test_module_qualified_checkpoints_exist_for_cgf_and_gaussian.

    The load needs no env: PPO.load restores the saved observation and action
    spaces (the eval scripts pass an env only because they then roll it out).
    """
    from stable_baselines3 import PPO

    path = _ANT_TAG_DIR / relative_path
    if not path.exists():
        pytest.skip(f"no {label} checkpoint on disk at {path}")

    recorded = _recorded_extractor_class(path)
    assert class_name in recorded, (
        f"{path} records {recorded}; this test targets {class_name}")

    model = PPO.load(str(path), device="cpu")
    extractor = model.policy.features_extractor
    assert type(extractor).__name__ == class_name

    if recorded.startswith("<class '__main__."):
        # A run launched as `python3 4_train_rl_st.py ...` put the class in
        # __main__, and cloudpickle serializes a __main__ class BY VALUE. Such
        # a zip carries its own copy of the extractor and needs no module
        # lookup at all, so a move cannot break it. Asserting class identity
        # here would be wrong: the reconstructed class is a distinct object.
        assert type(extractor).__module__ == "__main__"
    else:
        # The dangerous case, and 2074 of the 2788 zips on disk: the class was
        # pickled BY REFERENCE, so the load ran
        # getattr(import_module(module_name), class_name). It must resolve to
        # the very class the module exposes today.
        assert type(extractor) is getattr(ant_tag[module_name], class_name)
    # The observation contract the saved policy was built against.
    space = model.observation_space
    assert set(space.spaces) == {"obs", "particles", "weights"}
    # features_dim = base obs + the encoding, so the encoder really did
    # contribute features and the policy's first Linear still fits.
    assert extractor.features_dim > space["obs"].shape[0]
    assert (model.policy.mlp_extractor.policy_net[0].in_features
            == extractor.features_dim)


def test_module_qualified_checkpoints_exist_for_cgf_and_gaussian():
    """At least one checkpoint really is keyed to a module, not __main__.

    If every zip on disk said `__main__.<Class>`, the test above would not
    actually exercise the module lookup that a move breaks. This asserts the
    dangerous case is present in the corpus, so the gate has teeth.

    It is not a marginal case. Of the 2788 zips under
    experiments/ant_tag/runs/ on 2026-09-03, 1257 are keyed to
    4_train_rl_cgf and 817 to 4_train_rl_gaussian -- 2074 at risk -- against
    714 keyed to __main__ (which cloudpickle embedded by value and which a
    move therefore cannot break). No zip is keyed to 4_train_rl_st, so the
    ST arm's only coverage of the module-lookup path is the checkpoint the
    smoke-train test writes.
    """
    if not _RUNS_DIR.exists():
        pytest.skip("no runs/ directory on disk")
    keyed = set()
    for path in _RUNS_DIR.glob("*/*/models/*_agent.zip"):
        try:
            recorded = _recorded_extractor_class(path)
        except Exception:  # noqa: BLE001 - a truncated zip is not this test's job
            continue
        if not recorded.startswith("<class '__main__."):
            keyed.add(recorded)
    assert any("4_train_rl_cgf.WeightedCGFFeaturesExtractor" in entry
               for entry in keyed), keyed
    assert any("4_train_rl_gaussian.WeightedGaussianFeaturesExtractor" in entry
               for entry in keyed), keyed


# --------------------------------------------------------------------------
# 3. Golden numeric forward outputs
# --------------------------------------------------------------------------

#: The real Ant-Tag dict observation: 31-D base obs (qpos 15 + qvel 14 +
#: target_xy 2), 100 particles of 2 coordinates, one weight per particle.
_OBS_DIM = 31
_NUM_PARTICLES = 100
_PARTICLE_DIM = 2
_GOLDEN_SEED = 20260903
_BATCH = 3

# ==========================================================================
# GOLDEN LITERALS -- recorded on 2026-09-03 against the PRE-MOVE Ant-Tag
# code, with `python3 -m pytest` on this machine.
#
# REGENERATING THESE INSTEAD OF FIXING THE CODE DEFEATS THE ENTIRE PURPOSE
# OF THIS FILE. They exist so that moving an extractor into
# set_transformer/rl/ cannot quietly change what the policy sees. If one of
# them fails after a refactor, the refactor changed the encoder's arithmetic
# -- find out why. Only a deliberate, reviewed change to the encoder maths
# justifies new numbers, and then the commit must say so.
# ==========================================================================

_CGF_GOLDEN = {
    "linspace_all_dims": [
        [0.08528884, 0.21231104, -0.00612006, 0.01762971, 0.01742190, 0.02182292, 0.16726352, 0.21767101],
        [0.14916712, 0.14441548, 0.01850028, 0.01107215, 0.00804056, 0.02656202, 0.10751837, 0.21404235],
        [0.39502481, 0.19853552, 0.11368072, 0.02348744, -0.02034794, -0.02133344, -0.02375596, 0.06498282],
    ],
    "linspace_first_dim": [
        [0.08522239, 0.02638904, -0.00582739, -0.01004322, 0.01657970, 0.07381929, 0.16782691, 0.28679013],
        [0.14946733, 0.06797249, 0.01857938, -0.00125738, 0.00755761, 0.04314344, 0.10786518, 0.19377132],
        [0.39419875, 0.23660040, 0.11456717, 0.02802037, -0.02159378, -0.03949824, -0.02325911, 0.01267237],
    ],
    "spread": [
        [0.03569504, 0.01480673, -0.00076114, -0.01224474, -0.01179755, 0.00962356, 0.03908484, 0.05006621],
        [0.01857599, 0.00820965, 0.00422193, 0.00203664, 0.00285836, 0.01242257, 0.02551020, 0.02852088],
        [-0.03020138, -0.04264155, -0.02140657, 0.02023833, 0.05643033, 0.06693003, 0.04646067, 0.00681280],
    ],
    "random": [
        [1.01186132, 0.85770321, 0.22942546, 1.91016304, 0.46684462, 0.48412579, 0.24916698, 0.27255985],
        [0.73559070, 0.63518029, 0.21151152, 1.23608148, 0.52110595, 0.32530025, 0.35989133, 0.33845398],
        [0.34150136, 1.11019874, 0.02043071, 1.05519593, 1.18085170, 0.38758326, 0.46381402, 0.81264472],
    ],
}

_ST_GOLDEN = {
    True: [
        [0.33129507, -0.64387918, -0.53287017, -0.06956580, 0.38028365, -0.61633015, -0.50711429, -0.07659000],
        [0.28097954, -0.63554072, -0.51972103, -0.01920307, 0.33202043, -0.60496414, -0.49579802, -0.03261858],
        [0.23875013, -0.63284910, -0.50955296, 0.02498850, 0.29064122, -0.60165703, -0.48713222, 0.00874037],
    ],
    False: [
        [-0.06067315, -0.40480542, 0.22168627, 0.24208364, -0.18364693, -0.40919945, 0.21760117, 0.22808379],
        [-0.14377379, -0.32181871, 0.27354351, 0.24564603, -0.25948036, -0.32649848, 0.27168062, 0.23338336],
        [-0.23904656, -0.25521335, 0.33792272, 0.25325596, -0.34431618, -0.25981867, 0.32571363, 0.23768443],
    ],
}

_GAUSSIAN_GOLDEN = [
    [0.09444176, -0.07949370, 0.38281730, 0.61422032, -0.10751283],
    [0.03197220, -0.04397332, 0.34276617, 0.47589031, -0.07902230],
    [-0.17204145, -0.13578855, 0.42028704, 0.40101910, -0.02198861],
]

#: Small extractor geometries, chosen so the golden tables stay readable.
#: 8 CGF features is the smallest count "spread" accepts (8 directions).
_NUM_CGF_FEATURES = 8
_ST_KWARGS = dict(num_encodings=2, dim_encoder=4, num_inds=4, dim_hidden=16,
                  num_heads=2, ln=True)
_ARENA_SCALE = 4.5


def _golden_obs_space():
    """The fixed synthetic Dict space every golden test builds against."""
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf,
                                    (_NUM_PARTICLES, _PARTICLE_DIM),
                                    np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (_NUM_PARTICLES,), np.float32),
    })


def _golden_batch():
    """A fixed deterministic input batch, from an explicit torch.Generator.

    A local Generator, not the global RNG: the extractors consume the global
    RNG during construction, so a shared stream would make the input depend
    on which extractor was built first.
    """
    generator = torch.Generator().manual_seed(_GOLDEN_SEED)
    # The draw ORDER is part of the golden definition: the three tensors come
    # off one stream, so reordering these lines changes every literal below.
    obs = torch.randn(_BATCH, _OBS_DIM, generator=generator)
    # Coordinates of the same order as the 7.0-unit Ant-Tag arena.
    particles = 3.0 * torch.randn(_BATCH, _NUM_PARTICLES, _PARTICLE_DIM,
                                  generator=generator)
    raw_weights = torch.rand(_BATCH, _NUM_PARTICLES,
                             generator=generator) + 0.01
    return {
        "obs": obs,
        "particles": particles,
        "weights": raw_weights / raw_weights.sum(dim=1, keepdim=True),
    }


def _assert_matches_golden(actual, golden):
    torch.testing.assert_close(
        actual, torch.tensor(golden, dtype=actual.dtype),
        rtol=1e-6, atol=1e-6,
        msg=lambda text: (
            "Encoder output no longer matches the golden literals recorded "
            "on 2026-09-03 against the pre-move code. Fix the code, do not "
            f"re-record the numbers.\n{text}"),
    )


# `_golden_batch` is rebuilt per test rather than shared, so no test can
# mutate the input another one reads.
def _cgf_extractor(module, t_init_mode, t_frozen):
    torch.manual_seed(_GOLDEN_SEED)
    return module.WeightedCGFFeaturesExtractor(
        _golden_obs_space(),
        num_cgf_features=_NUM_CGF_FEATURES,
        arena_scale=_ARENA_SCALE,
        t_init_mode=t_init_mode,
        # 1.0, not the script default 0.1: it puts the CGF values well clear
        # of atol so the literals really test the arithmetic.
        t_init_scale=1.0,
        t_clamp=2.0,
        exp_arg_clamp=20.0,
        t_frozen=t_frozen,
    )


@pytest.mark.parametrize("t_init_mode", sorted(_CGF_GOLDEN))
@pytest.mark.parametrize("t_frozen", [False, True])
def test_cgf_forward_matches_golden(ant_tag, t_init_mode, t_frozen):
    """The weighted CGF encoder's arithmetic, pinned per t_init_mode.

    log sum_i w_i exp(<t_j, x_i / arena_scale>) has five separate steps that
    a refactor can silently reorder: the arena division, the NaN/inf
    sanitizing, the weight renormalization, the elementwise clamp on t and
    the clamp on the exponent argument. None of them raises when wrong; they
    just change what the policy sees. Each t_init_mode also builds t from a
    different recipe, and "spread" is the only one that asserts
    particle_dim == 2, so all four are covered.

    t_frozen only decides whether t is a buffer or a Parameter, so the
    forward output must be bit-identical either way -- that equality is part
    of what this pins.
    """
    extractor = _cgf_extractor(ant_tag["4_train_rl_cgf"], t_init_mode,
                               t_frozen)
    assert extractor.features_dim == _OBS_DIM + _NUM_CGF_FEATURES
    with torch.no_grad():
        output = extractor(_golden_batch())
    assert output.shape == (_BATCH, _OBS_DIM + _NUM_CGF_FEATURES)
    # The base observation is passed through untouched, ahead of the encoding.
    torch.testing.assert_close(output[:, :_OBS_DIM], _golden_batch()["obs"])
    _assert_matches_golden(output[:, _OBS_DIM:], _CGF_GOLDEN[t_init_mode])


@pytest.mark.parametrize("t_init_mode", sorted(_CGF_GOLDEN))
def test_cgf_t_frozen_controls_gradient_not_geometry(ant_tag, t_init_mode):
    """--t_frozen must fix t without changing its value or its state_dict key.

    Frozen registers `t_values` as a buffer and unfrozen as a Parameter. Both
    have to keep the same name, shape and initial value: the name is what a
    saved policy's state_dict is keyed by, and TNormLoggingCallback reads
    `extractor.t_values` by attribute for the cgf/t_norm_* metrics.
    """
    unfrozen = _cgf_extractor(ant_tag["4_train_rl_cgf"], t_init_mode, False)
    frozen = _cgf_extractor(ant_tag["4_train_rl_cgf"], t_init_mode, True)

    assert isinstance(unfrozen.t_values, torch.nn.Parameter)
    assert unfrozen.t_values.requires_grad
    assert not isinstance(frozen.t_values, torch.nn.Parameter)
    assert not frozen.t_values.requires_grad
    assert "t_values" in unfrozen.state_dict()
    assert "t_values" in frozen.state_dict()
    assert unfrozen.t_values.shape == (_NUM_CGF_FEATURES, _PARTICLE_DIM)
    torch.testing.assert_close(frozen.t_values, unfrozen.t_values.detach())
    # PPO must be able to move t in the unfrozen arm and not in the frozen one.
    assert [name for name, _ in unfrozen.named_parameters()] == ["t_values"]
    assert list(frozen.named_parameters()) == []


def test_cgf_spread_mode_rejects_non_2d_particles(ant_tag):
    """"spread" hardcodes 8 planar directions, so it must reject other dims.

    Odd-Even beliefs are 1-D. A shared extractor that silently accepted them
    under t_init_mode="spread" would build a meaningless t.
    """
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (10, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (10,), np.float32),
    })
    with pytest.raises(ValueError, match="2D particles"):
        ant_tag["4_train_rl_cgf"].WeightedCGFFeaturesExtractor(
            space, num_cgf_features=8, t_init_mode="spread")
    with pytest.raises(ValueError, match="divisible by 8"):
        ant_tag["4_train_rl_cgf"].WeightedCGFFeaturesExtractor(
            _golden_obs_space(), num_cgf_features=12, t_init_mode="spread")
    with pytest.raises(ValueError, match="Unknown t_init_mode"):
        ant_tag["4_train_rl_cgf"].WeightedCGFFeaturesExtractor(
            _golden_obs_space(), t_init_mode="not_a_mode")


@pytest.mark.parametrize("weight_channel", [True, False])
def test_st_forward_matches_golden(ant_tag, weight_channel):
    """The Set Transformer encoder's arithmetic, both weight conventions.

    --st_weight_channel appends the normalized weight scaled by N as a third
    input channel, so the encoder's input dim is particle_dim + 1. It has to
    agree with how a pretrained checkpoint was built, and it changes the
    numbers, so both settings are pinned. The literals also cover the
    scaling (w * N, a uniform belief feeding 1.0), which is easy to write as
    plain w and impossible to notice.
    """
    torch.manual_seed(_GOLDEN_SEED)
    extractor = ant_tag["4_train_rl_st"].SetTransformerFeaturesExtractor(
        _golden_obs_space(), arena_scale=_ARENA_SCALE,
        weight_channel=weight_channel, pretrained_st_model_path=None,
        st_frozen=False, **_ST_KWARGS)

    st_output_dim = _ST_KWARGS["num_encodings"] * _ST_KWARGS["dim_encoder"]
    assert extractor.features_dim == _OBS_DIM + st_output_dim
    assert extractor.dim_input == _PARTICLE_DIM + (1 if weight_channel else 0)
    assert extractor.num_particles == _NUM_PARTICLES

    with torch.no_grad():
        output = extractor(_golden_batch())
    assert output.shape == (_BATCH, _OBS_DIM + st_output_dim)
    torch.testing.assert_close(output[:, :_OBS_DIM], _golden_batch()["obs"])
    _assert_matches_golden(output[:, _OBS_DIM:], _ST_GOLDEN[weight_channel])
    # STFeatureLoggingCallback reads this attribute; it must be a detached
    # cache, not a buffer, or it would enter the saved policy state_dict.
    assert extractor.last_st_features is not None
    assert not extractor.last_st_features.requires_grad
    assert "last_st_features" not in extractor.state_dict()


def test_gaussian_forward_matches_golden(ant_tag):
    """The weighted mean/covariance encoder: 5 features for 2-D particles.

    This extractor has no parameters, so nothing about it is learned and any
    numeric drift is a bug rather than a different random draw. The five
    features are [mean_x, mean_y, var_x, var_y, cov_xy] in that order, and
    the order is what the policy's first Linear was trained against.
    """
    extractor = ant_tag["4_train_rl_gaussian"].WeightedGaussianFeaturesExtractor(
        _golden_obs_space(), arena_scale=_ARENA_SCALE)
    num_gaussian_features = 5
    assert extractor.features_dim == _OBS_DIM + num_gaussian_features
    assert list(extractor.parameters()) == []

    with torch.no_grad():
        output = extractor(_golden_batch())
    assert output.shape == (_BATCH, _OBS_DIM + num_gaussian_features)
    torch.testing.assert_close(output[:, :_OBS_DIM], _golden_batch()["obs"])
    _assert_matches_golden(output[:, _OBS_DIM:], _GAUSSIAN_GOLDEN)
    # Variances are clamped at 0: floating-point cancellation can push a
    # weighted variance slightly negative, and a negative "variance" feature
    # would be fed straight to the policy.
    assert bool((output[:, _OBS_DIM + 2:_OBS_DIM + 4] >= 0.0).all())


@pytest.mark.parametrize("arm", ["cgf", "gaussian", "st"])
def test_extractors_sanitize_nan_and_inf_particles(ant_tag, arm):
    """All three encoders must survive a NaN/inf particle and a zero-mass
    weight vector.

    A particle filter whose weights all underflow, or an env glitch that
    emits inf, must not put NaN into the policy: PPO would poison every
    subsequent gradient and the run would look like a learning failure. Each
    forward starts with nan_to_num plus a `+ 1e-8` in the weight
    normalization, and that guard is easy to drop in a rewrite.
    """
    batch = _golden_batch()
    batch["particles"][0, 0, 0] = float("nan")
    batch["particles"][0, 1, 0] = float("inf")
    batch["particles"][0, 2, 0] = float("-inf")
    batch["weights"][1] = 0.0                      # zero total mass
    batch["weights"][2, 0] = float("nan")

    if arm == "cgf":
        extractor = _cgf_extractor(ant_tag["4_train_rl_cgf"], "spread", False)
    elif arm == "gaussian":
        extractor = ant_tag["4_train_rl_gaussian"].WeightedGaussianFeaturesExtractor(
            _golden_obs_space(), arena_scale=_ARENA_SCALE)
    else:
        torch.manual_seed(_GOLDEN_SEED)
        extractor = ant_tag["4_train_rl_st"].SetTransformerFeaturesExtractor(
            _golden_obs_space(), arena_scale=_ARENA_SCALE, weight_channel=True,
            pretrained_st_model_path=None, st_frozen=False, **_ST_KWARGS)

    with torch.no_grad():
        output = extractor(batch)
    assert bool(torch.isfinite(output[:, _OBS_DIM:]).all()), (
        f"{arm} encoder emitted a non-finite feature from a degenerate "
        "particle set; the nan_to_num / epsilon guards are gone")

# --------------------------------------------------------------------------
# 3b. Golden numerics for the WEIGHT SANITIZE path
# --------------------------------------------------------------------------
# The block below the `forward` signature of both WeightedCGFFeaturesExtractor
# and SetTransformerFeaturesExtractor is byte-for-byte the same four lines:
#
#     weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
#     weights = torch.clamp(weights, min=0.0)
#     weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
#
# 4_train_rl_st.py's comment states outright that it must stay identical to
# the CGF arm's, because that is what makes the two arms information-matched.
# The tests above cannot see any of it: _golden_batch feeds weights that
# already sum to 1 and contain nothing degenerate, so the renormalization is
# a no-op on that input and DELETING IT ENTIRELY leaves every literal above
# unchanged. That was a real hole; these cases close it.
#
# PITFALLS.md section 3 is this exact class of bug: dividing by
# (sum + 1e-12) rather than the true sum turned a loss of 0.157 into 2831.
# The failure mode is silent -- no exception, no NaN, just different numbers
# reaching the policy -- so it has to be pinned numerically.


def _degenerate_weight_batches():
    """Six weight/particle vectors that each exercise one sanitize line.

    Built off _golden_batch, so `obs` and `particles` are unchanged except in
    the one case that perturbs particles.
    """
    base = _golden_batch()
    cases = {}

    def _copy():
        return {key: value.clone() for key, value in base.items()}

    # Sums to 3.7 per row, not 1. If the renormalization is dropped, every
    # CGF feature shifts by log(3.7) = +1.308.
    case = _copy()
    case["weights"] = case["weights"] * 3.7
    cases["unnormalized_scaled"] = case

    # Sharply non-uniform AND unnormalized: exponential decay at three
    # different rates. Catches a renormalization that is applied over the
    # wrong axis, which a uniform-ish vector would hide.
    case = _copy()
    index = torch.arange(_NUM_PARTICLES, dtype=torch.float32)
    case["weights"] = torch.stack([torch.exp(-index / scale)
                                   for scale in (3.0, 10.0, 40.0)])
    cases["unnormalized_decay"] = case

    # Negative mass, including one large negative entry.
    case = _copy()
    case["weights"][:, 0] = -0.5
    case["weights"][:, 7] = -1e-3
    case["weights"][1, 50] = -12.0
    cases["negative"] = case

    case = _copy()
    case["weights"][0, 0] = float("nan")
    case["weights"][1, 1] = float("inf")
    case["weights"][2, 2] = float("-inf")
    cases["nan_inf_weights"] = case

    case = _copy()
    case["particles"][0, 0, 0] = float("nan")
    case["particles"][1, 1, 0] = float("inf")
    case["particles"][2, 2, 1] = float("-inf")
    cases["nan_inf_particles"] = case

    # Total mass 1e-8, i.e. exactly the epsilon in the denominator, so the
    # normalized weights come out at HALF their true value. This is the only
    # case that pins the epsilon's magnitude: at 1e-12 the same input would
    # normalize to ~1.0 total mass and the CGF features would shift by
    # +log(2) = 0.693.
    case = _copy()
    case["weights"] = case["weights"] * 1e-8
    cases["near_epsilon_mass"] = case

    # Every particle dead. The epsilon is the only thing between this and a
    # 0/0 NaN reaching PPO.
    case = _copy()
    case["weights"] = torch.zeros_like(case["weights"])
    cases["zero_mass"] = case

    return cases


# ==========================================================================
# GOLDEN LITERALS -- recorded on 2026-09-03 against the PRE-MOVE Ant-Tag
# code. REGENERATING THESE INSTEAD OF FIXING THE CODE DEFEATS THE ENTIRE
# PURPOSE OF THIS FILE. See the banner above _CGF_GOLDEN.
# ==========================================================================

_CGF_SANITIZE_GOLDEN = {
    "unnormalized_scaled": [
        [0.08528884, 0.21231104, -0.00612018, 0.01762971, 0.01742190, 0.02182292, 0.16726352, 0.21767101],
        [0.14916712, 0.14441538, 0.01850016, 0.01107203, 0.00804056, 0.02656202, 0.10751837, 0.21404235],
        [0.39502481, 0.19853541, 0.11368062, 0.02348744, -0.02034806, -0.02133350, -0.02375596, 0.06498271],
    ],
    "unnormalized_decay": [
        [-0.01808659, 0.03375411, -0.07431272, -0.01600852, 0.04701960, 0.10819175, 0.33376172, 0.36310616],
        [0.38089928, 0.23231141, 0.12121017, 0.02909062, -0.02502072, -0.03229195, -0.04366248, 0.05118438],
        [0.31435913, 0.19689277, 0.08447520, 0.02331870, -0.01266491, -0.02275203, 0.00587451, 0.05020737],
    ],
    "negative": [
        [0.08354960, 0.21885782, -0.00741788, 0.01879010, 0.01801006, 0.01937138, 0.17071846, 0.21503909],
        [0.12963158, 0.13811621, 0.00963277, 0.00950466, 0.01093314, 0.03197961, 0.12093177, 0.22758798],
        [0.39554915, 0.19875209, 0.11389595, 0.02350898, -0.02041729, -0.02132291, -0.02410533, 0.06517171],
    ],
    "nan_inf_weights": [
        [0.08705435, 0.21394967, -0.00531666, 0.01797142, 0.01716027, 0.02089336, 0.16612299, 0.21602407],
        [0.14653438, 0.14199269, 0.01710777, 0.01051293, 0.00854172, 0.02831417, 0.10998941, 0.21794234],
        [0.39981472, 0.19286561, 0.11560258, 0.02234531, -0.02087693, -0.01814328, -0.02564270, 0.07151819],
    ],
    "nan_inf_particles": [
        [0.08676813, 0.21231104, -0.00529832, 0.01762971, 0.01710191, 0.02182292, 0.16559774, 0.21767101],
        [0.14185625, 0.14441548, 0.01464463, 0.01107215, 0.00952450, 0.02656202, 0.11565100, 0.21404235],
        [0.39502481, 0.20016426, 0.11368072, 0.02371518, -0.02034794, -0.02175282, -0.02375596, 0.06445587],
    ],
    "near_epsilon_mass": [
        [-0.60785824, -0.48083615, -0.69926721, -0.67551738, -0.67572528, -0.67132413, -0.52588367, -0.47547618],
        [-0.54398006, -0.54873168, -0.67464691, -0.68207502, -0.68510664, -0.66658515, -0.58562881, -0.47910482],
        [-0.29812238, -0.49461168, -0.57946646, -0.66965973, -0.71349514, -0.71448064, -0.71690315, -0.62816435],
    ],
    "zero_mass": [
        [-18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100],
        [-18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100],
        [-18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100, -18.42068100],
    ],
}

_ST_SANITIZE_GOLDEN = {
    "unnormalized_scaled": [
        [0.33129519, -0.64387929, -0.53287017, -0.06956577, 0.38028380, -0.61633003, -0.50711441, -0.07659024],
        [0.28097981, -0.63554084, -0.51972115, -0.01920316, 0.33202070, -0.60496414, -0.49579808, -0.03261882],
        [0.23875025, -0.63284910, -0.50955307, 0.02498847, 0.29064131, -0.60165703, -0.48713228, 0.00874025],
    ],
    "unnormalized_decay": [
        [0.34681690, -0.71733260, -0.51959991, -0.04862627, 0.39456233, -0.68624949, -0.49284723, -0.05635411],
        [0.30452129, -0.67591965, -0.52241886, -0.03081661, 0.35477990, -0.64622176, -0.49849090, -0.04299730],
        [0.23979980, -0.63589489, -0.50951028, 0.02499017, 0.29176697, -0.60480177, -0.48711297, 0.00871408],
    ],
    "negative": [
        [0.33170119, -0.64457345, -0.53301740, -0.06973627, 0.38068601, -0.61703253, -0.50724840, -0.07674563],
        [0.28070813, -0.63639152, -0.51958501, -0.01860142, 0.33186024, -0.60576928, -0.49567559, -0.03217188],
        [0.23877668, -0.63292956, -0.50956225, 0.02499640, 0.29067060, -0.60174119, -0.48714247, 0.00874710],
    ],
    "nan_inf_weights": [
        [0.33146882, -0.64408875, -0.53291810, -0.06964052, 0.38045722, -0.61654413, -0.50716007, -0.07666168],
        [0.28100014, -0.63568425, -0.51967156, -0.01913670, 0.33205527, -0.60510159, -0.49574879, -0.03257638],
        [0.23899728, -0.63286328, -0.50947714, 0.02485198, 0.29087585, -0.60168302, -0.48705009, 0.00861406],
    ],
    "nan_inf_particles": [
        [0.32832721, -0.64450264, -0.53190672, -0.06624344, 0.37717894, -0.61683655, -0.50632167, -0.07337967],
        [0.28775766, -0.63427687, -0.52163088, -0.02696016, 0.33806551, -0.60439694, -0.49749616, -0.03900400],
        [0.23855573, -0.63277578, -0.50956535, 0.02513534, 0.29045501, -0.60159254, -0.48714820, 0.00888216],
    ],
    "near_epsilon_mass": [
        [0.28444496, -0.78582060, -0.53385067, 0.01387599, 0.33864671, -0.75216818, -0.50730979, 0.00626752],
        [0.18962342, -0.76556170, -0.51506150, 0.08640569, 0.24813154, -0.73090041, -0.49279550, 0.07016653],
        [0.10481688, -0.75174510, -0.49919632, 0.16072348, 0.16418770, -0.72017014, -0.47823003, 0.14174283],
    ],
    "zero_mass": [
        [-0.01717599, -1.02230299, -0.47611484, 0.49208486, 0.08925074, -1.01362741, -0.41511795, 0.45804787],
        [-0.20772681, -0.95288563, -0.45347825, 0.56823611, -0.11978740, -0.94687891, -0.40264568, 0.54918987],
        [-0.35240775, -0.88542068, -0.43522575, 0.64575005, -0.27499270, -0.88319063, -0.38733697, 0.62765110],
    ],
}

#: The t_init_mode the sanitize literals were recorded against: the CLI
#: default, and the one 882 of the 1381 CGF checkpoints on disk were trained
#: with (the other 499 use "spread"). The sanitize block is shared by all
#: four modes, so one mode is enough to pin it.
_SANITIZE_T_INIT_MODE = "linspace_all_dims"


def _sanitize_st_extractor(ant_tag):
    torch.manual_seed(_GOLDEN_SEED)
    return ant_tag["4_train_rl_st"].SetTransformerFeaturesExtractor(
        _golden_obs_space(), arena_scale=_ARENA_SCALE, weight_channel=True,
        pretrained_st_model_path=None, st_frozen=False, **_ST_KWARGS)


@pytest.mark.parametrize("case", sorted(_CGF_SANITIZE_GOLDEN))
def test_cgf_weight_sanitize_matches_golden(ant_tag, case):
    """Numeric goldens for the CGF arm's weight/particle sanitize block.

    One case per line of that block: two unnormalized vectors pin the
    renormalization, `negative` pins the clamp at 0, the two nan/inf cases pin
    nan_to_num and its fill values, `near_epsilon_mass` pins the magnitude of
    the `+ 1e-8` in the denominator, and `zero_mass` pins the
    `clamp(min=1e-8)` inside the log.
    """
    extractor = _cgf_extractor(ant_tag["4_train_rl_cgf"],
                               _SANITIZE_T_INIT_MODE, False)
    with torch.no_grad():
        output = extractor(_degenerate_weight_batches()[case])
    _assert_matches_golden(output[:, _OBS_DIM:], _CGF_SANITIZE_GOLDEN[case])


@pytest.mark.parametrize("case", sorted(_ST_SANITIZE_GOLDEN))
def test_st_weight_sanitize_matches_golden(ant_tag, case):
    """The same seven cases through the Set Transformer arm.

    The ST arm carries its own copy of the sanitize block plus the `w * N`
    scaling. Pinning both arms on the same inputs is what keeps them
    information-matched: a change to one and not the other would leave every
    other test in this file green.
    """
    extractor = _sanitize_st_extractor(ant_tag)
    with torch.no_grad():
        output = extractor(_degenerate_weight_batches()[case])
    _assert_matches_golden(output[:, _OBS_DIM:], _ST_SANITIZE_GOLDEN[case])


@pytest.mark.parametrize("arm", ["cgf", "st"])
def test_negative_weights_behave_exactly_like_zero(ant_tag, arm):
    """clamp(min=0.0) must map a negative weight to 0, not to |w| or to w.

    Stronger than a literal on its own: whatever the encoder does, feeding
    negative mass must be indistinguishable from feeding zero mass at those
    positions. A dropped clamp lets a negative weight cancel real probability
    mass out of the normalizer, which no finiteness check would notice.
    """
    if arm == "cgf":
        extractor = _cgf_extractor(ant_tag["4_train_rl_cgf"],
                                   _SANITIZE_T_INIT_MODE, False)
    else:
        extractor = _sanitize_st_extractor(ant_tag)

    negative = _degenerate_weight_batches()["negative"]
    zeroed = {key: value.clone() for key, value in negative.items()}
    zeroed["weights"] = torch.where(zeroed["weights"] < 0.0,
                                    torch.zeros_like(zeroed["weights"]),
                                    zeroed["weights"])
    assert bool((negative["weights"] < 0.0).any())
    with torch.no_grad():
        torch.testing.assert_close(extractor(negative), extractor(zeroed),
                                   rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("arm", ["cgf", "st"])
def test_nan_to_num_fill_values_are_exactly_zero_one_minus_one(ant_tag, arm):
    """The nan_to_num fill values are part of the contract, not arbitrary.

    Weights use nan=0.0, posinf=0.0, neginf=0.0 -- a non-finite weight is
    treated as no evidence. Particles use nan=0.0, posinf=1.0, neginf=-1.0,
    applied AFTER the arena division, so an infinite coordinate is pinned to
    the arena edge rather than to inf. Changing any of those six numbers is
    silent, and the finiteness test below would still pass.
    """
    if arm == "cgf":
        extractor = _cgf_extractor(ant_tag["4_train_rl_cgf"],
                                   _SANITIZE_T_INIT_MODE, False)
    else:
        extractor = _sanitize_st_extractor(ant_tag)
    cases = _degenerate_weight_batches()

    weights_case = cases["nan_inf_weights"]
    equivalent = {key: value.clone() for key, value in weights_case.items()}
    equivalent["weights"] = torch.nan_to_num(
        equivalent["weights"], nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        torch.testing.assert_close(extractor(weights_case),
                                   extractor(equivalent),
                                   rtol=1e-6, atol=1e-6)

    particles_case = cases["nan_inf_particles"]
    equivalent = {key: value.clone() for key, value in particles_case.items()}
    # The fills are applied to the SCALED particles, so the equivalent raw
    # coordinate is arena_scale * (+/-1).
    equivalent["particles"] = torch.nan_to_num(
        equivalent["particles"], nan=0.0,
        posinf=_ARENA_SCALE * 1.0, neginf=_ARENA_SCALE * -1.0)
    with torch.no_grad():
        torch.testing.assert_close(extractor(particles_case),
                                   extractor(equivalent),
                                   rtol=1e-6, atol=1e-6)


def test_cgf_and_st_normalize_weights_identically(ant_tag):
    """The CGF and ST arms must derive the SAME normalized weight vector.

    This is the claim that makes the two arms information-matched, and it is
    stated only in a comment in 4_train_rl_st.py today.

    The ST arm's weights are read straight off the tensor it hands its
    encoder (a forward pre-hook), divided back by N. The CGF arm has no
    submodule to hook, and its forward exposes only linear functionals of w
    -- log sum_i w_i exp(<t_j, x_i>) cannot be inverted for 100 unknowns from
    8 features. So the particles are placed in two groups at normalized
    coordinates (+1, 0) and (-1, 0), which makes the CGF features an
    invertible 2x2 system in the two GROUP masses:

        exp(cgf_j) = W_A e^{tau_j} + W_B e^{-tau_j}

    With t_init_mode="linspace_all_dims" and 8 features, t_1 points along
    dim 1 alone, so exp(cgf_1) is the total mass directly, and t_0 = (-1, 0)
    gives the second equation. Group masses, not per-particle weights, are
    what is recoverable -- but they are enough: they depend on every
    sanitize step, and they are compared against the ST arm's own numbers on
    the same degenerate input.
    """
    cgf = _cgf_extractor(ant_tag["4_train_rl_cgf"], "linspace_all_dims", False)
    st = _sanitize_st_extractor(ant_tag)
    captured = {}
    st.encoder.register_forward_pre_hook(
        lambda module, args: captured.__setitem__("input", args[0].detach()))

    group = 40
    euler = float(np.e)
    for case, weights in (
            ("decay", _degenerate_weight_batches()["unnormalized_decay"]["weights"]),
            ("messy", _degenerate_weight_batches()["nan_inf_weights"]["weights"] * 3.7),
    ):
        particles = torch.zeros(_BATCH, _NUM_PARTICLES, _PARTICLE_DIM)
        particles[:, :group, 0] = _ARENA_SCALE
        particles[:, group:, 0] = -_ARENA_SCALE
        batch = {"obs": torch.zeros(_BATCH, _OBS_DIM),
                 "particles": particles, "weights": weights.clone()}

        with torch.no_grad():
            features = cgf(batch)[:, _OBS_DIM:]
            st(batch)

        total = torch.exp(features[:, 1])
        mass_b = ((torch.exp(features[:, 0]) - total / euler)
                  / (euler - 1.0 / euler))
        mass_a = total - mass_b

        channel = captured["input"][..., -1] / _NUM_PARTICLES
        torch.testing.assert_close(total, channel.sum(dim=1),
                                   rtol=1e-5, atol=1e-5, msg=f"{case}: total")
        torch.testing.assert_close(mass_a, channel[:, :group].sum(dim=1),
                                   rtol=1e-4, atol=1e-5, msg=f"{case}: group A")
        torch.testing.assert_close(mass_b, channel[:, group:].sum(dim=1),
                                   rtol=1e-4, atol=1e-5, msg=f"{case}: group B")
        # Both arms must end up with a genuine probability vector.
        torch.testing.assert_close(total, torch.ones(_BATCH),
                                   rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# 4. Golden seeded dict-obs rollout
# --------------------------------------------------------------------------

#: Recorded on 2026-09-03, pre-move, over the 24-step rollout below.
#:
#: These are COARSE summary statistics with a loose tolerance, on purpose.
#: MuJoCo's floating-point output differs slightly between machines and
#: builds, so pinning raw particle coordinates would fingerprint this
#: workstation's libmujoco rather than the pipeline. What must not change is
#: the WIRING: which filter is paired with the env, that the mapper feeds the
#: env's live geometry to the filter, that the weights are a normalized
#: posterior, that the belief concentrates on the arm the env actually chose,
#: and that obs masking still zeroes the target coordinates. A 10% band on
#: these aggregates catches all of that while tolerating platform noise.
_ROLLOUT_STEPS = 24
_ROLLOUT_MEAN_ABS_PARTICLE = 1.718993
_ROLLOUT_MEAN_PARTICLE = -1.718993
_ROLLOUT_MEDIAN_ESS = 9.795611


@pytest.fixture(scope="module")
def rollout(ant_tag):
    """One seeded 24-step rollout of the real cdens_terminal belief env."""
    cgf = ant_tag["4_train_rl_cgf"]
    variant = ant_tag["variants"].resolve("cdens_terminal")
    env = cgf.make_ant_tag_cgf_env(
        num_particles=_NUM_PARTICLES, rank=0, seed=123,
        distance_coeff=0.0, entropy_coeff=0.0, tag_bonus_coeff=0.0,
        # Full visibility, so the filter actually receives informative
        # updates inside 24 steps. At the env's real 1.0 radius the target
        # is out of range the whole time and the weights stay uniform, which
        # would test almost nothing about the weight path.
        initial_visibility_radius=100.0,
        obs_mask_indices=[-2, -1],
        apply_reward_shaping=False,
        env_id=variant.env_id,
        particle_filter_class=variant.particle_filter,
        target_speed_scale=0.0,
    )()
    try:
        # Seed the action space. An unseeded sample() would make every
        # recorded statistic a fresh random number.
        env.action_space.seed(4242)
        observation, _ = env.reset(seed=123)
        actions = [env.action_space.sample() for _ in range(_ROLLOUT_STEPS)]
        observations = [copy.deepcopy(observation)]
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            observations.append(copy.deepcopy(observation))
            if terminated or truncated:
                break
        yield env.observation_space, observations
    finally:
        env.close()


def test_rollout_dict_observation_space_is_exact(rollout):
    """The Dict observation contract every arm and eval script depends on.

    All three encoders, both eval scripts and every diagnostics probe read
    {"obs", "particles", "weights"} with these exact shapes and dtypes. A
    shared PFDictWithWeightsObservationWrapper that reordered the keys,
    changed a dtype to float64 or dropped the [0, 1] bound on weights would
    invalidate every saved policy's observation space, and SB3 rejects a
    mismatch at load time.
    """
    space, _ = rollout
    assert isinstance(space, gym.spaces.Dict)
    assert set(space.spaces) == {"obs", "particles", "weights"}
    assert space["obs"].shape == (_OBS_DIM,)
    assert space["obs"].dtype == np.float32
    assert space["particles"].shape == (_NUM_PARTICLES, _PARTICLE_DIM)
    assert space["particles"].dtype == np.float32
    assert bool(np.all(np.isneginf(space["particles"].low)))
    assert bool(np.all(np.isposinf(space["particles"].high)))
    assert space["weights"].shape == (_NUM_PARTICLES,)
    assert space["weights"].dtype == np.float32
    assert bool(np.all(space["weights"].low == 0.0))
    assert bool(np.all(space["weights"].high == 1.0))


def test_rollout_observations_are_well_formed(rollout):
    """Every step's dict obs must match the space, and the weights must be a
    normalized, finite, non-negative posterior.

    The encoders renormalize defensively, so a filter that started returning
    unnormalized or negative weights would be invisible in the features and
    would only show up as a worse policy. This asserts it at the source.
    """
    space, observations = rollout
    assert len(observations) == _ROLLOUT_STEPS + 1, "the episode ended early"
    for index, observation in enumerate(observations):
        assert set(observation) == {"obs", "particles", "weights"}, index
        for key in ("obs", "particles", "weights"):
            assert observation[key].dtype == np.float32, (index, key)
            assert observation[key].shape == space[key].shape, (index, key)
            assert np.isfinite(observation[key]).all(), (index, key)
        weights = observation["weights"]
        assert (weights >= 0.0).all(), index
        assert abs(float(weights.sum()) - 1.0) < 1e-5, index
        # obs_mask_indices=[-2, -1] hides the true target position, which is
        # the whole reason a belief encoder is needed here.
        assert observation["obs"][-2] == 0.0 and observation["obs"][-1] == 0.0


def test_rollout_summary_statistics_match_golden(rollout):
    """Coarse belief statistics, to catch a rewiring rather than MuJoCo noise.

    See the comment above _ROLLOUT_MEAN_ABS_PARTICLE for why these are loose
    aggregates and not raw coordinates. The two facts they pin are that the
    particle cloud sits on the arm the env actually sampled at this seed (the
    mean is negative in both coordinates, and equal to minus the mean
    magnitude, so no particle has crossed the origin), and that the weights
    are genuinely informative -- a median effective sample size near 10 of
    100, matching the counterweighted-den measurement. A uniform weight
    vector would read ESS = 100 and means the likelihood step is dead.
    """
    _, observations = rollout
    steps = observations[1:]
    mean_abs = float(np.mean([np.abs(o["particles"]).mean() for o in steps]))
    mean = float(np.mean([o["particles"].mean() for o in steps]))
    ess = np.array([1.0 / float(np.sum(o["weights"] ** 2)) for o in steps])

    assert mean_abs == pytest.approx(_ROLLOUT_MEAN_ABS_PARTICLE, rel=0.10)
    assert mean == pytest.approx(_ROLLOUT_MEAN_PARTICLE, rel=0.10)
    assert mean < 0.0
    assert float(np.median(ess)) == pytest.approx(_ROLLOUT_MEDIAN_ESS, rel=0.35)
    assert float(np.median(ess)) < 30.0, (
        "the particle weights are near-uniform; the filter's likelihood "
        "update is no longer reaching the belief")


def test_pf_dict_wrapper_seeding_is_deterministic(ant_tag):
    """Two independently built workers must produce the same PF stream.

    PFDictWithWeightsObservationWrapper derives a per-episode filter seed
    from (worker seed, episode index) through a SeedSequence, and it probes
    the particle dimension at construction with a separately derived seed so
    that probe cannot consume the first real episode's draws. Losing either
    half makes a run unreproducible in a way no other test would notice.
    """
    cgf = ant_tag["4_train_rl_cgf"]
    variant = ant_tag["variants"].resolve("cdens_terminal")
    kwargs = dict(
        num_particles=32, rank=3, seed=99, distance_coeff=0.0,
        entropy_coeff=0.0, tag_bonus_coeff=0.0,
        initial_visibility_radius=100.0, apply_reward_shaping=False,
        env_id=variant.env_id, particle_filter_class=variant.particle_filter,
        target_speed_scale=0.0)
    first = cgf.make_ant_tag_cgf_env(**kwargs)()
    second = cgf.make_ant_tag_cgf_env(**kwargs)()
    try:
        action = np.zeros(first.action_space.shape, dtype=np.float32)
        for reset_kwargs in ({"seed": 7}, {}):
            obs_a, _ = first.reset(**reset_kwargs)
            obs_b, _ = second.reset(**reset_kwargs)
            for key in obs_a:
                np.testing.assert_array_equal(obs_a[key], obs_b[key])
            for _ in range(4):
                obs_a = first.step(action)[0]
                obs_b = second.step(action)[0]
                np.testing.assert_array_equal(obs_a["particles"],
                                              obs_b["particles"])
                np.testing.assert_array_equal(obs_a["weights"],
                                              obs_b["weights"])
    finally:
        first.close()
        second.close()


# --------------------------------------------------------------------------
# 5. Variant registry golden
# --------------------------------------------------------------------------

#: (env id, particle filter class name, episode cap), hard-coded on
#: 2026-09-03. The cap is the one that matters: PITFALLS.md section 5 records
#: an Ant-Tag eval left at a 400-step default against a 200-step env, which
#: counted every timeout as a tag and read ~100% success. episode_cap() must
#: keep reading gym.spec(env_id).max_episode_steps, so a registration change
#: shows up here and not in a result table.
_VARIANT_TABLE = {
    "base": ("pdomains-ant-tag-v0", "AntTagParticleFilter", 400),
    "smart": ("pdomains-ant-tag-smart-v0", "SmartAntTagParticleFilter", 400),
    "smart_hard": ("pdomains-ant-tag-smart-hard-v0",
                   "SmartAntTagParticleFilter", 400),
    "ghost": ("pdomains-ant-tag-ghost-v0", "GhostAntTagParticleFilter", 400),
    "dens": ("pdomains-ant-tag-dens-v0", "TwinDenAntTagParticleFilter", 200),
    "cdens": ("pdomains-ant-tag-cdens-v0",
              "CounterweightedDenAntTagParticleFilter", 300),
    "cdens_hard": ("pdomains-ant-tag-cdens-hard-v0",
                   "CounterweightedDenAntTagParticleFilter", 300),
    "cdens_terminal": ("pdomains-ant-tag-cdens-terminal-v0",
                       "CounterweightedDenAntTagParticleFilter", 300),
    "cdens_nospook": ("pdomains-ant-tag-cdens-nospook-v0",
                      "CounterweightedDenAntTagParticleFilter", 300),
}

#: Variants whose target evades; only these respond to --evasion_curriculum
#: and --target_speed_scale.
_EVADING = {"smart", "smart_hard", "ghost", "dens", "cdens", "cdens_hard",
            "cdens_terminal", "cdens_nospook"}


def test_variant_registry_has_exactly_the_known_variants(ant_tag):
    """No variant may appear or vanish unnoticed.

    Seven scripts read this registry, and it is the single place the
    env/filter pairing is stated. A new entry is fine -- but it has to be
    added here too, which is the point: the reviewer then has to say what its
    cap and filter are.
    """
    variants = ant_tag["variants"]
    assert set(variants.VARIANTS) == set(_VARIANT_TABLE)
    assert set(variants.EVADING) == _EVADING
    assert _EVADING <= set(_VARIANT_TABLE)


@pytest.mark.parametrize("name", sorted(_VARIANT_TABLE))
def test_variant_env_filter_and_cap(ant_tag, name):
    """Each variant's env id, filter class and episode cap are pinned.

    The env/filter pairing is the one the registry exists to protect: a
    filter mirrors its env's target motion model, and if the two drift apart
    belief propagation diverges from the env with no error anywhere. The cap
    must come from the gym registration (PITFALLS.md section 5).
    """
    variants = ant_tag["variants"]
    env_id, filter_name, cap = _VARIANT_TABLE[name]
    variant = variants.resolve(name)
    assert variant.env_id == env_id
    assert variant.particle_filter.__name__ == filter_name
    assert variants.episode_cap(name) == cap
    # The cap really is read from the registration, not from a table here.
    assert int(gym.spec(env_id).max_episode_steps) == cap


def test_variant_lookup_and_run_subdir(ant_tag):
    """resolve() rejects a typo, and run_subdir keeps "base" bare.

    Existing run directories are named runs/ant_tag_<encoder> for the base
    variant and runs/ant_tag_<encoder>_<variant> otherwise. Changing that
    would orphan every run on disk from the scripts that read it.
    """
    variants = ant_tag["variants"]
    with pytest.raises(ValueError, match="Unknown variant"):
        variants.resolve("not_a_variant")
    assert variants.run_subdir("cgf", "base") == "ant_tag_cgf"
    assert variants.run_subdir("st", "cdens_terminal") == "ant_tag_st_cdens_terminal"
    assert variants.run_subdir("gaussian", "dens") == "ant_tag_gaussian_dens"


def test_variant_schedule_precedence_is_cli_then_variant_then_script(ant_tag):
    """resolve_schedule's precedence: CLI wins, then the variant's default,
    then the script's.

    The counterweighted-den variants carry a visibility curriculum that suits
    their geometry. If the variant default stopped applying, a cdens run
    would silently train at the base env's radius and the comparison against
    the recorded numbers would be void.
    """
    variants = ant_tag["variants"]
    assert variants.resolve_schedule(
        "cdens_terminal", "0:1,1:1", "default_curriculum", "script") == "0:1,1:1"
    assert variants.resolve_schedule(
        "cdens_terminal", None, "default_curriculum", "script") == (
        "0:100,0.2:100,0.5:1.0,1:1.0")
    assert variants.resolve_schedule(
        "base", None, "default_curriculum", "script") == "script"
    assert variants.resolve_schedule(
        "cdens_terminal", None, "default_evasion_curriculum",
        "script") == "0:0,0.2:0,0.5:1,1:1"


# --------------------------------------------------------------------------
# 6. Smoke train, all three arms
# --------------------------------------------------------------------------

#: Shared with the real CLI defaults where it is free to do so, and shrunk
#: only along the axes that cost wall time.
_SMOKE_KWARGS = dict(
    algorithm="PPO",
    total_timesteps=512,
    n_envs=1,
    ppo_n_steps=128,
    batch_size=64,
    n_epochs=2,
    num_particles=32,
    device="cpu",
    seed=0,
    # Large enough never to fire. The eval and checkpoint callbacks are still
    # constructed (and the eval env still built), which is the wiring worth
    # exercising; running 20 eval episodes of MuJoCo is not.
    eval_freq=10 ** 9,
    save_freq=10 ** 9,
    n_eval_episodes=1,
    use_vec_normalize=True,
    distance_coeff=1.0,
    entropy_coeff=0.0,
    obs_mask_indices=[-2, -1],
    # Exercise CurriculumCallback: it must actually change the training env's
    # visibility radius and reward coefficients partway through.
    curriculum_schedule=[(0.0, 100.0), (0.5, 1.0), (1.0, 1.0)],
    reward_schedule=[(0.0, 1.0, 0.0, 0.0), (0.5, 0.0, 0.0, 50.0)],
    evasion_schedule=[(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)],
    target_speed_scale=0.0,
    progress_bar=False,
)

_SMOKE_ARMS = {
    "cgf": ("train_ant_tag_cgf", "WeightedCGFFeaturesExtractor",
            dict(num_cgf_features=8, t_init_mode="spread", t_init_scale=0.1,
                 t_clamp=2.0, exp_arg_clamp=20.0, t_frozen=False)),
    "gaussian": ("train_ant_tag_gaussian", "WeightedGaussianFeaturesExtractor",
                 {}),
    "st": ("train_ant_tag_st", "SetTransformerFeaturesExtractor",
           dict(num_encodings=2, dim_encoder=4, num_inds=4, dim_hidden=16,
                num_heads=2, ln=True, weight_channel=True)),
}


@pytest.mark.slow
@pytest.mark.parametrize("arm", sorted(_SMOKE_ARMS))
def test_smoke_train_completes_and_saves_a_loadable_policy(ant_tag, arm,
                                                            tmp_path):
    """Each arm's real train_* function must run end to end and save a policy.

    This is the gate that catches everything the unit tests cannot: that the
    vec env, VecNormalize with norm_obs_keys=["obs"], the four callbacks, the
    curriculum router, the eval env and the save path still fit together
    after a move. It calls the scripts' own train_* functions -- not a
    hand-built PPO -- so the wiring under test is the wiring a real run uses.
    Only log_dir and model_save_path are redirected, into tmp_path.

    It is marked slow but is NOT deselected by default: it is the real gate.
    All three arms together take well under a minute at these sizes.
    """
    module_name = {"cgf": "4_train_rl_cgf", "gaussian": "4_train_rl_gaussian",
                   "st": "4_train_rl_st"}[arm]
    train_name, class_name, extra = _SMOKE_ARMS[arm]
    module = ant_tag[module_name]
    variant = ant_tag["variants"].resolve("cdens_terminal")

    model_path = tmp_path / arm / "models" / f"{arm}_agent.zip"
    getattr(module, train_name)(
        log_dir=str(tmp_path / arm / "logs") + "/",
        model_save_path=str(model_path),
        env_id=variant.env_id,
        particle_filter_class=variant.particle_filter,
        **_SMOKE_KWARGS, **extra)

    assert model_path.exists(), f"{train_name} saved no model"
    # VecNormalize statistics land next to the model; the eval scripts need
    # them, and a policy trained with normalized obs is not evaluable without.
    assert (model_path.parent / "vecnormalize.pkl").exists()

    recorded = _recorded_extractor_class(model_path)
    assert class_name in recorded, recorded

    from stable_baselines3 import PPO
    main_module = sys.modules["__main__"]
    grafted = False
    if recorded.startswith("<class '__main__."):
        # Under pytest the train_* functions live in a module, so this should
        # not happen; handled so the assertion below is about the class and
        # not about pytest's __main__.
        setattr(main_module, class_name, getattr(module, class_name))
        grafted = True
    try:
        reloaded = PPO.load(str(model_path), device="cpu")
    finally:
        if grafted:
            delattr(main_module, class_name)

    extractor = reloaded.policy.features_extractor
    assert type(extractor) is getattr(module, class_name), (
        "the saved policy's features extractor is not the class this arm's "
        "module exposes")
    assert set(reloaded.observation_space.spaces) == {"obs", "particles",
                                                      "weights"}
    assert reloaded.observation_space["particles"].shape == (32, _PARTICLE_DIM)
