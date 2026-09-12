"""The shared encoder interface (change 2 of the harness centralisation, 2026-09-12).

Every learned SB3 features extractor -- SetTransformerFeaturesExtractor,
WeightedCGFFeaturesExtractor, WeightedDeepSetFeaturesExtractor, PointNetFeaturesExtractor --
exposes five methods with one signature (`encoder_parameters`, `load_pretrained`, `freeze`,
`reference_state`, `encoder_state_dict`) and names its checkpoint-path constructor argument
in `PRETRAINED_PATH_KWARG`; `rl/pretrained_encoder.reload_pretrained` is the one
reload-after-PPO written against them. These tests pin the new names against the effect of
the old ones (`encoder.parameters()` / `parameters()`, `_load_pretrained_encoder`,
`freeze_encoder` / the by-hand ST freeze). The old names' own behaviour stays pinned by
tests/test_st_pretrained_load.py, test_cgf_readout_pretrained.py, test_ant_tag_pool_arms.py
and test_st_finetune_fixes.py.
"""

import contextlib
import inspect
import io

import gymnasium as gym
import numpy as np
import pytest
import torch

from set_transformer.models import DeepSetAE, PointNetAE
from set_transformer.models.pf_set_transformer import PFSetTransformer
from set_transformer.rl import pretrained_encoder as pe
from set_transformer.rl.feature_extractors import pooled
from set_transformer.rl.feature_extractors.cgf import WeightedCGFFeaturesExtractor
from set_transformer.rl.feature_extractors.pooled import (
    PointNetFeaturesExtractor,
    WeightedDeepSetFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.st import SetTransformerFeaturesExtractor

N, D, OBS, SCALE = 20, 2, 5, 4.5
ST_GEOMETRY = dict(num_encodings=4, dim_encoder=4, num_inds=8, dim_hidden=16, num_heads=2, ln=True)
POOL_GEOMETRY = dict(num_encodings=4, dim_encoder=4, dim_hidden=16)
CGF_KW = dict(num_cgf_features=8, arena_scale=SCALE, readout_hidden=16, readout_depth=1,
              feature_norm="running")


def _space():
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (N, D), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (N,), np.float32),
    })


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(3, N, generator=g)
    return {"obs": torch.randn(3, OBS, generator=g),
            "particles": torch.randn(3, N, D, generator=g) * 2.0,
            "weights": w / w.sum(1, keepdim=True)}


def _perturb(module):
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * 0.1)


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


# (class, extractor kwargs, checkpoint writer). The checkpoint always comes from a SIBLING
# model with perturbed weights, so a zero delta after loading is not vacuous.
def _st_checkpoint(path):
    torch.manual_seed(1)
    pf = PFSetTransformer(num_particles=N, dim_particles=D + 1, dim_output_particles=D, **ST_GEOMETRY)
    _perturb(pf)
    torch.save({"model_state_dict": pf.state_dict()}, path)


def _cgf_checkpoint(path):
    torch.manual_seed(1)
    sibling = _quiet(lambda: WeightedCGFFeaturesExtractor(_space(), **CGF_KW))
    _perturb(sibling)
    torch.save({"model_state_dict": sibling.state_dict()}, path)


def _deepset_checkpoint(path):
    torch.manual_seed(1)
    ae = DeepSetAE(num_particles=N, dim_particles=D, **POOL_GEOMETRY)
    _perturb(ae)
    torch.save(ae.state_dict(), path)


def _pointnet_checkpoint(path):
    torch.manual_seed(1)
    ae = PointNetAE(num_particles=N, dim_particles=D, **POOL_GEOMETRY)
    _perturb(ae)
    torch.save(ae.state_dict(), path)


ARMS = [
    pytest.param(SetTransformerFeaturesExtractor, dict(arena_scale=SCALE, weight_channel=True, **ST_GEOMETRY),
                 _st_checkpoint, id="st"),
    pytest.param(WeightedCGFFeaturesExtractor, CGF_KW, _cgf_checkpoint, id="cgf"),
    pytest.param(WeightedDeepSetFeaturesExtractor, dict(arena_scale=SCALE, weight_channel=False, **POOL_GEOMETRY),
                 _deepset_checkpoint, id="deepset"),
    pytest.param(PointNetFeaturesExtractor, dict(arena_scale=SCALE, weight_channel=False, **POOL_GEOMETRY),
                 _pointnet_checkpoint, id="pointnet"),
]


def _old_parameters(ext):
    """What each arm's old code treated as 'the encoder' (see the module docstring)."""
    if isinstance(ext, WeightedCGFFeaturesExtractor):
        return list(ext.parameters())            # the whole extractor is the encoder
    return list(ext.encoder.parameters())


@pytest.mark.parametrize("cls,kwargs,write_checkpoint", ARMS)
def test_interface_names_exist_and_the_path_kwarg_is_a_constructor_argument(cls, kwargs, write_checkpoint):
    for name in ("encoder_parameters", "load_pretrained", "freeze", "reference_state", "encoder_state_dict"):
        assert callable(getattr(cls, name)), f"{cls.__name__} lacks {name}"
    assert cls.PRETRAINED_PATH_KWARG in inspect.signature(cls.__init__).parameters


@pytest.mark.parametrize("cls,kwargs,write_checkpoint", ARMS)
def test_encoder_parameters_is_the_old_parameter_set(cls, kwargs, write_checkpoint):
    torch.manual_seed(0)
    ext = _quiet(lambda: cls(_space(), **kwargs))
    new, old = ext.encoder_parameters(), _old_parameters(ext)
    assert {id(p) for p in new} == {id(p) for p in old} and len(new) == len(old) > 0


@pytest.mark.parametrize("cls,kwargs,write_checkpoint", ARMS)
def test_load_pretrained_makes_the_live_state_equal_the_reference(cls, kwargs, write_checkpoint, tmp_path):
    path = tmp_path / "ckpt.pt"
    write_checkpoint(path)
    torch.manual_seed(0)
    ext = _quiet(lambda: cls(_space(), **kwargs))
    reference = ext.reference_state(str(path))
    assert set(reference) == set(ext.encoder_state_dict()), "reference and live dicts must share their keys"
    before, _ = pe.max_abs_delta(reference, ext.encoder_state_dict())
    assert before > 0.0                                   # the sibling really differs
    _quiet(lambda: ext.load_pretrained(str(path)))
    after, n = pe.max_abs_delta(reference, ext.encoder_state_dict())
    assert after == 0.0 and n == len(reference)
    _quiet(lambda: pe.verify_matches_checkpoint(reference, ext.encoder_state_dict(), str(path)))


@pytest.mark.parametrize("cls,kwargs,write_checkpoint", ARMS)
def test_freeze_has_the_old_freezes_effect(cls, kwargs, write_checkpoint):
    torch.manual_seed(0)
    ext = _quiet(lambda: cls(_space(), **kwargs))
    ext.train()
    ext.freeze()
    assert all(not p.requires_grad for p in ext.encoder_parameters())
    if isinstance(ext, SetTransformerFeaturesExtractor):
        # The by-hand freeze in the scripts: encoder.eval() + requires_grad False; plus the
        # constructor's no-grad flag, which the reload-after-PPO path never set before.
        assert ext.st_frozen is True and ext.encoder.training is False
    else:
        assert ext._frozen is True                          # what freeze_encoder() set
        ext.train()                                         # the override holds eval mode
        module = ext if isinstance(ext, WeightedCGFFeaturesExtractor) else ext.encoder
        assert module.training is False
    out = ext(_batch())
    if out.requires_grad:                                   # the pooled arms have no no_grad wrapper
        out.sum().backward()
    assert all(p.grad is None for p in ext.encoder_parameters())
    ext.freeze()                                            # idempotent
    assert all(not p.requires_grad for p in ext.encoder_parameters())


# ---------------------------------------------------------------------------
# The one reload-after-PPO
# ---------------------------------------------------------------------------

class _DictEnv(gym.Env):
    """A stand-in for the belief envs: the same three-key dict observation, random content."""

    def __init__(self):
        self.observation_space = _space()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def _obs(self):
        w = self.np_random.random(N).astype(np.float32)
        return {"obs": self.np_random.standard_normal(OBS).astype(np.float32),
                "particles": (self.np_random.standard_normal((N, D)) * 2).astype(np.float32),
                "weights": w / w.sum()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


def _tiny_ppo(cls, kwargs, path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    fe_kwargs = {**kwargs, cls.PRETRAINED_PATH_KWARG: str(path)}
    return PPO("MultiInputPolicy", DummyVecEnv([_DictEnv]),
               policy_kwargs=dict(features_extractor_class=cls, features_extractor_kwargs=fe_kwargs),
               n_steps=8, batch_size=8, device="cpu", seed=0, verbose=0)


@pytest.mark.parametrize("cls,kwargs,write_checkpoint", ARMS)
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "finetune"])
def test_reload_pretrained_reloads_verifies_freezes_and_scrubs_the_path(cls, kwargs, write_checkpoint, frozen, tmp_path):
    path = tmp_path / "ckpt.pt"
    write_checkpoint(path)
    model = _quiet(lambda: _tiny_ppo(cls, kwargs, path))
    ext = model.policy.features_extractor
    reference = ext.reference_state(str(path))
    # PITFALLS.md section 1: SB3's _build re-initialised the Linear layers the constructor
    # had just loaded, so before the reload the live encoder differs from the checkpoint.
    assert pe.max_abs_delta(reference, ext.encoder_state_dict())[0] > 0.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pe.reload_pretrained(model, str(path), frozen)
    text = buf.getvalue()
    assert pe.max_abs_delta(reference, ext.encoder_state_dict())[0] == 0.0
    assert f"{cls.__name__}: encoder RE-loaded after PPO construction" in text
    assert ("and re-frozen" in text) is frozen
    assert "Verified:" in text and "max|delta| = 0.0" in text
    assert all(p.requires_grad is (not frozen) for p in ext.encoder_parameters())
    assert model.policy_kwargs["features_extractor_kwargs"][cls.PRETRAINED_PATH_KWARG] is None


def test_reload_pretrained_fails_loudly_on_the_wrong_checkpoint(tmp_path):
    good, other = tmp_path / "good.pt", tmp_path / "other.pt"
    _cgf_checkpoint(good)
    torch.manual_seed(7)
    sibling = _quiet(lambda: WeightedCGFFeaturesExtractor(_space(), **CGF_KW))
    _perturb(sibling)
    torch.save({"model_state_dict": sibling.state_dict()}, other)
    model = _quiet(lambda: _tiny_ppo(WeightedCGFFeaturesExtractor, CGF_KW, good))
    _quiet(lambda: pe.reload_pretrained(model, str(good), frozen=False))
    with pytest.raises(AssertionError, match="does not match"):
        _quiet(lambda: pe.verify_matches_checkpoint(
            model.policy.features_extractor.reference_state(str(other)),
            model.policy.features_extractor.encoder_state_dict(), str(other)))


def test_the_old_reload_names_forward_to_the_one_reload(monkeypatch):
    calls = []
    monkeypatch.setattr(pe, "reload_pretrained", lambda *a, **k: calls.append((a, k)))
    sentinel = object()
    pe.reload_pretrained_cgf(sentinel, "p", True)
    pooled.reload_pretrained_pooled(sentinel, "q", False, verify=False)
    assert calls == [((sentinel, "p", True), {"verify": True}), ((sentinel, "q", False), {"verify": False})]


def test_policy_extractors_lists_the_shared_extractor_first_and_each_distinct_one_once():
    class _P:  # a SAC-shaped policy: actor and critic hold their own extractors
        pass
    shared, own = object(), object()
    policy = _P(); policy.features_extractor = shared
    policy.actor = _P(); policy.actor.features_extractor = shared
    policy.critic = _P(); policy.critic.features_extractor = own
    policy.critic_target = _P(); policy.critic_target.features_extractor = own
    model = _P(); model.policy = policy
    found = pe.policy_extractors(model)
    assert found[0] is shared and {id(e) for e in found} == {id(shared), id(own)}
    assert all(e is own for e in found[1:])
