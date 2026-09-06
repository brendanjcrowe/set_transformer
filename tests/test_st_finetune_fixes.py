"""The two finetune-collapse fixes in experiments/odd_even/4_train_rl_st.py.

Background (oddeven.md, 2026-09-05/06): a pretrained ST encoder finetuned
under PPO at the shared 3e-4 learning rate collapses in the first few hundred
thousand steps (frozen 0.878 steady exact-match, finetuned 0.31-0.48). Two
fixes were added, and each one fights an SB3 behaviour that would silently
undo it:

* ``--st_encoder_lr_scale`` (fix 1) puts the encoder in its own optimizer
  param group at a fraction of the head rate. SB3's
  ``BaseAlgorithm._update_learning_rate`` writes the scheduled rate into
  EVERY param group at the start of every ``train()``, so a plain second
  group is back at the head rate after the first update -- the fix would be
  a no-op with a correct-looking log line. And ``PPO.load`` rebuilds a
  one-group optimizer and refuses a two-group state dict, so every saved
  agent of a fix-1 run would be unloadable at eval time.
* ``--st_unfreeze_at`` (fix 2) keeps the reloaded encoder frozen and releases
  it from a callback. The reverse failures are: releasing early (the
  shared-LR finetune in disguise) or never (the frozen arm in disguise).

Each test below would fail if one of those regressions came back. Modelled on
tests/test_odd_even_pipeline.py: modules are loaded through the pipeline's
own _sibling loader, never by flat-name import (Gap 12 collision).
"""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_ROOT = Path(__file__).resolve().parents[1]
_ODD_EVEN_DIR = _ST_ROOT / "experiments" / "odd_even"
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("pdomains")
pytest.importorskip("stable_baselines3")

import torch  # noqa: E402

VARIANT = "oe50"
NS = 50
BASE_LR = 3e-4


@contextmanager
def _sys_path(*directories):
    saved = list(sys.path)
    saved_modules = set(sys.modules)
    try:
        for directory in directories:
            sys.path.insert(0, str(directory))
        yield
    finally:
        sys.path[:] = saved
        for name in set(sys.modules) - saved_modules:
            sys.modules.pop(name, None)


def _odd_even_sibling():
    key = "_oe_pipe_sibling_loader"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, _ODD_EVEN_DIR / "_sibling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def _load(name):
    with _sys_path(_ODD_EVEN_DIR):
        return _odd_even_sibling().load(name)


@pytest.fixture(scope="module")
def pieces(tmp_path_factory):
    """The ST arm module, a tiny distinctive checkpoint, and its reference."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from set_transformer.models.pf_set_transformer import PFSetTransformer

    st_module = _load("4_train_rl_st")
    torch.manual_seed(0)
    pf_st = PFSetTransformer(
        num_particles=NS, dim_particles=2, num_encodings=8, dim_encoder=8,
        num_inds=16, dim_hidden=64, num_heads=4, ln=True,
        dim_output_particles=1)
    for param in pf_st.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.1)
    path = tmp_path_factory.mktemp("oe_ft_ck") / "checkpoint_best.pt"
    torch.save({"model_state_dict": pf_st.state_dict()}, path)
    reference = {key[len("set_transformer."):]: value.clone()
                 for key, value in pf_st.state_dict().items()
                 if key.startswith("set_transformer.")}

    def make_venv():
        return DummyVecEnv([st_module.make_odd_even_belief_env(
            num_particles=NS, rank=0, seed=0, variant=VARIANT)])

    return st_module, make_venv, str(path), reference


def _encoder_delta(model, reference):
    prefix = "features_extractor.encoder."
    live = {key[len(prefix):]: value
            for key, value in model.policy.state_dict().items()
            if key.startswith(prefix)}
    assert live
    return max(float((reference[key] - live[key].cpu()).abs().max())
               for key in reference if key in live)


def _build(st_module, make_venv, path, frozen, lr_anneal=True):
    """Exactly the arm's construction: PPO(...), then the post-construct reload."""
    from stable_baselines3 import PPO

    model = PPO(
        "MultiInputPolicy", make_venv(),
        learning_rate=((lambda remaining: BASE_LR * remaining) if lr_anneal else BASE_LR),
        policy_kwargs=dict(
            features_extractor_class=st_module.SetTransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                num_encodings=8, dim_encoder=8, num_inds=16, dim_hidden=64,
                num_heads=4, ln=True, arena_scale=(NS - 1) / 2,
                weight_channel=True, pretrained_st_model_path=path,
                st_frozen=frozen)),
        n_steps=64, batch_size=32, n_epochs=2, device="cpu", verbose=0, seed=0)
    st_module.reload_pretrained_encoder(model, path, frozen)
    return model


def _group_lrs(model):
    return [group["lr"] for group in model.policy.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Fix 1: encoder learning-rate scale
# ---------------------------------------------------------------------------

def test_without_the_flag_the_optimizer_is_sb3s_own(pieces):
    """Default off: one plain group, the class SB3 built. Existing runs and
    the frozen / e2e arms must be byte-for-byte unaffected by the fix."""
    st_module, make_venv, path, _ = pieces
    model = _build(st_module, make_venv, path, frozen=False)
    assert type(model.policy.optimizer) is model.policy.optimizer_class
    assert len(model.policy.optimizer.param_groups) == 1


def test_encoder_lr_scale_partitions_every_parameter_exactly_once(pieces):
    st_module, make_venv, path, _ = pieces
    model = _build(st_module, make_venv, path, frozen=False)
    st_module.scale_encoder_learning_rate(model, 0.1)
    groups = model.policy.optimizer.param_groups
    assert len(groups) == 2
    encoder_ids = {id(p) for p in model.policy.features_extractor.encoder.parameters()}
    assert {id(p) for p in groups[0]["params"]} == encoder_ids
    all_ids = {id(p) for p in model.policy.parameters()}
    assert {id(p) for p in groups[0]["params"]} | {id(p) for p in groups[1]["params"]} == all_ids
    assert not (encoder_ids & {id(p) for p in groups[1]["params"]})


def test_encoder_lr_scale_survives_sb3s_schedule_write(pieces):
    """THE regression this fix exists to beat. SB3 overwrites every group's
    lr with the scheduled value at the start of each train(); the encoder
    group must come out at scale x that value, the heads at the value."""
    st_module, make_venv, path, _ = pieces
    model = _build(st_module, make_venv, path, frozen=False, lr_anneal=True)
    st_module.scale_encoder_learning_rate(model, 0.1)
    enc_lr, head_lr = _group_lrs(model)
    assert enc_lr == pytest.approx(BASE_LR * 0.1)
    assert head_lr == pytest.approx(BASE_LR)

    # What PPO.train() does first, half-way through an annealed run. (learn()
    # normally installs the logger the method records to.)
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(folder=None, format_strings=[]))
    model._current_progress_remaining = 0.5
    model._update_learning_rate(model.policy.optimizer)
    enc_lr, head_lr = _group_lrs(model)
    assert head_lr == pytest.approx(BASE_LR * 0.5), "anneal must still reach the heads"
    assert enc_lr == pytest.approx(BASE_LR * 0.5 * 0.1), (
        "SB3's update_learning_rate wiped the encoder's scale; the fix is a no-op")


def test_encoder_lr_scale_really_slows_the_encoder(pieces):
    """A 1000x smaller rate must move the encoder far less over the same
    updates (Adam's step is proportional to lr)."""
    st_module, make_venv, path, reference = pieces
    plain = _build(st_module, make_venv, path, frozen=False)
    plain.learn(total_timesteps=128)
    scaled = _build(st_module, make_venv, path, frozen=False)
    st_module.scale_encoder_learning_rate(scaled, 1e-3)
    scaled.learn(total_timesteps=128)
    d_plain, d_scaled = _encoder_delta(plain, reference), _encoder_delta(scaled, reference)
    assert d_plain > 0.0 and d_scaled > 0.0, "both encoders must train"
    assert d_scaled < 0.1 * d_plain, (d_plain, d_scaled)


def test_encoder_lr_scale_saves_a_state_a_plain_ppo_can_load(pieces, tmp_path):
    """PPO.load builds a one-group optimizer and refuses a two-group state
    dict, which would make the final agent, best_model and every checkpoint
    of a fix-1 run unloadable. The saved optimizer state must therefore
    look like a fresh single group over ALL parameters."""
    from stable_baselines3 import PPO

    st_module, make_venv, path, reference = pieces
    model = _build(st_module, make_venv, path, frozen=False)
    st_module.scale_encoder_learning_rate(model, 0.1)
    model.learn(total_timesteps=128)

    saved = model.policy.optimizer.state_dict()
    n_params = sum(1 for _ in model.policy.parameters())
    assert len(saved["param_groups"]) == 1
    assert saved["param_groups"][0]["params"] == list(range(n_params))
    assert saved["state"] == {}
    assert "lr_scale" not in saved["param_groups"][0]

    out = tmp_path / "agent"
    model.save(out)
    loaded = PPO.load(out, device="cpu")   # raises on a group-count mismatch
    live_before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    for key, value in loaded.policy.state_dict().items():
        assert torch.equal(value, live_before[key]), key
    assert _encoder_delta(loaded, reference) > 0.0, "the trained encoder was saved"


# ---------------------------------------------------------------------------
# Fix 2: freeze, then unfreeze
# ---------------------------------------------------------------------------

def test_unfreeze_callback_holds_the_encoder_until_the_step(pieces):
    """n_steps=64 with one env: rollout 1 is steps 0-64, then train(); the
    callback fires at num_timesteps >= 64, i.e. at the end of rollout 1, so
    the FIRST update already trains the encoder. With the release beyond the
    run, the encoder must end exactly at the checkpoint."""
    st_module, make_venv, path, reference = pieces

    never = _build(st_module, make_venv, path, frozen=True)
    cb_never = st_module.UnfreezeEncoderCallback(unfreeze_at=10**9)
    never.learn(total_timesteps=128, callback=cb_never)
    assert not cb_never.done
    assert not any(p.requires_grad for p in never.policy.features_extractor.encoder.parameters())
    assert _encoder_delta(never, reference) == 0.0, "encoder moved while frozen"

    # frozen=True here also sets the EXTRACTOR's st_frozen flag, which wraps
    # its forward in torch.no_grad(); the release must clear that too, or the
    # encoder is trainable in name only. (The script passes st_frozen=False
    # for fix 2 and freezes through the reload; the callback covers both.)
    released = _build(st_module, make_venv, path, frozen=True)
    assert released.policy.features_extractor.st_frozen
    cb = st_module.UnfreezeEncoderCallback(unfreeze_at=64)
    released.learn(total_timesteps=128, callback=cb)
    assert cb.done
    assert not released.policy.features_extractor.st_frozen
    assert all(p.requires_grad for p in released.policy.features_extractor.encoder.parameters())
    assert _encoder_delta(released, reference) > 0.0, (
        "encoder did not train after release: the parameters are not reaching "
        "the optimizer, or requires_grad was not flipped")


def test_unfreeze_callback_releases_exactly_at_the_boundary(pieces):
    """The encoder must be bit-identical to the checkpoint at the update
    BEFORE the release and moved at the one after. Recorded from a second
    callback at every rollout end, which is where PPO.train() runs next."""
    from stable_baselines3.common.callbacks import BaseCallback

    st_module, make_venv, path, reference = pieces

    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__()
            self.deltas = []

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            self.deltas.append((self.num_timesteps, _encoder_delta(self.model, reference)))

    model = _build(st_module, make_venv, path, frozen=True)
    recorder = Recorder()
    # Release at 128: rollouts end at 64, 128, 192, 256. The update after the
    # 64-rollout must not touch the encoder; the one after 128 must.
    model.learn(total_timesteps=256,
                callback=[recorder, st_module.UnfreezeEncoderCallback(unfreeze_at=128)])
    by_step = dict(recorder.deltas)
    assert by_step[64] == 0.0 and by_step[128] == 0.0, by_step
    assert by_step[192] > 0.0 and by_step[256] > 0.0, by_step


def test_fix1_and_fix2_compose(pieces):
    """The combined arm: frozen until the step, then released at the scaled
    rate, and the scale is still in place after the release."""
    st_module, make_venv, path, reference = pieces
    model = _build(st_module, make_venv, path, frozen=True, lr_anneal=False)
    st_module.scale_encoder_learning_rate(model, 0.1)
    model.learn(total_timesteps=128,
                callback=st_module.UnfreezeEncoderCallback(unfreeze_at=64))
    assert _encoder_delta(model, reference) > 0.0
    enc_lr, head_lr = _group_lrs(model)
    assert enc_lr == pytest.approx(BASE_LR * 0.1) and head_lr == pytest.approx(BASE_LR)


# ---------------------------------------------------------------------------
# CLI guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["--st_encoder_lr_scale", "0.1"],                       # no pretrained encoder
    ["--st_unfreeze_at", "5"],                               # no pretrained encoder
    ["--pretrained_st_model_path", "x.pt", "--st_frozen", "--st_unfreeze_at", "5"],
    ["--pretrained_st_model_path", "x.pt", "--st_frozen", "--st_encoder_lr_scale", "0.1"],
    ["--pretrained_st_model_path", "x.pt", "--st_encoder_lr_scale", "0"],
])
def test_cli_refuses_meaningless_fix_combinations(pieces, monkeypatch, argv):
    """A finetune fix on a random encoder, or on a permanently frozen one,
    is a misconfigured run that would train for hours and mean nothing.
    argparse must refuse it before anything is built (exit code 2)."""
    st_module, _, _, _ = pieces
    monkeypatch.setattr(sys, "argv", ["4_train_rl_st.py", "--variant", VARIANT] + argv)
    with pytest.raises(SystemExit) as excinfo:
        st_module.main()
    assert excinfo.value.code == 2
