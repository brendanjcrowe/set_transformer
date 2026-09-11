"""--st_encoder_lr_scale on the Ant-Tag ST arm (experiments/ant_tag/4_train_rl_st.py).

The Odd-Even arm's finetune-collapse fix 1 (tests/test_st_finetune_fixes.py)
was moved to set_transformer/rl/encoder_finetune.py on 2026-09-05 and wired
into the Ant-Tag arm, whose shared-rate finetune flipped between 0% and 10%
by seed on smart_hard. These tests pin the Ant-Tag wiring specifically: the
flag is off by default (existing runs unaffected), the scale survives SB3's
per-train() learning-rate write, the saved agent stays loadable by a plain
PPO.load, and the CLI refuses combinations that would make the flag a no-op.
Module loading follows tests/test_st_pretrained_load.py.
"""

import importlib
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_ROOT = Path(__file__).resolve().parents[1]
_ANT_TAG_DIR = _ST_ROOT / "experiments" / "ant_tag"
for _p in (str(_REPO_ROOT), str(_ST_ROOT), str(_ANT_TAG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco", reason="AntTag needs MuJoCo")

BASE_LR = 3e-4


@pytest.fixture(scope="module")
def pieces(tmp_path_factory):
    pytest.importorskip("pdomains")
    m = importlib.import_module("4_train_rl_st")
    variants = importlib.import_module("variants")
    from set_transformer.models.pf_set_transformer import PFSetTransformer
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.manual_seed(0)
    pf_st = PFSetTransformer(
        num_particles=100, dim_particles=3, num_encodings=8, dim_encoder=8,
        num_inds=16, dim_hidden=64, num_heads=4, ln=True,
        dim_output_particles=2)
    for param in pf_st.parameters():
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.1)
    path = tmp_path_factory.mktemp("ck") / "checkpoint_best.pt"
    torch.save({"model_state_dict": pf_st.state_dict()}, path)
    reference = {k[len("set_transformer."):]: v.clone()
                 for k, v in pf_st.state_dict().items()
                 if k.startswith("set_transformer.")}

    variant = variants.resolve("smart_hard")

    def make_venv():
        return DummyVecEnv([m.make_ant_tag_belief_env(
            num_particles=100, rank=0, seed=0, monitor_dir=None,
            initial_visibility_radius=100.0, obs_mask_indices=[-2, -1],
            apply_reward_shaping=False, env_id=variant.env_id,
            particle_filter_class=variant.particle_filter)])

    return m, make_venv, str(path), reference


def _encoder_delta(model, reference):
    prefix = "features_extractor.encoder."
    live = {k[len(prefix):]: v for k, v in model.policy.state_dict().items()
            if k.startswith(prefix)}
    assert live
    return max(float((reference[k] - live[k].cpu()).abs().max())
               for k in reference if k in live)


def _build(m, make_venv, path, lr_anneal=True):
    """The arm's construction: PPO(...) then the post-construct reload."""
    from stable_baselines3 import PPO
    model = PPO(
        "MultiInputPolicy", make_venv(),
        learning_rate=((lambda remaining: BASE_LR * remaining) if lr_anneal else BASE_LR),
        policy_kwargs=dict(
            features_extractor_class=m.SetTransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                num_encodings=8, dim_encoder=8, num_inds=16, dim_hidden=64,
                num_heads=4, ln=True, arena_scale=4.5, weight_channel=True,
                pretrained_st_model_path=path, st_frozen=False)),
        n_steps=64, batch_size=32, n_epochs=2, device="cpu", verbose=0, seed=0)
    extractor = model.policy.features_extractor
    extractor._load_pretrained_encoder(path, extractor.dim_input)
    return model


def _group_lrs(model):
    return [g["lr"] for g in model.policy.optimizer.param_groups]


def test_default_is_sb3s_own_single_group(pieces):
    m, make_venv, path, _ = pieces
    model = _build(m, make_venv, path)
    assert type(model.policy.optimizer) is model.policy.optimizer_class
    assert len(model.policy.optimizer.param_groups) == 1


def test_scale_partitions_parameters_and_survives_schedule_write(pieces):
    m, make_venv, path, _ = pieces
    model = _build(m, make_venv, path, lr_anneal=True)
    m.scale_encoder_learning_rate(model, 0.1)
    groups = model.policy.optimizer.param_groups
    assert len(groups) == 2
    encoder_ids = {id(p) for p in model.policy.features_extractor.encoder.parameters()}
    assert {id(p) for p in groups[0]["params"]} == encoder_ids
    assert ({id(p) for p in groups[0]["params"]} | {id(p) for p in groups[1]["params"]}
            == {id(p) for p in model.policy.parameters()})
    enc_lr, head_lr = _group_lrs(model)
    assert enc_lr == pytest.approx(BASE_LR * 0.1) and head_lr == pytest.approx(BASE_LR)

    from stable_baselines3.common.logger import configure
    model.set_logger(configure(folder=None, format_strings=[]))
    model._current_progress_remaining = 0.5
    model._update_learning_rate(model.policy.optimizer)
    enc_lr, head_lr = _group_lrs(model)
    assert head_lr == pytest.approx(BASE_LR * 0.5)
    assert enc_lr == pytest.approx(BASE_LR * 0.5 * 0.1), "SB3 wiped the encoder scale"


def test_scale_slows_the_encoder_and_saved_agent_loads(pieces, tmp_path):
    from stable_baselines3 import PPO
    m, make_venv, path, reference = pieces
    plain = _build(m, make_venv, path)
    plain.learn(total_timesteps=128)
    scaled = _build(m, make_venv, path)
    m.scale_encoder_learning_rate(scaled, 1e-3)
    scaled.learn(total_timesteps=128)
    d_plain, d_scaled = _encoder_delta(plain, reference), _encoder_delta(scaled, reference)
    assert d_plain > 0.0 and d_scaled > 0.0
    assert d_scaled < 0.1 * d_plain, (d_plain, d_scaled)

    saved = scaled.policy.optimizer.state_dict()
    assert len(saved["param_groups"]) == 1 and saved["state"] == {}
    out = tmp_path / "agent"
    scaled.save(out)
    loaded = PPO.load(out, device="cpu")    # raises on a group-count mismatch
    for key, value in loaded.policy.state_dict().items():
        assert torch.equal(value, scaled.policy.state_dict()[key]), key


@pytest.mark.parametrize("argv", [
    ["--st_encoder_lr_scale", "0.1"],                                      # no pretrained path
    ["--st_encoder_lr_scale", "0.1", "--pretrained_st_model_path", "x.pt", "--st_frozen"],
    ["--st_encoder_lr_scale", "0", "--pretrained_st_model_path", "x.pt"],
    ["--st_encoder_lr_scale", "0.1", "--pretrained_st_model_path", "x.pt", "--algorithm", "SAC"],
])
def test_cli_refuses_meaningless_combinations(pieces, monkeypatch, argv):
    m, _, _, _ = pieces
    monkeypatch.setattr(sys, "argv", ["4_train_rl_st.py", "--variant", "smart_hard", *argv])
    with pytest.raises(SystemExit) as exc:
        m.main()
    assert exc.value.code == 2
