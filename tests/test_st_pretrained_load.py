"""The pretrained ST encoder must survive PPO construction.

Regression test for a silent bug: SetTransformerFeaturesExtractor.__init__
loaded the pretrained encoder correctly, but SB3's ActorCriticPolicy._build
ends with `self.apply(partial(self.init_weights, ...))`, which walks the whole
policy INCLUDING the features extractor and re-initializes every Linear.
The pretrained weights were overwritten (measured max|delta| = 1.84) a moment
after being loaded, and --st_frozen then froze that noise. The log still said
"loaded encoder ... FROZEN", because those lines print during construction.

Two 6M-step runs were wasted on a random encoder before this was caught, so
the invariant is worth a test: after PPO construction the encoder must equal
the checkpoint exactly, and a frozen encoder must not move during learn().
"""
import importlib
import sys
from pathlib import Path

import pytest
import torch

_ANT_TAG = Path(__file__).resolve().parents[1] / "experiments" / "ant_tag"
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_ANT_TAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
# AntTag is a MuJoCo env; these envs use the modern `mujoco` binding, NOT the
# legacy mujoco_py, so guarding on mujoco_py would skip the test everywhere.
pytest.importorskip("mujoco", reason="AntTag needs MuJoCo")


@pytest.fixture(scope="module")
def pieces(tmp_path_factory):
    """A tiny pretrained checkpoint plus the vec env the RL arm builds."""
    pytest.importorskip("pdomains")
    m = importlib.import_module("4_train_rl_st")
    variants = importlib.import_module("variants")
    from set_transformer.models.pf_set_transformer import PFSetTransformer
    from stable_baselines3.common.vec_env import DummyVecEnv

    # A checkpoint in 3_train_st.py's format, with known non-default weights.
    torch.manual_seed(0)
    pf_st = PFSetTransformer(
        num_particles=100, dim_particles=3, num_encodings=8, dim_encoder=8,
        num_inds=32, dim_hidden=128, num_heads=4, ln=True,
        dim_output_particles=2)
    for param in pf_st.parameters():          # make every weight distinctive
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.1)
    path = tmp_path_factory.mktemp("ck") / "checkpoint_best.pt"
    torch.save({"model_state_dict": pf_st.state_dict()}, path)

    variant = variants.resolve("cdens_terminal")
    venv = DummyVecEnv([m.make_ant_tag_belief_env(
        num_particles=100, rank=0, seed=0, monitor_dir=None,
        initial_visibility_radius=100.0, obs_mask_indices=[-2, -1],
        apply_reward_shaping=False, env_id=variant.env_id,
        particle_filter_class=variant.particle_filter)])

    reference = {k[len("set_transformer."):]: v.clone()
                 for k, v in pf_st.state_dict().items()
                 if k.startswith("set_transformer.")}
    return m, venv, str(path), reference


def _encoder_delta(model, reference):
    prefix = "features_extractor.encoder."
    saved = {k[len(prefix):]: v for k, v in model.policy.state_dict().items()
             if k.startswith(prefix)}
    assert saved, "policy has no features_extractor.encoder.* parameters"
    return max(float((reference[k] - saved[k].cpu()).abs().max())
               for k in reference if k in saved)


def _build(m, venv, ck_path, frozen):
    from stable_baselines3 import PPO
    model = PPO("MultiInputPolicy", venv, policy_kwargs=dict(
        features_extractor_class=m.SetTransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            num_encodings=8, dim_encoder=8, num_inds=32, dim_hidden=128,
            num_heads=4, ln=True, arena_scale=7.0, weight_channel=True,
            pretrained_st_model_path=ck_path, st_frozen=frozen)),
        n_steps=64, batch_size=32, device="cpu", verbose=0, seed=0)
    # The reload train_ant_tag_st performs after construction.
    extractor = model.policy.features_extractor
    extractor._load_pretrained_encoder(ck_path, extractor.dim_input)
    if frozen:
        extractor.encoder.eval()
        for param in extractor.encoder.parameters():
            param.requires_grad_(False)
    return model


@pytest.mark.parametrize("frozen", [True, False])
def test_encoder_matches_checkpoint_after_ppo_construction(pieces, frozen):
    m, venv, ck_path, reference = pieces
    model = _build(m, venv, ck_path, frozen)
    assert _encoder_delta(model, reference) == 0.0, (
        "PPO construction overwrote the pretrained encoder; the reload after "
        "_build is missing or ineffective"
    )


def test_frozen_encoder_does_not_move_during_learn(pieces):
    m, venv, ck_path, reference = pieces
    model = _build(m, venv, ck_path, frozen=True)
    assert not any(p.requires_grad
                   for p in model.policy.features_extractor.encoder.parameters())
    model.learn(total_timesteps=128)
    assert _encoder_delta(model, reference) == 0.0, "frozen encoder changed"


def test_unfrozen_encoder_does_move_during_learn(pieces):
    """The counterpart: without --st_frozen the encoder must actually train,
    or 'finetune' would silently be the frozen arm."""
    m, venv, ck_path, reference = pieces
    model = _build(m, venv, ck_path, frozen=False)
    assert all(p.requires_grad
               for p in model.policy.features_extractor.encoder.parameters())
    model.learn(total_timesteps=128)
    assert _encoder_delta(model, reference) > 0.0, "encoder did not train"


def test_num_heads_mismatch_is_refused_when_the_checkpoint_records_it(pieces, tmp_path):
    """A strict load_state_dict cannot see num_heads: the MAB projections are
    dim_hidden x dim_hidden however the heads split them, so a 4-head
    checkpoint loads into an 8-head encoder without complaint and computes
    something else. 3_train_st.py checkpoints carry their TrainingConfig, so
    the extractor compares the recorded geometry and refuses."""
    m, venv, ck_path, _reference = pieces
    from set_transformer.training.config import TrainingConfig

    state = torch.load(ck_path, map_location="cpu", weights_only=False)["model_state_dict"]
    config = TrainingConfig(
        num_particles=100, dim_particles=2, num_encodings=8, dim_encoder=8,
        num_inds=32, dim_hidden=128, num_heads=4, use_layer_norm=True,
        weighted_particles=True)
    path = tmp_path / "with_config.pt"
    torch.save({"model_state_dict": state, "config": config}, path)

    common = dict(num_encodings=8, dim_encoder=8, num_inds=32, dim_hidden=128,
                  ln=True, arena_scale=7.0, weight_channel=True,
                  pretrained_st_model_path=str(path))
    # Matching geometry loads.
    m.SetTransformerFeaturesExtractor(venv.observation_space, num_heads=4, **common)
    # Only the head count differs: no shape changes, must still be refused.
    with pytest.raises(RuntimeError, match="num_heads"):
        m.SetTransformerFeaturesExtractor(venv.observation_space, num_heads=2, **common)
    # A checkpoint without a config (older files) still loads on shapes alone.
    m.SetTransformerFeaturesExtractor(venv.observation_space, num_heads=2, **{
        **common, "pretrained_st_model_path": ck_path})
