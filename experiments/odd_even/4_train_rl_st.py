"""
RL training with a Set Transformer belief encoder on the Odd-Even POMDP.

Second arm of the encoder comparison, alongside 4_train_rl_cgf.py (weighted
empirical CGF) and 4_train_rl_gaussian.py (weighted mean + variance). The arms
differ in the encoder alone: the env, the PPO loop and the run bookkeeping are
shared.

WHAT THIS DOMAIN ASKS OF THE ENCODER, which is not what Ant-Tag asks. The
particle support here is the constant set {1, ..., n} at every step of every
episode; the entire belief lives in the WEIGHTS. So the ST is being asked to
compress an n-vector of masses attached to fixed positions, not a 2-D cloud
whose shape carries the information. Expect the ranking against CGF to differ
from Ant-Tag. --st_weight_channel (default on) is therefore not an option
here in the way it is there: with it off, the encoder reads the same constant
set forever and can carry no belief information at all.

PRETRAINED WEIGHTS ARE OPTIONAL AND OFF BY DEFAULT, because the CGF baseline
learns its t_values from scratch under PPO and the matched ST run is likewise
trained end to end. --pretrained_st_model_path (from 3_train_st.py on a
2_collect_pf_dataset.py .npz) and --st_frozen exist for the pretrained
variant, mirroring the CGF arm's --t_frozen.

PITFALLS.md SECTION 1 APPLIES IN FULL, and it is the most expensive bug in
this repo. SB3's ActorCriticPolicy._build ends with

    self.apply(partial(self.init_weights, gain=...))

which walks the WHOLE policy -- features extractor included -- and
re-initializes every Linear. An encoder loaded in the extractor's __init__ is
overwritten a moment later, and the "loaded encoder ... FROZEN" line prints
BEFORE the overwrite, so the run looks correct. That cost two 6M-step runs,
both reported 0%. The shared trainer therefore reloads the encoder AFTER
PPO(...) returns, re-freezes it, and ASSERTS max|delta| == 0 against the
checkpoint (set_transformer.rl.pretrained_encoder.reload_pretrained). The
assertion is not decoration: it is the only thing that distinguishes a
working pretrained arm from a frozen-noise arm in the logs.

Since 2026-09-12 (change 4.4 of the harness centralisation) this file is an
ENTRY POINT: the flags, their resolution (geometry: CLI > checkpoint config >
the small 16 / 64 / 2 default), the run record and the PPO loop are the shared
ones in ``set_transformer.rl.train`` (domain ``odd_even``, encoder ``st``).
Every flag this script ever took still works; the run directory is still the
cwd-relative ``runs/odd_even_st_<variant>/...`` (``legacy_layout=True``) until
change 5. The names below are re-exported for the tests and diagnostics that
read them off this module.

Usage:
    python3 4_train_rl_st.py --variant oe50 --total_timesteps 500000 \\
        --num_encodings 8 --dim_encoder 8 --run_tag st_v1

    python3 4_train_rl_st.py --variant oe50 \\
        --pretrained_st_model_path experiments/.../checkpoint_best.pt \\
        --st_frozen
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-exported under this module's name (saved zips name the extractor class by module
# path; tests/test_st_finetune_fixes.py and tests/test_odd_even_pipeline.py read the rest
# off this module).
from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401
    OddEvenSTFeatureSentinel,
    make_odd_even_belief_env,
)
from set_transformer.rl.encoder_finetune import (  # noqa: E402,F401
    EncoderLRLoggingCallback,
    UnfreezeEncoderCallback,
    _ScaledLRParamGroup,
    _group_collapsing,
    scale_encoder_learning_rate,
)
from set_transformer.rl.feature_extractors.st import (  # noqa: E402,F401
    STFeatureLoggingCallback,
    SetTransformerFeaturesExtractor,
)
from set_transformer.rl.pretrained_encoder import (  # noqa: E402
    reload_pretrained,
    verify_matches_checkpoint,
)
from set_transformer.rl.train import main as _shared_main  # noqa: E402

#: The reload-after-PPO step, under this module's historical name.
reload_pretrained_encoder = reload_pretrained


def assert_encoder_matches_checkpoint(model, path: str) -> None:
    """max|delta| == 0 between the live encoder and the checkpoint.

    The verification snippet from PITFALLS.md section 1, run once per
    training start. Costs milliseconds and is the only positive evidence
    that the reload landed. The comparison itself lives in
    set_transformer.rl.pretrained_encoder, shared with the CGF arm.
    """
    extractor = model.policy.features_extractor
    verify_matches_checkpoint(extractor.reference_state(path),
                              extractor.encoder_state_dict(), path,
                              label="ST encoder")


def main(argv=None):
    """The shared command line with this arm's domain and encoder fixed."""
    return _shared_main(argv, domain="odd_even", encoder="st", legacy_layout=True,
                        prog="4_train_rl_st.py")


if __name__ == "__main__":
    main()
