"""
RL training with a pooling / moment belief encoder for Ant-Tag: DeepSet, PointNet or
k-moments (2026-09-11). One entry, three arms, selected by ``main(encoder=...)`` from the
two-line entry scripts 4_train_rl_deepset.py / 4_train_rl_pointnet.py / 4_train_rl_kmoments.py.

Same contract as the other arms: the AntTag particle filter, curriculum, reward shaping,
PPO settings and eval protocol are the CGF arm's, and ONLY the SB3 features extractor
changes. All three read the {"obs", "particles", "weights"} Dict observation, divide
particles by the arena half-width, keep PF weights in the measure, and hand the policy
[obs, encoder output]. A checkpoint from any of them evaluates through
eval_scripts/eval_true_reward_cgf.py's env.

    deepset   WeightedDeepSetFeaturesExtractor  learned; weighted-mean pool (or --pooling mean)
    pointnet  PointNetFeaturesExtractor         learned; masked-max pool (or --pooling max)
    kmoments  WeightedKMomentsFeaturesExtractor analytic; weighted mean + central moments 2..k

The learned arms take --pretrained_model_path (a DeepSetAE / PointNetAE state_dict or a
Trainer checkpoint), --frozen, --encoder_lr_scale and --unfreeze_at (PPO only; the ST
arm's finetune fixes), with the post-construction reload + verification of PITFALLS.md
section 1. Both AEs are unweighted (D inputs), so load them with --no_weight_channel; the
extractor says so.

Since 2026-09-12 (change 4.5 of the harness centralisation) this file is an ENTRY POINT:
the flags, their resolution, the run record and the PPO loop are the shared ones in
``set_transformer.rl.train`` (domain ``ant_tag``, encoder ``deepset`` / ``pointnet`` /
``kmoments``). Every flag these scripts ever took still works; the run directory is still
the cwd-relative ``runs/ant_tag_<encoder>[_<variant>]/<timestamp>_seed<seed>[_<run_tag>]/``
(``legacy_layout=True``) until change 5, model ``<encoder>_agent.zip``. Equivalently:

    python3 -m set_transformer.rl.train --domain ant_tag --encoder deepset --variant smart ...

The extractor classes are re-exported under this module's name for the zips the pooling
arms saved (SB3 pickles the class by module path).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401,E402 - registers pdomains-ant-tag-*

from set_transformer.rl.encoder_finetune import (  # noqa: E402,F401
    EncoderLRLoggingCallback,
    scale_encoder_learning_rate,
)
from set_transformer.rl.feature_extractors.pooled import (  # noqa: E402,F401
    PointNetFeaturesExtractor,
    WeightedDeepSetFeaturesExtractor,
    WeightedKMomentsFeaturesExtractor,
    reload_pretrained_pooled,
)
from set_transformer.rl.wrappers.particle_filter import (  # noqa: E402,F401
    PFDictWithWeightsObservationWrapper,
)
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter  # noqa: E402,F401

# The Ant-Tag domain pieces these arms shared with the CGF script, same objects.
from set_transformer.rl.domains.ant_tag import (  # noqa: E402,F401
    CurriculumCallback,
    CurriculumVisibilityWrapper,
    PFRewardShapingWrapper,
    _CurriculumRouter,
    _make_vec_env_from_fns,
    _make_vec_normalize,
    _resolve_reward_shaping,
    ant_tag_pf_interaction_mapper,
    get_ant_tag_arena_scale,
    get_ant_tag_pf_kwargs,
    get_env_visible_radius,
    make_ant_tag_cgf_env as make_ant_tag_belief_env,
)
from set_transformer.rl.curriculum import (  # noqa: E402,F401
    parse_curriculum as _parse_curriculum,
    parse_reward_schedule as _parse_reward_schedule,
)
from set_transformer.rl.run_records import (  # noqa: E402,F401
    default_run_dir as _default_run_dir,
    git_provenance as _git_provenance,
    tee_stdout_stderr as _tee_stdout_stderr,
    write_run_config as _write_run_config,
)
from set_transformer.rl import encoders as _encoders  # noqa: E402
from set_transformer.rl.train import main as _shared_main  # noqa: E402

#: encoder name -> (extractor class, learned?), the three arms this file serves; the
#: classes are the shared encoder table's.
ENCODERS = {
    name: (_encoders.ENCODERS[name].extractor_class, _encoders.ENCODERS[name].learned)
    for name in ("deepset", "pointnet", "kmoments")
}


def main(encoder: str = "deepset", argv=None):
    """The shared command line with this arm's domain and encoder fixed. --variant
    selects env id, particle filter and run subdir together, so the three cannot disagree."""
    if encoder not in ENCODERS:
        raise ValueError(f"encoder must be one of {sorted(ENCODERS)}, got {encoder!r}")
    return _shared_main(argv, domain="ant_tag", encoder=encoder, legacy_layout=True,
                        prog=f"4_train_rl_{encoder}.py")


if __name__ == "__main__":
    main()
