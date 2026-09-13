"""
RL training with a Set Transformer belief encoder for Ant-Tag.

Third arm of the encoder comparison, alongside 4_train_rl_cgf.py (weighted
empirical CGF) and 4_train_rl_gaussian.py (weighted mean + covariance). It
keeps the AntTag particle filter, curriculum, reward shaping, masking and
eval protocol of the CGF arm and swaps ONLY the belief encoder: a
SetTransformer (ISAB encoder + PMA/SAB decoder head) consumes the particle
set and emits num_encodings x dim_encoder features, which are concatenated
with the base observation exactly like the CGF features are.

WEIGHTS. The CGF and Gaussian encoders are both *weighted* -- they read
obs_dict["weights"]. The legacy ST extractors ignore the PF weights, which
on the counterweighted-den envs throws away real evidence: the alarm /
silence likelihood in CounterweightedDenAntTagParticleFilter.update()
multiplies weights by ALARM_EPS rather than deleting particles, and weights
stay non-uniform between resamples. So by default the normalized weight is
appended as a third input channel per particle (scaled by num_particles, so
a uniform belief feeds 1.0). Pass --no_st_weight_channel for the legacy
unweighted behavior; the arm is then no longer information-matched to CGF.

Pretrained weights are OPTIONAL and off by default. The CGF baseline learned
its t_values from scratch under PPO, so the matched ST run is likewise
trained end-to-end by PPO. --pretrained_st_model_path (steps 2+3 of the
pipeline: 2_collect_pf_dataset.py then 3_train_st.py), --st_frozen,
--st_encoder_lr_scale (the finetune-collapse fix) and --resume_from (fork a
run from one of its checkpoints; every progress-based schedule resumes at
the checkpoint's progress) exist for the pretrained and forked variants.

Since 2026-09-12 (change 4.5 of the harness centralisation) this file is an
ENTRY POINT: the flags, their resolution (geometry: CLI > checkpoint config >
the 32 / 128 / 2 default), the reload-after-PPO step with its max|delta| == 0
verification (PITFALLS.md section 1), the run record and the PPO loop are the
shared ones in ``set_transformer.rl.train`` (domain ``ant_tag``, encoder
``st``). Every flag this script ever took still works; the run directory is ``<output root>/ant_tag/<variant>/rl/st/<timestamp>_seed<seed>[_<run_tag>]/`` (change 5.2; the root is ``--output_root`` > ``$RL_BMDP_RUNS`` > the parent repo's ``runs/``, never the current directory). Equivalently:

    python3 -m set_transformer.rl.train --domain ant_tag --encoder st --variant smart ...

The names below are re-exported under this module's name: SB3 pickles a
policy's features-extractor CLASS into the saved zip by module path, so a
recorded ST checkpoint loads through
``getattr(import_module("4_train_rl_st"), "SetTransformerFeaturesExtractor")``;
the eval scripts and the tests read the rest off this module.

Usage (Counterweighted-Den terminal-phantom env):

    python3 4_train_rl_st.py --variant cdens_terminal \\
        --total_timesteps 6000000 --seed 0 --device cuda:0 \\
        --ppo_n_steps 4096 --n_epochs 10 --target_kl 0.03 --lr_anneal \\
        --reward_schedule "0:1:0:0,0.2:1:0:0,0.5:0:0:50,1:0:0:50" \\
        --eval_freq 40000 --n_eval_episodes 30 \\
        --run_tag terminal_v1_dist0_noent

(--curriculum and --evasion_curriculum come from the variant registry; pass
them explicitly only to override.)
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401,E402 - registers pdomains-ant-tag-*

# The belief encoder and its logging callback, as the SAME objects the package defines.
from set_transformer.rl.feature_extractors.st import (  # noqa: E402,F401
    STFeatureLoggingCallback,
    SetTransformerFeaturesExtractor,
)
from set_transformer.rl.encoder_finetune import (  # noqa: E402,F401
    EncoderLRLoggingCallback,
    scale_encoder_learning_rate,
)
from set_transformer.rl.pretrained_encoder import reload_pretrained  # noqa: E402,F401
from set_transformer.rl.wrappers.particle_filter import (  # noqa: E402,F401
    PFDictWithWeightsObservationWrapper,
)
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter  # noqa: E402,F401

# The Ant-Tag domain pieces this arm shared with the CGF script, same objects.
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
    default_run_dir as _shared_default_run_dir,
    git_provenance as _git_provenance,
    resume_vecnormalize_path as _default_resume_vecnormalize,
    tee_stdout_stderr as _tee_stdout_stderr,
    write_run_config as _write_run_config,
)
from set_transformer.rl.train import main as _shared_main  # noqa: E402


def _default_run_dir(seed: int, run_subdir: str = "ant_tag_st",
                      run_tag: str | None = None) -> str:
    """run_records.default_run_dir with this script's historical default subfolder."""
    return _shared_default_run_dir(seed, run_subdir, run_tag)


def main(argv=None):
    """The shared command line with this arm's domain and encoder fixed. --variant
    selects env id, particle filter and run subdir together, so the three cannot disagree."""
    return _shared_main(argv, domain="ant_tag", encoder="st",
                        prog="4_train_rl_st.py")


if __name__ == "__main__":
    main()
