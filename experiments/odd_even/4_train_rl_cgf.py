"""
RL training with a weighted CGF belief encoder on the Odd-Even POMDP.

Step 4 of the pipeline, and the BASELINE arm of the encoder comparison:

    CGF_j = log(sum_i w_i * exp(<t_j, x_i>))

where x_i is a particle (a centred, scaled state value) and w_i is its
particle-filter weight. The t_j directions are learned under PPO.

This module owns the training loop, the CLI and the run bookkeeping.
4_train_rl_st.py and 4_train_rl_gaussian.py import it and swap ONLY the
features extractor, exactly as the three Ant-Tag arms do. All three read the
same {"obs", "particles", "weights"} dict observation from
``set_transformer.rl.domains.odd_even.make_odd_even_belief_env``, so one eval script
(eval_scripts/eval_true_reward_odd_even.py) serves all three.

WEIGHTS ARE THE BELIEF HERE. The hidden state is static, so the particle
support never moves and every particle sits on a state value for the whole
episode. The exact posterior's effective sample size falls to about 1 of 50
within ~20 observations, which means an UNWEIGHTED reading of this particle
set sees the same near-uniform cloud over [1, n] at every step of every
episode and carries no information at all. This is a stronger version of the
Ant-Tag counterweighted-den case (ESS about 11 of 100).

--t_init_mode DEFAULTS TO spread_1d, not to the Ant-Tag default. On Ant-Tag
the spread init is what put the CGF features where the signal already was, so
PPO needed no ~10x growth of ||t_j|| to resolve it; the 0.1-scale linspace
init has to grow into place first. spread_1d is the 1-D analogue: log-spaced
magnitudes in BOTH signs, from 0.25 up to --t_init_max. It is a separate mode
from "spread", whose 8 planar directions and rho_hi=2.8 are intrinsically 2-D.

t IS TANH-BOUNDED AT 50 BY DEFAULT (2026-09-05), the ClusterHunt/LeastMass
pattern: t = t_bound * tanh(raw_t), smooth everywhere, so a probe never
freezes the way it does past a hard clamp. Why 50: same-parity neighbours are
2 / 24.5 = 0.0816 apart in normalized units and the CGF only resolves them
once t * 0.0816 is of order 2..4; the legacy clamp at 2.0 gave 0.16, which is
why every feature was a multiple of the posterior mean (domain_mds/oddeven.md,
2026-09-04/05). --t_init_max 40 makes the init span that range. The CGF block
is standardised per feature with running statistics (--feature_norm running)
because at wide t the raw K values reach magnitudes of tens; since 2026-09-06
those statistics are held fixed for each PPO collect + update cycle and
refreshed from the rollout buffer in between (--running_norm_update rollout,
RolloutFeatureNormCallback), so the stored and recomputed log-probs are
standardised identically -- PITFALLS.md section 8 item 5. The exact3M runs
predate all of this and used --t_param clamp --t_clamp 2.0 --feature_norm
none; run_config.json records which generation a run belongs to.

Know what this can and cannot buy before launching 3M steps: the offline mode
probe puts K'(t) at +-50 at 0.60 / 0.69 on the posterior mode, against the
Gaussian arm's 0.58 / 0.74 and the exact posterior's 0.997. This arm is the
correctly computed CGF, not a route to the oracle.

REWARD NORMALIZATION IS ON. -(pred - s*)^2 reaches -2401 at n=50, and
PITFALLS.md records a value loss of order 10^3 dominating a learned encoder's
parameters when the extractor is shared between the policy and value heads.
Training normalizes the reward; every eval reads the RAW reward.

Since 2026-09-12 (change 4.4 of the harness centralisation) this file is an
ENTRY POINT: the flags, their resolution, the run record and the PPO loop are
the shared ones in ``set_transformer.rl.train`` (domain ``odd_even``, encoder
``cgf``), and the belief env lives in ``set_transformer.rl.domains.odd_even``.
Every flag this script ever took still works; the run directory is ``<output root>/odd_even/<variant>/rl/cgf/<timestamp>_seed<seed>[_<run_tag>]/`` (change 5.2; the root is ``--output_root`` > ``$RL_BMDP_RUNS`` > the parent repo's ``runs/``, never the current directory). The three CGF flag helpers
``3_pretrain_st_belief.py`` borrows (``add_readout_and_pretrained_arguments``,
``resolve_cgf_geometry``, ``resolve_t_init_max``) live in
``set_transformer.rl.encoders`` since change 5.3a.

Usage:
    python3 4_train_rl_cgf.py --variant oe50 --total_timesteps 500000 \
        --run_tag cgf_v1

    python3 4_train_rl_cgf.py --list_variants
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-exported under this module's name: SB3 pickles a policy's features-extractor CLASS
# into the saved zip by module path, and a zip saved by an older version of this script
# may name it here.
from set_transformer.rl.feature_extractors.cgf import (  # noqa: E402,F401
    EncoderDriftLoggingCallback,
    RolloutFeatureNormCallback,
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
    cgf_raw_dim,
    matched_readout_hidden,
    non_readout_param_count,
)
# The rest of this module's historical surface, for the tests, diagnostics and eval that read
# these names off it (they live in the package since changes 1c / 1d).
from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401
    make_odd_even_belief_env,
    make_vec_env_from_fns,
    make_vec_normalize,
)
from set_transformer.rl.run_records import (  # noqa: E402,F401
    RUN_STATUS_FILENAME,
    TeeStream as _TeeStream,
    default_run_dir as _default_run_dir,
    git_provenance as _git_provenance,
    read_run_status,
    tee_stdout_stderr as _tee_stdout_stderr,
    write_run_config as _write_run_config,
    write_run_status,
)
from set_transformer.rl.train import main as _shared_main  # noqa: E402


def main(argv=None):
    """The shared command line with this arm's domain and encoder fixed."""
    return _shared_main(argv, domain="odd_even", encoder="cgf",
                        prog="4_train_rl_cgf.py")


# The three CGF flag helpers 3_pretrain_st_belief.py used to read off this module live in
# set_transformer.rl.encoders since change 5.3a; re-exported for anything that still looks here.
from set_transformer.rl.encoders import (  # noqa: E402,F401
    CGF_GEOMETRY_FLAGS,
    add_readout_and_pretrained_arguments,
    resolve_cgf_geometry,
    resolve_t_init_max,
)


if __name__ == "__main__":
    main()
