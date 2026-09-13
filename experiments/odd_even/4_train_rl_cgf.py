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
odd_even_belief_env.make_odd_even_belief_env, so one eval script
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
Every flag this script ever took still works; the run directory is ``<output root>/odd_even/<variant>/rl/cgf/<timestamp>_seed<seed>[_<run_tag>]/`` (change 5.2; the root is ``--output_root`` > ``$RL_BMDP_RUNS`` > the parent repo's ``runs/``, never the current directory). The three CGF flag helpers at the bottom
(``add_readout_and_pretrained_arguments``, ``resolve_cgf_geometry``,
``resolve_t_init_max``) are kept verbatim for ``3_pretrain_st_belief.py``,
which imports them so a checkpoint's geometry is spelled the same way on the
pretraining side; they go when that script moves onto
``set_transformer.rl.encoders`` (change 5).

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
# may name it here. The helper functions below need the sizing helpers too.
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


# ---------------------------------------------------------------------------
# Kept for experiments/odd_even/3_pretrain_st_belief.py (see the module docstring); the
# shared command line uses set_transformer.rl.encoders instead. Byte-identical to the
# pre-switch script.
# ---------------------------------------------------------------------------


#: CGF geometry flags a pretrained checkpoint carries in its ``config``. When
#: --pretrained_cgf_model_path is given and one of these is left at its
#: default, the checkpoint's value is used; an explicit value that disagrees
#: is an error, not an override (same rule as 4_train_rl_st.py's geometry).
CGF_GEOMETRY_FLAGS = ("num_cgf_features", "feature_mode", "t_param", "t_bound",
                      "t_clamp", "feature_norm", "readout_hidden", "readout_depth",
                      "readout_dim", "arena_scale", "t_init_mode", "t_init_max",
                      "x_embed_dim", "x_embed_hidden", "x_embed_depth")


def add_readout_and_pretrained_arguments(parser) -> None:
    """The readout MLP and the pretrained-encoder flags (2026-09-05).

    Shared with 3_pretrain_st_belief.py --encoder cgf so a checkpoint's
    geometry is spelled the same way in both scripts.
    """
    parser.add_argument(
        "--readout_hidden", type=int, default=0,
        help="Width of the readout MLP between the (normalised) CGF block and "
             "the policy. 0 (default) = no readout, the extractor as before. "
             "This is where a parameter-matched CGF arm keeps its budget; see "
             "--match_params.")
    parser.add_argument(
        "--readout_depth", type=int, default=0,
        help="Hidden layers in the readout MLP. 0 = none. --match_params "
             "with depth 0 uses 2 (ClusterHunt's).")
    parser.add_argument(
        "--readout_dim", type=int, default=None,
        help="Readout output width. Default 64, the ST arm's feature width, "
             "whatever --num_cgf_features is, so the policy heads are "
             "identical across arms.")
    parser.add_argument(
        "--match_params", type=int, default=None,
        help="Pick --readout_hidden so the encoder's parameter total (learned "
             "t + norm affine + readout) lands closest to this count, e.g. "
             "109448 for the small 2-SAB ST encoder. The chosen width and "
             "the exact total are printed and recorded in run_config.json.")
    parser.add_argument(
        "--pretrained_cgf_model_path", type=str, default=None,
        help="Checkpoint from 3_pretrain_st_belief.py --encoder cgf. Loaded "
             "into the extractor, RE-loaded after PPO construction and "
             "verified max|delta| == 0 (PITFALLS.md section 1). Geometry "
             "flags left at their defaults are taken from it.")
    parser.add_argument(
        "--x_embed_dim", type=int, default=0,
        help="Learned per-particle embedding phi: R^D -> R^d before the CGF, "
             "with t in R^d (idea 3, 2026-09-05). 0 = off, the plain CGF. NOTE "
             "this makes the arm a learned Deep-Set-family encoder, not a "
             "parameter-free statistic; report it as such.")
    parser.add_argument("--x_embed_hidden", type=int, default=64)
    parser.add_argument("--x_embed_depth", type=int, default=1)
    parser.add_argument(
        "--cgf_frozen", action="store_true",
        help="Freeze the WHOLE pretrained encoder: t, the running-norm "
             "statistics and the readout. Only PPO's heads learn -- the arm "
             "that matches the ST's --st_frozen. Requires a checkpoint.")


def resolve_t_init_max(args, tanh_default: float = 40.0) -> None:
    """Fill a None --t_init_max: ``tanh_default`` in tanh mode, t_clamp in
    clamp mode. Called AFTER resolve_cgf_geometry so a checkpoint's value
    wins, and BEFORE resolve_common so run_config.json records the number
    that ran. The extractor refuses t_init_max > t_clamp in clamp mode; this
    is what keeps a bare ``--t_param clamp`` run legal."""
    if args.t_init_max is None:
        args.t_init_max = (float(tanh_default) if args.t_param != "clamp"
                           else float(args.t_clamp))


def resolve_cgf_geometry(args, parser) -> None:
    """CLI > checkpoint config > default for the geometry flags, then size
    the readout if --match_params asks for it. Prints the resulting encoder
    parameter total so every run's size is on record."""
    from_ckpt = {}
    if args.pretrained_cgf_model_path:
        import torch
        checkpoint = torch.load(args.pretrained_cgf_model_path,
                                map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        from_ckpt = {k: config[k] for k in CGF_GEOMETRY_FLAGS if k in config}
        for key, ckpt_value in from_ckpt.items():
            given = getattr(args, key)
            if given == parser.get_default(key):
                setattr(args, key, ckpt_value)
            elif given != ckpt_value:
                parser.error(
                    f"--{key} {given!r} disagrees with the checkpoint's {key}="
                    f"{ckpt_value!r} ({args.pretrained_cgf_model_path}). Drop the "
                    "flag to take the checkpoint's geometry.")
        if args.match_params is not None:
            parser.error("--match_params sizes a NEW readout; with a pretrained "
                         "checkpoint the readout shape comes from the checkpoint.")
    particle_dim = 1  # Odd-Even: scalar state
    t_dim = args.x_embed_dim if args.x_embed_dim > 0 else particle_dim
    raw_dim = cgf_raw_dim(args.num_cgf_features, t_dim, args.feature_mode)
    fixed = non_readout_param_count(args.num_cgf_features, particle_dim,
                                    args.feature_mode, args.t_frozen, args.feature_norm,
                                    args.x_embed_dim, args.x_embed_hidden, args.x_embed_depth)
    if args.match_params is not None:
        if args.readout_depth <= 0:
            args.readout_depth = 2
        out_dim = (args.readout_dim if args.readout_dim is not None
                   else WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM)
        args.readout_hidden, total = matched_readout_hidden(
            args.match_params, raw_dim, args.readout_depth, out_dim, fixed)
        print(f"CGF readout sized to match {args.match_params:,} params: "
              f"hidden={args.readout_hidden} depth={args.readout_depth} "
              f"-> encoder total {total:,} ({100 * (total - args.match_params) / args.match_params:+.2f}%)")
    else:
        from set_transformer.rl.feature_extractors.cgf import readout_param_count
        out_dim = args.readout_dim if args.readout_dim is not None else (
            WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM if args.readout_depth > 0 else raw_dim)
        total = fixed + readout_param_count(raw_dim, args.readout_hidden,
                                            args.readout_depth, out_dim)
        print(f"CGF encoder parameters: {total:,} (raw block {raw_dim}, "
              f"readout hidden={args.readout_hidden} depth={args.readout_depth})")
    args.encoder_params = int(total)
    if from_ckpt:
        print(f"CGF geometry taken from checkpoint: "
              + ", ".join(f"{k}={getattr(args, k)!r}" for k in from_ckpt))




if __name__ == "__main__":
    main()
