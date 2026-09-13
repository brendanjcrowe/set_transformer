"""
Pretrain an encoder by reconstruction on a particle-filter dataset (pipeline step 3).

Since batch 7.2 of the harness centralisation (2026-09-13) this file is an ENTRY POINT of
the shared pretraining command, ``set_transformer/rl/pretrain.py``, with the objective fixed
to ``reconstruction``::

    python3 -m set_transformer.rl.pretrain --domain ant_tag --encoder st \\
        --objective reconstruction --data_path data/<variant>_pf_dataset.npz ...

is the same run. Every flag this script ever took still works, and a command that omits a
flag still gets the value this script always used (listed in ``LEGACY_DEFAULTS`` below;
the package command's own defaults are the RL trainer's, so a pretraining run and an RL run
with the same flags always agree on the encoder's geometry -- this script keeps its
historical ``--dim_encoder 2`` / ``--num_inds 32`` / ``--dim_hidden 128`` for fidelity to
recorded commands, so pass the geometry explicitly, as every recorded run did).

Step 3 of the pipeline:
  1) Train locomotion policy   (1_train_locomotion.py)
  2) Collect PF dataset        (2_collect_pf_dataset.py)
  3) Pretrain the encoder      (this script)
  4) Train RL with the encoder (4_train_rl_st.py --pretrained_st_model_path)

--encoder cgf pretrains the CGF arm's block instead (WeightedCGFFeaturesExtractor
through a PFDecoder, same loss / alignment / frame) and exports
checkpoints/checkpoint_best_cgf_arm.pt for 4_train_rl_cgf.py --pretrained_cgf_model_path.

Nothing here is env-specific: the command consumes whatever 2_collect_pf_dataset.py wrote,
for any env + particle filter pair; the domain and variant are read off the dataset's
metadata (or --domain / --variant) and decide only where the output lands:
<root>/runs/<domain>/<variant>/pretrain/<experiment_name>/<loss>_<timestamp>/ (--base_dir
overrides). The architecture flags must match what the RL feature extractor will build.

WEIGHTED SETS. A .npz from 2_collect_pf_dataset.py carries the PF weights alongside the
particles, and they are used by default (mass in the measure, never in the ground metric;
--ignore_weights for the unweighted ablation; see rl/pretrain_objectives/reconstruction.py).

To pretrain an encoder for 4_train_rl_st.py's defaults, match its geometry:
--num_encodings 8 --dim_encoder 8, weights on (so the RL side keeps its
default --st_weight_channel).

Usage:
    python3 3_train_st.py \\
        --data_path data/cdens_terminal_pf_dataset.npz \\
        --num_encodings 8 --dim_encoder 8 \\
        --sinkhorn_blur 0.01 --num_epochs 100
"""

import argparse
import sys
from pathlib import Path

# Same bootstrap as every other script in this directory: put the package root
# on sys.path so `set_transformer` resolves to the package rather than to the
# submodule directory of the same name.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.pretrain import main as _shared_main  # noqa: E402
from set_transformer.rl.pretrain_objectives.reconstruction import (  # noqa: E402,F401
    dataset_metadata as _dataset_metadata,
)

#: What this script used when the flag was omitted, where that differs from the package
#: command (whose defaults are the RL trainer's). Filled into the command line when absent.
LEGACY_DEFAULTS = {
    "--encoder": "st",
    "--experiment_name": "ant_tag_st",
    "--dim_encoder": "2",
    "--num_inds": "32",
    "--dim_hidden": "128",
    "--num_post_sab": "2",
}
#: The same, for --encoder cgf only.
LEGACY_CGF_DEFAULTS = {
    "--t_init_mode": "spread",
    "--feature_norm": "none",
}


def _value(argv: list[str], flag: str):
    """The value given for ``flag`` on the command line, or None."""
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _given(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in argv)


def with_legacy_defaults(argv: list[str]) -> list[str]:
    """``argv`` plus this script's historical defaults for every flag it does not set."""
    argv = list(argv)
    for flag, value in LEGACY_DEFAULTS.items():
        if not _given(argv, flag):
            argv += [flag, value]
    if _value(argv, "--encoder") == "cgf":
        for flag, value in LEGACY_CGF_DEFAULTS.items():
            if not _given(argv, flag):
                argv += [flag, value]
        t_param = _value(argv, "--t_param") or "clamp"
        if t_param in ("tanh", "polar"):
            # The script had no env to read a bound from, so it required the flag; the
            # package command would take the variant's registry rule instead.
            bound = _value(argv, "--t_bound")
            if bound is None or float(bound) <= 0:
                argparse.ArgumentParser(prog="3_train_st.py").error(
                    f"--t_param {t_param} needs --t_bound > 0 (see its help "
                    "for the sizing rule); this script has no env to read it from.")
            if not _given(argv, "--t_init_max"):
                # 4_train_rl_cgf.resolve_cgf_encoder_args: 0.8 * t_bound, whatever the init.
                argv += ["--t_init_max", str(0.8 * float(bound))]
    return argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv:
        return _shared_main(argv, objective="reconstruction", prog="3_train_st.py")
    # fallback_domain: a dataset that records no variant, placed with --base_dir, gets the
    # Ant-Tag encoder defaults -- this script always lived in experiments/ant_tag/ and had no
    # env to ask (the legacy defaults above fix the geometry anyway).
    return _shared_main(with_legacy_defaults(argv), objective="reconstruction", prog="3_train_st.py",
                        fallback_domain="ant_tag")


if __name__ == "__main__":
    main()
