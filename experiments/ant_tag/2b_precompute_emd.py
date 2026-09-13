"""Pipeline step 2b: precompute the pairwise-EMD matrix for latent metric alignment.

Since batch 7.5 of the harness centralisation (2026-09-13) this file is an ENTRY POINT of
``set_transformer/rl/precompute_emd.py``; every flag it took still works, and

    python3 -m set_transformer.rl.precompute_emd --data_path <dataset.npz> --sinkhorn_blur 0.01

is the same run. Env-generic: reads the ``.npz`` the collector wrote (Ant-Tag or Odd-Even),
loads it EXACTLY as the reconstruction pretraining will, and writes the pairwise
debiased-Sinkhorn matrix ``<stem>_emd.npy`` beside it plus a JSON sidecar. Only the ALIGNED
pretraining arm (``--align_lambda > 0``) needs this step; see the module's docstring for the
weights, blur / scaling and cost rules.

    python3 2b_precompute_emd.py --data_path data/cdens_terminal_pf_dataset.npz \\
        --sinkhorn_blur 0.01 --max_samples 20000
    python3 2b_precompute_emd.py --data_path ../odd_even/data/oe50_short_pf_dataset.npz \\
        --sinkhorn_blur 0.05
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.precompute_emd import (  # noqa: E402,F401 - re-exported
    DEGENERATE_OFFDIAG_STD,
    default_out_path,
)
from set_transformer.rl.precompute_emd import main as _shared_main  # noqa: E402


def main(argv=None):
    return _shared_main(argv, prog="2b_precompute_emd.py")


if __name__ == "__main__":
    main()
