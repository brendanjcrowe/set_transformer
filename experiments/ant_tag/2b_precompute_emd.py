"""Pipeline step 2b: precompute the pairwise-EMD matrix for latent metric alignment.

Env-generic: reads the ``.npz`` that ``2_collect_pf_dataset.py`` wrote (Ant-Tag
or Odd-Even), loads it EXACTLY as ``3_train_st.py`` will -- same
``(x - particle_centre) / particle_scale`` mapping, same weights -- and writes
the pairwise debiased-Sinkhorn matrix over the resulting clouds plus a JSON
sidecar recording what it was built from. Only the ALIGNED pretraining arm
(``3_train_st.py --align_lambda > 0``) needs this step.

    python3 2b_precompute_emd.py --data_path data/cdens_terminal_pf_dataset.npz \\
        --sinkhorn_blur 0.01 --max_samples 20000
    python3 2b_precompute_emd.py --data_path ../odd_even/data/oe50_short_pf_dataset.npz \\
        --sinkhorn_blur 0.05

Weights. When the dataset carries PF weights the matrix is between WEIGHTED
measures (mass in the measure, never in the metric), with each cloud's
debiasing self-term computed under its own weights. This is what makes the
matrix informative on the Odd-Even exact-support filter, where every cloud has
the same 50-state support and only the weights differ -- an unweighted matrix
there is identically zero, and the script refuses to write one unless told to.
``--ignore_weights`` gives the unweighted ablation and must then also be passed
to ``3_train_st.py``.

Blur and scaling MUST equal the ``--sinkhorn_blur`` / ``--sinkhorn_scaling``
you will train with: the alignment target and the reconstruction objective have
to measure the same geometry. ``3_train_st.py`` reads the sidecar and refuses a
mismatch, as it refuses a matrix built from a different dataset file
(``data_sha256``), a different frame (``particle_scale`` / ``particle_centre``)
or a different weightedness.

Cost is O(N^2) Sinkhorn calls and 4 N^2 bytes: about 1 h and 1.6 GB for 20k
clouds on an RTX 4070 Ti. ``--max_samples K`` builds the matrix over the first
K rows only; ``3_train_st.py`` then cuts the dataset to those same rows (the
sidecar records ``n_samples``). Resumable at row-block granularity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from set_transformer.data.dataset import get_dataset  # noqa: E402
from set_transformer.emd_matrix import (  # noqa: E402
    DEFAULT_BLUR,
    DEFAULT_SCALING,
    compute_matrix,
    dataset_sha256,
    matrix_stats,
    sidecar_path,
    write_sidecar,
)

#: Off-diagonal spread below which the alignment target carries no signal.
DEGENERATE_OFFDIAG_STD = 1e-6


def default_out_path(data_path: str) -> Path:
    data = Path(data_path)
    return data.with_name(data.stem + "_emd.npy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute the pairwise debiased-Sinkhorn matrix for the "
                    "latent metric-alignment loss (pipeline step 2b)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="The .npz from 2_collect_pf_dataset.py (any env).")
    parser.add_argument("--out_path", type=str, default=None,
                        help="Matrix destination (float32 memmap). Default: "
                             "<data stem>_emd.npy beside the dataset; the JSON "
                             "sidecar lands at the same stem.")
    parser.add_argument("--sinkhorn_blur", type=float, default=DEFAULT_BLUR,
                        help="Sinkhorn eps in NORMALIZED coordinate units; must "
                             "equal 3_train_st.py's --sinkhorn_blur.")
    parser.add_argument("--sinkhorn_scaling", type=float, default=DEFAULT_SCALING,
                        help="Epsilon-scaling decay; must equal 3_train_st.py's "
                             "--sinkhorn_scaling.")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Build over the first K rows only (0 = all). Cost is "
                             "quadratic, so this is the knob for large datasets.")
    parser.add_argument("--ignore_weights", action="store_true",
                        help="Unweighted ablation: treat every cloud as uniform. "
                             "Pass the same flag to 3_train_st.py.")
    parser.add_argument("--no_particle_scaling", action="store_true",
                        help="Raw coordinates (no centre, no scale). Only correct "
                             "if 3_train_st.py also runs with --no_particle_scaling.")
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--pair_chunk", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip the check of the manual debiasing against "
                             "geomloss(debias=True) on a sample of pairs.")
    parser.add_argument("--allow_degenerate", action="store_true",
                        help="Write the matrix even if its off-diagonal spread is "
                             "~0 (the alignment loss would have nothing to fit).")
    args = parser.parse_args()

    if args.sinkhorn_blur <= 0 or args.sinkhorn_scaling <= 0:
        parser.error("--sinkhorn_blur and --sinkhorn_scaling must be positive")

    out_path = Path(args.out_path) if args.out_path else default_out_path(args.data_path)
    max_samples = args.max_samples or None
    particle_scale = 1.0 if args.no_particle_scaling else None
    particle_centre = 0.0 if args.no_particle_scaling else None

    dataset = get_dataset(
        args.data_path, "cpu", load_weights=not args.ignore_weights,
        particle_scale=particle_scale, particle_centre=particle_centre,
        max_samples=max_samples,
    )
    points = dataset.data.numpy()
    weights = None if dataset.weights is None else dataset.weights.numpy()
    n = len(dataset)
    if args.ignore_weights and weights is None:
        print("NOTE: --ignore_weights given but the dataset has no weights.")

    print(f"Dataset: {args.data_path}")
    print(f"  {n} clouds x {dataset.num_particles} particles x {dataset.particle_dim} "
          f"coords, {'WEIGHTED' if weights is not None else 'uniform'}")
    print(f"  frame: (x - {dataset.particle_centre}) / {dataset.particle_scale}; "
          f"blur {args.sinkhorn_blur} (= {args.sinkhorn_blur * dataset.particle_scale:.4f} "
          f"env units), scaling {args.sinkhorn_scaling}")
    print(f"  matrix: {out_path} ({4 * n * n / 1e9:.2f} GB), {n * (n - 1) // 2} pairs")

    t0 = time.time()
    matrix = compute_matrix(
        points, out_path, blur=args.sinkhorn_blur, p=args.p, block=args.block,
        pair_chunk=args.pair_chunk, device=args.device, resume=not args.no_resume,
        verify=not args.no_verify, weights=weights, scaling=args.sinkhorn_scaling,
    )
    stats = matrix_stats(matrix)
    stats["wall_seconds"] = round(time.time() - t0, 1)
    print(json.dumps(stats, indent=2))

    if stats["offdiag_std"] < DEGENERATE_OFFDIAG_STD:
        msg = ("Pairwise distances are (nearly) constant -- the alignment target "
               "carries no signal on this dataset. On an exact-support filter "
               "this is what an UNWEIGHTED matrix looks like: every cloud has the "
               "same support. Drop --ignore_weights, or pass --allow_degenerate.")
        if not args.allow_degenerate:
            out_path.with_suffix(".progress.json").unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            raise SystemExit("ERROR: " + msg + " (matrix deleted)")
        print("WARNING: " + msg)

    extra = {
        "data_path": str(Path(args.data_path).resolve()),
        "data_sha256": dataset_sha256(Path(args.data_path)),
        "particle_scale": float(dataset.particle_scale),
        "particle_centre": float(dataset.particle_centre),
        "num_particles": int(dataset.num_particles),
        "particle_dim": int(dataset.particle_dim),
        "max_samples": max_samples,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "torch": torch.__version__,
    }
    write_sidecar(out_path, args.data_path, n, args.sinkhorn_blur, args.p, stats,
                  weighted=weights is not None, scaling=args.sinkhorn_scaling,
                  extra=extra)
    out_path.with_suffix(".progress.json").unlink(missing_ok=True)
    print(f"Wrote {out_path} and {sidecar_path(out_path)}")
    print("Next: 3_train_st.py --data_path {} --emd_matrix_path {} --align_lambda 0.2 "
          "--sinkhorn_blur {} --sinkhorn_scaling {}{}".format(
              args.data_path, out_path, args.sinkhorn_blur, args.sinkhorn_scaling,
              " --ignore_weights" if args.ignore_weights else ""))


if __name__ == "__main__":
    main()
