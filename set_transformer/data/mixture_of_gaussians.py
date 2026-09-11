"""Random 2D mixture-of-Gaussians point sets for set-autoencoder experiments.

Each sample is a point set of shape ``(num_particles, 2)`` drawn from a mixture of
``n`` Gaussians, where ``n`` itself is drawn uniformly from ``[n_min, n_max]``. Every
component gets a random mean, a random mixture weight, and a random *full* covariance
(random rotation applied to random anisotropic scales, so components are arbitrarily
oriented ellipses).

Unlike ``synthetic_sets`` (ten hand-designed shape families), the only latent factor
here is the number of Gaussians, so ``n`` doubles as the natural evaluation axis: the
reconstruction figure walks ``n = 1 .. n_max``.

CLI:
    python -m set_transformer.data.mixture_of_gaussians \
        --out data/mog_train.npz --num_samples 20000 --num_particles 100 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

N_MIN_DEFAULT = 1
N_MAX_DEFAULT = 10

# Component-parameter ranges. Means live in a box; each axis std in [lo, hi] before a
# random rotation makes the covariance full (off-diagonal) rather than axis-aligned.
MEAN_RANGE = 3.0
STD_RANGE = (0.15, 0.8)


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sample_mog(
    num_particles: int,
    n_components: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one point set of shape ``(num_particles, 2)`` from a random ``n``-Gaussian mix.

    Random means, random Dirichlet mixture weights, and a random full 2x2 covariance
    per component (``R diag(scales^2) R^T`` with a random rotation ``R``).
    """
    means = rng.uniform(-MEAN_RANGE, MEAN_RANGE, size=(n_components, 2))
    weights = rng.dirichlet(np.ones(n_components))

    covs = np.empty((n_components, 2, 2))
    for k in range(n_components):
        scales = rng.uniform(STD_RANGE[0], STD_RANGE[1], size=2)
        rot = _rotation(rng.uniform(0.0, np.pi))
        covs[k] = rot @ np.diag(scales ** 2) @ rot.T

    # Assign each particle to a component, then draw it from that component's Gaussian.
    assign = rng.choice(n_components, size=num_particles, p=weights)
    pts = np.empty((num_particles, 2), dtype=np.float32)
    for k in range(n_components):
        mask = assign == k
        m = int(mask.sum())
        if m:
            pts[mask] = rng.multivariate_normal(means[k], covs[k], size=m)
    return pts


def _sample_separated_means(
    n_components: int,
    rng: np.random.Generator,
    min_sep: float,
    box_half: float,
) -> np.ndarray:
    """Rejection-sample ``n_components`` means in ``[-box_half, box_half]^2`` with pairwise
    distance >= ``min_sep``; fall back to an evenly spaced ring if packing is too tight."""
    means: list[np.ndarray] = []
    for _ in range(2000):
        if len(means) == n_components:
            break
        cand = rng.uniform(-box_half, box_half, size=2)
        if all(np.linalg.norm(cand - m) >= min_sep for m in means):
            means.append(cand)
    while len(means) < n_components:
        k = len(means)
        radius = min(box_half * 0.9, min_sep / (2 * np.sin(np.pi / max(n_components, 2))))
        ang = 2 * np.pi * k / n_components + rng.uniform(0, 2 * np.pi)
        means.append(radius * np.array([np.cos(ang), np.sin(ang)]))
    return np.stack(means)


def _max_sep_for_n(n_components: int, box_half: float) -> float:
    """A conservative achievable nearest-neighbour spacing for ``n`` points in the box."""
    return (2 * box_half) / (np.sqrt(n_components) + 1)


def sample_mog_with_separability(
    num_particles: int,
    n_components: int,
    rng: np.random.Generator,
    separability: float,
    box_half: float = MEAN_RANGE,
) -> np.ndarray:
    """Draw one MoG set whose separability is controlled by ``separability`` in [0, 1].

    ``separability = 0`` reproduces :func:`sample_mog` exactly (means uniform in the box,
    wide std ``[0.15, 0.8]``, no separation constraint). As it rises to 1, the means are
    pushed as far apart as fit in the *fixed* ``[-box_half, box_half]`` domain (enforced
    ``min_sep = s * max_sep(n)``) and the component std shrinks toward ``[0.08, 0.22]``, so
    the sampled points become cleanly separated clusters. The domain never grows, so there
    is no scale shift across ``separability`` or ``n``.
    """
    s = float(np.clip(separability, 0.0, 1.0))
    min_sep = s * _max_sep_for_n(n_components, box_half)
    std_lo = 0.15 + s * (0.08 - 0.15)
    std_hi = 0.80 + s * (0.22 - 0.80)

    means = _sample_separated_means(n_components, rng, min_sep, box_half)
    weights = rng.dirichlet(np.ones(n_components))
    covs = np.empty((n_components, 2, 2))
    for k in range(n_components):
        scales = rng.uniform(std_lo, std_hi, size=2)
        rot = _rotation(rng.uniform(0.0, np.pi))
        covs[k] = rot @ np.diag(scales ** 2) @ rot.T

    assign = rng.choice(n_components, size=num_particles, p=weights)
    pts = np.empty((num_particles, 2), dtype=np.float32)
    for k in range(n_components):
        mask = assign == k
        m = int(mask.sum())
        if m:
            pts[mask] = rng.multivariate_normal(means[k], covs[k], size=m)
    return pts


def generate_mog_dataset_separability(
    num_samples: int,
    num_particles: int,
    seed: int = 0,
    n_min: int = N_MIN_DEFAULT,
    n_max: int = N_MAX_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like :func:`generate_mog_dataset` but each set also draws a separability
    ``s ~ Uniform[0, 1]`` (see :func:`sample_mog_with_separability`).

    Returns ``(points, n_components, separability)``.
    """
    rng = np.random.default_rng(seed)
    n_values = rng.integers(n_min, n_max + 1, size=num_samples)
    s_values = rng.uniform(0.0, 1.0, size=num_samples)
    points = np.empty((num_samples, num_particles, 2), dtype=np.float32)
    for i in range(num_samples):
        points[i] = sample_mog_with_separability(
            num_particles, int(n_values[i]), rng, float(s_values[i])
        )
    return points, n_values.astype(np.int64), s_values.astype(np.float32)


def sample_mog_separable(
    num_particles: int,
    n_components: int,
    rng: np.random.Generator,
    min_sep: float = 3.0,
    std_range: Tuple[float, float] = (0.12, 0.35),
    box_half: float | None = None,
) -> np.ndarray:
    """Draw one point set whose components are well-separated (visually distinct blobs).

    Means are rejection-sampled with a minimum pairwise distance ``min_sep``, and component
    stds are small relative to ``min_sep`` (3-sigma << min_sep/2) so the sampled points
    form cleanly separable clusters. Covariances are still full/random (oriented ellipses).

    ``box_half`` bounds the mean domain to ``[-box_half, box_half]^2``. If None it grows
    with ``n_components`` so any ``min_sep`` stays feasible; pin it (e.g. to ``MEAN_RANGE``)
    to keep the inputs inside the training domain, which requires a small enough ``min_sep``
    to pack ``n_components`` blobs in that box.
    """
    if box_half is None:
        box_half = max(MEAN_RANGE, 0.9 * min_sep * np.sqrt(n_components))
    means: list[np.ndarray] = []
    for _ in range(2000):
        if len(means) == n_components:
            break
        cand = rng.uniform(-box_half, box_half, size=2)
        if all(np.linalg.norm(cand - m) >= min_sep for m in means):
            means.append(cand)
    while len(means) < n_components:  # fallback: drop onto an evenly spaced ring
        k = len(means)
        radius = min_sep / (2 * np.sin(np.pi / max(n_components, 2)))
        ang = 2 * np.pi * k / n_components + rng.uniform(0, 2 * np.pi)
        means.append(radius * np.array([np.cos(ang), np.sin(ang)]))
    means_arr = np.stack(means)

    weights = rng.dirichlet(np.ones(n_components))
    covs = np.empty((n_components, 2, 2))
    for k in range(n_components):
        scales = rng.uniform(std_range[0], std_range[1], size=2)
        rot = _rotation(rng.uniform(0.0, np.pi))
        covs[k] = rot @ np.diag(scales ** 2) @ rot.T

    assign = rng.choice(n_components, size=num_particles, p=weights)
    pts = np.empty((num_particles, 2), dtype=np.float32)
    for k in range(n_components):
        mask = assign == k
        m = int(mask.sum())
        if m:
            pts[mask] = rng.multivariate_normal(means_arr[k], covs[k], size=m)
    return pts


def generate_mog_dataset(
    num_samples: int,
    num_particles: int,
    seed: int = 0,
    n_min: int = N_MIN_DEFAULT,
    n_max: int = N_MAX_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(points, n_components)`` where ``n_components`` is drawn uniformly in
    ``[n_min, n_max]`` per sample.

    Returns:
        points: (num_samples, num_particles, 2) float32
        n_components: (num_samples,) int64 in [n_min, n_max]
    """
    rng = np.random.default_rng(seed)
    n_values = rng.integers(n_min, n_max + 1, size=num_samples)
    points = np.empty((num_samples, num_particles, 2), dtype=np.float32)
    for i, n in enumerate(n_values):
        points[i] = sample_mog(num_particles, int(n), rng)
    return points, n_values.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_min", type=int, default=N_MIN_DEFAULT)
    parser.add_argument("--n_max", type=int, default=N_MAX_DEFAULT)
    args = parser.parse_args()

    points, n_components = generate_mog_dataset(
        args.num_samples, args.num_particles, args.seed, args.n_min, args.n_max
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, points=points, n_components=n_components)
    np.save(out_path.with_suffix(".points.npy"), points)
    print(f"Wrote {points.shape} to {out_path} (n in [{args.n_min}, {args.n_max}])")


if __name__ == "__main__":
    main()
