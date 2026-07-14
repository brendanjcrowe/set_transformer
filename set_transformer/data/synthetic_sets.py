"""Synthetic 2D point-set distributions for set-autoencoder experiments.

Each call to `sample_set` draws one point set of shape (num_particles, 2) from
one of the supported distribution families. Family parameters (means, widths,
orientations, mixture weights, noise levels) are themselves randomized within
sensible ranges so the training distribution covers a wide range of shapes.

CLI:
    python -m set_transformer.data.synthetic_sets \
        --out data/synthetic_train.npz \
        --num_samples 50000 --num_particles 100 --seed 0

The output `.npz` contains `points: (N, num_particles, 2)` and
`family_ids: (N,)`. A `.npy` variant containing only `points` is also written
to `<out>.points.npy` for compatibility with the existing trainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FAMILY_NAMES: List[str] = [
    "iso_gaussian",
    "aniso_gaussian",
    "mixture_2g",
    "mixture_3g",
    "uniform_square",
    "uniform_disk",
    "ring",
    "two_moons",
    "swiss_roll_2d",
    "spiral",
]

FAMILY_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(FAMILY_NAMES)}


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _sample_iso_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    mean = rng.uniform(-2.0, 2.0, size=2)
    std = rng.uniform(0.2, 1.0)
    return mean + std * rng.standard_normal((n, 2))


def _sample_aniso_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    mean = rng.uniform(-2.0, 2.0, size=2)
    scales = rng.uniform(0.2, 1.0, size=2)
    theta = rng.uniform(0.0, np.pi)
    pts = rng.standard_normal((n, 2)) * scales
    return mean + pts @ _rotation(theta).T


def _sample_mixture_gauss(
    n: int, rng: np.random.Generator, n_components: int
) -> np.ndarray:
    means = rng.uniform(-2.0, 2.0, size=(n_components, 2))
    stds = rng.uniform(0.15, 0.6, size=n_components)
    weights = rng.dirichlet(np.ones(n_components) * 2.0)
    component = rng.choice(n_components, size=n, p=weights)
    pts = means[component] + stds[component][:, None] * rng.standard_normal((n, 2))
    return pts


def _sample_uniform_square(n: int, rng: np.random.Generator) -> np.ndarray:
    side = rng.uniform(0.5, 2.5)
    center = rng.uniform(-1.5, 1.5, size=2)
    theta = rng.uniform(0.0, np.pi / 2)
    pts = rng.uniform(-side / 2, side / 2, size=(n, 2))
    return center + pts @ _rotation(theta).T


def _sample_uniform_disk(n: int, rng: np.random.Generator) -> np.ndarray:
    radius = rng.uniform(0.5, 2.0)
    center = rng.uniform(-1.5, 1.5, size=2)
    r = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * radius
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1) + center


def _sample_ring(n: int, rng: np.random.Generator) -> np.ndarray:
    inner = rng.uniform(0.4, 1.0)
    outer = inner + rng.uniform(0.2, 0.8)
    center = rng.uniform(-1.5, 1.5, size=2)
    r = np.sqrt(
        rng.uniform(inner ** 2, outer ** 2, size=n)
    )
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1) + center


def _sample_two_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.uniform(0.02, 0.15)
    n_upper = n // 2
    n_lower = n - n_upper
    theta_u = rng.uniform(0.0, np.pi, size=n_upper)
    upper = np.stack([np.cos(theta_u), np.sin(theta_u)], axis=-1)
    theta_l = rng.uniform(0.0, np.pi, size=n_lower)
    lower = np.stack([1.0 - np.cos(theta_l), -np.sin(theta_l) - 0.5], axis=-1)
    pts = np.concatenate([upper, lower], axis=0)
    pts = pts + rng.standard_normal(pts.shape) * noise
    # Random rotation + small translation to vary appearance.
    theta = rng.uniform(0.0, 2 * np.pi)
    return pts @ _rotation(theta).T + rng.uniform(-0.5, 0.5, size=2)


def _sample_swiss_roll_2d(n: int, rng: np.random.Generator) -> np.ndarray:
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0.0, 1.0, size=n))
    x = t * np.cos(t)
    y = t * np.sin(t)
    noise = rng.uniform(0.05, 0.3)
    pts = np.stack([x, y], axis=-1) * 0.15
    pts = pts + rng.standard_normal(pts.shape) * noise
    theta = rng.uniform(0.0, 2 * np.pi)
    return pts @ _rotation(theta).T + rng.uniform(-0.5, 0.5, size=2)


def _sample_spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    turns = rng.uniform(1.0, 2.5)
    t = np.linspace(0.0, turns * 2 * np.pi, n) + rng.uniform(-0.05, 0.05, size=n)
    r = 0.1 + 0.25 * t / (2 * np.pi)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)
    noise = rng.uniform(0.02, 0.08)
    pts = pts + rng.standard_normal(pts.shape) * noise
    theta = rng.uniform(0.0, 2 * np.pi)
    return pts @ _rotation(theta).T + rng.uniform(-0.5, 0.5, size=2)


_SAMPLERS = {
    "iso_gaussian": _sample_iso_gaussian,
    "aniso_gaussian": _sample_aniso_gaussian,
    "mixture_2g": lambda n, rng: _sample_mixture_gauss(n, rng, 2),
    "mixture_3g": lambda n, rng: _sample_mixture_gauss(n, rng, 3),
    "uniform_square": _sample_uniform_square,
    "uniform_disk": _sample_uniform_disk,
    "ring": _sample_ring,
    "two_moons": _sample_two_moons,
    "swiss_roll_2d": _sample_swiss_roll_2d,
    "spiral": _sample_spiral,
}


def sample_set(
    family: str, num_particles: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one point set of shape (num_particles, 2) from the named family."""
    if family not in _SAMPLERS:
        raise ValueError(f"Unknown family: {family}")
    return _SAMPLERS[family](num_particles, rng).astype(np.float32)


def generate_dataset(
    num_samples: int,
    num_particles: int,
    seed: int = 0,
    families: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (points, family_ids) arrays by mixing the given families.

    Args:
        num_samples: how many point sets to draw.
        num_particles: points per set.
        seed: deterministic seed.
        families: subset of FAMILY_NAMES to sample from (default: all).
        weights: per-family mixture weights (default: uniform).

    Returns:
        points: (num_samples, num_particles, 2) float32
        family_ids: (num_samples,) int64, mapping to FAMILY_NAMES
    """
    rng = np.random.default_rng(seed)
    fams = list(families) if families is not None else list(FAMILY_NAMES)
    if weights is None:
        w = np.ones(len(fams)) / len(fams)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    family_idx = rng.choice(len(fams), size=num_samples, p=w)

    points = np.empty((num_samples, num_particles, 2), dtype=np.float32)
    family_ids = np.empty(num_samples, dtype=np.int64)
    for i, fi in enumerate(family_idx):
        family = fams[int(fi)]
        points[i] = sample_set(family, num_particles, rng)
        family_ids[i] = FAMILY_TO_ID[family]
    return points, family_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic point-set data.")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Comma-separated subset of FAMILY_NAMES (default: all)",
    )
    args = parser.parse_args()

    families = args.families.split(",") if args.families else None
    points, family_ids = generate_dataset(
        num_samples=args.num_samples,
        num_particles=args.num_particles,
        seed=args.seed,
        families=families,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, points=points, family_ids=family_ids)
    np.save(out_path.with_suffix(".points.npy"), points)
    print(
        f"Wrote {points.shape[0]} sets ({points.shape[1]} particles each) to "
        f"{out_path} and {out_path.with_suffix('.points.npy')}"
    )


if __name__ == "__main__":
    main()
