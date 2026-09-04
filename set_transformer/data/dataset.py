"""Dataset utilities for POMDP data.

This module provides dataset classes and utility functions for loading and
processing POMDP (Partially Observable Markov Decision Process) data.
"""

import json
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class POMDPDataset(Dataset):
    """Dataset class for POMDP data."""

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: str = "cpu",
        particle_scale: float = 1.0,
        particle_centre: float = 0.0,
    ) -> None:
        """Initialize dataset.

        Args:
            data (Union[np.ndarray, torch.Tensor]): Data array of shape
                (num_samples, num_particles, particle_dim).
            weights (optional): Per-particle mass of shape
                (num_samples, num_particles), e.g. particle-filter weights.
                None (the default) means the sets are uniform and each sample
                is returned as a bare tensor, exactly as before. When given,
                each sample is returned as a (particles, weights) tuple.
            device (str, optional): Device to store data on. Defaults to "cpu".
            particle_scale (float, optional): Divide every coordinate by this
                once, at construction. Defaults to 1.0 (raw coordinates). Use
                it to match whatever normalization the downstream consumer of
                the encoder applies.
            particle_centre (float, optional): Subtract this from every
                coordinate BEFORE dividing by particle_scale, i.e. the stored
                mapping is (x - centre) / scale. Defaults to 0.0. Odd-Even's
                RL wrapper centres the state range on 0 before the extractor
                divides, so a dataset of raw states must be centred the same
                way or the encoder is pretrained one whole unit off the
                inputs it later receives.

        Raises:
            ValueError: If data is empty, has wrong shape, the weights do not
                line up with the particles, or particle_scale is not positive.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif not isinstance(data, torch.Tensor):
            raise TypeError("Data must be numpy array or torch tensor")

        if data.dim() != 3:
            raise ValueError(f"Data must have 3 dimensions, got {data.dim()}")
        if data.size(0) == 0:
            raise ValueError("Data cannot be empty")

        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights).float()
            elif not isinstance(weights, torch.Tensor):
                raise TypeError("Weights must be numpy array or torch tensor")
            if weights.dim() != 2:
                raise ValueError(
                    f"Weights must have 2 dimensions, got {weights.dim()}"
                )
            if weights.shape != data.shape[:2]:
                raise ValueError(
                    "Weights must have shape (num_samples, num_particles) "
                    f"matching the particles {tuple(data.shape[:2])}, got "
                    f"{tuple(weights.shape)}"
                )
            if bool((weights < 0).any()):
                raise ValueError("Weights must be non-negative")
            # Caught here, at load, rather than thousands of steps into an
            # epoch: an optimal-transport loss needs every set to carry
            # positive, finite mass. A float32 round trip of float64 particle
            # weights is the realistic way this breaks.
            if not bool(torch.isfinite(weights).all()):
                bad = torch.nonzero(~torch.isfinite(weights).all(dim=1))[:5]
                raise ValueError(
                    "Weights contain NaN or inf; first offending sample "
                    f"indices: {bad.flatten().tolist()}"
                )
            totals = weights.sum(dim=1)
            if bool((totals <= 0).any()):
                bad = torch.nonzero(totals <= 0)[:5]
                raise ValueError(
                    "Every particle set must carry positive total weight; "
                    f"first offending sample indices: {bad.flatten().tolist()}"
                )

        particle_scale = float(particle_scale)
        particle_centre = float(particle_centre)
        if not particle_scale > 0.0:
            raise ValueError(
                f"particle_scale must be positive, got {particle_scale}")
        if not np.isfinite(particle_centre):
            raise ValueError(
                f"particle_centre must be finite, got {particle_centre}")
        if particle_centre != 0.0:
            data = data - particle_centre
        if particle_scale != 1.0:
            data = data / particle_scale

        self.data = data
        self.weights = weights
        self.device = device
        self.particle_scale = particle_scale
        self.particle_centre = particle_centre

    @property
    def is_weighted(self) -> bool:
        """Whether samples carry per-particle mass."""
        return self.weights is not None

    @property
    def particle_dim(self) -> int:
        """Coordinate dimension of a single particle."""
        return int(self.data.shape[-1])

    @property
    def num_particles(self) -> int:
        """Number of particles per set."""
        return int(self.data.shape[1])

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of sample to get.

        Returns:
            The particle set at index idx, or a (particles, weights) tuple
            when the dataset is weighted.
        """
        particles = self.data[idx].to(self.device)
        if self.weights is None:
            return particles
        return particles, self.weights[idx].to(self.device)


def _stored_particle_centre(loaded) -> float:
    """The centre an .npz records, or 0.0.

    Collectors write it as a top-level ``particle_centre`` array; older
    Odd-Even datasets carry it only inside the ``metadata`` JSON. Either is
    honoured, so the datasets already on disk load in the RL frame without
    being recollected.
    """
    if "particle_centre" in loaded:
        return float(np.asarray(loaded["particle_centre"]).reshape(-1)[0])
    if "metadata" in loaded:
        try:
            meta = json.loads(str(loaded["metadata"]))
        except (TypeError, ValueError):
            return 0.0
        value = meta.get("particle_centre") if isinstance(meta, dict) else None
        if value is not None:
            return float(value)
    return 0.0


def get_dataset(
    data_path: str,
    device: str = "cpu",
    load_weights: bool = True,
    particle_scale: Optional[float] = None,
    particle_centre: Optional[float] = None,
) -> POMDPDataset:
    """Load dataset from file.

    Two on-disk formats are understood:

    * ``.npy`` — a bare array of shape (num_samples, num_particles, dim).
      Uniform (unweighted) sets. This is the legacy format.
    * ``.npz`` — an archive with a ``particles`` array of that same shape and,
      optionally, a ``weights`` array of shape (num_samples, num_particles).
      Any other arrays (collection metadata) are ignored here. An unnamed
      single-array archive is read as particles, for tolerance.

    Args:
        data_path (str): Path to data file.
        device (str, optional): Device to store data on. Defaults to "cpu".
        load_weights (bool, optional): Read the weights when the file has
            them. Set False to deliberately train on the unweighted set — the
            ablation, not the default.
        particle_scale (float, optional): Divide every coordinate by this
            before training. None (the default) means "use the scale recorded
            in the file", falling back to 1.0 when there is none.

            This exists because the RL feature extractor normalizes particles
            by the arena half-width before the encoder sees them. A dataset
            stores RAW env coordinates, so pretraining on it unscaled would
            train the encoder on inputs several times larger than the ones it
            is later handed — the pretrained weights would then be operating
            far outside their trained range. Pass 1.0 to opt out explicitly.
        particle_centre (float, optional): Subtract this before dividing.
            None (the default) means "use the centre recorded in the file"
            (top-level array or metadata JSON), falling back to 0.0.

    Returns:
        POMDPDataset: Dataset object.
    """
    loaded = np.load(data_path)
    if isinstance(loaded, np.ndarray):
        scale = 1.0 if particle_scale is None else float(particle_scale)
        centre = 0.0 if particle_centre is None else float(particle_centre)
        return POMDPDataset(loaded, None, device, particle_scale=scale,
                            particle_centre=centre)

    # NpzFile
    if "particles" in loaded:
        particles = loaded["particles"]
    elif len(loaded.files) == 1:
        particles = loaded[loaded.files[0]]
    else:
        raise KeyError(
            f"{data_path} has no 'particles' array; found {loaded.files}"
        )
    weights = loaded["weights"] if (load_weights and "weights" in loaded) else None

    if particle_scale is None:
        stored = loaded["particle_scale"] if "particle_scale" in loaded else None
        scale = 1.0 if stored is None else float(np.asarray(stored).reshape(-1)[0])
    else:
        scale = float(particle_scale)
    centre = (_stored_particle_centre(loaded) if particle_centre is None
              else float(particle_centre))
    return POMDPDataset(particles, weights, device, particle_scale=scale,
                        particle_centre=centre)


def get_data_loader(
    batch_size: int,
    data_path: str,
    device: str,
    train_split: float = 0.8,
    num_workers: int = 0,
    load_weights: bool = True,
    particle_scale: Optional[float] = None,
    particle_centre: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Create data loaders for training and evaluation.

    This function loads the dataset and splits it into training and evaluation sets,
    returning appropriate DataLoader objects for both.

    Args:
        batch_size (int): Batch size for data loaders.
        data_path (str): Path to data file.
        device (str): Device to load data on ('cpu' or 'cuda').
        train_split (float, optional): Fraction of data to use for training.
            Defaults to 0.8.
        num_workers (int, optional): Number of worker processes for data loading.
            Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader, int, int]: Training loader, evaluation loader,
            training size, and evaluation size.

    Raises:
        ValueError: If train_split is not between 0 and 1.
    """
    if not 0 < train_split < 1:
        raise ValueError("Train split must be between 0 and 1")

    # POMDPDataset moves each sample to `device` in __getitem__, which happens
    # inside the worker process. With CUDA and forked workers that raises
    # "Cannot re-initialize CUDA in forked subprocess" as soon as the parent
    # has touched CUDA. Workers therefore build CPU samples; the training loop
    # moves each batch to the device anyway, so nothing downstream changes.
    dataset_device = "cpu" if num_workers > 0 else device
    dataset = get_dataset(
        data_path, dataset_device, load_weights=load_weights,
        particle_scale=particle_scale, particle_centre=particle_centre,
    )
    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size

    # An unseeded split gives a different val set on every invocation, so
    # best_val_loss is not comparable across runs and a resumed run trains on
    # former val samples. The alignment loss (Phase 2) additionally indexes a
    # precomputed pairwise-EMD matrix by dataset row, which only works if the
    # split is reproducible. None keeps torch's global RNG (legacy behaviour).
    generator = (torch.Generator().manual_seed(int(seed))
                 if seed is not None else None)
    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, eval_loader, train_size, eval_size
