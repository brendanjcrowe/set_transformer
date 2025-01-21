"""Dataset utilities for POMDP data.

This module provides dataset classes and utility functions for loading and
processing POMDP (Partially Observable Markov Decision Process) data.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class POMDPDataset(Dataset):
    """Dataset class for POMDP data."""

    def __init__(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """Initialize dataset.

        Args:
            data (Union[np.ndarray, torch.Tensor]): Data array of shape
                (num_samples, num_particles, particle_dim).

        Raises:
            ValueError: If data is empty or has wrong shape.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif not isinstance(data, torch.Tensor):
            raise TypeError("Data must be numpy array or torch tensor")

        if data.dim() != 3:
            raise ValueError(f"Data must have 3 dimensions, got {data.dim()}")
        if data.size(0) == 0:
            raise ValueError("Data cannot be empty")

        self.data = data

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of sample to get.

        Returns:
            torch.Tensor: Sample at index idx.
        """
        return self.data[idx]


def get_dataset(data_path: str) -> POMDPDataset:
    """Load dataset from file.

    Args:
        data_path (str): Path to data file.

    Returns:
        POMDPDataset: Dataset object.
    """
    data = np.load(data_path)
    return POMDPDataset(data)


def get_data_loader(
    batch_size: int,
    data_path: str,
    device: str,
    train_split: float = 0.8
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

    Returns:
        Tuple[DataLoader, DataLoader, int, int]: Training loader, evaluation loader,
            training size, and evaluation size.

    Raises:
        ValueError: If train_split is not between 0 and 1.
    """
    if not 0 < train_split < 1:
        raise ValueError("Train split must be between 0 and 1")

    dataset = get_dataset(data_path)
    dataset.data = dataset.data.to(device)

    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda")
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == "cuda")
    )

    return train_loader, eval_loader, train_size, eval_size
