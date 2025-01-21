"""Dataset utilities for POMDP data.

This module provides dataset classes and utility functions for loading and
processing POMDP (Partially Observable Markov Decision Process) data.
"""

import os
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class POMDPDataset(Dataset):
    """Dataset class for POMDP data.

    This class wraps numpy arrays containing POMDP data into a PyTorch Dataset,
    making it compatible with PyTorch's data loading utilities.

    Args:
        numpy_array (npt.NDArray): Input numpy array containing POMDP data
    """

    def __init__(self, numpy_array: npt.NDArray) -> None:
        # Convert numpy array to torch tensor
        self.data = torch.from_numpy(numpy_array)

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            torch.Tensor: The requested sample
        """
        return self.data[index]


def get_dataset(device: str = "cpu") -> Dataset:
    """Load the POMDP dataset from disk.

    Args:
        device (str, optional): Device to load the data to. Defaults to "cpu".

    Returns:
        Dataset: The loaded POMDP dataset
    """
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./numpy/data.npy"
    )
    data = np.load(data_path)
    dataset = POMDPDataset(data)
    return dataset


def get_data_loader(
    batch_size: int = 32, 
    device: str = "cuda", 
    training_split: float = 0.9
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Create data loaders for training and evaluation.

    This function loads the dataset and splits it into training and evaluation sets,
    returning appropriate DataLoader objects for both.

    Args:
        batch_size (int, optional): Batch size for data loading. Defaults to 32.
        device (str, optional): Device to load the data to. Defaults to "cuda".
        training_split (float, optional): Fraction of data to use for training. Defaults to 0.9.

    Returns:
        Tuple[DataLoader, DataLoader, int, int]: A tuple containing:
            - Training data loader
            - Evaluation data loader
            - Size of training set
            - Size of evaluation set
    """
    dataset = get_dataset(device=device)
    training_size = int(training_split * len(dataset))
    eval_size = len(dataset) - training_size
    train_set, eval_set = random_split(dataset, [training_size, eval_size])
    return DataLoader(train_set, shuffle=True, batch_size=batch_size), DataLoader(
        eval_set, shuffle=True, batch_size=batch_size
    ), training_size, eval_size
