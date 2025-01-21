import os
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class POMDPDataset(Dataset):
    def __init__(self, numpy_array: npt.NDArray) -> None:
        # Convert numpy array to torch tensor
        self.data = torch.from_numpy(numpy_array)

    def __len__(self) -> int:
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        # Return a single sample
        return self.data[index]


def get_dataset(device: str = "cpu") -> Dataset:

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./numpy/data.npy"
    )
    data = np.load(data_path)

    dataset = POMDPDataset(data)

    return dataset


def get_data_loader(
    batch_size: int = 32, device: str = "cuda", training_split: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    dataset = get_dataset(device=device)
    training_size = int(training_split * len(dataset))
    eval_size = len(dataset) - training_size
    train_set, eval_set = random_split(dataset, [training_size, eval_size])
    return DataLoader(train_set, shuffle=True, batch_size=batch_size), DataLoader(
        eval_set, shuffle=True, batch_size=batch_size
    ), training_size, eval_size
