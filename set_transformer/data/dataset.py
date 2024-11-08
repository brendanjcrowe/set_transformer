import os

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset


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


def get_dataset(device: str = "cuda") -> Dataset:

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./numpy/data.npy"
    )
    data = np.load(data_path)

    dataset = POMDPDataset(data)

    return dataset


def get_data_loader(batch_size: int = 32, device: str = "cuda") -> DataLoader:
    return DataLoader(get_dataset(device=device), shuffle=True, batch_size=batch_size)
