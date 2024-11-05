import json
import argparse
import os
from typing import Optional, Dict

import numpy as np
import numpy.typing as npt
import torch

def load_raw_data(data_file_name: str = "data.json", relative_data_path: str = "./raw", absolute_data_path: Optional[str] = None) -> npt.NDArray:
    if absolute_data_path is not None:
        data_path = os.path.join(absolute_data_path, data_file_name)
    else:
        data_path = os.path.join(__file__, relative_data_path, data_file_name)

    with open(data_path, "rb") as file:
        raw_data = json.load(file)

    return raw_data

def raw_to_numpy(raw_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> npt.NDArray:
    return np.array(
        [
            [
                [
                    float(x) for particle_id, particle in particles.items()
                    for var_nam, x in particle.itmes()

                ]
                for timestep, particles in traj.items()
            ]
            for traj_num, traj in raw_data.items()
        ],
        dtype=np.float32
    )

def numpy_to_torch(numpy_data: npt.NDArray) -> torch.Tensor:
    return torch.Tensor(numpy_data, dtype=torch.float32)

def raw_to_torch(raw_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> torch.Tensor:
    return numpy_to_torch(raw_to_numpy(raw_data))

parser = argparse.ArgumentParser()
parser.add_argument("--data_file_name", default="data.json", required=False, type=str)
parser.add_argument("--relative_data_path", default="./raw", required=False, type=str)
parser.add_argument("--absolute_data_path", default=None, required=False, type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    raw_data = load_raw_data(data_file_name=args.data_file_name, relative_data_path=args.relative_data_path, absolute_data_path=args.absolute_data_path)
    numpy_data = raw_to_numpy(raw_data)
    np.save(numpy_data, "./numpy/data.npy")
    torch_data = numpy_to_torch(numpy_data)
    torch.save(torch_data, "./torch/data.pyt")

