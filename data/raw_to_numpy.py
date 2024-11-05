import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
import torch


def load_raw_data(
    data_file_name: str = "data.json",
    relative_data_path: str = "raw/",
    absolute_data_path: Optional[str] = None,
) -> npt.NDArray:
    if absolute_data_path is not None:
        data_path = os.path.join(absolute_data_path, data_file_name)
    else:

        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            relative_data_path,
            data_file_name,
        )

    with open(data_path, "rb") as file:
        raw_data = json.load(file)

    return raw_data


def raw_to_numpy(
    raw_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
) -> npt.NDArray:
    return np.array(
        [
            [
                np.array([
                    float(x)
                    for particle_id, particle in particles.items()
                    for var_nam, x in particle.items()
                ], dtype=np.float32)
                for timestep, particles in traj.items()
            ]
            for traj_num, traj in raw_data.items()
        ],
        dtype=np.object_,
    )

parser = argparse.ArgumentParser()
parser.add_argument("--data_file_name", default="data.json", required=False, type=str)
parser.add_argument("--relative_data_path", default="./raw", required=False, type=str)
parser.add_argument("--absolute_data_path", default=None, required=False, type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    raw_data = load_raw_data(
        data_file_name=args.data_file_name,
        relative_data_path=args.relative_data_path,
        absolute_data_path=args.absolute_data_path,
    )
    data_path = os.path.dirname(os.path.abspath(__file__))
    numpy_data = raw_to_numpy(raw_data)
    print(numpy_data.shape)
    np.save(os.path.join(data_path, "./numpy/data.npy"), numpy_data)
