"""Convert raw POMDP data to numpy format.

This script loads raw POMDP data from JSON files and converts it to numpy arrays
for efficient processing. The raw data consists of particle filter states across
multiple trajectories and timesteps.
"""

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
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Load raw POMDP data from a JSON file.

    Args:
        data_file_name (str, optional): Name of the JSON file. Defaults to "data.json".
        relative_data_path (str, optional): Relative path to data directory. Defaults to "raw/".
        absolute_data_path (Optional[str], optional): Absolute path to data directory. Defaults to None.

    Returns:
        Dict[str, Dict[str, Dict[str, Dict[str, float]]]]: Raw data in the format:
            {trajectory_id: {timestep: {particle_id: {variable: value}}}}
    """
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
    """Convert raw POMDP data to a numpy array.

    This function flattens the hierarchical structure of the raw data into a 3D array
    where each element represents a particle's state variables at a specific timestep
    in a trajectory.

    Args:
        raw_data (Dict[str, Dict[str, Dict[str, Dict[str, float]]]]): Raw data in the format:
            {trajectory_id: {timestep: {particle_id: {variable: value}}}}

    Returns:
        npt.NDArray: Numpy array of shape (num_samples, num_particles, num_variables)
            where num_samples = num_trajectories * num_timesteps
    """
    return np.array(
        [
            [
                [float(x) for var_name, x in particle.items()]
                for particle_id, particle in particles.items()
            ]
            for traj_num, traj in raw_data.items()
            for timestep, particles in traj.items()
        ],
        dtype=np.float32,
    )


# Set up command line argument parser
parser = argparse.ArgumentParser(description="Convert raw POMDP data to numpy format")
parser.add_argument(
    "--data_file_name", 
    default="data.json", 
    required=False, 
    type=str,
    help="Name of the input JSON file"
)
parser.add_argument(
    "--relative_data_path", 
    default="./raw", 
    required=False, 
    type=str,
    help="Relative path to the data directory"
)
parser.add_argument(
    "--absolute_data_path", 
    default=None, 
    required=False, 
    type=str,
    help="Absolute path to the data directory (overrides relative_data_path if provided)"
)


if __name__ == "__main__":
    args = parser.parse_args()
    
    # Load raw data from JSON
    raw_data = load_raw_data(
        data_file_name=args.data_file_name,
        relative_data_path=args.relative_data_path,
        absolute_data_path=args.absolute_data_path,
    )
    
    # Convert to numpy and save
    data_path = os.path.dirname(os.path.abspath(__file__))
    numpy_data = raw_to_numpy(raw_data)
    print(f"Converted data shape: {numpy_data.shape}")
    np.save(os.path.join(data_path, "./numpy/data.npy"), numpy_data)
