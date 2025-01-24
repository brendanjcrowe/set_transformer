"""Tests for raw data conversion utilities."""

import os
import json
import tempfile
import numpy as np
import pytest

from set_transformer.data.raw_to_numpy import load_raw_data, raw_to_numpy


@pytest.fixture
def sample_raw_data():
    """Create sample raw data in the expected format."""
    return {
        "traj_0": {
            "t_0": {
                "p_0": {"x": 1.0, "y": 2.0},
                "p_1": {"x": 3.0, "y": 4.0}
            },
            "t_1": {
                "p_0": {"x": 5.0, "y": 6.0},
                "p_1": {"x": 7.0, "y": 8.0}
            }
        },
        "traj_1": {
            "t_0": {
                "p_0": {"x": 9.0, "y": 10.0},
                "p_1": {"x": 11.0, "y": 12.0}
            }
        }
    }


@pytest.fixture
def temp_data_file(sample_raw_data):
    """Create a temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_raw_data, f)
    yield f.name
    os.unlink(f.name)


def test_load_raw_data(temp_data_file):
    """Test loading raw data from JSON file."""
    data = load_raw_data(
        data_file_name=os.path.basename(temp_data_file),
        absolute_data_path=os.path.dirname(temp_data_file)
    )
    assert isinstance(data, dict)
    assert "traj_0" in data
    assert "traj_1" in data
    assert len(data["traj_0"]) == 2  # Two timesteps
    assert len(data["traj_1"]) == 1  # One timestep


def test_raw_to_numpy(sample_raw_data):
    """Test converting raw data to numpy array."""
    array = raw_to_numpy(sample_raw_data)
    
    # Expected shape: (num_samples, num_particles, num_variables)
    # num_samples = num_trajectories * num_timesteps = 2 * 2 = 3
    # num_particles = 2
    # num_variables = 2 (x, y)
    assert array.shape == (3, 2, 2)
    assert array.dtype == np.float32
    
    # Check specific values
    np.testing.assert_array_almost_equal(
        array[0],  # First sample (traj_0, t_0)
        np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_raw_to_numpy_empty():
    """Test converting empty raw data."""
    with pytest.raises(ValueError):
        raw_to_numpy({})


def test_raw_to_numpy_invalid_format():
    """Test converting invalid raw data format."""
    invalid_data = {"traj_0": {"t_0": "invalid"}}
    with pytest.raises((TypeError, ValueError)):
        raw_to_numpy(invalid_data)
