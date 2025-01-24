"""ModelNet40 dataset utilities for point cloud processing.

This module provides utilities for loading and processing the ModelNet40 dataset,
including functions for data augmentation (rotation and scaling) and standardization.
The dataset consists of 3D point clouds representing different object categories.
"""

from typing import Generator, Tuple

import numpy as np
import numpy.typing as npt
import h5py


def rotate_z(theta: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
    """Rotate point cloud around the z-axis.

    Args:
        theta (npt.NDArray): Rotation angles in radians of shape (batch_size, 1)
        x (npt.NDArray): Point cloud of shape (batch_size, num_points, 3)

    Returns:
        npt.NDArray: Rotated point cloud of shape (batch_size, num_points, 3)
    """
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:, :, 2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:, :, 0], 2)
    yy = np.expand_dims(x[:, :, 1], 2)
    outx = cos_t * xx - sin_t * yy
    outy = sin_t * xx + cos_t * yy
    return np.concatenate([outx, outy, outz], axis=2)


def augment(x: npt.NDArray) -> npt.NDArray:
    """Apply data augmentation to point cloud.

    Applies random rotation around z-axis and random scaling to the point cloud.
    Rotation range: [-0.1π, 0.1π] radians
    Scale range: [0.8, 1.25] independently for each dimension

    Args:
        x (npt.NDArray): Point cloud of shape (batch_size, num_points, 3)

    Returns:
        npt.NDArray: Augmented point cloud of shape (batch_size, num_points, 3)
    """
    bs = x.shape[0]
    # rotation
    min_rot, max_rot = -0.1, 0.1
    thetas = np.random.uniform(min_rot, max_rot, [bs, 1]) * np.pi
    rotated = rotate_z(thetas, x)
    # scaling
    min_scale, max_scale = 0.8, 1.25
    scale = np.random.rand(bs, 1, 3) * (max_scale - min_scale) + min_scale
    return rotated * scale


def standardize(x: npt.NDArray) -> npt.NDArray:
    """Standardize point cloud by centering and scaling.

    Clips outliers based on mean absolute value, then applies zero-mean unit-variance
    standardization.

    Args:
        x (npt.NDArray): Point cloud of shape (batch_size, num_points, 3)

    Returns:
        npt.NDArray: Standardized point cloud of shape (batch_size, num_points, 3)
    """
    clipper = np.mean(np.abs(x), (1, 2), keepdims=True)
    z = np.clip(x, -100 * clipper, 100 * clipper)
    mean = np.mean(z, (1, 2), keepdims=True)
    std = np.std(z, (1, 2), keepdims=True)
    return (z - mean) / std


class ModelFetcher:
    """Data loader for ModelNet40 dataset.

    This class handles loading and preprocessing of the ModelNet40 dataset,
    providing iterators for both training and test data.

    Args:
        fname (str): Path to the HDF5 file containing the dataset
        batch_size (int): Number of samples per batch
        down_sample (int, optional): Factor by which to downsample points. Defaults to 10.
        do_standardize (bool, optional): Whether to standardize the data. Defaults to True.
        do_augmentation (bool, optional): Whether to apply data augmentation. Defaults to False.
    """

    def __init__(
        self, 
        fname: str, 
        batch_size: int, 
        down_sample: int = 10, 
        do_standardize: bool = True, 
        do_augmentation: bool = False
    ) -> None:
        self.fname = fname
        self.batch_size = batch_size
        self.down_sample = down_sample

        with h5py.File(fname, 'r') as f:
            self._train_data = np.array(f['tr_cloud'])
            self._train_label = np.array(f['tr_labels'])
            self._test_data = np.array(f['test_cloud'])
            self._test_label = np.array(f['test_labels'])

        self.num_classes = np.max(self._train_label) + 1
        self.num_train_batches = len(self._train_data) // self.batch_size
        self.num_test_batches = len(self._test_data) // self.batch_size

        self.prep1 = standardize if do_standardize else lambda x: x
        self.prep2 = (lambda x: augment(self.prep1(x))) if do_augmentation else self.prep1

        assert len(self._train_data) > self.batch_size, \
            'Batch size larger than number of training examples'

        # select the subset of points to use throughout beforehand
        self.perm = np.random.permutation(self._train_data.shape[1])[::self.down_sample]

    def train_data(self) -> Generator[Tuple[npt.NDArray, npt.NDArray, npt.NDArray], None, None]:
        """Get iterator over training data.

        Shuffles the data before each epoch.

        Yields:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: 
                - Point cloud batch of shape (batch_size, num_points, 3)
                - Batch cardinality of shape (batch_size,)
                - Labels of shape (batch_size,)
        """
        rng_state = np.random.get_state()
        np.random.shuffle(self._train_data)
        np.random.set_state(rng_state)
        np.random.shuffle(self._train_label)
        return self.next_train_batch()

    def next_train_batch(self) -> Generator[Tuple[npt.NDArray, npt.NDArray, npt.NDArray], None, None]:
        """Iterator over training batches.

        Yields:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: 
                - Point cloud batch of shape (batch_size, num_points, 3)
                - Batch cardinality of shape (batch_size,)
                - Labels of shape (batch_size,)
        """
        start = 0
        end = self.batch_size
        N = len(self._train_data)
        perm = self.perm
        batch_card = len(perm) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield self.prep2(self._train_data[start:end, perm]), batch_card, self._train_label[start:end]
            start = end
            end += self.batch_size

    def test_data(self) -> Generator[Tuple[npt.NDArray, npt.NDArray, npt.NDArray], None, None]:
        """Get iterator over test data.

        Yields:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: 
                - Point cloud batch of shape (batch_size, num_points, 3)
                - Batch cardinality of shape (batch_size,)
                - Labels of shape (batch_size,)
        """
        return self.next_test_batch()

    def next_test_batch(self) -> Generator[Tuple[npt.NDArray, npt.NDArray, npt.NDArray], None, None]:
        """Iterator over test batches.

        Yields:
            Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: 
                - Point cloud batch of shape (batch_size, num_points, 3)
                - Batch cardinality of shape (batch_size,)
                - Labels of shape (batch_size,)
        """
        start = 0
        end = self.batch_size
        N = len(self._test_data)
        batch_card = (self._train_data.shape[1] // self.down_sample) * np.ones(self.batch_size, dtype=np.int32)
        while end < N:
            yield self.prep1(self._test_data[start:end, 1::self.down_sample]), batch_card, self._test_label[start:end]
            start = end
            end += self.batch_size
