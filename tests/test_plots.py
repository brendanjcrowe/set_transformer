"""Tests for plotting utilities."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from set_transformer.plots import scatter, draw_ellipse, scatter_mog


@pytest.fixture
def sample_data():
    """Generate sample 2D data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 2)


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    np.random.seed(42)
    return np.random.randint(0, 3, 100)


def test_scatter_unlabeled(sample_data):
    """Test scatter plot without labels."""
    fig, ax = plt.subplots()
    ulabels, colors = scatter(sample_data, ax=ax)
    assert ulabels is None
    assert colors is None
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    plt.close()


def test_scatter_labeled(sample_data, sample_labels):
    """Test scatter plot with labels."""
    fig, ax = plt.subplots()
    ulabels, colors = scatter(sample_data, labels=sample_labels, ax=ax)
    assert len(ulabels) == len(np.unique(sample_labels))
    assert len(colors) == len(ulabels)
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    plt.close()


def test_draw_ellipse():
    """Test drawing confidence ellipses."""
    fig, ax = plt.subplots()
    pos = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 2]])
    draw_ellipse(pos, cov, ax=ax)
    # Check that ellipses were added to the plot
    assert len(ax.patches) == 5  # One ellipse per confidence level
    plt.close()


def test_scatter_mog(sample_data, sample_labels):
    """Test mixture of Gaussians visualization."""
    fig, ax = plt.subplots()
    K = len(np.unique(sample_labels))
    mu = np.random.randn(K, 2)
    cov = np.array([np.eye(2) for _ in range(K)])
    scatter_mog(sample_data, sample_labels, mu, cov, ax=ax)
    # Check that both points and ellipses were plotted
    assert len(ax.collections) > 0  # Points were plotted
    assert len(ax.patches) > 0  # Ellipses were plotted
    plt.close()


def test_scatter_input_validation():
    """Test input validation for scatter function."""
    with pytest.raises(ValueError):
        # Test with wrong dimensionality
        scatter(np.random.randn(100, 3))  # Should be (N, 2)

    with pytest.raises(ValueError):
        # Test with mismatched labels
        scatter(np.random.randn(100, 2), labels=np.random.randint(0, 2, 50))  # Wrong length
