"""Tests for plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Ellipse

from set_transformer.plots import (
    draw_ellipse,
    scatter,
    scatter_mog,
    visualize_particle_filter_reconstruction,
)


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


@pytest.fixture
def sample_particles():
    """Generate sample original and reconstructed particles for testing.

    Returns:
        tuple: (original_particles, reconstructed_particles)
            - original_particles: Ground truth particles (N, 2)
            - reconstructed_particles: Reconstructed particles with some noise (N, 2)
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate ground truth particles from a mixture of 2 Gaussians
    N = 100  # Number of particles

    # First Gaussian
    n1 = N // 2
    mean1 = np.array([1.0, 1.0])
    cov1 = np.array([[0.5, 0.3], [0.3, 0.5]])
    particles1 = np.random.multivariate_normal(mean1, cov1, n1)

    # Second Gaussian
    n2 = N - n1
    mean2 = np.array([-1.0, -1.0])
    cov2 = np.array([[0.3, -0.2], [-0.2, 0.3]])
    particles2 = np.random.multivariate_normal(mean2, cov2, n2)

    # Combine particles
    original_particles = np.vstack([particles1, particles2])

    # Create reconstructed particles by adding noise
    noise = np.random.normal(0, 0.2, original_particles.shape)
    reconstructed_particles = original_particles + noise

    return original_particles, reconstructed_particles


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
        scatter(
            np.random.randn(100, 2), labels=np.random.randint(0, 2, 50)
        )  # Wrong length


def test_visualize_particle_filter_reconstruction(sample_particles):
    """Test the particle filter visualization function."""
    original_particles, reconstructed_particles = sample_particles

    # Test basic visualization
    visualize_particle_filter_reconstruction(
        original_particles, reconstructed_particles, title="Test Visualization"
    )

    # Clear the plot
    plt.close()

    # Test with custom axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    visualize_particle_filter_reconstruction(
        original_particles,
        reconstructed_particles,
        ax=(ax1, ax2),
        title="Test with Custom Axes",
    )

    # Clear the plot
    plt.close()

    # Test with different particle counts
    fewer_particles = original_particles[::2]  # Take every other particle
    fewer_reconstructed = reconstructed_particles[::2]

    visualize_particle_filter_reconstruction(
        fewer_particles, fewer_reconstructed, title="Test with Fewer Particles"
    )

    # Clear the plot
    plt.close()


def test_visualize_particle_filter_reconstruction_errors():
    """Test error handling in the visualization function."""
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        invalid_particles = np.random.randn(10, 3)  # 3D particles
        visualize_particle_filter_reconstruction(invalid_particles, invalid_particles)

    # Test with mismatched particle counts
    with pytest.raises(ValueError):
        particles1 = np.random.randn(10, 2)
        particles2 = np.random.randn(20, 2)
        visualize_particle_filter_reconstruction(particles1, particles2)
