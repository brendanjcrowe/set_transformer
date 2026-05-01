"""
Proof of Concept: Set Transformer for Particle Reconstruction

This script trains a Set Transformer to reconstruct particle representations
of belief distributions in the Odd-Even POMDP domain.
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from set_transformer.loss import SinkhornLoss
from set_transformer.models import SetTransformer
from set_transformer.modules import ISAB, MAB, PMA, SAB


class ParticleReconstructionDataset(Dataset):
    """
    Dataset for training Set Transformer to reconstruct particles from input particles.
    
    Each sample contains:
    - Input: Particle set sampled from the true belief distribution
    - Target: Another particle set sampled from the same distribution (what we want to reconstruct)
    - Ground Truth: True belief distribution for evaluation
    """
    
    def __init__(self, 
                 pomdp_config,
                 num_samples: int = 1000,
                 num_particles: int = 100):
        """
        Args:
            pomdp_config: OddEvenPOMDPConfig instance
            num_samples: Number of training samples to generate
            num_particles: Number of particles per sample
        """
        self.pomdp_config = pomdp_config
        self.num_samples = num_samples
        self.num_particles = num_particles
        
        # Create a shared RNG for generating different seeds for each POMDP
        # This ensures each POMDP gets a different mean
        base_seed = pomdp_config.seed if pomdp_config.seed is not None else 42
        self.dataset_rng = np.random.RandomState(base_seed)
        
        # Generate training data
        self.samples = self._generate_samples()
        
    def _generate_samples(self) -> List[Dict]:
        """Generate training samples"""
        
        samples = []
        
        for i in range(self.num_samples):
            # Create a new POMDP instance for each sample with a different seed
            # This ensures each POMDP gets a different random mean
            sample_config = OddEvenPOMDPConfig(
                n=self.pomdp_config.n,
                mean=None,  # Let it be random for each sample
                std_dev=self.pomdp_config.std_dev,
                belief_resolution=self.pomdp_config.belief_resolution,
                seed=self.dataset_rng.randint(0, 2**31)  # Use a different seed for each POMDP
            )
            pomdp = OddEvenPOMDP(sample_config)
            
            # Sample particles from the true belief distribution over modes
            # Particles are sampled from valid_numbers according to observation probabilities
            input_particles = pomdp.rng.choice(pomdp.valid_numbers, size=self.num_particles, p=pomdp.observation_probs)
            input_particles = input_particles.reshape(-1, 1)  # Shape: (num_particles, 1)
            
            # Target particles (same distribution, but different sample)
            # This is what we want to reconstruct
            target_particles = pomdp.rng.choice(pomdp.valid_numbers, size=self.num_particles, p=pomdp.observation_probs)
            target_particles = target_particles.reshape(-1, 1)  # Shape: (num_particles, 1)
            
            # Create sample
            sample = {
                'input_particles': input_particles.astype(np.float32),
                'target_particles': target_particles.astype(np.float32),
                'particles': target_particles.astype(np.float32),  # Alias for compatibility
                'belief': pomdp.belief.astype(np.float32),  # Full belief distribution
                'belief_points': pomdp.belief_points.astype(np.float32),  # Support points
                'true_mode': pomdp.mean,  # True mode (mean of the Gaussian)
                'hidden_param': pomdp.hidden_param,
                'pomdp_info': pomdp.get_info()
            }
            
            samples.append(sample)
            
        return samples

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_particles': torch.FloatTensor(sample['input_particles']),
            'target_particles': torch.FloatTensor(sample['target_particles']),
            'particles': torch.FloatTensor(sample['particles']),  # Alias for compatibility
            'belief': torch.FloatTensor(sample['belief']),
            'belief_points': torch.FloatTensor(sample['belief_points']),
            'true_mode': sample['true_mode'],
            'hidden_param': sample['hidden_param'],
            'pomdp_info': sample['pomdp_info']
        }


class ParticleReconstructionModel(nn.Module):
    """
    Model that uses Set Transformer to reconstruct particles from input particles.
    
    Architecture:
    - Input: Set of particles sampled from true distribution
    - Set Transformer: Processes the input particles as a set
    - Output: Reconstructed particles representing the same distribution
    """
    
    def __init__(self,
                 particle_dim: int = 1,
                 st_hidden_dim: int = 128,
                 st_num_heads: int = 4,
                 st_num_inds: int = 32,
                 st_num_outputs: int = 1,
                 num_particles: int = 100):
        super().__init__()
        
        self.num_particles = num_particles
        self.particle_dim = particle_dim
        
        # Input projection for particles
        self.input_projection = nn.Linear(particle_dim, st_hidden_dim)
        
        # Set Transformer to process input particles
        self.set_transformer = SetTransformer(
            dim_input=st_hidden_dim,
            num_outputs=st_num_outputs,
            dim_output=st_hidden_dim,  # Required parameter
            dim_hidden=st_hidden_dim,
            num_heads=st_num_heads,
            num_inds=st_num_inds,
            ln=bool(st_num_inds)  # Use layer norm if num_inds > 0
        )
        
        # Decoder to generate output particles
        self.particle_decoder = nn.Sequential(
            nn.Linear(st_hidden_dim * st_num_outputs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_particles * particle_dim)
        )
        
    def forward(self, input_particles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_particles: Shape (batch_size, num_input_particles, particle_dim)
            
        Returns:
            particles: Shape (batch_size, num_particles, particle_dim)
        """
        batch_size = input_particles.size(0)
        
        # Project input particles
        projected_particles = self.input_projection(input_particles)  # (batch_size, num_input_particles, st_hidden_dim)
        
        # Process as a set with Set Transformer
        st_output = self.set_transformer(projected_particles)  # (batch_size, num_outputs, dim_hidden)
        
        # Flatten for decoder
        st_output_flat = st_output.view(batch_size, -1)  # (batch_size, num_outputs * dim_hidden)
        
        # Generate output particles
        particles_flat = self.particle_decoder(st_output_flat)  # (batch_size, num_particles * particle_dim)
        
        # Reshape to particle format
        particles = particles_flat.view(batch_size, self.num_particles, self.particle_dim)
        
        return particles


def particles_to_distribution(particles: torch.Tensor, belief_points: torch.Tensor) -> torch.Tensor:
    """
    Convert particles back to a belief distribution over modes.
    
    Args:
        particles: Shape (batch_size, num_particles, particle_dim)
        belief_points: Shape (batch_size, belief_resolution) or (belief_resolution,)
        
    Returns:
        belief: Shape (batch_size, belief_resolution) representing belief over modes
    """
    batch_size, num_particles, particle_dim = particles.shape
    
    # Handle belief_points shape - it could be (batch_size, belief_resolution) or (belief_resolution,)
    if belief_points.dim() == 1:
        belief_resolution = belief_points.shape[0]
        belief_points_expanded = belief_points.unsqueeze(0).unsqueeze(0)  # (1, 1, belief_resolution)
    else:
        belief_resolution = belief_points.shape[1]
        belief_points_expanded = belief_points.unsqueeze(1)  # (batch_size, 1, belief_resolution)
    
    # For each particle, find the closest belief point
    # particles: (batch_size, num_particles, 1)
    particles_expanded = particles.squeeze(-1).unsqueeze(-1)  # (batch_size, num_particles, 1)
    
    # Compute distances from each particle to each belief point
    distances = torch.abs(particles_expanded - belief_points_expanded)  # (batch_size, num_particles, belief_resolution)
    
    # Find closest belief point for each particle
    closest_indices = torch.argmin(distances, dim=-1)  # (batch_size, num_particles)
    
    # Count particles for each belief point
    belief = torch.zeros(batch_size, belief_resolution, device=particles.device)
    
    for b in range(batch_size):
        for p in range(num_particles):
            belief[b, closest_indices[b, p]] += 1.0
    
    # Normalize to get probabilities
    belief = belief / num_particles
    
    return belief


def compute_loss(predicted_particles: torch.Tensor, 
                target_particles: torch.Tensor,
                target_belief: torch.Tensor,
                belief_points: torch.Tensor,
                sinkhorn_loss_fn: SinkhornLoss) -> Dict[str, torch.Tensor]:
    """
    Compute loss between predicted and target particles.
    
    Args:
        predicted_particles: Shape (batch_size, num_particles, particle_dim)
        target_particles: Shape (batch_size, num_particles, particle_dim)
        target_belief: Shape (batch_size, belief_resolution)
        belief_points: Shape (belief_resolution,)
        sinkhorn_loss_fn: SinkhornLoss instance
        
    Returns:
        Dictionary of loss components
    """
    # Sinkhorn loss for particles (permutation-invariant)
    particle_sinkhorn_loss = sinkhorn_loss_fn(predicted_particles, target_particles)
    
    # Distribution-level loss (convert particles to distributions and compare)
    pred_belief = particles_to_distribution(predicted_particles, belief_points)
    belief_mse = nn.MSELoss()(pred_belief, target_belief)
    
    # Mode prediction loss (compare predicted mode vs target mode)
    pred_mode = torch.sum(belief_points.unsqueeze(0) * pred_belief, dim=-1)  # Expected value
    target_mode = torch.sum(belief_points.unsqueeze(0) * target_belief, dim=-1)  # Expected value
    mode_mse = nn.MSELoss()(pred_mode, target_mode)
    
    # KL divergence between distributions
    pred_belief_clamped = torch.clamp(pred_belief, 1e-8, 1.0)
    target_belief_clamped = torch.clamp(target_belief, 1e-8, 1.0)
    kl_div = torch.sum(target_belief_clamped * torch.log(target_belief_clamped / pred_belief_clamped), dim=-1)
    kl_div = torch.mean(kl_div)
    
    # Combined loss
    total_loss = particle_sinkhorn_loss + belief_mse + mode_mse + 0.1 * kl_div
    
    return {
        'total_loss': total_loss,
        'particle_sinkhorn_loss': particle_sinkhorn_loss,
        'belief_mse': belief_mse,
        'mode_mse': mode_mse,
        'kl_div': kl_div
    }


def train_model(model: ParticleReconstructionModel,
               train_loader: DataLoader,
               val_loader: DataLoader,
               num_epochs: int = 100,
               learning_rate: float = 1e-3,
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
               sinkhorn_blur: float = 0.5) -> Dict:
    """Train the particle reconstruction model"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Create Sinkhorn loss function
    sinkhorn_loss_fn = SinkhornLoss(p=2, blur=sinkhorn_blur, reduction="mean")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_losses_epoch = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_particles = batch['input_particles'].to(device)
            target_particles = batch['target_particles'].to(device)
            target_belief = batch['belief'].to(device)
            belief_points = batch['belief_points'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_particles = model(input_particles)
            
            # Compute loss
            losses = compute_loss(predicted_particles, target_particles, target_belief, belief_points, sinkhorn_loss_fn)
            
            # Backward pass
            losses['total_loss'].backward()
            optimizer.step()
            
            train_loss += losses['total_loss'].item()
            train_losses_epoch.append(losses['total_loss'].item())
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_particles = batch['input_particles'].to(device)
                target_particles = batch['target_particles'].to(device)
                target_belief = batch['belief'].to(device)
                belief_points = batch['belief_points'].to(device)
                
                predicted_particles = model(input_particles)
                losses = compute_loss(predicted_particles, target_particles, target_belief, belief_points, sinkhorn_loss_fn)
                
                val_loss += losses['total_loss'].item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Early stopping
        if epoch > 20 and avg_val_loss > min(val_losses[-20:]):
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model
    }


def evaluate_model(model: ParticleReconstructionModel,
                 test_loader: DataLoader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 sinkhorn_blur: float = 0.5) -> Dict:
    """Evaluate the trained model"""
    
    model.eval()
    total_loss = 0.0
    mode_errors = []
    belief_errors = []
    
    # Create Sinkhorn loss function
    sinkhorn_loss_fn = SinkhornLoss(p=2, blur=sinkhorn_blur, reduction="mean")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_particles = batch['input_particles'].to(device)
            target_particles = batch['target_particles'].to(device)
            target_belief = batch['belief'].to(device)
            belief_points = batch['belief_points'].to(device)
            # true_mode is a list of scalars, convert to tensor
            if isinstance(batch['true_mode'], torch.Tensor):
                true_mode = batch['true_mode'].to(device).float()
            else:
                true_mode = torch.tensor(batch['true_mode'], dtype=torch.float32).to(device)
            
            predicted_particles = model(input_particles)
            losses = compute_loss(predicted_particles, target_particles, target_belief, belief_points, sinkhorn_loss_fn)
            
            total_loss += losses['total_loss'].item()
            
            # Convert particles to distributions for analysis
            pred_belief = particles_to_distribution(predicted_particles, belief_points)
            
            # Compute mode prediction error
            pred_mode = torch.sum(belief_points.unsqueeze(0) * pred_belief, dim=-1)
            mode_error = torch.abs(pred_mode - true_mode)
            mode_errors.append(mode_error.cpu().numpy().flatten())  # Flatten to ensure consistent shape
            
            # Compute belief distribution error
            belief_error = torch.mean(torch.abs(pred_belief - target_belief), dim=0)
            belief_errors.append(belief_error.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    avg_mode_error = np.mean(np.concatenate(mode_errors))  # Concatenate all mode errors
    avg_belief_error = np.mean(belief_errors, axis=0)
    
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Average Mode Error: {avg_mode_error:.4f}')
    print(f'Average Belief Error: {avg_belief_error.mean():.4f}')
    
    return {
        'test_loss': avg_loss,
        'mode_errors': mode_errors,
        'belief_errors': belief_errors,
        'avg_mode_error': avg_mode_error,
        'avg_belief_error': avg_belief_error
    }

def visualize_results(model: ParticleReconstructionModel, 
                      pomdp_config: OddEvenPOMDPConfig,
                      num_particles: int = 100,
                      n_tests: int = 10, 
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Visualize the results of the model"""
    
    # Create a RNG for generating different seeds for each POMDP
    # This ensures each visualization gets a different random mean
    base_seed = pomdp_config.seed if pomdp_config.seed is not None else 42
    viz_rng = np.random.RandomState(base_seed + 9999)  # Offset to avoid overlap with dataset seeds
    
    model.eval()
    with torch.no_grad():
        for _ in range(n_tests):
            # Create a new POMDP instance with a different seed for each visualization
            sample_config = OddEvenPOMDPConfig(
                n=pomdp_config.n,
                mean=None,  # Let it be random for each visualization
                std_dev=pomdp_config.std_dev,
                belief_resolution=pomdp_config.belief_resolution,
                seed=viz_rng.randint(0, 2**31)  # Use a different seed for each POMDP
            )
            pomdp = OddEvenPOMDP(sample_config)
            
            # Sample input particles from the true distribution
            input_particles = pomdp.rng.choice(pomdp.valid_numbers, size=num_particles, p=pomdp.observation_probs)
            input_particles = input_particles.reshape(-1, 1).astype(np.float32)
            
            # Get prediction
            input_particles_tensor = torch.FloatTensor(input_particles).unsqueeze(0).to(device)
            predicted_particles = model(input_particles_tensor)
            
            visualize_distribution(pomdp, predicted_particles[0], input_particles)
            
            
def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(P||Q)"""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    # Normalize
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return np.sum(p_safe * np.log(p_safe / q_safe))


def compute_descriptive_stats(particles: np.ndarray) -> dict:
    """Compute descriptive statistics for a particle set"""
    particles_flat = particles.flatten()
    return {
        'mean': np.mean(particles_flat),
        'std': np.std(particles_flat),
        'median': np.median(particles_flat),
        'min': np.min(particles_flat),
        'max': np.max(particles_flat),
        'q25': np.percentile(particles_flat, 25),
        'q75': np.percentile(particles_flat, 75)
    }


def visualize_distribution(pomdp: OddEvenPOMDP, predicted_particles: torch.Tensor, input_particles: np.ndarray):
    """Visualize the distributions as side-by-side histograms with statistics"""
    
    # Convert tensors to numpy
    if isinstance(predicted_particles, torch.Tensor):
        predicted_particles_np = predicted_particles.cpu().numpy().flatten()
    else:
        predicted_particles_np = predicted_particles.flatten()
    
    input_particles_np = input_particles.flatten()
    
    # Compute descriptive statistics
    input_stats = compute_descriptive_stats(input_particles_np)
    predicted_stats = compute_descriptive_stats(predicted_particles_np)
    
    # Create histograms
    # Use the same bins for both histograms for fair comparison
    all_values = np.concatenate([input_particles_np, predicted_particles_np])
    bins = np.linspace(all_values.min(), all_values.max(), 30)
    
    # Compute KL divergence using histograms
    input_hist, _ = np.histogram(input_particles_np, bins=bins, density=True)
    predicted_hist, _ = np.histogram(predicted_particles_np, bins=bins, density=True)
    
    # Normalize histograms to probabilities
    input_hist = input_hist / (input_hist.sum() + 1e-10)
    predicted_hist = predicted_hist / (predicted_hist.sum() + 1e-10)
    
    kl_div = compute_kl_divergence(input_hist, predicted_hist)
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot input particles histogram
    ax1.hist(input_particles_np, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(input_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {input_stats['mean']:.2f}")
    ax1.axvline(input_stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {input_stats['median']:.2f}")
    ax1.set_xlabel('Particle Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Input Particles Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"Mean: {input_stats['mean']:.3f}\n"
    stats_text += f"Std: {input_stats['std']:.3f}\n"
    stats_text += f"Median: {input_stats['median']:.3f}\n"
    stats_text += f"Min: {input_stats['min']:.3f}\n"
    stats_text += f"Max: {input_stats['max']:.3f}\n"
    stats_text += f"Q25: {input_stats['q25']:.3f}\n"
    stats_text += f"Q75: {input_stats['q75']:.3f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, family='monospace')
    
    # Plot predicted particles histogram
    ax2.hist(predicted_particles_np, bins=bins, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(predicted_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {predicted_stats['mean']:.2f}")
    ax2.axvline(predicted_stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {predicted_stats['median']:.2f}")
    ax2.set_xlabel('Particle Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Predicted Particles Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f"Mean: {predicted_stats['mean']:.3f}\n"
    stats_text += f"Std: {predicted_stats['std']:.3f}\n"
    stats_text += f"Median: {predicted_stats['median']:.3f}\n"
    stats_text += f"Min: {predicted_stats['min']:.3f}\n"
    stats_text += f"Max: {predicted_stats['max']:.3f}\n"
    stats_text += f"Q25: {predicted_stats['q25']:.3f}\n"
    stats_text += f"Q75: {predicted_stats['q75']:.3f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, family='monospace')
    
    # Add KL divergence as title
    fig.suptitle(f'Particle Distributions Comparison (KL Divergence: {kl_div:.4f}, Hidden: {pomdp.hidden_param})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Set Transformer for Particle Reconstruction')
    
    # POMDP config
    parser.add_argument('--n', type=int, default=10, help='Maximum number in range [1, n]')
    parser.add_argument('--mean', type=float, default=None, help='Mean of Gaussian (random if None)')
    parser.add_argument('--std_dev', type=float, default=2.0, help='Standard deviation of Gaussian')
    parser.add_argument('--belief_resolution', type=int, default=100, help='Number of discrete belief points for mode estimation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Dataset config
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--num_particles', type=int, default=100, help='Number of particles per sample')
    
    # Model config
    parser.add_argument('--st_hidden_dim', type=int, default=128, help='Set Transformer hidden dimension')
    parser.add_argument('--st_num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--st_num_inds', type=int, default=32, help='Number of inducing points')
    parser.add_argument('--st_num_outputs', type=int, default=1, help='Number of Set Transformer outputs')
    
    # Training config
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sinkhorn_blur', type=float, default=0.5, help='Sinkhorn loss blur parameter')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    # Output config
    parser.add_argument('--save_dir', type=str, default='./particle_reconstruction_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f'Using device: {device}')
    
    # Create POMDP config
    pomdp_config = OddEvenPOMDPConfig(
        n=args.n,
        mean=args.mean,
        std_dev=args.std_dev,
        belief_resolution=args.belief_resolution,
        seed=args.seed
    )
    
    # Create datasets
    print('Generating training data...')
    train_dataset = ParticleReconstructionDataset(
        pomdp_config=pomdp_config,
        num_samples=args.num_samples,
        num_particles=args.num_particles
    )
    
    # Split into train/val/test
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.15 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}')
    
    # Create model
    model = ParticleReconstructionModel(
        particle_dim=1,
        st_hidden_dim=args.st_hidden_dim,
        st_num_heads=args.st_num_heads,
        st_num_inds=args.st_num_inds,
        st_num_outputs=args.st_num_outputs,
        num_particles=args.num_particles
    )
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Train model
    print('Starting training...')
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        sinkhorn_blur=args.sinkhorn_blur
    )
    
    # Evaluate model
    print('Evaluating model...')
    eval_results = evaluate_model(
        model=results['model'],
        test_loader=test_loader,
        device=device,
        sinkhorn_blur=args.sinkhorn_blur
    )
    
    
    visualize_results(
        model=results['model'], 
        pomdp_config=pomdp_config,
        num_particles=args.num_particles,
        device=device
    )
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save model
    torch.save(results['model'].state_dict(), os.path.join(args.save_dir, 'model.pth'))
    
    # Save training history
    np.save(os.path.join(args.save_dir, 'train_losses.npy'), results['train_losses'])
    np.save(os.path.join(args.save_dir, 'val_losses.npy'), results['val_losses'])
    
    # Save evaluation results
    np.save(os.path.join(args.save_dir, 'test_loss.npy'), eval_results['test_loss'])
    np.save(os.path.join(args.save_dir, 'belief_errors.npy'), eval_results['belief_errors'])
    
    print(f'Results saved to {args.save_dir}')
    print(f'Final test loss: {eval_results["test_loss"]:.4f}')
    print(f'Average belief error: {eval_results["avg_belief_error"]}')


if __name__ == '__main__':
    main()
