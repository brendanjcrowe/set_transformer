"""
Proof of Concept: Set Transformer for Particle Reconstruction

This script trains a Set Transformer to reconstruct particle representations
of belief distributions in the Odd-Even POMDP domain.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from set_transformer.loss import SinkhornLoss
from set_transformer.models import SetTransformer
from set_transformer.modules import ISAB, MAB, PMA, SAB


class ParticleReconstructionDataset(Dataset):
    """
    Dataset for training Set Transformer to reconstruct particles from belief distributions.
    
    Each sample contains:
    - Input: Observation history from the POMDP
    - Target: Particle set representing the belief distribution over modes
    - Ground Truth: True belief distribution for evaluation
    """
    
    def __init__(self, 
                 pomdp_config,
                 num_samples: int = 10000,
                 num_particles: int = 100,
                 observation_history_length: int = 5):
        """
        Args:
            pomdp_config: OddEvenPOMDPConfig instance
            num_samples: Number of training samples to generate
            num_particles: Number of particles per sample
            observation_history_length: Length of observation history to use as input
        """
        self.pomdp_config = pomdp_config
        self.num_samples = num_samples
        self.num_particles = num_particles
        self.obs_history_length = observation_history_length
        
        # Generate training data
        self.samples = self._generate_samples()
        
    def _generate_samples(self) -> List[Dict]:
        """Generate training samples"""
        from odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig
        
        samples = []
        
        for i in range(self.num_samples):
            # Create a new POMDP instance for each sample
            pomdp = OddEvenPOMDP(self.pomdp_config)
            
            # Generate observation history and update belief
            obs_history = []
            for _ in range(self.obs_history_length):
                obs = pomdp.get_observation()
                pomdp.update_belief(obs)
                obs_history.append(obs)
            
            # Sample particles from the current belief distribution over modes
            particles = self._sample_particles_from_belief(pomdp.belief, pomdp.belief_points, self.num_particles)
            
            # Create sample
            sample = {
                'obs_history': np.array(obs_history, dtype=np.float32),
                'particles': particles.astype(np.float32),
                'belief': pomdp.belief.astype(np.float32),  # Full belief distribution
                'belief_points': pomdp.belief_points.astype(np.float32),  # Support points
                'true_mode': pomdp.mean,  # True mode (mean of the Gaussian)
                'hidden_param': pomdp.hidden_param,
                'pomdp_info': pomdp.get_info()
            }
            
            samples.append(sample)
            
        return samples
    
    def _sample_particles_from_belief(self, belief: np.ndarray, belief_points: np.ndarray, num_particles: int) -> np.ndarray:
        """
        Sample particles from a belief distribution over modes.
        
        For the mode prediction POMDP, particles represent possible mode values.
        Each particle is a scalar value representing a possible mode.
        """
        # Sample particles according to the belief distribution
        # belief_points: discrete support points for the belief
        # belief: probabilities over these support points
        
        # Sample indices according to belief probabilities
        particle_indices = np.random.choice(len(belief_points), size=num_particles, p=belief)
        
        # Get the corresponding mode values
        particles = belief_points[particle_indices]
        
        # Add some noise to make particles more realistic (represent uncertainty)
        noise_std = 0.1  # Small noise to represent uncertainty within each belief point
        particles = particles + np.random.normal(0, noise_std, particles.shape)
        
        # Clip to valid range [1, n]
        particles = np.clip(particles, 1, self.pomdp_config.n)
        
        return particles.reshape(-1, 1)  # Shape: (num_particles, 1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'obs_history': torch.LongTensor(sample['obs_history']),  # Convert to LongTensor for embedding
            'particles': torch.FloatTensor(sample['particles']),
            'belief': torch.FloatTensor(sample['belief']),
            'belief_points': torch.FloatTensor(sample['belief_points']),
            'true_mode': sample['true_mode'],
            'hidden_param': sample['hidden_param'],
            'pomdp_info': sample['pomdp_info']
        }


class ParticleReconstructionModel(nn.Module):
    """
    Model that uses Set Transformer to reconstruct particles from observation history.
    
    Architecture:
    - Input: Observation history (sequence of integers)
    - Encoder: Embedding layer for observations
    - Set Transformer: Processes the embedded observations as a set
    - Output: Particles representing the belief distribution
    """
    
    def __init__(self,
                 obs_history_length: int,
                 embedding_dim: int = 64,
                 st_hidden_dim: int = 128,
                 st_num_heads: int = 4,
                 st_num_inds: int = 32,
                 st_num_outputs: int = 1,
                 num_particles: int = 100,
                 particle_dim: int = 1):
        super().__init__()
        
        self.obs_history_length = obs_history_length
        self.num_particles = num_particles
        self.particle_dim = particle_dim
        
        # Embedding layer for observations
        self.obs_embedding = nn.Embedding(21, embedding_dim)  # Assuming max obs value is 20
        
        # Set Transformer
        self.set_transformer = SetTransformer(
            dim_input=embedding_dim,
            num_outputs=st_num_outputs,
            dim_output=st_hidden_dim,  # Required parameter
            dim_hidden=st_hidden_dim,
            num_heads=st_num_heads,
            num_inds=st_num_inds,
            ln=bool(st_num_inds)  # Use layer norm if num_inds > 0
        )
        
        # Decoder to generate particles
        self.particle_decoder = nn.Sequential(
            nn.Linear(st_hidden_dim * st_num_outputs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_particles * particle_dim)
        )
        
    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_history: Shape (batch_size, obs_history_length)
            
        Returns:
            particles: Shape (batch_size, num_particles, particle_dim)
        """
        batch_size = obs_history.size(0)
        
        # Embed observations
        embedded_obs = self.obs_embedding(obs_history)  # (batch_size, obs_history_length, embedding_dim)
        
        # Process as a set with Set Transformer
        st_output = self.set_transformer(embedded_obs)  # (batch_size, num_outputs, dim_hidden)
        
        # Flatten for decoder
        st_output_flat = st_output.view(batch_size, -1)  # (batch_size, num_outputs * dim_hidden)
        
        # Generate particles
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
            obs_history = batch['obs_history'].to(device)
            target_particles = batch['particles'].to(device)
            target_belief = batch['belief'].to(device)
            belief_points = batch['belief_points'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_particles = model(obs_history)
            
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
                obs_history = batch['obs_history'].to(device)
                target_particles = batch['particles'].to(device)
                target_belief = batch['belief'].to(device)
                belief_points = batch['belief_points'].to(device)
                
                predicted_particles = model(obs_history)
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
            obs_history = batch['obs_history'].to(device)
            target_particles = batch['particles'].to(device)
            target_belief = batch['belief'].to(device)
            belief_points = batch['belief_points'].to(device)
            true_mode = batch['true_mode'].to(device)
            
            predicted_particles = model(obs_history)
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


def main():
    parser = argparse.ArgumentParser(description='Train Set Transformer for Particle Reconstruction')
    
    # POMDP config
    parser.add_argument('--n', type=int, default=20, help='Maximum number in range [1, n]')
    parser.add_argument('--mean', type=float, default=None, help='Mean of Gaussian (random if None)')
    parser.add_argument('--std_dev', type=float, default=2.0, help='Standard deviation of Gaussian')
    parser.add_argument('--belief_resolution', type=int, default=100, help='Number of discrete belief points for mode estimation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Dataset config
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--num_particles', type=int, default=100, help='Number of particles per sample')
    parser.add_argument('--obs_history_length', type=int, default=5, help='Length of observation history')
    
    # Model config
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension for observations')
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
    from odd_even_pomdp import OddEvenPOMDPConfig
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
        num_particles=args.num_particles,
        observation_history_length=args.obs_history_length
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
        obs_history_length=args.obs_history_length,
        embedding_dim=args.embedding_dim,
        st_hidden_dim=args.st_hidden_dim,
        st_num_heads=args.st_num_heads,
        st_num_inds=args.st_num_inds,
        st_num_outputs=args.st_num_outputs,
        num_particles=args.num_particles,
        particle_dim=1
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
