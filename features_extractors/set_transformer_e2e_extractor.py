import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Assuming set_transformer.modules can be found (e.g., if set_transformer is installed or in PYTHONPATH)
from set_transformer.modules import ISAB, PMA, SAB


class SetTransformerNetwork(nn.Module):
    """
    Set Transformer network that processes sets of particles.
    This implementation follows the architecture from the Set Transformer paper.
    Configurable for different particle dimensions and Set Transformer hyperparameters.
    """
    def __init__(
        self,
        dim_particle_input: int, # Dimension of each particle (e.g., x, y, weight)
        dim_st_output: int,      # Output dimension of the Set Transformer
        dim_hidden: int = 128,
        num_heads: int = 4,
        num_inds: int = 32,
        num_st_outputs: int = 1, # Corresponds to PMA num_seeds
        ln: bool = False
    ):
        super(SetTransformerNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            ISAB(dim_particle_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, num_st_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_st_output)
        )
        
        self.dim_particle_input = dim_particle_input
        self.dim_st_output = dim_st_output
        self.num_st_outputs = num_st_outputs # Number of vectors from PMA
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Set Transformer.
        
        Args:
            x: Input tensor of shape [batch_size, num_particles, dim_particle_input]
            
        Returns:
            output: Tensor of shape [batch_size, num_st_outputs * dim_st_output]
        """
        x = self.encoder(x)
        x = self.decoder(x)
        # Reshape to [batch_size, num_st_outputs * dim_st_output]
        # If num_st_outputs is 1, this is equivalent to [batch_size, dim_st_output]
        return x.view(-1, self.num_st_outputs * self.dim_st_output)


class CustomSetTransformerExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Stable Baselines 3 that integrates a Set Transformer
    to process particle filter beliefs from a Dict observation space.
    The observation space must contain:
    - 'obs': The standard environment observation.
    - 'particles': The particle set, shape (num_particles, particle_dim).
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        # Set Transformer parameters
        st_output_dim: int = 64,
        st_hidden_dim: int = 128,
        st_num_heads: int = 4,
        st_num_inds: int = 32,
        st_num_outputs: int = 1, # Corresponds to PMA num_seeds in SetTransformerNetwork
        st_ln: bool = False,
        # MLP for 'obs'
        obs_mlp_hidden_dims: list[int] = [64, 64],
        # Combined features
        features_dim: int = 128
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("CustomSetTransformerExtractor expects a Dict observation space.")
        if "obs" not in observation_space.spaces:
            raise ValueError("Observation space Dict must contain an 'obs' key.")
        if "particles" not in observation_space.spaces:
            raise ValueError("Observation space Dict must contain a 'particles' key.")

        super(CustomSetTransformerExtractor, self).__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        self.obs_dim = observation_space["obs"].shape[0]
        self.num_particles = observation_space["particles"].shape[0]
        self.particle_dim = observation_space["particles"].shape[1]
        
        # Create the Set Transformer
        self.set_transformer = SetTransformerNetwork(
            dim_particle_input=self.particle_dim,
            dim_st_output=st_output_dim,
            dim_hidden=st_hidden_dim,
            num_heads=st_num_heads,
            num_inds=st_num_inds,
            num_st_outputs=st_num_outputs,
            ln=st_ln
        )
        
        # Create MLP for original observations
        obs_mlp_layers = []
        current_dim = self.obs_dim
        for hidden_dim in obs_mlp_hidden_dims:
            obs_mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            obs_mlp_layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.obs_net = nn.Sequential(*obs_mlp_layers)
        
        # Calculate combined dimension before the final layer
        combined_features_input_dim = current_dim + (st_num_outputs * st_output_dim)

        # Final layer to project to features_dim
        self.combined_net = nn.Sequential(
            nn.Linear(combined_features_input_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process both original observations and particle filter states.
        """
        # Process original observations
        obs_features = self.obs_net(observations["obs"])
        
        # Process particles with Set Transformer
        particles = observations["particles"]
        # Add batch dimension if needed (SB3 usually provides batched observations)
        # However, for consistency and direct calls, this check can be useful.
        # if len(particles.shape) == 2: # [num_particles, particle_dim]
        #     particles = particles.unsqueeze(0) # [1, num_particles, particle_dim]
        
        particle_features = self.set_transformer(particles) # [batch_size, st_num_outputs * st_output_dim]
        
        # Combine features
        # Ensure obs_features also has a batch dimension if particles were unbatched and processed.
        # SB3 handles batching, so usually obs_features and particle_features are already batched.
        combined = torch.cat([obs_features, particle_features], dim=1)
        return self.combined_net(combined) 