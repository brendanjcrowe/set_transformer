import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from set_transformer.models import SetTransformer


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
        st_num_outputs: int = 1,  # PMA num_seeds in the underlying SetTransformer
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
        
        self.st_num_outputs = st_num_outputs
        self.st_output_dim = st_output_dim

        self.set_transformer = SetTransformer(
            dim_input=self.particle_dim,
            num_outputs=st_num_outputs,
            dim_output=st_output_dim,
            num_inds=st_num_inds,
            dim_hidden=st_hidden_dim,
            num_heads=st_num_heads,
            ln=st_ln,
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
        
        # SetTransformer returns [batch_size, st_num_outputs, st_output_dim];
        # flatten the last two dims into a single feature vector for SB3.
        particles = observations["particles"]
        particle_features = self.set_transformer(particles).flatten(start_dim=1)

        combined = torch.cat([obs_features, particle_features], dim=1)
        return self.combined_net(combined)
