import os

import numpy as np
import torch

# It's better to load the specific SetTransformer model class that was used for pretraining.
# For now, let's assume it's the one from set_transformer.models
# This might need adjustment based on how the pretrained model was saved.
from set_transformer.models import SetTransformer


class PretrainedSetTransformerProcessor:
    """
    Processes particle filter beliefs using a *pretrained* Set Transformer model.
    This is not an SB3 FeatureExtractor itself, but a helper to be used by one or by a wrapper.
    """
    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading pretrained Set Transformer model from {model_path} onto device {self.device}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        try:
            # Attempt to load the model. 
            # This assumes the model was saved as a full SetTransformer object.
            # If only state_dict was saved, this needs to be adapted by first instantiating
            # the model with correct parameters, then loading state_dict.
            self.model = torch.load(model_path, map_location=self.device)
            if not isinstance(self.model, SetTransformer):
                 print(f"Warning: Loaded model from {model_path} is not an instance of set_transformer.models.SetTransformer. Ensure it has the expected interface (forward method, dim_input, dim_output attributes).")
            
            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Please ensure the model_path points to a valid PyTorch model file saved as a complete object (not just state_dict), or modify this loader.")
            raise

        # Try to get model dimensions. These might not exist if the loaded model is not the expected type.
        try:
            # These attributes (dim_input, dim_output) are specific to the SetTransformer class in set_transformer.models
            # If your pretrained model has different attribute names for these, they need to be adapted.
            self.dim_particle_input = self.model.dim_input
            self.st_output_dim = self.model.dim_output 
            # For SetTransformer from set_transformer.models, the PMA output (num_outputs) and final linear layer output (dim_output) are coupled.
            # The effective feature dimension coming out is num_outputs * dim_output, but here it's simplified.
            # The loaded model's forward pass should produce [batch, feature_dim]
        except AttributeError as e:
            print(f"Warning: Could not access dim_input or dim_output attributes from the loaded model. {e}")
            print("Defaulting dim_particle_input to 3 and st_output_dim to 64. This might be incorrect.")
            self.dim_particle_input = 3 # Default, assuming (x,y,weight)
            self.st_output_dim = 64 # Default common ST output size
        
        print(f"Pretrained model loaded. Expected particle input dim: {self.dim_particle_input}, ST output dim: {self.st_output_dim}")
    
    def process_particles(self, particle_set: torch.Tensor) -> np.ndarray:
        """
        Process a batch of particle sets through the Set Transformer model.
        
        Args:
            particle_set: Tensor of particle features, shape [batch_size, num_particles, particle_feature_dim].
                          If particle_feature_dim doesn't match self.dim_particle_input, an error will occur.
                          Must be on the same device as the model.
            
        Returns:
            features: NumPy array of processed features, shape [batch_size, self.st_output_dim]
        """
        if particle_set.shape[-1] != self.dim_particle_input:
            raise ValueError(f"Input particle dimension {particle_set.shape[-1]} does not match model's expected input dimension {self.dim_particle_input}")
        
        particle_set = particle_set.to(self.device) # Ensure tensor is on the correct device

        with torch.no_grad():
            features = self.model(particle_set) # Expected output: [batch_size, st_output_dim]
        
        return features.cpu().numpy()

    def process_particles_numpy(self, particles_np: np.ndarray, weights_np: np.ndarray | None = None) -> np.ndarray:
        """
        Convenience method to process NumPy arrays of particles and optional weights.
        
        Args:
            particles_np: NumPy array of particle coordinates, shape [num_particles, coord_dim] (e.g., [N, 2] for x,y).
            weights_np: Optional NumPy array of particle weights, shape [num_particles]. 
                        If provided and self.dim_particle_input requires weights (e.g., 3 for x,y,w),
                        weights will be appended to particles.
                        
        Returns:
            features: NumPy array of processed features, shape [1, self.st_output_dim] (batched with batch_size=1).
        """
        if weights_np is not None and self.dim_particle_input == particles_np.shape[1] + 1:
            # Assuming weights make up the last dimension if dim_particle_input matches
            particle_features_np = np.column_stack((particles_np, weights_np.reshape(-1, 1)))
        elif self.dim_particle_input == particles_np.shape[1]:
            particle_features_np = particles_np
        else:
            raise ValueError(
                f"Particle coord_dim ({particles_np.shape[1]}) + optional weights (1 if provided else 0) "
                f"does not match model's expected input dim ({self.dim_particle_input})."
            )

        # Convert to tensor and add batch dimension
        particle_set_torch = torch.tensor(particle_features_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return self.process_particles(particle_set_torch).squeeze(0) # Remove batch dim for single input 