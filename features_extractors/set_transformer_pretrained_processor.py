import os

import numpy as np
import torch
import torch.nn as nn

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
            # First try loading as a complete object, then try as state_dict
            loaded = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(loaded, dict) and any(k.startswith('set_transformer.') or k.startswith('input_projection.') for k in loaded.keys()):
                # It's a state_dict, likely from ParticleReconstructionModel
                print("Detected state_dict format. Reconstructing model...")
                # Extract set_transformer state_dict
                st_state_dict = {k.replace('set_transformer.', ''): v 
                                for k, v in loaded.items() 
                                if k.startswith('set_transformer.')}
                input_proj_state_dict = {k.replace('input_projection.', ''): v 
                                        for k, v in loaded.items() 
                                        if k.startswith('input_projection.')}
                
                # Reconstruct the SetTransformer - need to infer dimensions from state_dict
                # Get dim_input from first layer weight shape
                if input_proj_state_dict:
                    # input_projection maps particle_dim -> st_hidden_dim
                    particle_dim = input_proj_state_dict['weight'].shape[1]
                    st_hidden_dim = input_proj_state_dict['weight'].shape[0]
                else:
                    # Try to infer from set_transformer
                    if st_state_dict and 'encoder.0.weight' in st_state_dict:
                        st_hidden_dim = st_state_dict['encoder.0.weight'].shape[0]
                        particle_dim = 1  # Default for OddEvenPOMDP
                    else:
                        raise ValueError("Cannot infer model dimensions from state_dict")
                
                # Get num_outputs and other params from state_dict structure
                # Look for PMA layer to get num_outputs
                num_outputs = 1  # Default
                if any('pma' in k.lower() for k in st_state_dict.keys()):
                    # Try to infer from PMA layer
                    for k in st_state_dict.keys():
                        if 'pma' in k.lower() and 'weight' in k:
                            num_outputs = st_state_dict[k].shape[0]
                            break
                
                # Create input projection
                self.input_projection = nn.Linear(particle_dim, st_hidden_dim)
                if input_proj_state_dict:
                    self.input_projection.load_state_dict(input_proj_state_dict)
                
                # Create SetTransformer - need to infer other params
                num_heads = 4  # Default
                num_inds = 32  # Default
                dim_output = st_hidden_dim  # Usually same as hidden
                
                self.model = SetTransformer(
                    dim_input=st_hidden_dim,
                    num_outputs=num_outputs,
                    dim_output=dim_output,
                    dim_hidden=st_hidden_dim,
                    num_heads=num_heads,
                    num_inds=num_inds,
                    ln=bool(num_inds)
                )
                self.model.load_state_dict(st_state_dict)
                
                self.dim_particle_input = particle_dim
                self.st_output_dim = dim_output * num_outputs
                
            elif isinstance(loaded, SetTransformer):
                # It's a complete SetTransformer object
                self.model = loaded
                self.input_projection = None
                self.dim_particle_input = self.model.dim_input
                self.st_output_dim = self.model.dim_output
            else:
                # Try to use it as-is (might be a wrapper model)
                self.model = loaded
                self.input_projection = None
                if hasattr(self.model, 'set_transformer'):
                    # It's a wrapper model with set_transformer attribute
                    self.dim_particle_input = getattr(self.model, 'particle_dim', 1)
                    self.st_output_dim = self.model.set_transformer.dim_output
                else:
                    raise ValueError(f"Unknown model format: {type(loaded)}")
            
            self.model.eval()  # Set to evaluation mode
            self.model.to(self.device)
            if self.input_projection is not None:
                self.input_projection.eval()
                self.input_projection.to(self.device)

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Dimensions should be set during loading, but verify
        if not hasattr(self, 'dim_particle_input') or not hasattr(self, 'st_output_dim'):
            try:
                if hasattr(self.model, 'dim_input'):
                    self.dim_particle_input = self.model.dim_input
                if hasattr(self.model, 'dim_output'):
                    self.st_output_dim = self.model.dim_output
            except:
                print("Warning: Could not determine model dimensions. Using defaults.")
                self.dim_particle_input = 1  # Default for OddEvenPOMDP
                self.st_output_dim = 128  # Default
        
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
        
        particle_set = particle_set.to(self.device) # Ensure tensor is on the same device

        with torch.no_grad():
            # If we have input_projection, use it first
            if self.input_projection is not None:
                # Project particles to hidden dimension
                projected = self.input_projection(particle_set)
                # Process through SetTransformer
                features = self.model(projected)  # [batch_size, num_outputs, dim_output]
            elif hasattr(self.model, 'set_transformer'):
                # It's a wrapper model
                features = self.model.set_transformer(particle_set)
            else:
                # Direct SetTransformer
                features = self.model(particle_set)
            
            # Flatten if needed (if features is [batch, num_outputs, dim_output])
            if len(features.shape) == 3:
                features = features.view(features.shape[0], -1)  # [batch, num_outputs * dim_output]
        
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