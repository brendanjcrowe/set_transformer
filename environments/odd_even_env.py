"""
Gymnasium environment wrapper for OddEvenPOMDP
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig


class OddEvenPOMDPEnv(gym.Env):
    """
    Gymnasium environment wrapper for OddEvenPOMDP.
    
    Observation: The particles from the POMDP (shape: [n_particles,])
    Action: Integer prediction of the mean (discrete action space)
    Reward: Negative squared error between predicted and true mean
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, pomdp_config: OddEvenPOMDPConfig, max_steps: int = 100):
        """
        Args:
            pomdp_config: Configuration for the OddEvenPOMDP
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        self.pomdp_config = pomdp_config
        self.max_steps = max_steps
        self.pomdp = OddEvenPOMDP(pomdp_config)
        self.step_count = 0
        
        # Action space: discrete actions from 0 to n_dist_size-1 (predicting the mean, 0-indexed)
        # The action will be converted to 1-indexed in step()
        self.action_space = spaces.Discrete(pomdp_config.n_dist_size)
        
        # Observation space: particles from the POMDP
        # The particles are 1D (just values), so shape is (n_particles,)
        # Ensure particles is initialized
        if not hasattr(self.pomdp, 'particles') or self.pomdp.particles is None:
            # Initialize particles if not already done
            if hasattr(self.pomdp, '_init_particle_set'):
                self.pomdp.particles = self.pomdp._init_particle_set()
            else:
                # Fallback: create initial particles
                self.pomdp.particles = np.random.uniform(1, pomdp_config.n_dist_size, size=pomdp_config.n_particles)
        
        self.observation_space = spaces.Box(
            low=1.0,
            high=float(pomdp_config.n_dist_size),
            shape=(pomdp_config.n_particles,),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.pomdp.reset(new_seed=seed)
        else:
            self.pomdp.reset()
        
        self.step_count = 0
        
        # Ensure particles are initialized
        if not hasattr(self.pomdp, 'particles') or self.pomdp.particles is None:
            if hasattr(self.pomdp, '_init_particle_set'):
                self.pomdp.particles = self.pomdp._init_particle_set()
            else:
                # Fallback: create initial particles
                self.pomdp.particles = np.random.uniform(1, self.pomdp_config.n_dist_size, size=self.pomdp_config.n_particles)
        
        # Get initial particles
        obs = self.pomdp.particles.astype(np.float32)
        info = {
            'true_mean': self.pomdp.mean,
            'hidden_param': self.pomdp.hidden_param,
            'pomdp_info': self.pomdp.get_info()
        }
        
        return obs, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Integer action (predicted mean, 0-indexed, so we add 1)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert action from 0-indexed to 1-indexed (since mean is in range [1, n])
        predicted_mean = int(action) + 1
        
        # Step the POMDP
        obs, reward, done, _ = self.pomdp.step(predicted_mean)
        
        self.step_count += 1
        
        # Convert to gymnasium format
        obs = obs.astype(np.float32)
        terminated = done
        truncated = self.step_count >= self.max_steps
        
        info = {
            'true_mean': self.pomdp.mean,
            'predicted_mean': predicted_mean,
            'hidden_param': self.pomdp.hidden_param,
            'pomdp_info': self.pomdp.get_info()
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            self.pomdp.render(mode='human')
        return None


# Register the environment with gymnasium
register(
    id='OddEvenPOMDP-v0',
    entry_point='environments.odd_even_env:OddEvenPOMDPEnv',
    max_episode_steps=100,
)

