"""
Odd-Even POMDP Implementation

This implements a variant where:
- Numbers range from 1 to n (hyperparameter)
- Hidden parameter: either "odd" or "even" (fixed throughout episode)
- True distribution: Gaussian distribution constrained to odd/even numbers only
- Prediction task: predict the mode (peak) of the distribution
- Mean and standard deviation are hyperparameters with defaults
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class OddEvenPOMDPConfig:
    """Configuration for the Odd-Even POMDP"""
    n: int = 20  # Maximum number in range [1, n]
    mean: Optional[float] = None  # Mean of Gaussian (random if None)
    std_dev: float = 2.0  # Standard deviation of Gaussian
    seed: Optional[int] = None  # Random seed for reproducibility
    belief_resolution: int = 100  # Number of discrete belief points for mode estimation


class OddEvenPOMDP:
    """
    Mode Prediction POMDP where:
    - Hidden parameter is either "odd" or "even"
    - Observations are drawn from Gaussian distributions constrained to odd/even integers
    - Agent must predict the mode of the distribution
    """
    
    def __init__(self, config: OddEvenPOMDPConfig):
        self.config = config
        self.n = config.n
        self.std_dev = config.std_dev
        self.belief_resolution = config.belief_resolution
        
        # Initialize random number generator
        self.rng = np.random.RandomState(config.seed)
        
        # Set mean randomly if not provided
        if config.mean is None:
            self.mean = self.rng.uniform(1, self.n)
        else:
            self.mean = config.mean
            
        # Hidden parameter (fixed throughout episode)
        self.hidden_param = self.rng.choice(['odd', 'even'])
        
        # Generate valid odd and even numbers in range [1, n]
        self.odd_numbers = np.array([i for i in range(1, self.n + 1) if i % 2 == 1])
        self.even_numbers = np.array([i for i in range(1, self.n + 1) if i % 2 == 0])
        
        # Create discrete belief space for mode estimation
        self.belief_points = np.linspace(1, self.n, self.belief_resolution)
        
        # Initialize uniform belief over possible modes
        self.belief = np.ones(self.belief_resolution) / self.belief_resolution
        
        # Pre-compute probabilities for efficiency
        self._compute_probabilities()
        
    def _compute_probabilities(self):
        """Pre-compute observation probabilities for the hidden parameter"""
        if self.hidden_param == 'odd':
            # Compute Gaussian probabilities for odd numbers
            self.observation_probs = np.exp(-0.5 * ((self.odd_numbers - self.mean) / self.std_dev) ** 2)
            self.observation_probs = self.observation_probs / np.sum(self.observation_probs)  # Normalize
            self.valid_numbers = self.odd_numbers
        else:  # even
            # Compute Gaussian probabilities for even numbers
            self.observation_probs = np.exp(-0.5 * ((self.even_numbers - self.mean) / self.std_dev) ** 2)
            self.observation_probs = self.observation_probs / np.sum(self.observation_probs)  # Normalize
            self.valid_numbers = self.even_numbers
        
    def get_observation(self) -> int:
        """
        Generate an observation based on the hidden parameter.
        
        Returns:
            int: An observation (odd or even integer in range [1, n])
        """
        return self.rng.choice(self.valid_numbers, p=self.observation_probs)
    
    def get_reward(self, predicted_mode: float) -> float:
        """
        Get reward for predicting a mode.
        
        Args:
            predicted_mode: The predicted mode value
            
        Returns:
            float: Negative squared error as reward
        """
        error = predicted_mode - self.mean
        return -error ** 2  # Negative squared error (higher reward for better predictions)
    
    def _compute_observation_probability(self, observation: int, mode: float) -> float:
        """
        Compute probability of observation given a specific mode.
        
        Args:
            observation: The observed integer
            mode: The mode of the Gaussian distribution
            
        Returns:
            float: Probability of observation given mode
        """
        # Check if observation is consistent with hidden parameter
        if self.hidden_param == 'odd' and observation % 2 == 0:
            return 0.0
        if self.hidden_param == 'even' and observation % 2 == 1:
            return 0.0
            
        # Compute Gaussian probability
        prob = np.exp(-0.5 * ((observation - mode) / self.std_dev) ** 2)
        
        # Normalize over valid numbers for this hidden parameter
        if self.hidden_param == 'odd':
            valid_nums = self.odd_numbers
        else:
            valid_nums = self.even_numbers
            
        normalization = np.sum(np.exp(-0.5 * ((valid_nums - mode) / self.std_dev) ** 2))
        
        return prob / normalization if normalization > 0 else 0.0
    
    def update_belief(self, observation: int) -> np.ndarray:
        """
        Update belief state using Bayes' rule.
        
        Args:
            observation: New observation
            
        Returns:
            np.ndarray: Updated belief state over possible modes
        """
        # Compute likelihood for each possible mode
        likelihoods = np.array([self._compute_observation_probability(observation, mode) 
                               for mode in self.belief_points])
        
        # Bayes' rule: P(mode|obs) = P(obs|mode) * P(mode) / P(obs)
        # P(obs) = sum over all modes of P(obs|mode) * P(mode)
        p_obs = np.sum(likelihoods * self.belief)
        
        if p_obs == 0:
            # If observation is impossible, return uniform belief
            self.belief = np.ones(self.belief_resolution) / self.belief_resolution
        else:
            # Updated belief
            self.belief = likelihoods * self.belief / p_obs
        
        return self.belief
    
    def get_optimal_prediction(self) -> float:
        """
        Get the optimal mode prediction given current belief state.
        
        Returns:
            float: Optimal mode prediction (expected value of belief)
        """
        return np.sum(self.belief_points * self.belief)
    
    def get_max_likelihood_prediction(self) -> float:
        """
        Get the maximum likelihood mode prediction.
        
        Returns:
            float: Mode with highest belief probability
        """
        max_idx = np.argmax(self.belief)
        return self.belief_points[max_idx]
    
    def reset(self, new_seed: Optional[int] = None):
        """
        Reset the POMDP with a new hidden parameter.
        
        Args:
            new_seed: Optional new random seed
        """
        if new_seed is not None:
            self.rng = np.random.RandomState(new_seed)
        
        # Choose new hidden parameter
        self.hidden_param = self.rng.choice(['odd', 'even'])
        
        # Reset belief to uniform
        self.belief = np.ones(self.belief_resolution) / self.belief_resolution
        
        # Recompute probabilities
        self._compute_probabilities()
        
    def get_info(self) -> dict:
        """
        Get information about the current POMDP instance.
        
        Returns:
            dict: Information about the POMDP configuration and state
        """
        return {
            'n': self.n,
            'mean': self.mean,
            'std_dev': self.std_dev,
            'hidden_param': self.hidden_param,
            'odd_numbers': self.odd_numbers.tolist(),
            'even_numbers': self.even_numbers.tolist(),
            'valid_numbers': self.valid_numbers.tolist(),
            'observation_probs': self.observation_probs.tolist(),
            'belief_points': self.belief_points.tolist(),
            'current_belief': self.belief.tolist()
        }


def run_example():
    """Example usage of the ModePredictionPOMDP"""
    print("Mode Prediction POMDP Example")
    print("=" * 50)
    
    # Create POMDP with default configuration
    config = OddEvenPOMDPConfig(n=10, std_dev=1.5, seed=42)
    pomdp = OddEvenPOMDP(config)
    
    print(f"Configuration: n={config.n}, mean={pomdp.mean:.2f}, std_dev={config.std_dev}")
    print(f"Hidden parameter: {pomdp.hidden_param}")
    print(f"Valid numbers: {pomdp.valid_numbers}")
    print()
    
    print("Generating observations and updating belief:")
    print("-" * 40)
    
    # Generate some observations and update belief
    for step in range(8):
        obs = pomdp.get_observation()
        pomdp.update_belief(obs)
        optimal_pred = pomdp.get_optimal_prediction()
        ml_pred = pomdp.get_max_likelihood_prediction()
        reward = pomdp.get_reward(optimal_pred)
        
        print(f"Step {step + 1}: obs={obs}, optimal_pred={optimal_pred:.2f}, "
              f"ml_pred={ml_pred:.2f}, reward={reward:.3f}")
    
    print(f"\nTrue mean: {pomdp.mean:.2f}")
    print(f"Hidden parameter: {pomdp.hidden_param}")
    print(f"Final optimal prediction: {pomdp.get_optimal_prediction():.2f}")
    print(f"Final ML prediction: {pomdp.get_max_likelihood_prediction():.2f}")


if __name__ == "__main__":
    run_example()
