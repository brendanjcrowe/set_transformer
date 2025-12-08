"""
Utility functions for OddEvenPOMDP environment and particle filter interaction.
"""

import numpy as np


def odd_even_pf_interaction_mapper(base_env_obs, base_env_info, base_env_action=None, unwrapped_env=None):
    """
    Maps environment observations and actions to particle filter predict/update arguments.
    
    For OddEvenPOMDP:
    - predict: No special arguments needed (action is the predicted mean, but we don't use it for prediction)
    - update: The observation is the particles from the POMDP
    
    Args:
        base_env_obs: Observation from the base environment (particles from POMDP)
        base_env_info: Info dict from the base environment
        base_env_action: Action taken (predicted mean, 0-indexed)
        unwrapped_env: The unwrapped environment (OddEvenPOMDPEnv)
    
    Returns:
        dict with 'predict_args' and 'update_args' keys
    """
    predict_args = {}
    update_args = {}
    
    # For prediction, we don't need any special arguments
    # The action (predicted mean) could be used, but for now we just use process noise
    
    # For update, the observation is the particles from the POMDP
    # The particle filter will compute the mean of these particles
    
    return {
        'predict_args': predict_args,
        'update_args': update_args
    }

