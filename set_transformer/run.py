"""Training and evaluation script for Set Transformer on mixture modeling.

This script implements training, testing, and visualization of Set Transformer and
DeepSet models for learning mixture of Gaussian distributions. The models learn to
predict mixture parameters (weights and component parameters) from sets of points.

The script supports several modes:
- 'bench': Generate benchmark dataset
- 'train': Train a model
- 'test': Evaluate a trained model
- 'plot': Visualize model predictions
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from .models import SetTransformer, DeepSet
from .mixture_of_mvns import MixtureOfMVNs
from .mvn_diag import MultivariateNormalDiag


# Default configuration
DEFAULT_CONFIG = {
    'mode': 'train',
    'num_bench': 100,
    'net': 'set_transformer',
    'B': 10,
    'N_min': 300,
    'N_max': 600,
    'K': 4,
    'gpu': '0',
    'lr': 1e-3,
    'run_name': 'trial',
    'num_steps': 50000,
    'test_freq': 200,
    'save_freq': 400
}

# Initialize global variables with defaults
B = DEFAULT_CONFIG['B']
N_min = DEFAULT_CONFIG['N_min']
N_max = DEFAULT_CONFIG['N_max']
K = DEFAULT_CONFIG['K']
D = 2  # Data dimension

mvn = MultivariateNormalDiag(D)
mog = MixtureOfMVNs(mvn)
dim_output = 2*D

print(B, N_min, N_max, K, D)

# Initialize model
if DEFAULT_CONFIG['net'] == 'set_transformer':
    net = SetTransformer(D, K, dim_output).cuda()
elif DEFAULT_CONFIG['net'] == 'deepset':
    net = DeepSet(D, K, dim_output).cuda()
else:
    raise ValueError('Invalid net {}'.format(DEFAULT_CONFIG['net']))

benchfile = os.path.join('benchmark', 'mog_{:d}.pkl'.format(K))


def generate_benchmark() -> None:
    """Generate benchmark datasets."""
    benchmark_dir = os.environ.get('BENCHMARK_DIR', 'benchmark')
    if not os.path.exists(benchmark_dir):
        os.makedirs(benchmark_dir)

    # Generate mixture of Gaussians data
    data = []
    ll = 0.
    for N in tqdm(N_list):
        X, labels, pi, params = mog.sample(B, N, K, return_gt=True)
        ll += mog.log_prob(X, pi, params).item()
        data.append(X)
    bench = [data, ll/DEFAULT_CONFIG['num_bench']]
    torch.save(bench, benchfile)


save_dir = os.path.join('results', DEFAULT_CONFIG['net'], DEFAULT_CONFIG['run_name'])


def train(args: argparse.Namespace) -> None:
    """Train the model.

    Args:
        args (argparse.Namespace): Training arguments.
    """
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Set up model
    if args.net == 'set_transformer':
        net = SetTransformer(D, K, dim_output).to(device)
    elif args.net == 'deepset':
        net = DeepSet(D, K, dim_output).to(device)
    else:
        raise ValueError('Invalid net {}'.format(args.net))
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    # Generate benchmark data
    if not os.path.isfile(benchfile):
        generate_benchmark()
    bench = torch.load(benchfile)
    
    # Training loop
    pbar = tqdm(range(args.num_steps))
    for step in pbar:
        optimizer.zero_grad()
        
        # Generate batch
        N = np.random.randint(N_min, N_max)
        X = mog.sample(args.B, N, K)
        X = X.to(device)
        
        # Forward pass
        pi, params = mvn.parse(net(X))
        loss = -mog.log_prob(X, pi, params).mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Testing
        if (step + 1) % args.test_freq == 0:
            test_loss = test(bench, verbose=False)
            pbar.set_description(f'Step {step + 1}, Test Loss: {test_loss:.4f}')
        
        # Save checkpoint
        if (step + 1) % args.save_freq == 0:
            torch.save({'state_dict':net.state_dict()},
                    os.path.join(save_dir, 'model.tar'))

    torch.save({'state_dict':net.state_dict()},
        os.path.join(save_dir, 'model.tar'))


def test(bench: List[torch.Tensor], verbose: bool = True) -> float:
    """Test the model on benchmark data.

    Args:
        bench (List[torch.Tensor]): List of benchmark tensors.
        verbose (bool, optional): Whether to print results. Defaults to True.

    Returns:
        float: Average test loss.
    """
    device = next(net.parameters()).device
    losses = []
    for X in bench[0]:
        X = X.to(device)
        with torch.no_grad():
            pi, params = mvn.parse(net(X))
            loss = -mog.log_prob(X, pi, params).mean()
        losses.append(loss.item())
    
    avg_loss = np.mean(losses)
    if verbose:
        print(f'Test Loss: {avg_loss:.4f}')
    
    return avg_loss


def plot() -> None:
    """Visualize model predictions.

    Generates random sets of points and visualizes the predicted mixture
    components along with the data points.
    """
    net.eval()
    X = mog.sample(B, np.random.randint(N_min, N_max), K)
    pi, params = mvn.parse(net(X))
    ll, labels = mog.log_prob(X, pi, params, return_labels=True)
    fig, axes = plt.subplots(2, B//2, figsize=(7*B//5,5))
    mog.plot(X, labels, params, axes)
    plt.show()


if __name__ == '__main__':
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate Set Transformer on mixture modeling')
    parser.add_argument('--mode', type=str, default=DEFAULT_CONFIG['mode'],
                      help='Mode: bench, train, test, or plot')
    parser.add_argument('--num_bench', type=int, default=DEFAULT_CONFIG['num_bench'],
                      help='Number of benchmark datasets to generate')
    parser.add_argument('--net', type=str, default=DEFAULT_CONFIG['net'],
                      help='Network architecture: set_transformer or deepset')
    parser.add_argument('--B', type=int, default=DEFAULT_CONFIG['B'],
                      help='Batch size')
    parser.add_argument('--N_min', type=int, default=DEFAULT_CONFIG['N_min'],
                      help='Minimum number of points per set')
    parser.add_argument('--N_max', type=int, default=DEFAULT_CONFIG['N_max'],
                      help='Maximum number of points per set')
    parser.add_argument('--K', type=int, default=DEFAULT_CONFIG['K'],
                      help='Number of mixture components')
    parser.add_argument('--gpu', type=str, default=DEFAULT_CONFIG['gpu'],
                      help='GPU device ID')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                      help='Learning rate')
    parser.add_argument('--run_name', type=str, default=DEFAULT_CONFIG['run_name'],
                      help='Name of this run')
    parser.add_argument('--num_steps', type=int, default=DEFAULT_CONFIG['num_steps'],
                      help='Number of training steps')
    parser.add_argument('--test_freq', type=int, default=DEFAULT_CONFIG['test_freq'],
                      help='Test frequency in steps')
    parser.add_argument('--save_freq', type=int, default=DEFAULT_CONFIG['save_freq'],
                      help='Model save frequency in steps')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Update global variables with command line arguments
    B = args.B
    N_min = args.N_min
    N_max = args.N_max
    K = args.K

    if args.mode == 'bench':
        generate_benchmark()
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        bench = torch.load(benchfile)
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        if args.net == 'set_transformer':
            net = SetTransformer(D, K, dim_output).cuda()
        elif args.net == 'deepset':
            net = DeepSet(D, K, dim_output).cuda()
        else:
            raise ValueError('Invalid net {}'.format(args.net))
        net.load_state_dict(ckpt['state_dict'])
        test(bench)
    elif args.mode == 'plot':
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        if args.net == 'set_transformer':
            net = SetTransformer(D, K, dim_output).cuda()
        elif args.net == 'deepset':
            net = DeepSet(D, K, dim_output).cuda()
        else:
            raise ValueError('Invalid net {}'.format(args.net))
        net.load_state_dict(ckpt['state_dict'])
        plot()
