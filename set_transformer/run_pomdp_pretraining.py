import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .mixture_of_mvns import MixtureOfMVNs
from .models import DeepSet, PFSetTransformer, SetTransformer
from .mvn_diag import MultivariateNormalDiag
from tqdm import tqdm
from set_transformer.data.dataset import get_data_loader
from set_transformer.loss import ChamferDistanceLoss, SinkhornLoss


def train(args, train_loader, eval_loader, train_size, eval_size):
    """Train the model on POMDP data.

    Args:
        args: Training arguments.
        train_loader: Training data loader.
        eval_loader: Evaluation data loader.
        train_size: Number of training samples.
        eval_size: Number of evaluation samples.
    """
    save_dir = os.path.join('results', args.net, args.run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(
        logging.FileHandler(
            os.path.join(save_dir, "train_" + time.strftime("%Y%m%d-%H%M") + ".log"),
            mode="w",
        )
    )
    logger.info(str(args) + "\n")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.net == "pf_set_transformer":
        net = PFSetTransformer(
            num_particles=args.K,
            dim_partciles=4,
            num_encodings=8,
            dim_encoder=2,
            num_inds=32,
            dim_hidden=128,
            num_heads=4,
            ln=True,
        ).to(device)
    else:
        raise ValueError("Invalid net {}".format(args.net))

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = SinkhornLoss(p=2, blur=0.5) # For example, if you’re reconstructing particles
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps
    )

    tick = time.time()
    eval_loss = []
    line, loss = test(net, eval_loader, eval_size)
    eval_loss.append(loss)
    print(line)
    for t in range(1, args.num_steps + 1):
        net.train()
        print(f"Timestep: {t}")
        for batch in train_loader:

            optimizer.zero_grad()

            # Forward pass
            batch = batch.to(device)
            outputs = net(batch)
            # Calculate the reconstruction loss (outputs vs. input batch)
            loss = criterion(
                outputs, batch
            )  # batch is the target since we are reconstructing
            # Backward pass and optimize
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        if t % args.test_freq == 0:
            line = "step {}, lr {:.3e}, ".format(t, optimizer.param_groups[0]["lr"])
            linex, loss = test(net, eval_loader, eval_size, verbose=False)
            line += linex
            eval_loss.append(loss)
            line += " ({:.3f} secs)".format(time.time() - tick)
            tick = time.time()
            logger.info(line)

        if t % args.save_freq == 0:
            torch.save(
                {"state_dict": net.state_dict()},
                os.path.join(save_dir, f"model_{t}.tar")
            )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
    plot(eval_loader, eval_loss[-1])


def test(net, eval_loader, eval_size, verbose=True):
    """Test the model on evaluation data.

    Args:
        net: Model to test.
        eval_loader: Evaluation data loader.
        eval_size: Number of evaluation samples.
        verbose: Whether to print results.

    Returns:
        tuple: Test results string and average loss.
    """
    device = next(net.parameters()).device
    net.eval()
    total_loss = 0.0
    criterion = SinkhornLoss(reduction="sum")

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            outputs = net(batch)
            loss = criterion(outputs, batch)
            total_loss += loss.item()

    avg_loss = total_loss / eval_size
    result = f"Test Loss: {avg_loss:.4f}"
    if verbose:
        print(result)

    return result, avg_loss


def plot(eval_loader, loss):
    """Plot results.

    Args:
        eval_loader: Evaluation data loader.
        loss: Loss value to display.
    """
    device = next(net.parameters()).device
    net.eval()

    with torch.no_grad():
        X = next(iter(eval_loader))
        X = X.to(device)
        Z = net(X)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X.cpu().numpy()[:, 0], X.cpu().numpy()[:, 1])
    plt.title('Input')

    plt.subplot(122)
    plt.scatter(Z.cpu().numpy()[:, 0], Z.cpu().numpy()[:, 1])
    plt.title(f'Latent (Loss: {loss:.4f})')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--num_bench", type=int, default=100)
    parser.add_argument("--net", type=str, default="set_transformer")
    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--N_min", type=int, default=300)
    parser.add_argument("--N_max", type=int, default=600)
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_name", type=str, default="trial")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--test_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    B = args.B
    N_min = args.N_min
    N_max = args.N_max
    num_particles = args.K

    dim_particles = 4
    # mvn = MultivariateNormalDiag(D)
    # mog = MixtureOfMVNs(mvn)
    # dim_output = 2*D

    save_dir = os.path.join("results", args.net, args.run_name + "_" + time.strftime("%Y-%m-DD:hh-mm-ss"))

    if args.mode == "bench":
        generate_benchmark()
    elif args.mode == "train":
        train_loader, eval_loader, train_size, eval_size = get_data_loader(
            batch_size=args.batch_size, device="cuda"
        )
        train(args, train_loader, eval_loader, train_size, eval_size)
    elif args.mode == "test":
        bench = torch.load(benchfile)
        ckpt = torch.load(os.path.join(save_dir, "model.tar"))
        net.load_state_dict(ckpt["state_dict"])
        test(bench)
    elif args.mode == "plot":
        ckpt = torch.load(os.path.join(save_dir, "model.tar"))
        net.load_state_dict(ckpt["state_dict"])
        plot()
