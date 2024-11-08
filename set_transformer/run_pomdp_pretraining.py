import argparse
import logging
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mixture_of_mvns import MixtureOfMVNs
from models import DeepSet, SetTransformer
from mvn_diag import MultivariateNormalDiag
from tqdm import tqdm

from set_transformer.data.dataset import get_data_loader

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
K = args.K

D = 4
# mvn = MultivariateNormalDiag(D)
# mog = MixtureOfMVNs(mvn)
# dim_output = 2*D


# def generate_benchmark():
#     if not os.path.isdir('benchmark'):
#         os.makedirs('benchmark')
#     N_list = np.random.randint(N_min, N_max, args.num_bench)
#     data = []
#     ll = 0.
#     for N in tqdm(N_list):
#         X, labels, pi, params = mog.sample(B, N, K, return_gt=True)
#         ll += mog.log_prob(X, pi, params).item()
#         data.append(X)
#     bench = [data, ll/args.num_bench]
#     torch.save(bench, benchfile)

save_dir = os.path.join("results", args.net, args.run_name)


def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    training_dataloader, eval_dataloader, training_size, eval_size = get_data_loader(batch_size=64, device="cuda")
    print(training_size, eval_size)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(
        logging.FileHandler(
            os.path.join(save_dir, "train_" + time.strftime("%Y%m%d-%H%M") + ".log"),
            mode="w",
        )
    )
    logger.info(str(args) + "\n")

    if args.net == "set_transformer":
        net = SetTransformer(D, K, D).cuda()
    elif args.net == "deepset":
        net = DeepSet(D, K, D).cuda()
    else:
        raise ValueError("Invalid net {}".format(args.net))

    criterion = nn.MSELoss()  # For example, if you’re reconstructing particles
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    tick = time.time()
    eval_loss = []
    line, loss = test(net, eval_dataloader, eval_size)
    eval_loss.append(loss)
    print(line)
    for t in range(1, args.num_steps + 1):
        net.train()
        print(f"Timestep: {t}")
        for batch in training_dataloader:

            optimizer.zero_grad()

            # Forward pass
            batch = batch.to("cuda")
            outputs = net(batch)
            # Calculate the reconstruction loss (outputs vs. input batch)
            loss = criterion(
                outputs, batch
            )  # batch is the target since we are reconstructing

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        scheduler.step()

        if t % args.test_freq == 0:
            line = "step {}, lr {:.3e}, ".format(t, optimizer.param_groups[0]["lr"])
            linex, loss = test(net, eval_dataloader, eval_size, verbose=False)
            line +=linex
            eval_loss.append(eval_loss)
            line += " ({:.3f} secs)".format(time.time() - tick)
            tick = time.time()
            logger.info(line)

        if t % args.save_freq == 0:
            torch.save(
                {"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar")
            )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
    plot(args.num_steps, eval_loss)

def test(net, eval_data, eval_size, verbose=True):
    net.eval()
    avg_loss = 0.0
    criterion = nn.MSELoss(reduction="sum")

    for batch in eval_data:
        batch = batch.to("cuda")
        outputs = net(batch)
        loss = criterion(outputs, batch).item()
        avg_loss += loss

    avg_loss /= eval_size
    rmse = np.sqrt(avg_loss)
    line = "Eval RMSE Loss: {:.4f}".format(rmse)

    if verbose:
        logger = logging.getLogger(args.run_name)
        logger.addHandler(
            logging.FileHandler(os.path.join(save_dir, "test.log"), mode="w")
        )
        logger.info(line)

    return line, rmse


def plot(t_steps, loss):
    fig, ax = plt.subplots()
    ax.plot(np.arange(t_steps), loss)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Eval loss")


if __name__ == "__main__":
    if args.mode == "bench":
        generate_benchmark()
    elif args.mode == "train":
        train()
    elif args.mode == "test":
        bench = torch.load(benchfile)
        ckpt = torch.load(os.path.join(save_dir, "model.tar"))
        net.load_state_dict(ckpt["state_dict"])
        test(bench)
    elif args.mode == "plot":
        ckpt = torch.load(os.path.join(save_dir, "model.tar"))
        net.load_state_dict(ckpt["state_dict"])
        plot()
