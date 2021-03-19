import argparse
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from .data_views import get_data_loaders
from .modules.FullHistory import FullHistory

from .utils import create_output_folder

def run_loop(model, dl, loss_func, step=False, optimizer=None):
    total_loss = 0
    for trip, hist, y in dl:
        pred = model(trip, hist)
        loss = loss_func(pred, y)

        if step:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss

    return (total_loss / len(dl)).item()

def get_predictions(model, dl):
    model.eval()
    with torch.no_grad():
        trips = []
        hists = []
        preds = []
        ys = []
        for trip, hist, y in dl:
            trips.append(trip)
            hists.append(hist)
            ys.append(y)
            preds.append(model(trip, hist))
        return torch.cat(trips), torch.cat(hists), torch.cat(ys), torch.cat(preds)

def graph_results(model, dl):
    trips, hists, ys, preds = get_predictions(model, dl)
    plt.ioff()
    absolute_errors = (preds-ys).reshape(-1).numpy()
    plt.hist(absolute_errors, bins=50)
    create_output_folder("plots")
    plt.savefig("output/plots/absolute_errors.png")

def run_experiment(
        encode_hist=True,
        loss='mse',
        l1_beta=10,
        learning_rate=1e-3,
        epochs=50,
        batch_size=16,
        dropout_rate=0,
        num_layers=4,
        layer_size=32,
        quick=False):

    writer = SummaryWriter()
    train_dl, test_dl = get_data_loaders(batch_size=batch_size, quick=quick)

    # Set loss func
    loss_funcs = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'smoothl1': nn.SmoothL1Loss(beta=l1_beta),
    }
    loss_func = loss_funcs[loss]

    model = FullHistory(encode_hist=encode_hist, dropout_rate=dropout_rate, num_layers=num_layers, layer_size=layer_size)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = run_loop(model, train_dl, loss_func, step=True, optimizer=opt)

        model.eval()
        with torch.no_grad():
            mse_loss = run_loop(model, test_dl, loss_funcs['mse'])
            mae_loss = run_loop(model, test_dl, loss_funcs['mae'])

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test/mse", mse_loss, epoch)
        writer.add_scalar("Loss/test/mae", mae_loss, epoch)
        print(f"Epoch {epoch}:\t training: {train_loss:,.1f}\t mse: {mse_loss:,.1f} \t mae: {mae_loss:,.1f}")

    if not quick:
        if not os.path.exists("output"):
            os.mkdir("output")
        graph_results(model, test_dl)
        create_output_folder("params")
        torch.save(model.state_dict(), "output/params/full_history.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Network structure
    p.add_argument("--encode_hist", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--layer-size", type=int, default=32)
    p.add_argument("--dropout-rate", type=float, default=0.5)

    # Optimizer
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--loss", choices=["mse", "mae", "smoothl1"], default="mse")
    p.add_argument("--l1-beta", type=float, default=10)

    # Training
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)

    p.add_argument('--quick', action="store_true")

    args = p.parse_args()

    if args.quick:
        args.epochs = 2

    run_experiment(
        **vars(args)
    )
