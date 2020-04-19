#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019 Lukas Lang
#
# This file is part of PNPBI.
#
#    PNPBI is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PNPBI is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PNPBI. If not, see <http://www.gnu.org/licenses/>.
"""Trains a CNN image reconstruction network using the PG method."""
import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from tqdm import tqdm

from pnpbi.data import NoisyCTDataset
from pnpbi.model import DnCNN, PG
from pnpbi.util.torch import helper
from pnpbi.util import utils


# Specify default arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/phantom/images',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='checkpoint.pth.tar',
                    help="Optional, name of the file in --model_dir \
                    containing last checkpoint")


def imshow(img):
    """Plot images."""
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()


def train(model, optimizer, loss_fn, loader, params, device) -> float:
    """Train model for one epoch.

    Args:
    ----
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn: A loss function.
        loader (torch.utils.data.DataLoader): The dataloader to use.
        params (pnpbi.util.utils.Params): Training parameters.
        device (torch.device): The device to use.

    Return:
    ------
        loss (float): The training loss.
    """
    # Set model to training mode.
    model.train()

    # Train model.
    loss_avg = utils.RunningAvg()

    with tqdm(total=len(loader)) as t:
        for step, (inputs, labels) in enumerate(loader):
            # Move to GPU if available.
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Compute forward pass and evaluate loss.
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Clear previously computed gradients.
            optimizer.zero_grad()

            # Compute gradients and perform update step.
            loss.backward()
            optimizer.step()

            # Update average loss.
            loss_avg.add(loss.item())

            # Update progress.
            t.set_postfix(loss='{:.3e}'.format(loss_avg()))
            t.update()

    # Return loss after current epoch.
    return loss_avg()


def evaluate(model, loss_fn, loader, params, device) -> float:
    """Evaluate model.

    Args:
    ----
        model (torch.nn.Module): The model.
        loss_fn: A loss function.
        loader (torch.utils.data.DataLoader): The dataloader to use.
        params (pnpbi.util.utils.Params): Training parameters.
        device (torch.device): The device to use.

    Return:
    ------
        loss (float): The value of the evaluated loss function.
    """
    # Set model to evaluate mode.
    model.eval()

    # Compute validation loss.
    loss_avg = utils.RunningAvg()
    with torch.no_grad():
        for inputs, labels in loader:
            # Move to GPU if available.
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            batch_loss = loss_fn(outputs, labels)
            loss_avg.add(batch_loss.item())

    return loss_avg()


def train_and_evaluate(model, optimizer, device, train_loader, valid_loader,
                       loss_fn, params, model_dir, restore_file=None):
    """Train model and evaluate after each epoch.

    Args:
    ----
        model (torch.nn.Module): Model.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): The device to use.
        train_loader (torch.utils.data.DataLoader): Training data.
        valid_loader (torch.utils.data.DataLoader): Validation data.
        loss_fn: A loss function.
        params (pnpbi.util.utils.Params): Training parameters.
        model_dir (str): Directory where checkpoint is found/created.
        restore_file (str): Filename of checkpoint (e.g. xyz.pth.tar).
    """
    # Number of already finished epochs (relevant if checkpoint is loaded).
    fin_epochs = 0

    # Load parameters from file.
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file)
        logging.info(f"Loading checkpoint from '{restore_path}'.")
        fin_epochs = utils.load_checkpoint(restore_path, model,
                                           optimizer, device)

    # Init lists to store losses during epochs.
    train_loss, val_loss = [], []

    for epoch in range(fin_epochs, params.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        # Train one epoch (one full pass over training set)
        loss = train(model, optimizer, loss_fn, train_loader, params, device)
        train_loss.append(loss)

        # Evaluate.
        loss = evaluate(model, loss_fn, valid_loader, params, device)
        val_loss.append(loss)

        # Save/overwrite checkpoint.
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optimizer_dict': optimizer.state_dict()},
                              model_dir)

        # Log losses and tau.
        msg = f"Training loss: {train_loss[-1]:.3e}, " \
            f"Validation loss: {val_loss[-1]:.3e}, tau={model.tau:.3e}"
        logging.info(msg)

        # Plot training and validation losses.
        if params.plot:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(train_loss, '-', label='Training loss')
            plt.plot(val_loss, '--', label='Validation loss')
            plt.legend(frameon=False)

            # Plot first result.
            with torch.no_grad():
                inputs, labels = next(iter(valid_loader))

                # Move to GPU if available.
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Compute reconstruction.
                outputs = model(inputs)

                plt.subplot(2, 1, 2)
                disp_images = torch.cat((labels, outputs), 2)
                imshow(torchvision.utils.make_grid(disp_images,
                                                   normalize=True))
            plt.show()


if __name__ == '__main__':

    # Load parameters from JSON file.
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Check and use GPU if available.
    cuda = torch.cuda.is_available()
    torch_device = torch.device('cuda' if cuda else 'cpu')

    # Init random seed for reproducible experiments.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Init logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Set up problem and load datasets.
    logging.info(f"Loading datasets from '{args.data_dir}'.")

    # Define data.
    image_size = (100, 100)

    # Set noise level.
    sigma = 0.05

    # Set up operators, functional, and gradient.
    pb = helper.setup_reconstruction_problem(image_size)
    K, Kadj, G, gradG, data_size = pb

    # Define training set.
    trainset = NoisyCTDataset(K, args.data_dir, mode='train',
                              image_size=image_size, sigma=sigma)
    trainset = data.Subset(trainset, list(range(0, 100)))
    train_loader = data.DataLoader(trainset, batch_size=params.batch_size,
                                   shuffle=True,
                                   pin_memory=torch.cuda.is_available(),
                                   num_workers=params.num_workers)

    # Define test set.
    valset = NoisyCTDataset(K, args.data_dir, mode='test',
                            image_size=image_size, sigma=sigma)
    valset = data.Subset(valset, list(range(0, 10)))
    valid_loader = data.DataLoader(valset, batch_size=params.batch_size,
                                   shuffle=False,
                                   pin_memory=torch.cuda.is_available(),
                                   num_workers=params.num_workers)

    # Create model and push to GPU is available.
    denoising_model = DnCNN(D=6, C=64).to(torch_device)
    model = PG(denoising_model, image_size, gradG=gradG,
               tau=2e-5, niter=5).to(torch_device)

    # Define optimisation problem.
    loss_fn = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    logging.info(f"Starting training for {params.num_epochs} epoch(s).")
    logging.info(f"Parameters: {params}.")

    train_and_evaluate(model, optimizer, torch_device,
                       train_loader, valid_loader, loss_fn, params,
                       args.model_dir, args.restore_file)

    logging.info(f"Training for {params.num_epochs} epoch(s) completed.")
