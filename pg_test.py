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
import torch.utils.data as data
import torchvision

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
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()
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
    device = torch.device('cuda' if cuda else 'cpu')

    # Init random seed for reproducible experiments.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Init logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Set up problem and load datasets.
    logging.info(f"Loading datasets from '{args.data_dir}'.")

    # Define data.
    image_size = (40, 40)

    # Set noise level.
    sigma = 0.05

    # Set up operators, functional, and gradient.
    pb = helper.setup_reconstruction_problem(image_size)
    K, Kadj, G, gradG, data_size = pb

    # Define test set.
    valset = NoisyCTDataset(K, args.data_dir, mode='test',
                            image_size=image_size, sigma=sigma)
    valset = data.Subset(valset, list(range(0, 4)))
    valid_loader = data.DataLoader(valset, batch_size=params.batch_size,
                                   shuffle=False,
                                   num_workers=params.num_workers)

    # Create model and push to GPU is available.
    denoising_model = DnCNN(D=6, C=64).to(device)
    model = PG(denoising_model, image_size, gradG=gradG,
               tau=2e-5, niter=5).to(device)

    # Load model from file.
    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file)
        logging.info(f"Loading checkpoint from '{restore_path}'.")
        utils.load_checkpoint(restore_path, model, device=device)

    # Create Landweber iteration.
    class Identity(torch.nn.Module):
        """Identity operator."""

        def forward(self, input):
            """Return input."""
            return input
    lw_model = PG(Identity(), image_size, gradG=gradG, tau=2e-5, niter=20)

    # Set models to evaluate mode.
    model.eval()
    lw_model.eval()

    # Plot results.
    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Compute reconstruction.
            outputs = model(inputs)

            # Display results.
            disp_images = torch.cat((labels, outputs), 2)
            imshow(torchvision.utils.make_grid(disp_images, normalize=True))

            # Compute Landweber reconstruction.
            lw_outputs = lw_model(inputs)

            # Display results.
            disp_images = torch.cat((labels, lw_outputs), 2)
            imshow(torchvision.utils.make_grid(disp_images, normalize=True))
