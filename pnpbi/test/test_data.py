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
from pnpbi.data import NoisyCTDataset
from pnpbi.model import DnCNN, PG
from pnpbi.util.torch import helper
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import unittest


def imshow(img):
    """Plot images."""
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()
    plt.show()


class TestData(unittest.TestCase):

    def test_create_op_functions(self):

        # Define data dir.
        data_dir = 'data/phantom_test/images'

        # Check and use GPU if available.
        cuda = torch.cuda.is_available()

        # Use CPU for testing.
        device = torch.device('cuda' if cuda else 'cpu')

        # Init random seed for reproducible experiments.
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Define data.
        image_size = (100, 100)

        # Set noise level.
        sigma = 0.05

        # Set up operators, functional, and gradient.
        pb = helper.setup_reconstruction_problem(image_size)
        K, Kadj, G, gradG, data_size = pb

        # Define training set.
        trainset = NoisyCTDataset(K, data_dir, mode='train',
                                  image_size=image_size, sigma=sigma)
        train_loader = data.DataLoader(trainset, batch_size=4,
                                       shuffle=True,
                                       pin_memory=torch.cuda.is_available(),
                                       num_workers=1)

        # Plot results.
        for inputs, labels in train_loader:
            # Move to GPU if available.
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Check data.
            inputs_check = K(labels)

            # Compute reconstruction.
            outputs = Kadj(inputs)

            # Display results.
            disp_images = torch.cat((labels, outputs),
                                    2).to(torch.device('cpu'))
            imshow(torchvision.utils.make_grid(disp_images, normalize=True))

            # Display results.
            disp_images = torch.cat((inputs, inputs_check),
                                    2).to(torch.device('cpu'))
            imshow(torchvision.utils.make_grid(disp_images, normalize=True))

    def test_create_op_functions_cnn(self):

        # Define data dir.
        data_dir = 'data/phantom_test/images'

        # Check and use GPU if available.
        cuda = torch.cuda.is_available()

        # Use CPU for testing.
        device = torch.device('cuda' if cuda else 'cpu')

        # Init random seed for reproducible experiments.
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Define data.
        image_size = (100, 100)

        # Set noise level.
        sigma = 0.05

        # Set up operators, functional, and gradient.
        pb = helper.setup_reconstruction_problem(image_size)
        K, Kadj, G, gradG, data_size = pb

        # Define training set.
        trainset = NoisyCTDataset(K, data_dir, mode='train',
                                  image_size=image_size, sigma=sigma)
        train_loader = data.DataLoader(trainset, batch_size=4,
                                       shuffle=True,
                                       pin_memory=torch.cuda.is_available(),
                                       num_workers=1)

        # Create model and push to GPU is available.
        denoising_model = DnCNN(D=6, C=64).to(device)
        model = PG(denoising_model, image_size, gradG=gradG,
                   tau=2e-5, niter=5).to(device)
        model.eval()

        # Plot results.
        for inputs, labels in train_loader:
            # Move to GPU if available.
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Check data.
            inputs_check = K(labels)

            # Compute reconstruction.
            outputs = Kadj(inputs)

            # Compute forward pass.
            with torch.no_grad():
                pg_outputs = model(inputs)

            # Display results.
            disp_images = torch.cat((labels, outputs),
                                    2).to(torch.device('cpu'))
            imshow(torchvision.utils.make_grid(disp_images, normalize=True))

            # Display results.
            disp_images = torch.cat((inputs, inputs_check),
                                    2).to(torch.device('cpu'))
            imshow(torchvision.utils.make_grid(disp_images, normalize=True))

            # Display results.
            disp_images = torch.cat((labels, pg_outputs),
                                    2).to(torch.device('cpu'))
            imshow(torchvision.utils.make_grid(disp_images, normalize=True))


if __name__ == '__main__':
    unittest.main()
