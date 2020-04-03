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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from pnpbi.dncnn.data import NoisyCTDataset
from pnpbi.dncnn import model
from pnpbi.util.torch import helper

# Define data.
image_dir = './data/phantom/images'
image_size = (40, 40)

# Set noise level.
sigma = 0.05

# Define path for model.
model_path = './pg_phantom.pth'

# Set up operators, functional, and gradient.
pb = helper.setup_reconstruction_problem(image_size)
Kfun, Kadjfun, G, gradG, data_size = pb

# Define training set.
trainset = NoisyCTDataset(Kfun, image_dir, mode='train',
                          image_size=image_size, sigma=sigma)
trainset = torch.utils.data.Subset(trainset, list(range(0, 40)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Define test set.
testset = NoisyCTDataset(Kfun, image_dir, mode='test',
                         image_size=image_size, sigma=sigma)
testset = torch.utils.data.Subset(testset, list(range(0, 10)))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# Create model and load if present.
denoising_model = model.DnCNN(D=6, C=64)
net = model.PG(denoising_model, image_size, gradG=gradG, tau=1e-5, niter=3)
net.load_state_dict(torch.load(model_path))
net.eval()


def imshow(img):
    """Plot images."""
    # Unnormalise for plotting.
    # img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()
    plt.show()


# Plot results.
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)

        imshow(torchvision.utils.make_grid(images, normalize=True))
        imshow(torchvision.utils.make_grid(labels, normalize=True))
        imshow(torchvision.utils.make_grid(Kadjfun(images), normalize=True))
        imshow(torchvision.utils.make_grid(outputs, normalize=True))
