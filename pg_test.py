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
from pnpbi.dncnn.data import NoisyBSDSDataset
from pnpbi.dncnn import model
from pnpbi.util.torch import functionals
from pnpbi.util.torch import operators

# Define data.
image_dir = './data/phantom/images'
image_size = (100, 100)
sigma = 30

# Define path for model.
model_path = './pg_phantom.pth'

# Define training set.
trainset = NoisyBSDSDataset(image_dir, mode='train',
                            image_size=image_size, sigma=sigma)
trainset = torch.utils.data.Subset(trainset, list(range(0, 40)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Define test set.
testset = NoisyBSDSDataset(image_dir, mode='test',
                           image_size=image_size, sigma=sigma)
testset = torch.utils.data.Subset(testset, list(range(0, 10)))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


# Define identity operator for denoising.
def K(x: torch.Tensor):
    """Identity operator."""
    return x


# Define adjoint.
Kadj = K

# Create function handles for use with torch.
operators.create_op_functions(K, Kadj, image_size, image_size)

# Create data fidelity and its gradient.
G, gradG = functionals.OpSqNormDataTerm(K, Kadj)

# Create model and load if present.
denoising_model = model.DnCNN(D=6, C=64)
net = model.PG(denoising_model, image_size, gradG=gradG, tau=0.01, niter=5)
net.load_state_dict(torch.load(model_path))


def imshow(img):
    """Plot images."""
    # Unnormalise for plotting.
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()
    plt.show()


# Plot results.
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        imshow(torchvision.utils.make_grid(images))
        imshow(torchvision.utils.make_grid(labels))
        # imshow(torchvision.utils.make_grid(Kadjfun(images.squeeze(1))))
        imshow(torchvision.utils.make_grid(outputs))
