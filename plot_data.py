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
"""Plots the training set."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from pnpbi.dncnn.data import NoisyBSDSDataset
from pnpbi.util import operators

# Define data.
image_dir = './data/BSDS300/images'
image_size = (100, 100)
sigma = 30

# Define path for model.
model_path = './pg_BSDS300.pth'

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


def imshow(img):
    # Unnormalise for plotting.
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()
    plt.show()


for data in testloader:
    images, labels = data
    imshow(torchvision.utils.make_grid(images))
    imshow(torchvision.utils.make_grid(Kadj(images)))
    imshow(torchvision.utils.make_grid(labels))
