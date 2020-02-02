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
"""Evaluates a CNN image denoiser."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from pnpbi.dncnn.data import NoisyBSDSDataset, NoisyBSDSDatasetTV, NoisyCTDataset
from pnpbi.util import radon


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()
    plt.show()


# Define data.
image_dir = './data/BSDS300/images'
image_size = (100, 100)
sigma = 30
alpha = 1

testset = NoisyBSDSDatasetTV(image_dir, mode='test',
                             image_size=image_size, sigma=sigma, alpha=alpha)
testset = NoisyBSDSDataset(image_dir, mode='test',
                           image_size=image_size, sigma=sigma)


# Define angles.
nang = 180
angles = np.linspace(0, np.pi, nang, False)

# Define Radon transform and adjoint.
K, Kadj, ndet = radon.radon2d(*image_size, angles)

def Kfun(x):
    return torch.from_numpy(K(x.numpy()))

def Kadjfun(x):
    return torch.from_numpy(Kadj(x.numpy()))

# def Kfun(x):
#     nimg = x.shape[0]
#     y = torch.ones((nimg, nang, ndet))
#     for k in range(nimg):
#         y[k] = torch.from_numpy(K(x[k].numpy()))
#     return y


# def Kadjfun(x):
#     nimg = x.shape[0]
#     print(x.shape)
#     y = torch.ones((nimg, 1, *image_size))
#     for k in range(nimg):
#         y[k, 0] = torch.from_numpy(Kadj(x[k][0].numpy()))
#     return y


testset = NoisyCTDataset(Kfun, image_dir, mode='test',
                         image_size=image_size, sigma=sigma)

testset = torch.utils.data.Subset(testset, list(range(0, 4)))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# image, label = testset.__getitem__(0)
# imshow(torchvision.utils.make_grid(image, normalize=True))
# imshow(torchvision.utils.make_grid(label))

for data in testloader:
    images, labels = data
    imshow(torchvision.utils.make_grid(images, normalize=True))
    imshow(torchvision.utils.make_grid(labels))
