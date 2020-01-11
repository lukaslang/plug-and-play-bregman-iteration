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
from pnpbi.dncnn.data import NoisyBSDSDataset
from pnpbi.dncnn.model import PnpBi


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Define data.
image_dir = './data/BSDS300/images'
image_size = (100, 100)
sigma = 30

testset = NoisyBSDSDataset(image_dir, mode='test',
                           image_size=image_size, sigma=sigma)
testset = torch.utils.data.Subset(testset, list(range(0, 10)))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


# Define data fidelity and its gradient.
def G(x, y):
    """Compute data fidelity function."""
    return torch.sum((x - y) ** 2) / 2


def gradG(x, y):
    """Compute gradient of data fidelity function."""
    return x - y


PATH = './PnpBi_BSDS300.pth'
net = PnpBi(gradG=gradG, tau=1e-1, niter=5)
net.load_state_dict(torch.load(PATH))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        imshow(torchvision.utils.make_grid(images))
        imshow(torchvision.utils.make_grid(labels))
        imshow(torchvision.utils.make_grid(outputs))

        # plt.figure()
        # ax = plt.subplot(1, 3, 1)
        # plt.imshow(images[0, 0], cmap='gray')
        # plt.colorbar()
        # ax = plt.subplot(1, 3, 2)
        # plt.imshow(labels[0, 0], cmap='gray')
        # plt.colorbar()
        # ax = plt.subplot(1, 3, 3)
        # plt.imshow(outputs[0, 0], cmap='gray')
        # plt.colorbar()
        # plt.show()
