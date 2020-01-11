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
"""Linearised Bregman iteration for image denoising."""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pnpbi.util import TvDenoiser
from pnpbi.util import derivatives
import torch
import torchvision
import torchvision.transforms as transforms
from pnpbi.dncnn.data import NoisyBSDSDataset
from pnpbi.dncnn.model import DnCNN, DUDnCNN

# Define data.
image_dir = './data/BSDS300/images'
image_size = (128, 128)
sigma = 30

testset = NoisyBSDSDataset(image_dir, mode='test',
                           image_size=image_size, sigma=sigma)
testset = torch.utils.data.Subset(testset, list(range(0, 5)))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

PATH = './DnCNN_BSDS300.pth'
# net = DnCNN()
net = DUDnCNN()
net.load_state_dict(torch.load(PATH))


transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.squeeze(npimg), cmap='gray')
    plt.colorbar()


def linBregmanIteration():
    """Compute linearised Bregman iteration for denoising problem."""
    # Load phantom image.
    f = Image.open('data/brain.png').convert('L').resize(image_size)
    f = transform(f)
    fdelta = f + 2 / 255 * sigma * torch.randn(f.shape)

    fdelta, f = testset.__getitem__(0)

    # Add noise.
    # m, n = f.shape
    # fdelta = f + 0.05 * np.random.randn(m, n)

    # Define data fidelity and its gradient.
    def G(x, y):
        """Compute data fidelity function."""
        return torch.sum((x - y) ** 2) / 2

    def gradG(x, y):
        """Compute gradient of data fidelity function."""
        return x - y

    # Initialise data.
    tau = 1
    x = torch.zeros_like(fdelta)
    y = - tau * gradG(x, fdelta)

    # Define regularisation parameter.
    # alpha = 1

    plt.figure()
    ax = plt.subplot(2, 2, 1)
    imshow(f)
    ax.set_title('f')
    ax = plt.subplot(2, 2, 2)
    imshow(y)
    ax.set_title('y')
    ax = plt.subplot(2, 2, 3)
    imshow(x)
    ax.set_title('x')
    ax = plt.subplot(2, 2, 4)
    imshow(y - x)
    ax.set_title('y - x')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Run Bregman iteration.
    nbiter = 30
    for i in range(nbiter):

        # Denoise.
        with torch.no_grad():
            x = net(y.unsqueeze(0)).squeeze(0)

        # Update y.
        y -= tau * gradG(x, fdelta)

        plt.figure()
        ax = plt.subplot(2, 2, 1)
        imshow(f)
        ax.set_title('f')
        ax = plt.subplot(2, 2, 2)
        imshow(y)
        ax.set_title('y')
        ax = plt.subplot(2, 2, 3)
        imshow(x)
        ax.set_title('x')
        ax = plt.subplot(2, 2, 4)
        imshow(y - x)
        ax.set_title('y - x')
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    linBregmanIteration()
