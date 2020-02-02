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
"""Neural network models for image denoising."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PnpBi(nn.Module):
    """An iterative model."""

    def __init__(self, image_size, gradG, tau=1, niter=1):
        """Initialise model.

        Constructor takes the number of blocks and the number of channels.

        Args:
        ----
            D (int): The number of Conv-Bias-Normalisation-ReLU blocks.
            C (int): The number of channels.
        """
        super(PnpBi, self).__init__()
        self.image_size = image_size
        self.gradG = gradG
        self.tau = nn.Parameter(torch.Tensor([tau]))
        self.niter = niter

        # Define denoising model.
        self.model = DnCNN()

    def forward(self, fdelta):
        """Compute the forward pass for given put data.

        Args:
        ----
            x (Tensor): A noisy input image.

        Return:
        ------
            x (Tensor): The denoised image.
        """
        # Initialise data.
        x = torch.zeros(fdelta.shape[0], 1, *self.image_size)
        y = - self.tau * self.gradG(x, fdelta)

        # Run Bregman iteration.
        for i in range(self.niter):

            # Denoise.
            x = self.model.forward(y)

            # Update y.
            y = y - self.tau * self.gradG(x, fdelta)

        return x


class DnCNN(nn.Module):
    """A convolutional neural network for image denoising."""

    def __init__(self, D=6, C=64):
        """Initialise model.

        Constructor takes the number of blocks and the number of channels.

        Args:
        ----
            D (int): The number of Conv-Bias-Normalisation-ReLU blocks.
            C (int): The number of channels.
        """
        super(DnCNN, self).__init__()
        self.D = D

        # Create convolutional layers.
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(1, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for x in range(D)])
        self.conv.append(nn.Conv2d(C, 1, 3, padding=1))

        # Use He's initialization.
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # Add batch normalization.
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])

        # Initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        """Compute the forward pass for given put data.

        Args:
        ----
            x (Tensor): A noisy input image.

        Return:
        ------
            x (Tensor): The denoised image.
        """
        D = self.D
        h = F.relu(self.conv[0](x))
        for i in range(D):
            h = F.relu(self.bn[i](self.conv[i+1](h)))
        x = self.conv[D+1](h) + x
        return x


class DUDnCNN(nn.Module):
    """A convolutional neural network for image denoising."""

    def __init__(self, D=6, C=64):
        """Initialise model.

        Constructor takes the number of blocks and the number of channels.

        Args:
        ----
            D (int): The number of Conv-Bias-Normalisation-ReLU blocks.
            C (int): The number of channels.
        """
        super(DUDnCNN, self).__init__()
        self.D = D

        # compute k(max_pool) and l(max_unpool)
        k = [0]
        k.extend([i for i in range(D//2)])
        k.extend([k[-1] for _ in range(D//2, D+1)])
        l = [0 for _ in range(D//2+1)]
        l.extend([i for i in range(D+1-(D//2+1))])
        l.append(l[-1])

        # holes and dilations for convolution layers
        holes = [2**(kl[0]-kl[1])-1 for kl in zip(k, l)]
        dilations = [i+1 for i in holes]

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(
            nn.Conv2d(1, C, 3, padding=dilations[0], dilation=dilations[0]))
        self.conv.extend([nn.Conv2d(C, C, 3,
                                    padding=dilations[i+1],
                                    dilation=dilations[i+1])
                          for i in range(D)])
        self.conv.append(
            nn.Conv2d(C, 1, 3, padding=dilations[-1], dilation=dilations[-1]))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(
                self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        """Compute the forward pass for given put data.

        Args:
        ----
            x (Tensor): A noisy input image.

        Return:
        ------
            x (Tensor): The denoised image.
        """
        D = self.D
        h = F.relu(self.conv[0](x))
        h_buff = []

        for i in range(D//2 - 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))
            h_buff.append(h)

        for i in range(D//2 - 1, D//2 + 1):
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1](h)
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))

        for i in range(D//2 + 1, D):
            j = i - (D//2 + 1) + 1
            torch.backends.cudnn.benchmark = True
            h = self.conv[i+1]((h + h_buff[-j]) / np.sqrt(2))
            torch.backends.cudnn.benchmark = False
            h = F.relu(self.bn[i](h))

        y = self.conv[D+1](h) + x
        return y
