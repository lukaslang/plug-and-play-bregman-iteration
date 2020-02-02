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
"""Trains a CNN image denoiser."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pnpbi.dncnn.data import NoisyBSDSDataset, NoisyCTDataset
from pnpbi.dncnn.model import PnpBi
from pnpbi.util import radon
from pnpbi.util.operators import LinearOperator
from pytorchviz.torchviz import make_dot, make_dot_from_trace

# Define data.
image_dir = './data/BSDS300/images'
image_size = (100, 100)
sigma = 30

# trainset = NoisyBSDSDataset(image_dir, mode='train',
#                             image_size=image_size, sigma=sigma)
# trainset = torch.utils.data.Subset(trainset, list(range(0, 40)))
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = NoisyBSDSDataset(image_dir, mode='test',
#                            image_size=image_size, sigma=sigma)
# testset = torch.utils.data.Subset(testset, list(range(0, 10)))
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)


# # Define data fidelity and its gradient.
# def G(x, y):
#     """Compute data fidelity function."""
#     return torch.sum((x - y) ** 2) / 2


# def gradG(x, y):
#     """Compute gradient of data fidelity function."""
#     return x - y


# Define angles.
nangles = 180
angles = np.linspace(0, np.pi, nangles, False)

# Create operators.
K, Kadj, ndet = radon.radon2d(*image_size, angles)

# Create function.
Op = LinearOperator.apply


def Kfun(x):
    nimg = x.shape[0]
    y = torch.ones((nimg, nangles, ndet))
    for k in range(nimg):
        # y[k] = torch.from_numpy(K(x[k].numpy()))
        y[k] = Op(x[k][0], K, Kadj)
    return y


def Kadjfun(x):
    nimg = x.shape[0]
    y = torch.ones((nimg, *image_size))
    for k in range(nimg):
        # y[k] = torch.from_numpy(Kadj(x[k].numpy()))
        y[k] = Op(x[k][0], Kadj, K)
    return y


trainset = NoisyCTDataset(Kfun, image_dir, mode='train',
                          image_size=image_size, sigma=sigma)
trainset = torch.utils.data.Subset(trainset, list(range(0, 40)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=1)


# Define data fidelity and its gradient.
def G(x, y):
    """Compute data fidelity function."""
    return torch.sum((Kfun(x) - y) ** 2) / 2


def gradG(x, y):
    """Compute gradient of data fidelity function."""
    return Kadjfun(Kfun(x) - y)

    # print('gradG')
    # print(x.shape)
    # nimg = x.shape[0]
    # g = torch.ones((nimg, *image_size))
    # for k in range(nimg):
    #     g[k] = Kadjfun(Kfun(x[k][0]) - y[k][0])
    # print(y.shape)
    # return g.unsqueeze(1)


net = PnpBi(image_size, gradG, tau=1e-1, niter=1)

x = torch.zeros(1, 1, nangles, ndet, requires_grad=False)
#make_dot(net(x), params=dict(net.named_parameters()))
out = net(x)
make_dot(out)  # plot graph of variable, not of a nn.Module


criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print('inputs')
        print(inputs.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('Learned tau is {0:0.3f}'.format(net.tau[0]))
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

PATH = './PnpBi_BSDS300.pth'
torch.save(net.state_dict(), PATH)
