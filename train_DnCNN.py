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
import torch
import torch.nn as nn
import torch.optim as optim
from pnpbi.dncnn.data import NoisyBSDSDataset, NoisyBSDSDatasetTV
from pnpbi.dncnn.model import DnCNN, DUDnCNN

# Define data.
image_dir = './data/BSDS300/images'
image_size = (128, 128)
sigma = 30
alpha = 1

# trainset = NoisyBSDSDataset(image_dir, mode='train',
#                             image_size=image_size, sigma=sigma)
trainset = NoisyBSDSDatasetTV(image_dir, mode='train',
                              image_size=image_size, sigma=sigma, alpha=alpha)
# trainset = torch.utils.data.Subset(trainset, list(range(0, 100)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# testset = NoisyBSDSDataset(image_dir, mode='test',
#                           image_size=image_size, sigma=sigma)
testset = NoisyBSDSDatasetTV(image_dir, mode='test',
                             image_size=image_size, sigma=sigma, alpha=alpha)
# testset = torch.utils.data.Subset(testset, list(range(0, 10)))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


# net = DnCNN()
net = DUDnCNN()

criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

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
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

PATH = './DnCNN_BSDS300.pth'
torch.save(net.state_dict(), PATH)