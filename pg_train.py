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
import torch.nn as nn
import torch.optim as optim
import torchvision
from pnpbi.dncnn.data import NoisyCTDataset
from pnpbi.dncnn import model
from pnpbi.util.torch import helper

# Define data.
image_dir = './data/phantom/images'
image_size = (50, 50)

# Set noise level.
sigma = 0.01

# Define path for model.
model_path = './pg_phantom.pth'


def imshow(img):
    """Plot images."""
    # Unnormalise for plotting.
    # img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.colorbar()


# Set up operators, functional, and gradient.
pb = helper.setup_reconstruction_problem(image_size)
Kfun, Kadjfun, G, gradG, data_size = pb

# Define training set.
trainset = NoisyCTDataset(Kfun, image_dir, mode='train',
                          image_size=image_size, sigma=sigma)
trainset = torch.utils.data.Subset(trainset, list(range(0, 100)))
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
# denoising_model = model.DUDnCNN(D=6, C=64)
net = model.PG(denoising_model, image_size, gradG=gradG, tau=2e-5, niter=3)
# net.load_state_dict(torch.load(model_path))
net.train()

# Define optimisation problem.
criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.05)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Define training parameters.
num_epochs = 50
print_every = 5
train_losses, test_losses = [], []

# Train model.
for epoch in range(num_epochs):
    running_loss = 0.0
    for step, data in enumerate(trainloader, 0):
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
        if step % print_every == 0:    # print every 10 mini-batches
            test_loss = 0
            net.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    outputs = net(inputs)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()

            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))

            # Print losses.
            print(f"Epoch {epoch + 1}/{num_epochs} "
                  f"Training loss: {running_loss / print_every:.5f} "
                  f"Validation loss: {test_loss / len(testloader):.5f} ")
            print(f"Learned tau is {net.tau:.3e}")

            # Plot loss.
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(train_losses, label='Training loss')
            plt.plot(test_losses, label='Validation loss')
            plt.legend(frameon=False)

            # Plot first result.
            with torch.no_grad():
                inputs, labels = next(iter(testloader))
                outputs = net(inputs)

                plt.subplot(2, 1, 2)
                imshow(torchvision.utils.make_grid(torch.cat((labels,
                                                              outputs), 2),
                                                   normalize=True))

            plt.show()

            running_loss = 0.0
            net.train()

print('Finished Training')
torch.save(net.state_dict(), model_path)
