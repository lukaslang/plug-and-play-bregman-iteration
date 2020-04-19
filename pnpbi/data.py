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
"""Provides datasets for denoising training."""
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from pnpbi.util import TvDenoiser
from pnpbi.util import derivatives
import matplotlib.pyplot as plt
from pnpbi.util import radon
import pydicom
from pydicom.data import get_testdata_files


class NoisyDataset(Dataset):
    """Noisy images dataset."""

    def __init__(self, train, sigma=30):
        self.sigma = sigma
        transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        self.dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=train,
                                                    download=True,
                                                    transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clean, label = self.dataset.__getitem__(idx)
        noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)
        return noisy, clean


class NoisyCTDataset(Dataset):

    def __init__(self, K, root_dir, mode='train', image_size=(100, 100), sigma=0):
        super(NoisyCTDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)
        self.K = K

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyCTDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.image_size),
            # convert it to a tensor
            transforms.ToTensor(),
            # normalize it to the range [0, 1]
            transforms.Normalize([0], [1])
        ])
        clean = transform(clean)

        # Generate data and add noise.
        data = self.K(clean.unsqueeze(0)).squeeze(0)
        noisy = data + self.sigma**2 * data.var() * data.max() * torch.randn(data.shape)

        return noisy, clean

class NoisyDCIMDataset(Dataset):

    def __init__(self, K, root_dir, mode='train', image_size=(100, 100), sigma=0):
        super(NoisyDCIMDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)
        self.K = K

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        dataset = pydicom.dcmread(img_path, force=True)
        dataset.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        clean = Image.fromarray(dataset.pixel_array)

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.image_size),
            # convert it to a tensor
            transforms.ToTensor(),
            # normalize it to the range [0, 1]
            transforms.Normalize([0], [1])
        ])
        clean = transform(clean)

        # Generate data and add noise.
        data = self.K(clean.unsqueeze(0)).squeeze(0)
        noisy = data + self.sigma**2 * data.var() * data.max() * torch.randn(data.shape)

        return noisy, clean

class NoisyBSDSDataset(Dataset):

    def __init__(self, root_dir, mode='train', image_size=(100, 100), sigma=30):
        super(NoisyBSDSDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert('RGB')
        # random crop
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])

        clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])
        transform = transforms.Compose([
            transforms.Grayscale(),
            # convert it to a tensor
            transforms.ToTensor(),
            # normalize it to the range [−1, 1]
            transforms.Normalize([.5], [.5])
        ])
        clean = transform(clean)

        noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)
        return noisy, clean


class NoisyBSDSDatasetTV(Dataset):

    def __init__(self, root_dir, mode='train', image_size=(100, 100),
                 sigma=30, alpha=0.1):
        super(NoisyBSDSDatasetTV, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

        self.alpha = alpha

        # Create derivative operators.
        self.Dx, self.Dy = derivatives.vecderiv2dfw(*image_size, 1, 1)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean = Image.open(img_path).convert('RGB')
        # random crop
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])

        clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])

        transform = transforms.Compose([
            transforms.Grayscale(),
            # convert it to a tensor
            transforms.ToTensor(),
            # normalize it to the range [−1, 1]
            transforms.Normalize([.5], [.5])
        ])
        clean = transform(clean)

        denoiser = TvDenoiser.TvDenoiser(np.asarray(clean)[0], self.alpha,
                                         self.Dx, self.Dy)

        niter = 100
        clean_tv = denoiser.denoise(np.asarray(clean)[0], niter)
        clean_tv = torch.from_numpy(clean_tv).unsqueeze(0).float()

        noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)
        return noisy, clean_tv
