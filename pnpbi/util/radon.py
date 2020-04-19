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
"""Encapsulates use of ASTRA Toolbox."""
import astra
import numpy as np
import torch

from pnpbi.util.torch.operators import LinearOperator


class RadonTransform:
    """Radon transform class to use with torch. Runs on CPU."""

    def __init__(self, R, Radj, data_size: tuple):
        """Constructor.

        Args:
        ----
            R (function): A function that computes the 2D Radon transform for
            a numpy array.

            Radj (function): A function that computes the backprojection for
            a numpy array.

            data_size (tuple): The data size, i.e. (nangles, ndet).
        """
        def Rfun(x: torch.Tensor) -> torch.Tensor:
            x_np = x.detach().cpu().numpy()
            y_np = R(x_np)
            return x.new(y_np)

        def Radjfun(y: torch.Tensor) -> torch.Tensor:
            y_np = y.numpy()
            x_np = Radj(y_np)
            return y.new(x_np)

        self.R = Rfun
        self.Radj = Radjfun
        self.data_size = data_size

        # Create torch function for linear operator.
        self.Op = LinearOperator.apply

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 2D Radon transform for a torch.Tensor.

        Args:
        ----
            x (torch.Tensor): A tensor of shape (k, 1, *image_size).

        Return:
        ------
            x (torch.Tensor): A tensor of shape (k, 1, *data_size).
        """
        # Number of images.
        nimg = x.shape[0]

        # Create output tensor.
        y = x.new_ones((nimg, 1, *self.data_size))

        # Apply Radon transform for each image.
        for k in range(nimg):
            y[k][0] = self.Op(x[k][0], self.R, self.Radj)

        # Return result.
        return y


class BackProjection:
    """Backprojection class to use with torch. Runs on CPU."""

    def __init__(self, R, Radj, image_size: tuple):
        """Constructor.

        Args:
        ----
            R (function): A function that computes the 2D Radon transform for
            a numpy array.

            Radj (function): A function that computes the backprojection for
            a numpy array.

            image_size (tuple): The size of the original image, i.e. (m, n).
        """
        def Rfun(x: torch.Tensor) -> torch.Tensor:
            x_np = x.detach().cpu().numpy()
            y_np = R(x_np)
            return x.new(y_np)

        def Radjfun(y: torch.Tensor) -> torch.Tensor:
            y_np = y.detach().cpu().numpy()
            x_np = Radj(y_np)
            return y.new(x_np)

        self.R = Rfun
        self.Radj = Radjfun
        self.image_size = image_size

        # Create torch function for linear operator.
        self.Op = LinearOperator.apply

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 2D backprojection for a torch.Tensor.

        Args:
        ----
            x (torch.Tensor): A tensor of shape (k, 1, *data_size).

        Return:
        ------
            x (torch.Tensor): A tensor of shape (k, 1, *image_size).
        """
        # Number of images.
        nimg = x.shape[0]

        # Create output tensor.
        y = x.new_ones((nimg, 1, *self.image_size))

        # Apply Radon transform for each image.
        for k in range(nimg):
            y[k][0] = self.Op(x[k][0], self.Radj, self.R)

        # Return result.
        return y


def radon2d(m: int, n: int, angles: np.array, cuda=False):
    """Create operator for the 2D Radon transform.

    K takes an array of shape (m, n) and returns an array of shape
    (nangles, ndet), where nangles is the number of measurement angles, i.e.
    angles.size, and ndet is the number of detectors, calculated as:

    >>> ndet = np.ceil(np.hypot(m, n)).astype(int)

    Args:
    ----
        m (int): Number of rows. m > 1.
        n (int): Number of columns. n > 1.
        angles (np.array): An array specifying measurement angles in rad in the
        range [0, pi].
        cuda (bool): Uses GPU (default is device 0) if True.

    Return:
    ------
        K: A function that takes an array of shape (m, n).
        Kadj: A function that takes an array of shape (nangles, ndet).
        ndet: The number of detectors used.
    """
    ndet = np.ceil(1.5 * np.hypot(m, n)).astype(int)
    det_spacing = 1
    vol_geom = astra.create_vol_geom(m, n)
    proj_geom = astra.create_proj_geom('parallel',
                                       det_spacing, ndet, angles)

    proj_type = 'cuda' if cuda else 'linear'
    alg = 'BP_CUDA' if cuda else 'BP'

    def K(x: np.array) -> np.array:
        # Create sinogram.
        proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)
        sino_id, sino = astra.create_sino(x, proj_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)
        return sino

    def Kadj(sino: np.array) -> np.array:
        # Create objects for the reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom)
        sino_id = astra.data2d.create('-sino', proj_geom, sino)
        proj_id = astra.create_projector(proj_type, proj_geom, vol_geom)

        # Set up the parameters for the backpropagation reconstruction.
        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)

        # Run algorithm.
        astra.algorithm.run(alg_id)

        # Get the result
        rec = astra.data2d.get(rec_id)

        # Clean up and return result.
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)
        return rec

    return K, Kadj, ndet
