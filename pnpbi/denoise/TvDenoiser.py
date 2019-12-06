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
import numpy as np
import matplotlib.pyplot as plt


class TvDenoiser:
    """A TV-based denoiser using the primal-dual hybrid gradient algorithm."""

    def __init__(self, x: np.array, alpha: float, Dx: np.array, Dy: np.array):
        """Constructor takes an image, a regularisation parameter, and
        derivative operators.

        Args:
            x (np.array): A grey-scale image in matrix form.
            alpha (float): A regularisation parameter.
            Dx (np.array): A matrix.
            Dy (np.array): A matrix.
        """
        self.dims = x.shape
        self.x = x.flatten()
        self.alpha = alpha
        self.Dx = Dx
        self.Dy = Dy

    def __apply_op(self, x):
        return np.stack((self.Dx * x, self.Dy * x), axis=1)

    def __apply_op_adjoint(self, y):
        return self.Dx.transpose() * y[:, 0] + self.Dy.transpose() * y[:, 1]

    def denoise(self, x: np.array, niter: int) -> np.array:
        """Takes an initial image and returns a denoised image.

        Args:
            x0 (np.array): A grey-scale image in matrix form.

        Returns:
            x (np.array): The denoised image in matrix form.
        """
        tau = 1.0 / np.sqrt(8)
        sigma = 1.0 / np.sqrt(8)

        x = x.flatten()
        y = np.zeros((np.prod(self.dims), 2))

        k = 1
        xprev = np.copy(x)
        while(k < niter):

            # Update primal variables.
            x = xprev - tau * self.__apply_op_adjoint(y)
            x = (x + tau * self.x) / (1 + tau)
            xprev = 2 * x - xprev

            # Update dual variables.
            y = y + sigma * self.__apply_op(xprev)
            norm = np.hypot(y[:, 0], y[:, 1])
            y = y / np.maximum(1.0, norm / self.alpha)[:, None]

            # Call logging function.

            # Call termination check function.

            # Increase iteration count.
            k += 1
        return x.reshape(self.dims)
