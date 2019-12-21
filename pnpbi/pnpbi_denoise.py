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


def linBregmanIteration():
    """Compute linearised Bregman iteration for denoising problem."""
    # Load phantom image.
    n = 256
    f = np.asarray(Image.open('data/brain.png')
                   .convert('L')
                   .resize((n, n)), dtype=float)
    f = f / np.max(f)
    m, n = f.shape

    # Add noise.
    m, n = f.shape
    fdelta = f + 0.05 * np.random.randn(m, n)

    # Create derivative operators.
    Dx, Dy = derivatives.vecderiv2dfw(m, n, 1, 1)

    # Define data fidelity and its gradient.
    def G(x: np.array, y: np.array) -> np.array:
        """Compute data fidelity function."""
        return np.sum((x - y)**2) / 2

    def gradG(x: np.array, y: np.array) -> np.array:
        """Compute gradient of data fidelity function."""
        return x - y

    # Initialise data.
    tau = 0.01
    x = np.zeros_like(fdelta)
    w = - tau * gradG(x, fdelta)

    # Define regularisation parameter.
    alpha = 1

    # Run Bregman iteration.
    nbiter = 20
    for i in range(nbiter):

        # Define denoiser.
        dn = TvDenoiser.TvDenoiser(w, alpha * tau, Dx, Dy)
        niter = 100

        # Denoise.
        x = dn.denoise(x, niter)

        # Update w.
        w -= tau * gradG(x, fdelta)

        plt.figure()
        ax = plt.subplot(2, 2, 1)
        plt.imshow(f, cmap='gray')
        ax.set_title('f')
        ax = plt.subplot(2, 2, 2)
        plt.imshow(w, cmap='gray')
        ax.set_title('w')
        ax = plt.subplot(2, 2, 3)
        plt.imshow(x, cmap='gray')
        ax.set_title('x')
        ax = plt.subplot(2, 2, 4)
        plt.imshow(w - x, cmap='gray')
        ax.set_title('w - x')
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    linBregmanIteration()
