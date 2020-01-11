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
"""Linearised Bregman iteration for Radon transform data."""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pnpbi.util import TvDenoiser
from pnpbi.util import derivatives
from pnpbi.util import radon


def linBregmanIteration():
    """Compute linearised Bregman iteration for Radon inversion problem."""
    # Load phantom image.
    n = 256
    f = np.asarray(Image.open('data/brain.png')
                   .convert('L')
                   .resize((n, n)), dtype=float)
    f = f / np.max(f)
    m, n = f.shape

    # Define angles.
    angles = np.linspace(0, np.pi, 180, False)

    # Define Radon transform and adjoint.
    K, Kadj = radon.radon2d(m, n, angles)

    # Generate data and add noise.
    y = K(f)
    ydelta = y + 0.05 * np.random.randn(*y.shape)

    # Define data fidelity and its gradient.
    def G(x: np.array) -> np.array:
        """Compute data fidelity function."""
        return np.sum((K(x) - ydelta) ** 2) / 2

    def gradG(x: np.array) -> np.array:
        """Compute gradient of data fidelity function."""
        return Kadj(K(x) - ydelta)

    # Show image.
    plt.figure()
    plt.imshow(ydelta, cmap='gray')
    plt.show()

    # Create derivative operators.
    Dx, Dy = derivatives.vecderiv2dfw(m, n, 1, 1)

    # Initialise solution.
    x = np.zeros_like(f)

    # Initialise data.
    tau = 2e-5
    x = np.zeros_like(f)
    w = - tau * gradG(x)

    # Define regularisation parameter.
    alpha = 5

    # Run Bregman iteration.
    nbiter = 100
    for i in range(nbiter):

        # Define denoiser.
        dn = TvDenoiser.TvDenoiser(w, alpha * tau, Dx, Dy)
        niter = 100

        # Denoise.
        x = dn.denoise(np.zeros_like(w), niter)

        plt.figure()
        ax = plt.subplot(2, 2, 1)
        plt.imshow(f, cmap='gray')
        plt.colorbar()
        ax.set_title('f')
        ax = plt.subplot(2, 2, 2)
        plt.imshow(w, cmap='gray')
        plt.colorbar()
        ax.set_title('w')
        ax = plt.subplot(2, 2, 3)
        plt.imshow(x, cmap='gray')
        plt.colorbar()
        ax.set_title('x')
        ax = plt.subplot(2, 2, 4)
        plt.imshow(w - x, cmap='gray')
        plt.colorbar()
        ax.set_title('w - x')
        plt.tight_layout()
        plt.show()
        plt.close()

        # Update w.
        w -= tau * gradG(x)


if __name__ == '__main__':
    linBregmanIteration()
