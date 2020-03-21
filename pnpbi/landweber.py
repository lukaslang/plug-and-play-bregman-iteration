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
"""Module for Landweber iteration."""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pnpbi.util import radon


def landweber():
    """Compute a solution by Landweber iteration."""
    # Load phantom image.
    f = np.asarray(Image.open('data/phantom.png').convert('L'),
                   dtype=np.double)
    f = f / np.max(f)
    m, n = f.shape

    # Define angles.
    angles = np.linspace(0, np.pi, 180, False)

    # Define Radon transform and adjoint.
    K, Kadj, ndet = radon.radon2d(m, n, angles)

    # Generate data and add noise.
    y = K(f)
    ydelta = y + 5 * np.random.randn(*y.shape)

    # Show image.
    plt.figure()
    plt.imshow(ydelta, cmap='gray')
    plt.colorbar()
    plt.show()

    # Initialise solution.
    x = np.zeros_like(f)

    # Define stepsize parameter.
    omega = 2e-5

    # Run Landweber iteration.
    nliter = 30
    for i in range(nliter):

        x = x - omega * Kadj(K(x) - ydelta)

        plt.figure()
        plt.imshow(x, cmap='gray')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    landweber()
