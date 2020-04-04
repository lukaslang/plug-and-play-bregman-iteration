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
from pnpbi.util import radon
from PIL import Image
import unittest
import matplotlib.pyplot as plt


class TestRadon(unittest.TestCase):

    def test_radon2d(self):
        # Set image size.
        m, n = 39, 23

        # Define angles.
        angles = np.linspace(0, np.pi, 180, False)

        # Create operators.
        K, Kadj, ndet = radon.radon2d(m, n, angles)

        # Apply to dummy image.
        f = K(np.ones((m, n)))
        np.testing.assert_allclose(f.shape[0], angles.size)

        # Apply to dummy data.
        f = Kadj(np.ones_like(f))
        np.testing.assert_allclose(f.shape, (m, n))

    def test_radon2d_adjointness(self):
        # Set image size.
        m, n = 39, 23

        # Define angles.
        angles = np.linspace(0, np.pi, 180, False)

        # Create operators.
        K, Kadj, ndet = radon.radon2d(m, n, angles)

        # Apply to dummy image.
        f = K(np.ones((m, n)))
        np.testing.assert_allclose(f.shape[0], angles.size)

        # Apply to dummy data.
        f = Kadj(np.ones_like(f))
        np.testing.assert_allclose(f.shape, (m, n))

        # Create random matrix.
        x = np.random.rand(m, n)
        p, q = K(x).shape

        # Create second random matrix.
        y = np.random.rand(p, q)

        # Check adjointness up to certain relative tolerance.
        np.testing.assert_allclose(np.dot(K(x).flatten(), y.flatten()),
                                   np.dot(x.flatten(), Kadj(y).flatten()),
                                   1e-3)

    def test_radon2d_plot(self):

        img = Image.open('data/phantom.png').convert('L')
        img = np.array(img)

        # Set image size.
        m, n = img.shape

        # Define angles.
        angles = np.linspace(0, np.pi, 180, False)

        # Create operators.
        K, Kadj, ndet = radon.radon2d(m, n, angles)

        # Plot original image.
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.show()

        # Apply to dummy image.
        f = K(img)

        # Plot data.
        plt.imshow(f, cmap='gray')
        plt.colorbar()
        plt.show()

        # Apply to dummy data.
        f = Kadj(f)

        # Plot result of backprojection.
        plt.imshow(f, cmap='gray')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    unittest.main()
