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
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pnpbi.denoise import TvDenoiser
from pnpbi.util import derivatives
import unittest


class TestTvDenoiser(unittest.TestCase):

    def test_denoise(self):
        m, n = 99, 77
        alpha = 1.0

        x = np.random.randn(m, n)

        # Create derivative operators.
        hx, hy = 1, 1
        Dx, Dy = derivatives.vecderiv2dfw(m, n, hx, hy)

        dn = TvDenoiser.TvDenoiser(x, alpha, Dy, Dy)

        niter = 1
        y = dn.denoise(x, niter)
        np.testing.assert_allclose(y.shape, x.shape)
        np.testing.assert_allclose(y, x)

    def test_denoise_img(self):
        x = np.asarray(Image.open('data/cat.jpg').convert('L'), dtype=float)
        x = x / np.max(x)

        # Add noise.
        m, n = x.shape
        x += 0.05 * np.random.randn(m, n)

        # Show image.
        plt.figure()
        plt.imshow(x, cmap='gray')
        plt.show()

        # Create derivative operators.
        hx, hy = 1, 1
        Dx, Dy = derivatives.vecderiv2dfw(m, n, hx, hy)

        alpha = 0.1
        dn = TvDenoiser.TvDenoiser(x, alpha, Dx, Dy)
        niter = 100
        y = dn.denoise(x, niter)

        plt.figure()
        plt.imshow(y, cmap='gray')
        plt.show()


if __name__ == '__main__':
    unittest.main()
