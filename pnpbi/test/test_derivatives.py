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
from pnpbi.util import derivatives
import unittest


class TestDerivatives(unittest.TestCase):

    def test_deriv2dfw(self):
        m, n = 4, 3
        hx, hy = 2, 2

        Lx = np.array([[-1, 1, 0, 0],
                       [0, -1, 1, 0],
                       [0, 0, -1, 1],
                       [0, 0, 0, 0]]) / hx

        Ly = np.array([[-1, 1, 0],
                       [0, -1, 1],
                       [0, 0, 0]]) / hy

        Dx, Dy = derivatives.deriv2dfw(m, n, hx, hy)
        np.testing.assert_allclose(Dx.toarray(), Lx)
        np.testing.assert_allclose(Dy.toarray(), Ly)

    def test_deriv2dfw_one_by_one(self):
        m, n = 1, 1
        hx, hy = 1, 1

        Lx = 0 / hx
        Ly = 0 / hy

        Dx, Dy = derivatives.deriv2dfw(m, n, hx, hy)
        np.testing.assert_allclose(Dx.toarray(), Lx)
        np.testing.assert_allclose(Dy.toarray(), Ly)

    def test_deriv2dfw_apply_to_ones(self):
        m, n = 5, 4
        hx, hy = 2, 2

        # Create derivative operators.
        Dx, Dy = derivatives.deriv2dfw(m, n, hx, hy)

        # Apply to constant matrix.
        f = np.ones((n, m))
        fx = f * Dx.transpose()
        fy = Dy * f

        np.testing.assert_allclose(fx, np.zeros((n, m)))
        np.testing.assert_allclose(fy, np.zeros((n, m)))

    def test_deriv2dfw_adjoint(self):
        m, n = 4, 5
        hx, hy = 2, 2

        # Create derivative operators.
        Dx, Dy = derivatives.deriv2dfw(m, n, hx, hy)

        # Create random matrix.
        f = np.random.rand(n, m)

        # Create second random matrix.
        g = np.random.rand(n, m)

        # Apply to constant matrix.
        fx = f * Dx.transpose()
        fy = Dy * f

        # Compute divergence.
        gx = g * Dx
        gy = Dy.transpose() * g

        np.testing.assert_allclose(np.dot(fx.flatten(), g.flatten()),
                                   np.dot(f.flatten(), gx.flatten()))
        np.testing.assert_allclose(np.dot(fy.flatten(), g.flatten()),
                                   np.dot(f.flatten(), gy.flatten()))

    def test_vecderiv2dfw(self):
        m, n = 4, 3
        hx, hy = 2, 2

        Lx = np.array([[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) / hx

        Ly = np.array([[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) / hy

        Dx, Dy = derivatives.vecderiv2dfw(m, n, hx, hy)
        np.testing.assert_allclose(Dx.toarray(), Lx)
        np.testing.assert_allclose(Dy.toarray(), Ly)

    def test_vecderiv2dfw_adjoint(self):
        m, n = 4, 5
        hx, hy = 2, 2

        # Create derivative operators.
        Dx, Dy = derivatives.vecderiv2dfw(m, n, hx, hy)

        # Create random matrix.
        f = np.random.rand(n * m)

        # Create second random matrix.
        g = np.random.rand(n * m)

        # Apply to constant matrix.
        fx = Dx * f
        fy = Dy * f

        # Compute divergence.
        gx = Dx.transpose() * g
        gy = Dy.transpose() * g

        np.testing.assert_allclose(np.dot(fx.flatten(), g.flatten()),
                                   np.dot(f.flatten(), gx.flatten()))
        np.testing.assert_allclose(np.dot(fy.flatten(), g.flatten()),
                                   np.dot(f.flatten(), gy.flatten()))


if __name__ == '__main__':
    unittest.main()
