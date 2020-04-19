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
import numpy as np
import torch
import torch.autograd as tag
import unittest

from PIL import Image
from pnpbi.util import radon
from pnpbi.util.torch.operators import LinearOperator


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

    def test_radon2d_cuda(self):
        # Set image size.
        m, n = 39, 23

        # Define angles.
        angles = np.linspace(0, np.pi, 180, False)

        # Check if GPU is available.
        cuda = torch.cuda.is_available()

        # Create operators.
        K, Kadj, ndet = radon.radon2d(m, n, angles, cuda)

        # Apply to dummy image.
        f = K(np.ones((m, n)))
        np.testing.assert_allclose(f.shape[0], angles.size)

        # Apply to dummy data.
        f = Kadj(np.ones_like(f))
        np.testing.assert_allclose(f.shape, (m, n))

    def test_radon2d_adjointness_cuda(self):
        # Set image size.
        m, n = 39, 23

        # Define angles.
        angles = np.linspace(0, np.pi, 180, False)

        # Check if GPU is available.
        cuda = torch.cuda.is_available()

        # Create operators.
        K, Kadj, ndet = radon.radon2d(m, n, angles, cuda)

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

    def test_radon2d_operators_cuda(self):
        # Define image size.
        image_size = (31, 23)

        # Define angles.
        nangles = 180
        angles = np.linspace(0, np.pi, nangles, False)

        # Check if GPU is available.
        cuda = torch.cuda.is_available()
        device = torch.device('cuda') if cuda else 'cpu'

        # Create operators.
        R, Radj, ndet = radon.radon2d(*image_size, angles, cuda)
        data_size = (nangles, ndet)

        # Create instances for use with torch.
        K = radon.RadonTransform(R, Radj, data_size)
        Kadj = radon.BackProjection(R, Radj, image_size)

        # Create random matrix.
        x = torch.randn(1, 1, *image_size).to(device)

        # Create second random matrix.
        y = torch.randn(1, 1, *data_size).to(device)

        # Check adjointness up to certain relative tolerance.
        ip1 = torch.dot(K(x).flatten(), y.flatten())
        ip2 = torch.dot(x.flatten(), Kadj(y).flatten())
        torch.allclose(ip1, ip2)

    def test_LinearOperator_radon_cuda(self):
        # Set image size.
        image_size = 5, 4

        # Define angles.
        nangles = 180
        angles = np.linspace(0, np.pi, nangles, False)

        # Check if GPU is available.
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        # Create operators.
        R, Radj, ndet = radon.radon2d(*image_size, angles, cuda)
        data_size = (nangles, ndet)

        # Create instances for use with torch.
        K = radon.RadonTransform(R, Radj, data_size)
        Kadj = radon.BackProjection(R, Radj, image_size)

        # Apply to dummy input.
        x = torch.randn((1, 1, *image_size), requires_grad=True,
                        dtype=torch.double, device=device)
        f = K(x)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.allclose(x.grad, Kadj(x.new_ones(1, 1, *data_size)))

        def op_fun(x):
            out = LinearOperator.apply(x, K, Kadj)
            return out.sum()

        # Check for anomalies.
        with tag.detect_anomaly():
            x = torch.randn(1, 1, *image_size, requires_grad=True,
                            dtype=torch.double, device=device)
            out = op_fun(x)
            out.backward()

    def test_LinearOperator_radon_gradcheck(self):
        # Set image size.
        image_size = (5, 4)

        # Define angles.
        nangles = 180
        angles = np.linspace(0, np.pi, nangles, False)

        # Create operators.
        R, Radj, ndet = radon.radon2d(*image_size, angles)
        data_size = (nangles, ndet)

        # Create instances for use with torch.
        K = radon.RadonTransform(R, Radj, data_size)
        Kadj = radon.BackProjection(R, Radj, image_size)

        # Apply to dummy input.
        x = torch.randn((1, 1, *image_size), requires_grad=True,
                        dtype=torch.double)
        f = K(x)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.allclose(x.grad, Kadj(x.new_ones(1, 1, *data_size)))

        def op_fun(x):
            out = LinearOperator.apply(x, K, Kadj)
            return out.sum()

        # Check for anomalies.
        with tag.detect_anomaly():
            x = torch.randn(1, 1, *image_size, requires_grad=True,
                            dtype=torch.double)
            out = op_fun(x)
            out.backward()

        # Check numerical gradient up to certain tolerance.
        # Due to inaccuracy of adjoint this check fails.
        x = torch.randn(1, 1, *image_size, requires_grad=True,
                        dtype=torch.double)
        tag.gradcheck(lambda t: K(t), x)


if __name__ == '__main__':
    unittest.main()
