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
from pnpbi.util import radon
from pnpbi.util.torch.operators import LinearOperator
import scipy.sparse as sparse
import numpy as np
import torch
import torch.autograd as tag
import unittest


class TestOperator(unittest.TestCase):

    def test_LinearOperator_Matrix(self):
        # Set image size.
        m, n = 19, 23

        # Create an underdetermined system.
        p = 5
        M = sparse.spdiags(np.ones((p, )), 0, p, m)

        def K(x):
            return M @ x

        def Kadj(y):
            return M.T @ y

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), Kadj(np.ones((p, n))))

        # Check gradient.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)

    def test_LinearOperator_Matrix_Transpose(self):
        # Set image size.
        m, n = 21, 37

        # Create an underdetermined system.
        p = 7
        M = sparse.spdiags(np.ones((p, )), 0, p, m)

        def K(x):
            return M @ x

        def Kadj(y):
            return M.T @ y

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(p, n, requires_grad=True, dtype=torch.double)
        f = Op(x, Kadj, K)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), K(np.ones((m, n))))

        # Check gradient.
        x = torch.randn(p, n, requires_grad=True, dtype=torch.double)
        tag.gradcheck(lambda t: Op(t, Kadj, K), x)

    def test_LinearOperator_Random_Matrix(self):
        # Set image size.
        m, n = 19, 23

        # Create an underdetermined random system.
        p = 5
        M = sparse.rand(p, m, density=0.1, dtype=np.double)

        def K(x):
            return M @ x

        def Kadj(y):
            return M.T @ y

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), Kadj(np.ones((p, n))))

        # Check gradient.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)

    def test_LinearOperator_radon(self):
        # Set image size.
        m, n = 5, 4

        # Define angles.
        nangles = 180
        angles = np.linspace(0, np.pi, nangles, False)

        # Check if GPU is available.
        cuda = torch.cuda.is_available()

        # Create operators.
        K, Kadj, ndet = radon.radon2d(m, n, angles, cuda)

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(),
                                   Kadj(np.ones((nangles, ndet))))

        def op_fun(x):
            out = LinearOperator.apply(x, K, Kadj)
            return out.sum()

        # Check for anomalies.
        with tag.detect_anomaly():
            x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
            out = op_fun(x)
            out.backward()

    def test_LinearOperator_radon_gradcheck(self):
        # Set image size.
        m, n = 5, 4

        # Define angles.
        nangles = 180
        angles = np.linspace(0, np.pi, nangles, False)

        # Create operators.
        K, Kadj, ndet = radon.radon2d(m, n, angles)

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(),
                                   Kadj(np.ones((nangles, ndet))))

        def op_fun(x):
            out = LinearOperator.apply(x, K, Kadj)
            return out.sum()

        # Check for anomalies.
        with tag.detect_anomaly():
            x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
            out = op_fun(x)
            out.backward()

        # Check numerical gradient up to certain tolerance.
        # Due to inaccuracy of adjoint this check fails.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)


if __name__ == '__main__':
    unittest.main()
