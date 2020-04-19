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
from pnpbi.util.torch import operators
import scipy.sparse as sparse
import numpy as np
import torch
import torch.autograd as tag
import unittest

class NumpyMatrixMultOperator:

    def __init__(self, M):
        self.M = M

    def __call__(self, x):
        x_np = x.detach().numpy()
        res = self.M @ x_np
        return torch.from_numpy(res)

class TorchMatrixMultOperator:

    def __init__(self, M):
        self.M = M

    def __call__(self, x):
        return torch.mm(self.M, x)


class TestOperator(unittest.TestCase):

    def test_LinearOperator_Identity(self):
        # Set image size.
        m, n = 19, 23

        # Define operators.
        def K(x):
            return x

        def Kadj(y):
            return y

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.testing.assert_allclose(Kadj(x.new_ones((m, n))), x.grad)

        # Check gradient.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)

    def test_LinearOperator_Identity_cuda(self):
        # Set image size.
        m, n = 19, 23

        # Define operators.
        def K(x):
            return x

        def Kadj(y):
            return y

        # Create function.
        Op = LinearOperator.apply

        # Check if GPU is available.
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True,
                        dtype=torch.double, device=device)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.testing.assert_allclose(Kadj(x.new_ones((m, n))), x.grad)

        # Check gradient.
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)

    def test_LinearOperator_Matrix(self):
        # Set image size.
        m, n = 19, 23

        # Create an underdetermined system.
        p = 5
        M = sparse.spdiags(np.ones((p, )), 0, p, m)

        # Define operators.
        K = NumpyMatrixMultOperator(M)
        Kadj = NumpyMatrixMultOperator(M.T)

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.testing.assert_allclose(Kadj(x.new_ones((p, n))), x.grad)

        # Check gradient.
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)

    def test_LinearOperator_Matrix_Transpose(self):
        # Set image size.
        m, n = 21, 37

        # Create an underdetermined system.
        p = 7
        M = sparse.spdiags(np.ones((p, )), 0, p, m)

        # Define operators.
        K = NumpyMatrixMultOperator(M)
        Kadj = NumpyMatrixMultOperator(M.T)

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(p, n, requires_grad=True, dtype=torch.double)
        f = Op(x, Kadj, K)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.testing.assert_allclose(K(x.new_ones((m, n))), x.grad)

        # Check gradient.
        tag.gradcheck(lambda t: Op(t, Kadj, K), x)

    def test_LinearOperator_Random_Matrix(self):
        # Set image size.
        m, n = 19, 23

        # Create an underdetermined random system.
        p = 5
        M = sparse.rand(p, m, density=0.1, dtype=np.double)

        # Define operators.
        K = NumpyMatrixMultOperator(M)
        Kadj = NumpyMatrixMultOperator(M.T)

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.testing.assert_allclose(Kadj(x.new_ones((p, n))), x.grad)

        # Check gradient.
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)

    def test_LinearOperator_Random_Matrix_torch(self):
        # Set image size.
        m, n = 19, 23

        # Create an underdetermined random system.
        p = 5
        M = torch.randn(p, m, dtype=torch.double)

        # Define operators.
        K = TorchMatrixMultOperator(M)
        Kadj = TorchMatrixMultOperator(M.T)

        # Create function.
        Op = LinearOperator.apply

        # Apply to dummy input.
        x = torch.randn(m, n, requires_grad=True, dtype=torch.double)
        f = Op(x, K, Kadj)

        # Check for simple loss.
        loss = f.sum()
        loss.backward()
        torch.testing.assert_allclose(Kadj(x.new_ones((p, n))), x.grad)

        # Check gradient.
        tag.gradcheck(lambda t: Op(t, K, Kadj), x)


if __name__ == '__main__':
    unittest.main()
