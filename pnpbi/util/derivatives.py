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
"""Module provides helper functions for derivatives."""
import numpy as np
import scipy.sparse as sparse


def deriv2dfw(m: int, n: int, hx: float, hy: float) -> (np.array, np.array):
    """Create 2D forward finite difference matrices.

    Takes the number of columns m, the number of rows n, and spatial scaling
    parameters hx and hy, and creates first order forward difference matrices
    Dx and Dy with Neumann boundary conditions.

    The gradient of a matrix f of size [n, m] is then given by

    >>> gradf = np.stack((f * Dx.transpose(), Dy * f), axis=2)

    For v = (v1, v2), the adjoint -div(v) is then given by

    >>> divv = -(v1 * Dx + Dy.transpose() * v2)

    Args:
    ----
        m (int): Number of columns. m > 1.
        n (int): Number of rows. n > 1.
        hx (float): Spatial scaling parameter in column axis.
        hy (float): Spatial scaling parameter in row axis.

    Return:
    ------
        Dx (np.array): Array of shape (m, m).
        Dy (np.array): Array of shape (n, n).
    """
    v1 = -np.ones((1, m)) / hx
    v1[0, m - 1] = 0
    v2 = np.ones((1, m)) / hx
    Dx = sparse.spdiags(np.vstack((v1, v2)), [0, 1], m, m)

    v1 = -np.ones((1, n)) / hy
    v1[0, n - 1] = 0
    v2 = np.ones((1, n)) / hy
    Dy = sparse.spdiags(np.vstack((v1, v2)), [0, 1], n, n)
    return Dx, Dy


def vecderiv2dfw(m: int, n: int, hx: float, hy: float) -> (np.array, np.array):
    """Create 2D forward finite difference matrices for vectorised input.

    Takes the number of columns m, the number of rows n, and spatial scaling
    parameters hx and hy, and creates first order forward difference matrices
    Dx and Dy with Neumann boundary conditions.

    The gradient of a matrix f in vector form, i.e. shape (n*m, 1) is then
    given by

    >>> gradf = np.stack((Dx * f, Dy * f), axis=1)

    For v = (v1, v2), the adjoint -div(v) is then given by

    >>> divv = -(Dx.transpose() * v1 + Dy.transpose() * v2).

    Args:
    ----
        m (int): Number of columns. m > 1.
        n (int): Number of rows. n > 1.
        hx (float): Spatial scaling parameter in column axis.
        hy (float): Spatial scaling parameter in row axis.

    Return:
    ------
        Dx (np.array): Array of shape (n*m, n*m).
        Dy (np.array): Array of shape (n*m, n*m).
    """
    Dx, Dy = deriv2dfw(m, n, hx, hy)
    Dx = sparse.kron(Dx, sparse.identity(n))
    Dy = sparse.kron(sparse.identity(m), Dy)
    return Dx, Dy
