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
"""Provides helpers to encapsulate functionals and their derivatives."""
import numpy as np


def OpSqNormDataTerm(K, Kadj, ydelta):
    """Return function handles for squared norm data fidelity functional.

    A linear operator and its adjoint are given.

    Args:
    ----
        K, Kadj: Function handles for a linear operator and its adjoint.
        ydelta (np.array): The (noisy) data as an array of shape (o, p).

    Return:
    ------
        G: A function that takes an array of shape (m, n) and returns a scalar.
        gradG: A function that takes an array of shape (m, n) and returns an
            array of shape (m, n).
    """
    # Define data fidelity.
    def G(x: np.array) -> np.array:
        """Compute data fidelity function."""
        return np.sum((K(x) - ydelta) ** 2) / 2

    # Define gradient of data fidelity.
    def gradG(x: np.array) -> np.array:
        """Compute gradient of data fidelity function."""
        return Kadj(K(x) - ydelta)

    # Return both functions.
    return G, gradG


def SqNormDataTerm(ydelta):
    """Return function handles for squared norm data fidelity functional.

    Args:
    ----
        ydelta (np.array): The (noisy) data as an array of shape (o, p).

    Return:
    ------
        G: A function that takes an array of shape (m, n) and returns a scalar.
        gradG: A function that takes an array of shape (m, n) and returns an
            array of shape (m, n).
    """
    # Define data fidelity.
    def G(x: np.array) -> np.array:
        """Compute data fidelity function."""
        return np.sum((x - ydelta)**2) / 2

    # Defint gradient of data fidelity.
    def gradG(x: np.array) -> np.array:
        """Compute gradient of data fidelity function."""
        return x - ydelta

    # Return both functions.
    return G, gradG
