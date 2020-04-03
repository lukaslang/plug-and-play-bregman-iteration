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
"""Provides helper functions to set up problems."""
import numpy as np
from pnpbi.util import functionals
from pnpbi.util import radon


def setup_denoising_problem(f: np.array, sigma: float):
    """Set up denoising problem."""
    # Generate data and add noise.
    ydelta = f + sigma**2 * f.var() * f.max() * np.random.randn(*f.shape)

    # Define data fidelity and its gradient.
    G, gradG = functionals.SqNormDataTerm(ydelta)

    return ydelta, G, gradG


def setup_reconstruction_problem(f: np.array, sigma: float):
    """Set up reconstruction problem using Radon transform."""
    m, n = f.shape

    # Define angles.
    nangles = 180
    angles = np.linspace(0, np.pi, nangles, False)

    # Define Radon transform and adjoint.
    K, Kadj, ndet = radon.radon2d(m, n, angles)

    # Generate data and add noise.
    y = K(f)

    # Add noise.
    ydelta = y + sigma**2 * y.var() * y.max() * np.random.randn(*y.shape)

    # Define data fidelity and its gradient.
    G, gradG = functionals.OpSqNormDataTerm(K, Kadj, ydelta)

    return ydelta, G, gradG
